# Gemma3
import torch
import os
import io
import json
import logging
from enum import Enum
from typing import List, Any, Dict, Optional, Union, Tuple
from PIL import Image
from datasets import Dataset
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, Gemma3ForCausalLM
from transformers import TrainingArguments, Trainer
from transformers import logging as transformers_logging
from peft import LoraConfig, get_peft_model, PeftModel
from pydantic import BaseModel, Field, validator
from src.finetuning_scripts.diagnostic_scripts.gemma_trainer_diagnostic import patch_model_for_tied_weights
from src.llm_callers.llm_providers import CustomLLMProvider
from src.models.data_collators.gemma_three_collator import Gemma3DataCollator

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
transformers_logging.set_verbosity_error()


class TrainingMode(Enum):
    """Training mode enumeration for different fine-tuning strategies."""
    NONE = "none"
    LORA = "lora"
    FULL = "full"
    FROZEN = "frozen"
    DONUT_STYLE = "donut_style"
    VISION_ONLY = "vision_only"


class LoraTrainingConfig(BaseModel):
    """Configuration for LoRA training parameters.

    Example:
        config = LoraTrainingConfig(
            target_layers="all",
            target_types=["attention", "mlp"],
            rank=32,
            alpha=64
        )
    """
    target_layers: str = Field(default="all", description="Which layers to target: 'all', 'vision', 'language'")
    target_types: List[str] = Field(default=["attention", "mlp"], description="Types of modules to include")
    rank: int = Field(default=16, ge=1, le=256, description="LoRA rank")
    alpha: int = Field(default=32, ge=1, description="LoRA alpha parameter")
    dropout: float = Field(default=0.1, ge=0.0, le=1.0, description="LoRA dropout rate")
    n_layers: Optional[int] = Field(default=None, description="Number of layers for 'last_n' mode")
    pattern_matching: Optional[Dict[str, List[str]]] = Field(default=None, description="Custom pattern matching")

    class Config:
        extra = "forbid"


class TrainingParameters(BaseModel):
    """Training parameters with mode-specific recommendations.

    Example:
        params = TrainingParameters(
            learning_rate=1e-4,
            batch_size=8,
            num_epochs=5,
            gradient_accumulation_steps=2
        )
    """
    learning_rate: float = Field(default=2e-4, gt=0, description="Learning rate")
    batch_size: int = Field(default=4, ge=1, description="Training batch size")
    gradient_accumulation_steps: int = Field(default=4, ge=1, description="Gradient accumulation steps")
    num_epochs: int = Field(default=3, ge=1, description="Number of training epochs")
    warmup_ratio: float = Field(default=0.1, ge=0.0, le=1.0, description="Warmup ratio")
    weight_decay: float = Field(default=0.01, ge=0.0, description="Weight decay")
    lr_scheduler_type: str = Field(default="cosine", description="Learning rate scheduler type")

    class Config:
        extra = "forbid"


class ModelInfo(BaseModel):
    """Comprehensive model information structure."""
    model_name: str
    training_mode: str
    is_finetuned: bool
    device: str
    torch_dtype: str
    use_vision: bool
    total_parameters: int
    trainable_parameters: int
    trainable_ratio: float
    unfrozen_modules_count: int
    max_input_sequence_length: int
    max_output_length: int


class FineTuningStrategy:
    """Strategy class for different fine-tuning approaches."""

    @staticmethod
    def apply_donut_style(model: Union[Gemma3ForConditionalGeneration, Gemma3ForCausalLM],
                          logger: logging.Logger) -> List[str]:
        """Apply Donut-style training: freeze text decoder, train vision encoder.

        This strategy is optimal for document transcription tasks where we want to
        adapt the vision components while keeping the language model frozen.

        :param model: The Gemma3 model to configure
        :type model: Union[Gemma3ForConditionalGeneration, Gemma3ForCausalLM]
        :param logger: Logger instance for reporting
        :type logger: logging.Logger
        :return: List of unfrozen module names
        :rtype: List[str]

        Example:
            unfrozen_modules = FineTuningStrategy.apply_donut_style(
                model=my_model,
                logger=logging.getLogger(__name__)
            )
        """
        total_params = 0
        trainable_params = 0

        # First freeze everything
        for param in model.parameters():
            param.requires_grad = False
            total_params += param.numel()

        # Define vision-related patterns
        vision_patterns = [
            'vision_model', 'vision_tower', 'visual_encoder', 'vision_processor',
            'image_processor', 'patch_embed', 'vision_proj', 'visual_projection',
            'mm_projector', 'cross_attn', 'adapter'
        ]

        unfrozen_modules = []
        for name, param in model.named_parameters():
            if any(pattern in name.lower() for pattern in vision_patterns):
                param.requires_grad = True
                trainable_params += param.numel()
                unfrozen_modules.append(name)

        # Also unfreeze the final projection layer for better adaptation
        for name, param in model.named_parameters():
            if 'lm_head' in name.lower() or 'output_projection' in name.lower():
                param.requires_grad = True
                trainable_params += param.numel()
                unfrozen_modules.append(name)

        logger.info(f"Donut-style training setup:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Trainable ratio: {trainable_params / total_params:.2%}")
        logger.info(f"  Unfrozen modules: {len(unfrozen_modules)}")

        return unfrozen_modules

    @staticmethod
    def apply_vision_only(model: Union[Gemma3ForConditionalGeneration, Gemma3ForCausalLM],
                          logger: logging.Logger) -> List[str]:
        """Apply vision-only training: only train pure vision components.

        More aggressive than donut_style - only unfreezes core vision components.

        :param model: The Gemma3 model to configure
        :type model: Union[Gemma3ForConditionalGeneration, Gemma3ForCausalLM]
        :param logger: Logger instance for reporting
        :type logger: logging.Logger
        :return: List of unfrozen module names
        :rtype: List[str]
        """
        total_params = 0
        trainable_params = 0

        # Freeze everything first
        for param in model.parameters():
            param.requires_grad = False
            total_params += param.numel()

        # Only unfreeze pure vision components
        vision_only_patterns = [
            'vision_model', 'vision_tower', 'visual_encoder',
            'patch_embed', 'vision_proj', 'visual_projection'
        ]

        unfrozen_modules = []
        for name, param in model.named_parameters():
            if any(pattern in name.lower() for pattern in vision_only_patterns):
                param.requires_grad = True
                trainable_params += param.numel()
                unfrozen_modules.append(name)

        logger.info(f"Vision-only training setup:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Trainable ratio: {trainable_params / total_params:.2%}")
        logger.info(f"  Unfrozen modules: {len(unfrozen_modules)}")

        return unfrozen_modules


class Gemma3Provider(CustomLLMProvider):
    """Enhanced Gemma 3 provider with multiple training strategies including Donut-style.

    Supports flexible training configurations from LoRA to full fine-tuning,
    with specialized Donut-style training for document transcription tasks.

    :param model_name: Hugging Face model identifier
    :type model_name: str
    :param device: Device placement - "auto", "cuda", "mps", or "cpu"
    :type device: str
    :param torch_dtype: Model precision (bfloat16 recommended for CUDA)
    :type torch_dtype: torch.dtype
    :param use_vision: Enable multimodal capabilities
    :type use_vision: bool
    :param enable_pan_scan: Enable adaptive image cropping
    :type enable_pan_scan: bool
    :param lora_adapter: Path to pre-trained LoRA adapter
    :type lora_adapter: Optional[str]
    :param cache_dir: Custom cache directory
    :type cache_dir: Optional[str]
    :param max_input_sequence: Maximum input sequence length
    :type max_input_sequence: int
    :param max_output_length: Maximum generation length
    :type max_output_length: int

    Example:
        # Basic usage for inference
        provider = Gemma3Provider(
            model_name="google/gemma-3-1b-it",
            device="cuda",
            use_vision=True
        )

        # Donut-style training for document transcription
        provider.prepare_for_training(training_mode="donut_style")
        trainer = provider.train(
            training_data=training_data,
            output_dir="./donut-gemma-model"
        )
    """

    def __init__(self,
                 model_name: str = "google/gemma-3-1b-it",
                 device: str = "auto",
                 torch_dtype: torch.dtype = torch.bfloat16,
                 use_vision: bool = True,
                 enable_pan_scan: bool = True,
                 lora_adapter: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 max_input_sequence: int = 2048,
                 max_output_length: int = 512) -> None:

        self.model_name = model_name
        self.torch_dtype = torch_dtype
        self.use_vision = use_vision
        self.enable_pan_scan = enable_pan_scan
        self.cache_dir = cache_dir
        self.lora_adapter_path = lora_adapter
        self.max_input_sequence_length = max_input_sequence
        self.max_output_length = max_output_length

        # Model components
        self.model: Optional[Union[Gemma3ForConditionalGeneration, Gemma3ForCausalLM]] = None
        self.processor: Optional[AutoProcessor] = None
        self.is_finetuned: bool = False
        self.training_mode: TrainingMode = TrainingMode.NONE
        self.unfrozen_modules: List[str] = []

        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.device = self._setup_device(device=device)

        # Initialize model
        self._load_model()

        # Load adapter if provided
        if self.lora_adapter_path:
            self.load_finetuned_model(model_path=self.lora_adapter_path)

    def _setup_device(self, device: str) -> str:
        """Setup device with CUDA, MPS, and CPU support.

        :param device: Device preference string - "auto", "cuda", "mps", or "cpu"
        :type device: str
        :return: Validated device string
        :rtype: str
        """
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"

        # Validate requested device
        if device == "cuda" and not torch.cuda.is_available():
            self.logger.warning("CUDA requested but not available, using CPU")
            return "cpu"
        elif device == "mps" and not torch.backends.mps.is_available():
            self.logger.warning("MPS requested but not available, using CPU")
            return "cpu"

        return device

    def _get_device_dtype(self) -> torch.dtype:
        """Get optimal dtype for current device.

        :return: Compatible torch dtype for the current device
        :rtype: torch.dtype
        """
        if self.device == "mps":
            return torch.float32  # MPS has issues with mixed precision
        elif self.device == "cuda":
            return self.torch_dtype  # CUDA handles bfloat16 well
        else:
            return torch.float32  # CPU default

    def _load_model(self) -> None:
        """Load Gemma 3 model and processor.

        Loads the appropriate model class based on vision capabilities and
        sets up the processor for tokenization and image processing.
        """
        self.logger.info(f"Loading {self.model_name} on {self.device}")

        # Determine model class
        if self.use_vision:
            model_class = Gemma3ForConditionalGeneration
        else:
            model_class = Gemma3ForCausalLM

        # Get device-appropriate dtype
        effective_dtype = self._get_device_dtype()

        # Load model
        model_kwargs = {
            "pretrained_model_name_or_path": self.model_name,
            "torch_dtype": effective_dtype,
            "device_map": self.device if self.device != "mps" else None,
            "cache_dir": self.cache_dir,
            "trust_remote_code": True,
            "attn_implementation": "sdpa"
        }

        self.model = model_class.from_pretrained(**model_kwargs)

        # Move to device if needed
        if self.device == "mps":
            self.model = self.model.to(device=self.device, dtype=effective_dtype)

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=True
        )

        self.logger.info("Model loaded successfully")

    def get_training_recommendations(self, training_mode: TrainingMode) -> TrainingParameters:
        """Get training recommendations based on training mode.

        :param training_mode: The training mode to get recommendations for
        :type training_mode: TrainingMode
        :return: Recommended training parameters
        :rtype: TrainingParameters

        Example:
            recommendations = provider.get_training_recommendations(TrainingMode.DONUT_STYLE)
            print(f"Recommended LR: {recommendations.learning_rate}")
        """
        recommendations = {
            TrainingMode.DONUT_STYLE: TrainingParameters(
                learning_rate=1e-4,
                batch_size=8,
                gradient_accumulation_steps=2,
                num_epochs=5,
                warmup_ratio=0.05,
                weight_decay=0.01,
                lr_scheduler_type="cosine"
            ),
            TrainingMode.VISION_ONLY: TrainingParameters(
                learning_rate=5e-5,
                batch_size=16,
                gradient_accumulation_steps=1,
                num_epochs=3,
                warmup_ratio=0.1,
                weight_decay=0.001,
                lr_scheduler_type="linear"
            ),
            TrainingMode.LORA: TrainingParameters(
                learning_rate=2e-4,
                batch_size=4,
                gradient_accumulation_steps=4,
                num_epochs=3,
                warmup_ratio=0.1,
                weight_decay=0.01,
                lr_scheduler_type="cosine"
            ),
            TrainingMode.FULL: TrainingParameters(
                learning_rate=5e-5,
                batch_size=2,
                gradient_accumulation_steps=8,
                num_epochs=2,
                warmup_ratio=0.05,
                weight_decay=0.01,
                lr_scheduler_type="cosine"
            )
        }

        return recommendations.get(training_mode, recommendations[TrainingMode.LORA])

    def prepare_for_training(self,
                             training_mode: str = "lora",
                             lora_config: Optional[LoraTrainingConfig] = None,
                             enable_gradient_checkpointing: bool = True) -> None:
        """Prepare model for training with specified approach.

        :param training_mode: Training approach - "lora", "full", "frozen", "donut_style", "vision_only"
        :type training_mode: str
        :param lora_config: LoRA configuration parameters
        :type lora_config: Optional[LoraTrainingConfig]
        :param enable_gradient_checkpointing: Enable gradient checkpointing for memory efficiency
        :type enable_gradient_checkpointing: bool

        Example:
            # Donut-style training for document transcription
            provider.prepare_for_training(
                training_mode="donut_style",
                enable_gradient_checkpointing=True
            )

            # LoRA training with custom config
            lora_config = LoraTrainingConfig(rank=32, target_layers="all")
            provider.prepare_for_training(
                training_mode="lora",
                lora_config=lora_config
            )
        """
        self.training_mode = TrainingMode(training_mode)

        # Enable gradient checkpointing if requested
        if enable_gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            self.logger.info("Gradient checkpointing enabled")

        if self.training_mode == TrainingMode.LORA:
            self._setup_lora_training(lora_config=lora_config or LoraTrainingConfig())
        elif self.training_mode == TrainingMode.FULL:
            self._setup_full_training()
        elif self.training_mode == TrainingMode.FROZEN:
            self._setup_frozen_training()
        elif self.training_mode in [TrainingMode.DONUT_STYLE, TrainingMode.VISION_ONLY]:
            self._setup_donut_style_training()

    def _setup_donut_style_training(self) -> None:
        """Setup Donut-style training: focus on vision components for document transcription."""
        if self.training_mode == TrainingMode.DONUT_STYLE:
            self.unfrozen_modules = FineTuningStrategy.apply_donut_style(
                model=self.model,
                logger=self.logger
            )
        elif self.training_mode == TrainingMode.VISION_ONLY:
            self.unfrozen_modules = FineTuningStrategy.apply_vision_only(
                model=self.model,
                logger=self.logger
            )

        self.is_finetuned = True

    def _setup_lora_training(self, lora_config: LoraTrainingConfig) -> None:
        """Setup LoRA training configuration.

        :param lora_config: LoRA configuration parameters
        :type lora_config: LoraTrainingConfig
        """
        # Create LoRA config
        peft_lora_config = self._create_lora_config(lora_config=lora_config)

        # Apply LoRA
        self.model = get_peft_model(self.model, peft_lora_config)
        self.model.print_trainable_parameters()
        self.is_finetuned = True

    def _setup_full_training(self) -> None:
        """Setup full model fine-tuning."""
        # Enable all parameters
        for param in self.model.parameters():
            param.requires_grad = True

        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Full fine-tuning with {total_params:,} parameters")

    def _setup_frozen_training(self) -> None:
        """Setup training with frozen base model (only train specific layers)."""
        # Freeze all parameters first
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze specific layers
        unfrozen_count = 0
        for name, param in self.model.named_parameters():
            if any(pattern in name.lower() for pattern in ['vision', 'cross_attn', 'lm_head']):
                param.requires_grad = True
                unfrozen_count += 1

        self.logger.info(f"Frozen training with {unfrozen_count} unfrozen parameters")

    def _create_lora_config(self, lora_config: LoraTrainingConfig) -> LoraConfig:
        """Create flexible LoRA configuration.

        :param lora_config: LoRA training configuration
        :type lora_config: LoraTrainingConfig
        :return: PEFT LoRA configuration
        :rtype: LoraConfig
        """
        # Default pattern matching if not provided
        if lora_config.pattern_matching is None:
            pattern_matching = {
                'attention': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                'mlp': ['gate_proj', 'up_proj', 'down_proj'],
                'vision': ['vision', 'visual', 'img', 'patch'],
                'language': ['language_model', 'decoder']
            }
        else:
            pattern_matching = lora_config.pattern_matching

        # Discover modules
        all_modules = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                for category, patterns in pattern_matching.items():
                    if any(p in name.lower() for p in patterns):
                        all_modules[name] = category
                        break

        # Filter based on criteria
        target_modules = []
        for name, category in all_modules.items():
            if lora_config.target_layers == "all":
                if category in lora_config.target_types:
                    target_modules.append(name)
            elif lora_config.target_layers == "vision" and category == "vision":
                target_modules.append(name)
            elif lora_config.target_layers == "language" and category == "language":
                target_modules.append(name)

        if not target_modules:
            raise ValueError(f"No modules found for {lora_config.target_layers}/{lora_config.target_types}")

        self.logger.info(f"LoRA targeting {len(target_modules)} modules")

        return LoraConfig(
            r=lora_config.rank,
            lora_alpha=lora_config.alpha,
            lora_dropout=lora_config.dropout,
            target_modules=target_modules,
            bias="none"
        )

    def train(self,
              training_data: List[Dict[str, Any]],
              validation_data: Optional[List[Dict[str, Any]]] = None,
              output_dir: str = "./output",
              training_params: Optional[TrainingParameters] = None,
              logging_steps: int = 10,
              save_steps: int = 100,
              eval_steps: int = 100,
              fp16: bool = False,
              bf16: bool = False,
              report_to: Optional[str] = None) -> Trainer:
        """Train the model with specified configuration.

        :param training_data: Training examples with text, images, and targets. Expected format:
            [{"text": "prompt", "target": "response", "images": ["path.jpg"]}, ...]
        :type training_data: List[Dict[str, Any]]
        :param validation_data: Optional validation examples in same format as training_data
        :type validation_data: Optional[List[Dict[str, Any]]]
        :param output_dir: Directory to save model checkpoints and final model
        :type output_dir: str
        :param training_params: Training parameters (uses mode recommendations if None)
        :type training_params: Optional[TrainingParameters]
        :param logging_steps: Steps between logging outputs
        :type logging_steps: int
        :param save_steps: Steps between saving checkpoints
        :type save_steps: int
        :param eval_steps: Steps between evaluations
        :type eval_steps: int
        :param fp16: Use FP16 mixed precision training
        :type fp16: bool
        :param bf16: Use BF16 mixed precision training
        :type bf16: bool
        :param report_to: Reporting integration ("wandb", "tensorboard", etc.)
        :type report_to: Optional[str]
        :return: Configured Trainer instance
        :rtype: Trainer

        Example:
            # Train with automatic parameter selection
            trainer = provider.train(
                training_data=train_data,
                validation_data=val_data,
                output_dir="./gemma-rtl-model"
            )

            # Train with custom parameters
            custom_params = TrainingParameters(learning_rate=1e-4, batch_size=8)
            trainer = provider.train(
                training_data=train_data,
                training_params=custom_params,
                output_dir="./custom-model"
            )
        """
        self.model = patch_model_for_tied_weights(self.model)
        # Get training parameters
        if training_params is None:
            training_params = self.get_training_recommendations(training_mode=self.training_mode)

        # Prepare datasets
        train_dataset = self._prepare_training_dataset(training_data=training_data)
        eval_dataset = None
        if validation_data:
            eval_dataset = self._prepare_training_dataset(training_data=validation_data)

        # Determine precision settings
        use_fp16 = fp16 and self.device == "cuda"
        use_bf16 = bf16 and self.device == "cuda"

        # Create training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=training_params.batch_size,
            per_device_eval_batch_size=training_params.batch_size,
            gradient_accumulation_steps=training_params.gradient_accumulation_steps,
            num_train_epochs=training_params.num_epochs,
            learning_rate=training_params.learning_rate,
            warmup_ratio=training_params.warmup_ratio,
            weight_decay=training_params.weight_decay,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps if eval_dataset else None,
            eval_strategy="steps" if eval_dataset else "no",
            save_total_limit=3,
            load_best_model_at_end=eval_dataset is not None,
            metric_for_best_model="loss" if eval_dataset else None,
            fp16=use_fp16,
            bf16=use_bf16,
            gradient_checkpointing=(self.training_mode in [TrainingMode.FULL, TrainingMode.DONUT_STYLE]),
            optim="adamw_torch",
            lr_scheduler_type=training_params.lr_scheduler_type,
            report_to=report_to,
            remove_unused_columns=False,
            dataloader_pin_memory=self.device == "cuda",
            dataloader_num_workers=0 if self.device == "mps" else 4,
        )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=Gemma3DataCollator(
                processor=self.processor,
                max_input_length=self.max_input_sequence_length
            ),
            tokenizer=self.processor.tokenizer
        )

        # Log training setup
        self.logger.info(f"Training setup:")
        self.logger.info(f"  Mode: {self.training_mode.value}")
        self.logger.info(f"  Batch size: {training_params.batch_size}")
        self.logger.info(f"  Learning rate: {training_params.learning_rate}")
        self.logger.info(f"  Epochs: {training_params.num_epochs}")

        # Train
        trainer.train()

        # Save final model
        trainer.save_model()
        self.processor.save_pretrained(save_directory=output_dir)

        # Save training info
        training_info = {
            "training_mode": self.training_mode.value,
            "unfrozen_modules": self.unfrozen_modules,
            "model_name": self.model_name,
            "training_args": training_args.to_dict()
        }

        with open(f"{output_dir}/training_info.json", "w") as f:
            json.dump(training_info, f, indent=2)

        return trainer

    def call_llm(self, inline_data: List[Any]) -> List[Dict[str, Any]]:
        """Generate responses for input data.

        :param inline_data: List of inputs - strings or dicts with text/images. Expected formats:
            - ["text prompt", "another prompt"]
            - [{"text": "prompt", "images": ["path.jpg"]}, ...]
        :type inline_data: List[Any]
        :return: List of generation results with keys: generated_text, input_tokens, output_tokens, model_name
        :rtype: List[Dict[str, Any]]

        Example:
            # Text only
            results = provider.call_llm(["Transcribe this document"])

            # With images
            results = provider.call_llm([{
                "text": "Transcribe this Hebrew manuscript",
                "images": ["manuscript.jpg"]
            }])
        """
        if not self.model or not self.processor:
            raise RuntimeError("Model not loaded")

        results = []
        for data_item in inline_data:
            # Parse input
            messages, images = self._parse_input(data_item=data_item)

            # Generate response
            result = self._generate(
                messages=messages,
                images=images,
                original_input=data_item
            )
            results.append(result)

        return results

    def _parse_input(self, data_item: Any) -> Tuple[List[Dict[str, Any]], Optional[List[Image.Image]]]:
        """Parse various input formats into messages and images.

        :param data_item: Input data in string or dict format
        :type data_item: Any
        :return: Tuple of (messages, images)
        :rtype: Tuple[List[Dict[str, Any]], Optional[List[Image.Image]]]
        """
        if isinstance(data_item, str):
            messages = [{"role": "user", "content": [{"type": "text", "text": data_item}]}]
            return messages, None

        elif isinstance(data_item, dict):
            text = data_item.get("text", "")
            images = data_item.get("images")

            messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]

            if images:
                processed_images = []
                if not isinstance(images, list):
                    images = [images]

                for img in images:
                    if isinstance(img, str):
                        if img.startswith("http"):
                            import requests
                            response = requests.get(img)
                            img = Image.open(io.BytesIO(response.content))
                        else:
                            img = Image.open(img)
                    processed_images.append(img)

                return messages, processed_images

            return messages, None

        else:
            raise ValueError(f"Unsupported input type: {type(data_item)}")

    def _generate(self,
                  messages: List[Dict[str, Any]],
                  images: Optional[List[Image.Image]],
                  original_input: Any) -> Dict[str, Any]:
        """Generate response using model.

        :param messages: Conversation messages in chat format
        :type messages: List[Dict[str, Any]]
        :param images: Optional list of PIL images
        :type images: Optional[List[Image.Image]]
        :param original_input: Original input for metadata tracking
        :type original_input: Any
        :return: Generation result with text and token counts
        :rtype: Dict[str, Any]
        """
        # Embed images in messages if present
        if images:
            messages = self._embed_images_in_messages(messages=messages, images=images)

        # Tokenize
        inputs = self.processor.apply_chat_template(
            [messages],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=False,
            do_pan_and_scan=self.enable_pan_scan and self.use_vision,
            max_length=self.max_input_sequence_length,
            truncation=True,
            padding=True
        )

        # Move to device
        inputs = {k: v.to(self.model.device) if torch.is_tensor(v) else v
                  for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_output_length,
                do_sample=False,
                temperature=1.0,
                cache_implementation="static"
            )

        # Decode
        input_length = inputs["input_ids"].shape[-1]
        generated_tokens = outputs[0][input_length:]
        response_text = self.processor.decode(generated_tokens, skip_special_tokens=True)

        return {
            "generated_text": response_text,
            "input_tokens": input_length,
            "output_tokens": len(generated_tokens),
            "model_name": self.model_name,
            "training_mode": self.training_mode.value
        }

    def _embed_images_in_messages(self,
                                  messages: List[Dict[str, Any]],
                                  images: List[Image.Image]) -> List[Dict[str, Any]]:
        """Embed images into message structure for processing.

        :param messages: Original message list
        :type messages: List[Dict[str, Any]]
        :param images: Images to embed in messages
        :type images: List[Image.Image]
        :return: Updated messages with embedded images
        :rtype: List[Dict[str, Any]]
        """
        updated_messages = []
        img_idx = 0

        for msg in messages:
            updated_msg = {"role": msg["role"], "content": []}

            for content in msg["content"]:
                if content["type"] == "text" and img_idx < len(images):
                    # Add image before text
                    updated_msg["content"].append({
                        "type": "image",
                        "image": images[img_idx]
                    })
                    img_idx += 1

                updated_msg["content"].append(content)

            updated_messages.append(updated_msg)

        return updated_messages

    def _prepare_training_dataset(self, training_data: List[Dict[str, Any]]) -> Dataset:
        """Prepare dataset for training from raw data.

        :param training_data: Raw training data with text, target, and optional images
        :type training_data: List[Dict[str, Any]]
        :return: HuggingFace Dataset formatted for training
        :rtype: Dataset
        """
        formatted_data = []

        for item in training_data:
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": item.get("text", "")}]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": item["target"]}]
                }
            ]

            formatted_data.append({
                "messages": messages,
                "images": item.get("images", [])
            })

        return Dataset.from_list(formatted_data)

    def load_finetuned_model(self, model_path: str) -> None:
        """Load a fine-tuned model from disk.

        :param model_path: Path to the saved fine-tuned model directory
        :type model_path: str

        Example:
            provider.load_finetuned_model("./my-finetuned-model")
        """
        self.model = PeftModel.from_pretrained(self.model, model_path)
        self.is_finetuned = True
        self.logger.info(f"Loaded fine-tuned model from {model_path}")

    def save_model(self, output_dir: str, save_processor: bool = True) -> None:
        """Save the current model state to disk.

        :param output_dir: Directory to save model files
        :type output_dir: str
        :param save_processor: Whether to save the processor along with model
        :type save_processor: bool

        Example:
            provider.save_model("./my-saved-model", save_processor=True)
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save model
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(output_dir)
        else:
            torch.save(self.model.state_dict(), f"{output_dir}/pytorch_model.bin")

        # Save processor
        if save_processor and self.processor:
            self.processor.save_pretrained(output_dir)

        # Save training info
        training_info = {
            "training_mode": self.training_mode.value,
            "unfrozen_modules": self.unfrozen_modules,
            "model_name": self.model_name,
            "is_finetuned": self.is_finetuned
        }

        with open(f"{output_dir}/training_info.json", "w") as f:
            json.dump(training_info, f, indent=2)

        self.logger.info(f"Model saved to {output_dir}")

    def get_model_info(self) -> ModelInfo:
        """Get comprehensive model information.

        :return: Complete model information including parameters and training state
        :rtype: ModelInfo

        Example:
            info = provider.get_model_info()
            print(f"Trainable parameters: {info.trainable_parameters:,}")
            print(f"Training mode: {info.training_mode}")
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return ModelInfo(
            model_name=self.model_name,
            training_mode=self.training_mode.value,
            is_finetuned=self.is_finetuned,
            device=self.device,
            torch_dtype=str(self.torch_dtype),
            use_vision=self.use_vision,
            total_parameters=total_params,
            trainable_parameters=trainable_params,
            trainable_ratio=trainable_params / total_params if total_params > 0 else 0,
            unfrozen_modules_count=len(self.unfrozen_modules),
            max_input_sequence_length=self.max_input_sequence_length,
            max_output_length=self.max_output_length
        )

    def reset_training_state(self) -> None:
        """Reset model to inference state after training.

        Disables gradient checkpointing and sets all parameters to non-trainable.

        Example:
            # After training, reset for inference
            provider.reset_training_state()
        """
        if hasattr(self.model, 'gradient_checkpointing_disable'):
            self.model.gradient_checkpointing_disable()

        # Set all parameters to not require gradients
        for param in self.model.parameters():
            param.requires_grad = False

        # Reset training flags
        self.training_mode = TrainingMode.NONE
        self.unfrozen_modules = []

        self.logger.info("Model reset to inference state")

    def __repr__(self) -> str:
        """String representation of the provider."""
        return (f"Gemma3Provider("
                f"model='{self.model_name}', "
                f"mode='{self.training_mode.value}', "
                f"device='{self.device}', "
                f"vision={self.use_vision}, "
                f"finetuned={self.is_finetuned})")

    def __del__(self) -> None:
        """Cleanup when provider is destroyed.."""
        if hasattr(self, 'model') and self.model is not None:
            # Move model to CPU to free GPU memory
            if self.device in ['cuda', 'mps']:
                self.model.cpu()
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()