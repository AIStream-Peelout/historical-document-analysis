import numpy as np
from PIL import Image
from transformers import AutoProcessor
from typing import List, Dict, Any
import torch

class Gemma3DataCollator:
    def __init__(self, processor: AutoProcessor, max_input_length: int = 2048):
        """
        Initialize data collator for Gemma 3 training.

        :param processor: Gemma 3 processor instance
        :type processor: AutoProcessor
        :param max_length: Maximum sequence length
        :type max_length: int
        """
        self.processor = processor
        self.max_length = max_input_length

    def load_image_safely(self, img_path):
        """
        Load image and ensure file handle is closed immediately.
        """
        # Use context manager to ensure file closes
        with Image.open(img_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Create a complete copy in memory
            img_array = np.array(img, dtype=np.uint8).copy()

        # File is now closed, return new Image from array
        return Image.fromarray(img_array)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of training examples.
        :param batch: Batch of training examples
        :type batch: List[Dict[str, Any]]
        :return: Collated batch tensors
        :rtype: Dict[str, torch.Tensor]
        """
        messages_batch = [item["messages"] for item in batch]
        images_batch = [item.get("images", []) for item in batch]

        # Process images
        all_images = []
        for img_list in images_batch:
            if img_list:
                for img_path in img_list:
                    if isinstance(img_path, str):
                        all_images.append(self.load_image_safely(img_path))
                    else:
                        all_images.append(img_path)
            else:
                all_images.append(None)

        # Process with chat template - DO NOT pass images separately
        # Images should be embedded in messages_batch structure
        processed = self.processor.apply_chat_template(
            messages_batch,  # Conversation with embedded images
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=False,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            do_pan_and_scan=True
        )

        labels = processed["input_ids"].clone()

        return {
            "input_ids": processed["input_ids"],
            "attention_mask": processed.get("attention_mask", torch.ones_like(labels)),
            "pixel_values": processed.get("pixel_values"),
            "labels": labels
        }