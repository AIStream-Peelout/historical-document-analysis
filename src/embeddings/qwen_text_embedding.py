# qwen_text_embedding.py
"""Single-vector text embeddings with Qwen3-Embedding-0.6B.

Replaces the mean-pooled ColBERT-style (colnomic) vectors, which were never a
valid single-vector embedding and whose unpinned dependencies silently drifted
from the query-time embedder. Everything here is pinned:

- Model: ``Qwen/Qwen3-Embedding-0.6B`` at :data:`MODEL_REVISION`.
- Output: 1024-dim, L2-normalized, cosine similarity.
- ASYMMETRIC usage: documents are embedded with NO instruction prefix
  (:meth:`QwenTextEmbedding.get_text_embeddings_batch`); queries use
  :data:`QUERY_INSTRUCTION` (:meth:`QwenTextEmbedding.get_query_embedding`).
  The serving repo (genizah_search) must mirror this exact convention; it can
  assert compatibility via the canary entry in the index ``_meta``
  (see :func:`build_index_meta`).
"""

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
MODEL_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
EMBEDDING_DIMS = 1024

# Query-side instruction (serving side). Documents get NO prefix.
QUERY_INSTRUCTION = (
    "Instruct: Given a search query, retrieve relevant passages\nQuery: "
)

# Fixed multilingual reference string; its document-mode embedding is stored in
# each index's _meta so any consumer can verify at startup that its embedder
# reproduces the indexed vector space (cosine > 0.99 against the stored vector).
CANARY_STRING = (
    "genizah-embedding-canary-v1: Cairo Genizah semantic search. "
    "שלום עליכם, כתובה מן הגניזה הקהירית משנת ד'תתנ\"ו. "
    "السلام عليكم، وثيقة من جنيزة القاهرة. "
    "A ketubba fragment from the Ben Ezra synagogue, Fustat."
)


class QwenTextEmbedding:
    """Text-only embedding model wrapping Qwen3-Embedding-0.6B.

    Exposes the same ``get_embeddings(image, text, use_cache)`` call signature
    as the legacy ``NomicsEmbedding`` so it drops into
    :class:`ElasticsearchGenizahProcessor` unchanged, plus a batched path
    (:meth:`get_text_embeddings_batch`) the processor prefers when available.

    :param model_name: HF model id. Defaults to :data:`MODEL_ID`.
    :param revision: Pinned HF revision (commit hash). Defaults to
        :data:`MODEL_REVISION`.
    :param device: Torch device string; auto-detected (cuda > mps > cpu) when
        ``None``.
    :param batch_size: Maximum texts per forward pass (kept small — attention
        memory scales with batch x seq^2 and long docs must not be batched).
    :param max_seq_length: Truncation limit in tokens. Deliberately below the
        model's 32k context: on MPS a 32k-token sequence materializes
        multi-GB attention tensors per layer and a batch of them can swap the
        whole machine. 8k covers the longest bibliography pages.
    :param token_budget: Cap on padded tokens per forward pass
        (``batch_len * longest_seq_in_batch``). Long docs therefore run alone.
    """

    # The processor checks these to decide how to build the text to embed.
    text_only: bool = True
    image_only: bool = False

    def __init__(
        self,
        model_name: str = MODEL_ID,
        revision: str = MODEL_REVISION,
        device: Optional[str] = None,
        batch_size: int = 2,
        max_seq_length: int = 8192,
        token_budget: int = 8192,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.revision = revision
        self.device = device or self._get_device()
        self.batch_size = batch_size
        self.token_budget = token_budget
        self.model = SentenceTransformer(
            model_name,
            revision=revision,
            device=self.device,
        )
        if max_seq_length < self.model.max_seq_length:
            self.model.max_seq_length = max_seq_length
        self.max_seq_length = self.model.max_seq_length
        logger.info(
            "QwenTextEmbedding initialized: %s@%s on %s (max_seq_length=%d, "
            "batch_size=%d, token_budget=%d)",
            model_name, revision[:12], self.device, self.max_seq_length,
            self.batch_size, self.token_budget,
        )

    @staticmethod
    def _get_device() -> str:
        """Pick the best available torch device.

        :return: ``"cuda:0"``, ``"mps"`` or ``"cpu"``.
        """
        if torch.cuda.is_available():
            return "cuda:0"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _group_by_token_budget(self, texts: List[str]) -> List[List[int]]:
        """Split texts into contiguous groups bounded by the padded-token budget.

        A group's memory cost is ``len(group) * longest_seq_in_group`` (padding
        to the longest member), so that product is capped by ``token_budget``
        and group size by ``batch_size``. The longest permissible document
        (``max_seq_length`` tokens) always ends up alone in its group.

        :param texts: Raw document texts, already order-stable.
        :return: List of groups of indices into ``texts``.
        """
        encoded = self.model.tokenizer(
            texts, truncation=True, max_length=self.max_seq_length
        )["input_ids"]
        lengths = [len(ids) for ids in encoded]

        groups: List[List[int]] = []
        current: List[int] = []
        current_max = 0
        for i, length in enumerate(lengths):
            padded = (len(current) + 1) * max(current_max, length)
            if current and (padded > self.token_budget or len(current) >= self.batch_size):
                groups.append(current)
                current = []
                current_max = 0
            current.append(i)
            current_max = max(current_max, length)
        if current:
            groups.append(current)
        return groups

    def get_text_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Embed documents in document mode (no instruction prefix).

        Texts are chunked by :meth:`_group_by_token_budget` so a single long
        document is never batched with others — this keeps peak attention
        memory flat regardless of document length mix.

        :param texts: Raw document texts. Truncation happens at
            ``max_seq_length`` tokens (8k by default), not at 512 tokens.
        :return: L2-normalized float32 array of shape ``(len(texts), 1024)``.
        """
        results: List[np.ndarray] = []
        for group in self._group_by_token_budget(texts):
            embeddings = self.model.encode(
                [texts[i] for i in group],
                batch_size=len(group),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            results.append(embeddings.astype(np.float32))
        return np.concatenate(results, axis=0)

    def get_embeddings(
        self,
        image: Any = None,
        text: str = "",
        use_cache: bool = False,
    ) -> np.ndarray:
        """Embed a single document text (legacy processor-compatible signature).

        :param image: Ignored — this is a text-only model.
        :param text: Document text to embed (document mode, no prefix).
        :param use_cache: Ignored (kept for signature compatibility).
        :return: L2-normalized float32 array of shape ``(1, 1024)``.
        """
        return self.get_text_embeddings_batch([text])

    def get_query_embedding(self, query: str) -> np.ndarray:
        """Embed a search query in query mode (with instruction prefix).

        Serving-side convention only; never use this for indexing documents.

        :param query: The user's search query.
        :return: L2-normalized float32 array of shape ``(1024,)``.
        """
        embedding = self.model.encode(
            [query],
            prompt=QUERY_INSTRUCTION,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embedding.astype(np.float32)[0]


def build_index_meta(embedding_model: QwenTextEmbedding) -> Dict[str, Any]:
    """Build the ``_meta`` block recorded on each re-embedded index.

    Records the exact embedding contract (model, revision, dims, normalization,
    query instruction) plus a canary: :data:`CANARY_STRING` and its
    document-mode embedding. A consumer verifies its embedder by re-embedding
    the canary string and requiring cosine > 0.99 against ``canary.vector``.

    :param embedding_model: The initialized embedding model to canary against.
    :return: Dict suitable for the index mapping's ``_meta`` field.
    """
    import sentence_transformers
    import transformers

    canary_vector = embedding_model.get_text_embeddings_batch([CANARY_STRING])[0]
    vector_list = [round(float(v), 8) for v in canary_vector.tolist()]
    vector_hash = hashlib.sha256(
        json.dumps([round(v, 6) for v in vector_list]).encode()
    ).hexdigest()
    return {
        "embedding_model": embedding_model.model_name,
        "embedding_model_revision": embedding_model.revision,
        "dims": EMBEDDING_DIMS,
        "normalization": "l2",
        "similarity": "cosine",
        "max_seq_length": embedding_model.max_seq_length,
        "document_mode": "raw text, no instruction prefix",
        "query_instruction": QUERY_INSTRUCTION,
        "library_versions": {
            "sentence_transformers": sentence_transformers.__version__,
            "transformers": transformers.__version__,
            "torch": torch.__version__,
        },
        "canary": {
            "string": CANARY_STRING,
            "vector": vector_list,
            "vector_sha256_6dp": vector_hash,
        },
    }
