# qwen_text_embedding.py
"""Single-vector text embeddings with Qwen3-Embedding-0.6B and its Genizah fine-tune.

Replaces the mean-pooled ColBERT-style (colnomic) vectors, which were never a
valid single-vector embedding and whose unpinned dependencies silently drifted
from the query-time embedder. Everything here is pinned:

- Model for new indices (v9 on): :data:`FINETUNED_MODEL_NAME` at
  :data:`FINETUNED_MODEL_REVISION`, a fine-tune of the base model with the same
  architecture and output size. Its vectors are NOT compatible with base-model
  vectors: an index and the queries against it must come from the same model.
  The Hub repo is private; the token is read from ``HF1_TOKEN`` (or
  ``HF_TOKEN``) at load time and never logged.
- Base model, kept for verifying the indices it built (``genizah_merged_v8``
  and earlier, ``bibliography_text_only_0.7``): :data:`MODEL_ID` at
  :data:`MODEL_REVISION`.
- Output: 1024-dim, last-token pooling, L2-normalized, cosine similarity, fp32,
  ``max_seq_length`` 8192.
- ASYMMETRIC usage: documents are embedded with NO instruction prefix
  (:meth:`QwenTextEmbedding.get_text_embeddings_batch`); queries always pass
  :data:`QUERY_PROMPT` explicitly (:meth:`QwenTextEmbedding.get_query_embedding`).
  The serving repo (genizah_search) must mirror this exact convention; it can
  assert compatibility via the canaries in the index ``_meta``
  (see :func:`build_index_meta`).
"""

import hashlib
import json
import logging
import os
import shutil
import subprocess
from functools import lru_cache
from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np
import torch

from src.embeddings.semantic_text import text_sha256
from src.embeddings.text_representation import (
    TEXT_VERSION_SEMANTIC_V3,
    TEXT_VERSION_SEMANTIC_V4,
    DocumentCanary,
    check_text_version,
)

logger = logging.getLogger(__name__)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# ``_meta.text_hash``: how every document's ``embedded_text_sha256`` and ``document_canary.text_sha256`` are computed
# (``semantic_text.text_sha256``).
TEXT_HASH_RULE = "sha256 of the UTF-8 bytes of the embedded text (embedded_text_sha256)"
# Golden fixture of each semantic version's fragment text function (shared with genizah_search); its sha256 goes into
# ``_meta.text_function.golden_fixture_sha256``. ``v8`` has none: its text is the production text itself.
GOLDEN_FIXTURES: Dict[str, str] = {
    TEXT_VERSION_SEMANTIC_V3: os.path.join(_REPO_ROOT, "tests", "data", "semantic_v3_golden.jsonl"),
    TEXT_VERSION_SEMANTIC_V4: os.path.join(_REPO_ROOT, "tests", "data", "semantic_v4_golden.jsonl"),
}

# Base model: built every index up to genizah_merged_v8 / bibliography_text_only_0.7.
MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
MODEL_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
# Fine-tuned model for new indices. Pin the commit sha (tag v3-final-run1), never the tag.
FINETUNED_MODEL_NAME = "isaacmg/genizah-embed-qwen3-0.6b"
# v10 embedder: tag v10-lora-s0211 = v4-final-lora-run1 step 211 (LoRA r16/α32 merged into
# Qwen/Qwen3-Embedding-0.6B@97b0c614; trained on semantic_v4 with scholarly frames OFF; data v4-final).
# Previous pin dbfab73f1e91… = v3-final-run1 (index v9, abandoned). Pin the sha, never the tag.
FINETUNED_MODEL_REVISION = "af0252255d3e9b8b8432e09de7c5f0689d7ada34"
EMBEDDING_DIMS = 1024
MAX_SEQ_LENGTH = 8192
# Environment variables holding a Hub token for the private fine-tune, in order of preference.
HF_TOKEN_ENV_VARS = ("HF1_TOKEN", "HF_TOKEN")

# Query-side prefix (serving side), exactly what the fine-tune was trained with. Documents get NO prefix.
QUERY_PROMPT = (
    "Instruct: Given a search query, retrieve relevant passages\nQuery: "
)
QUERY_INSTRUCTION = QUERY_PROMPT  # pre-v9 name, kept for existing imports

# Fixed multilingual reference string; its document-mode embedding is stored in
# each index's _meta so any consumer can verify at startup that its embedder
# reproduces the indexed vector space (cosine > 0.99 against the stored vector).
CANARY_STRING = (
    "genizah-embedding-canary-v1: Cairo Genizah semantic search. "
    "שלום עליכם, כתובה מן הגניזה הקהירית משנת ד'תתנ\"ו. "
    "السلام عليكم، وثيقة من جنيزة القاهرة. "
    "A ketubba fragment from the Ben Ezra synagogue, Fustat."
)


def hf_token_from_env() -> Optional[str]:
    """Hub token for the private fine-tune, read from the environment at load time.

    :return: ``HF1_TOKEN``, else ``HF_TOKEN``, else ``None`` (public models need none).
    """
    return os.environ.get(HF_TOKEN_ENV_VARS[0]) or os.environ.get(HF_TOKEN_ENV_VARS[1])


class QwenTextEmbedding:
    """Text-only embedding model wrapping Qwen3-Embedding-0.6B (default: the Genizah fine-tune).

    Exposes the same ``get_embeddings(image, text, use_cache)`` call signature
    as the legacy ``NomicsEmbedding`` so it drops into
    :class:`ElasticsearchGenizahProcessor` unchanged, plus a batched path
    (:meth:`get_text_embeddings_batch`) the processor prefers when available.

    :param model_name: HF model id. Defaults to :data:`FINETUNED_MODEL_NAME`;
        pass :data:`MODEL_ID` to verify a base-model index.
    :param revision: Pinned HF revision (commit hash). Defaults to
        :data:`FINETUNED_MODEL_REVISION`; :data:`MODEL_REVISION` for the base model.
    :param device: Torch device string; auto-detected (cuda > mps > cpu) when
        ``None``.
    :param batch_size: Maximum texts per forward pass (kept small — attention
        memory scales with batch x seq^2 and long docs must not be batched).
    :param max_seq_length: Truncation limit in tokens, always set explicitly on
        load (the fine-tune was trained at 384 tokens, a limit its saved config
        may carry).
        Deliberately below the model's 32k context: on MPS a 32k-token sequence
        materializes multi-GB attention tensors per layer and a batch of them can
        swap the whole machine. 8k covers the longest bibliography pages and is
        what the fine-tune was evaluated at.
    :param token_budget: Cap on padded tokens per forward pass
        (``batch_len * longest_seq_in_batch``). Long docs therefore run alone.
    """

    # The processor checks these to decide how to build the text to embed.
    text_only: bool = True
    image_only: bool = False

    def __init__(
        self,
        model_name: str = FINETUNED_MODEL_NAME,
        revision: str = FINETUNED_MODEL_REVISION,
        device: Optional[str] = None,
        batch_size: int = 2,
        max_seq_length: int = MAX_SEQ_LENGTH,
        token_budget: int = 8192,
    ) -> None:
        """Load the pinned model (parameters are documented on the class)."""
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.revision = revision
        self.device = device or self._get_device()
        self.batch_size = batch_size
        self.token_budget = token_budget
        token = hf_token_from_env()
        if token and not os.environ.get("HF_TOKEN"):
            # sentence-transformers 5.6 forwards ``token`` to the weights/config/tokenizer loaders but not to its
            # AutoProcessor probe (processor_config.json), which a private repo answers with 401 instead of 404.
            # huggingface_hub reads HF_TOKEN for every request, so export it for this process (never logged).
            os.environ["HF_TOKEN"] = token
        self.model = SentenceTransformer(
            model_name,
            revision=revision,
            device=self.device,
            token=token,
            model_kwargs={"dtype": torch.float32},  # the evaluated contract is fp32
            processor_kwargs={"token": token} if token else None,
        )
        self.model.max_seq_length = max_seq_length
        # Documents are embedded raw. The fine-tune's saved "query" prompt is the
        # stock Qwen one ("Given a web search query ..."), not the prompt it was
        # trained with, so no saved prompt may ever apply implicitly.
        self.model.default_prompt_name = None
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
        """Embed a search query in query mode (with :data:`QUERY_PROMPT`).

        Serving-side convention only; never use this for indexing documents.
        The prefix is passed explicitly: never ``prompt_name="query"``, which
        would apply the saved stock web-search prompt.

        :param query: The user's search query.
        :return: L2-normalized float32 array of shape ``(1024,)``.
        """
        embedding = self.model.encode(
            [query],
            prompt=QUERY_PROMPT,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embedding.astype(np.float32)[0]


def _vector_fields(vector: np.ndarray) -> Dict[str, Any]:
    """Serialise a vector for ``_meta`` with a hash of its 6-decimal rounding.

    :param vector: 1-D embedding.
    :return: ``{"vector": [...], "vector_sha256_6dp": "..."}``.
    """
    vector_list = [round(float(v), 8) for v in np.asarray(vector).reshape(-1).tolist()]
    vector_hash = hashlib.sha256(
        json.dumps([round(v, 6) for v in vector_list]).encode()
    ).hexdigest()
    return {"vector": vector_list, "vector_sha256_6dp": vector_hash}


def file_sha256(path: str, chunk_size: int = 1 << 20) -> str:
    """sha256 hex digest of a file's bytes, read in chunks (the merged file is about 1 GB).

    :param path: File path.
    :param chunk_size: Bytes per read.
    :return: Hex digest.
    :rtype: str
    """
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def golden_fixture_sha256(text_version: str) -> Optional[str]:
    """sha256 of the golden fixture that pins a text version's fragment text function (:data:`GOLDEN_FIXTURES`).

    :param text_version: Text representation version.
    :return: Hex digest, or ``None`` for ``v8`` (no fixture) or when the fixture file is not in this checkout.
    :rtype: Optional[str]
    :raises ValueError: For an unknown text version.
    """
    path = GOLDEN_FIXTURES.get(check_text_version(text_version))
    return file_sha256(path) if path and os.path.isfile(path) else None


def git_commit_state(repo_root: str = _REPO_ROOT, run: Callable[..., Any] = subprocess.run) -> Optional[str]:
    """``git rev-parse HEAD`` of a checkout, with ``-dirty`` appended when ``git status --porcelain`` lists anything
    (untracked files included: the text function may live in one).

    Read-only: ``--no-optional-locks`` keeps ``git status`` from refreshing the index file while another process
    (GitHub Desktop, a commit) may hold it.

    :param repo_root: Checkout to describe.
    :param run: ``subprocess.run`` (replaceable in tests).
    :return: ``<sha>`` or ``<sha>-dirty``; ``None`` when git is not installed or ``repo_root`` is not a checkout.
    :rtype: Optional[str]
    """
    if shutil.which("git") is None:
        return None
    git = ["git", "--no-optional-locks", "-C", repo_root]
    head = run(git + ["rev-parse", "HEAD"], capture_output=True, text=True)
    status = run(git + ["status", "--porcelain"], capture_output=True, text=True)
    if head.returncode != 0 or status.returncode != 0:
        return None
    commit = head.stdout.strip()
    return f"{commit}-dirty" if status.stdout.strip() else commit


@lru_cache(maxsize=1)
def hda_commit() -> Optional[str]:
    """This checkout's commit (:func:`git_commit_state`), computed once per process (at the first ``_meta`` build).

    :return: ``<sha>``, ``<sha>-dirty`` or ``None``.
    :rtype: Optional[str]
    """
    return git_commit_state()


def text_function_block(text_version: str, include_scholarly_frames: bool = False) -> Dict[str, Any]:
    """``_meta.text_function``: which text function, render flag, fixture and code produced the embedded texts.

    :param text_version: Text representation version.
    :param include_scholarly_frames: The render flag the build actually used (False: KTIV scholarly frames left out;
        the default is the indexers' default, ``index_merged_genizah.INDEX_SCHOLARLY_FRAMES``).
    :return: ``{"version", "scholarly_frames", "golden_fixture_sha256", "hda_commit"}``.
    :rtype: Dict[str, Any]
    :raises ValueError: For an unknown text version.
    """
    return {
        "version": check_text_version(text_version),
        "scholarly_frames": include_scholarly_frames,
        "golden_fixture_sha256": golden_fixture_sha256(text_version),
        "hda_commit": hda_commit(),
    }


def build_index_meta(
    embedding_model: QwenTextEmbedding,
    text_version: str,
    document_canary: Optional[DocumentCanary],
    *,
    text_function: Optional[Mapping[str, Any]] = None,
    source: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the ``_meta`` block recorded on each index.

    Records the embedding contract: ``model_name`` / ``model_revision`` (sha),
    ``query_prompt``, ``text_representation_version``, ``text_hash`` (the rule
    behind every document's ``embedded_text_sha256``), ``text_function`` (text
    version, render flag, golden-fixture sha and commit), ``source`` (the input
    file, when given), plus three canaries a consumer re-embeds and compares
    (cosine > 0.99):

    - ``canary``: :data:`CANARY_STRING` in document mode (the site's startup
      gate reads this one, together with ``embedding_model`` /
      ``embedding_model_revision``, so those pre-v9 keys stay);
    - ``query_canary``: the same string in query mode (checks the query prefix);
    - ``document_canary``: one real record's ``doc_id``, its versioned text, the
      text's ``text_sha256`` and its document-mode vector (checks the text
      function and the model).

    A builder adds ``counts`` at the end of a build
    (``elastic_index_genizah.write_index_counts``).

    :param embedding_model: The initialized embedding model to canary against.
    :param text_version: Text representation version the index is built with.
    :param document_canary: The record to canary (an eligible one), or ``None``
        when the build has none (``document_canary`` is then ``None`` and the
        verifier fails it).
    :param text_function: ``_meta.text_function`` (:func:`text_function_block`
        with the render flag the build uses); default: the block for
        ``text_version`` with the processor's default flag (frames off).
    :param source: ``_meta.source``, the input the build reads (the merged
        indexer passes ``{"merged_path", "merged_records", "merged_sha256"}``);
        left out when ``None``.
    :return: Dict suitable for the index mapping's ``_meta`` field.
    :raises ValueError: For an unknown text version, a ``text_function`` of
        another version, or an empty canary text.
    """
    import sentence_transformers
    import transformers

    check_text_version(text_version)
    if text_function is not None and text_function.get("version") != text_version:
        raise ValueError(f"text_function is for {text_function.get('version')!r}, the index for {text_version!r}")
    string_vector = embedding_model.get_text_embeddings_batch([CANARY_STRING])[0]
    query_vector = embedding_model.get_query_embedding(CANARY_STRING)
    document_block: Optional[Dict[str, Any]] = None
    if document_canary is not None:
        if not document_canary.text.strip():
            raise ValueError(f"document canary {document_canary.doc_id!r} has an empty text; never embed one")
        document_vector = embedding_model.get_text_embeddings_batch([document_canary.text])[0]
        document_block = {
            "doc_id": document_canary.doc_id,
            "text": document_canary.text,
            "text_sha256": text_sha256(document_canary.text),
            **_vector_fields(document_vector),
        }
    meta = {
        "model_name": embedding_model.model_name,
        "model_revision": embedding_model.revision,
        "query_prompt": QUERY_PROMPT,
        "text_representation_version": text_version,
        "text_hash": TEXT_HASH_RULE,
        "text_function": dict(text_function) if text_function is not None else text_function_block(text_version),
        "embedding_model": embedding_model.model_name,
        "embedding_model_revision": embedding_model.revision,
        "dims": EMBEDDING_DIMS,
        "normalization": "l2",
        "similarity": "cosine",
        "max_seq_length": embedding_model.max_seq_length,
        "document_mode": "raw text, no instruction prefix",
        "query_instruction": QUERY_PROMPT,
        "library_versions": {
            "sentence_transformers": sentence_transformers.__version__,
            "transformers": transformers.__version__,
            "torch": torch.__version__,
        },
        "canary": {"string": CANARY_STRING, **_vector_fields(string_vector)},
        "query_canary": {"string": CANARY_STRING, "prompt": QUERY_PROMPT, **_vector_fields(query_vector)},
        "document_canary": document_block,
    }
    if source is not None:
        meta["source"] = dict(source)
    return meta
