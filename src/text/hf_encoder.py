"""Mean-pooled HuggingFace text encoders (M2).

Uses `transformers` AutoModel + mean pooling + optional L2 norm — equivalent to
the default sentence-transformers pipeline for the candidate checkpoints, without
depending on sentence-transformers/torchcodec (broken on this Windows+CUDA stack).
"""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence

# hf_xet native wheel can SIGILL/crash on some Windows hosts when fetching weights
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    summed = torch.sum(last_hidden * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


class HFMeanPoolEncoder:
    """Frozen HF encoder with mean pooling over non-padding tokens."""

    def __init__(
        self,
        model_name: str,
        *,
        text_prefix: str = "",
        batch_size: int = 32,
        normalise_l2: bool = True,
        device: Optional[str] = None,
        max_length: int = 512,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.text_prefix = text_prefix or ""
        self.batch_size = int(batch_size)
        self.normalise_l2 = bool(normalise_l2)
        self.max_length = int(max_length)
        self.device = device
        # Prefer smaller encode batches on CPU to avoid Windows process crashes
        if str(device).startswith("cpu"):
            self.batch_size = min(self.batch_size, 8)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)
        self.embedding_dim = int(self.model.config.hidden_size)

    @torch.inference_mode()
    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: Optional[int] = None,
        show_progress_bar: bool = False,  # API compat with ST; ignored
        convert_to_numpy: bool = True,
        normalize_embeddings: Optional[bool] = None,
    ) -> np.ndarray:
        del show_progress_bar  # unused
        bs = int(batch_size or self.batch_size)
        do_norm = self.normalise_l2 if normalize_embeddings is None else bool(normalize_embeddings)
        prepared = [f"{self.text_prefix}{t}" if self.text_prefix else str(t) for t in texts]
        chunks: list[torch.Tensor] = []
        for i in range(0, len(prepared), bs):
            batch = prepared[i : i + bs]
            tok = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            tok = {k: v.to(self.device) for k, v in tok.items()}
            out = self.model(**tok)
            emb = mean_pool(out.last_hidden_state, tok["attention_mask"])
            if do_norm:
                emb = F.normalize(emb, p=2, dim=1)
            chunks.append(emb.detach().cpu())
        if not chunks:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        mat = torch.cat(chunks, dim=0)
        if convert_to_numpy:
            return mat.numpy().astype(np.float32)
        return mat

    def get_sentence_embedding_dimension(self) -> int:
        return self.embedding_dim


def load_text_encoder(
    model_name: str,
    *,
    text_prefix: str = "",
    batch_size: int = 32,
    normalise_l2: bool = True,
    device: Optional[str] = None,
) -> Any:
    """Factory used by bake-off and TextEncoderV1."""
    return HFMeanPoolEncoder(
        model_name,
        text_prefix=text_prefix,
        batch_size=batch_size,
        normalise_l2=normalise_l2,
        device=device,
    )
