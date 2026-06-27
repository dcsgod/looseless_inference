"""
models.py — Load the draft model, target model, and their shared tokenizer.

Both Qwen2.5-0.5B and Qwen2.5-7B-Instruct use the same tokenizer family,
so a single tokenizer instance is shared across both models.

Design notes:
- dtype is bfloat16 by default (numerically stable, same memory as float16 on Ampere+)
- load_in_4bit / load_in_8bit use bitsandbytes for low-VRAM environments
- device_map="auto" lets HF Accelerate distribute across available GPUs
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelPair:
    """Container for the draft model, target model, and shared tokenizer."""
    draft: PreTrainedModel
    target: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    draft_name: str
    target_name: str

    def __repr__(self) -> str:
        return (
            f"ModelPair(\n"
            f"  draft  = {self.draft_name}  [{_param_count(self.draft):.1f}B params]\n"
            f"  target = {self.target_name} [{_param_count(self.target):.1f}B params]\n"
            f"  vocab_size = {self.tokenizer.vocab_size}\n"
            f")"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_models(config: dict) -> ModelPair:
    """
    Load draft + target models and the shared tokenizer according to *config*.

    Args:
        config: Parsed YAML config dict (see configs/default.yaml).

    Returns:
        ModelPair with both models in eval mode on the requested device/dtype.

    Raises:
        ValueError: If the two models have different vocabulary sizes (would
                    break the probability comparison in rejection sampling).
    """
    model_cfg = config["models"]
    draft_name: str = model_cfg["draft"]
    target_name: str = model_cfg["target"]
    dtype_str: str = model_cfg.get("dtype", "bfloat16")
    load_in_4bit: bool = model_cfg.get("load_in_4bit", False)
    load_in_8bit: bool = model_cfg.get("load_in_8bit", False)
    device_cfg: dict = config.get("device", {})

    torch_dtype = _resolve_dtype(dtype_str)
    quant_config: Optional[BitsAndBytesConfig] = _build_quant_config(
        load_in_4bit, load_in_8bit, torch_dtype
    )

    logger.info("Loading tokenizer from %s …", target_name)
    tokenizer = AutoTokenizer.from_pretrained(
        target_name,
        trust_remote_code=True,
        padding_side="left",
    )
    # Ensure a pad token exists (required for batch inference)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading draft model %s …", draft_name)
    draft_model = _load_single_model(
        draft_name,
        torch_dtype=torch_dtype,
        quant_config=quant_config,
        device_map=device_cfg.get("draft", "auto"),
    )

    logger.info("Loading target model %s …", target_name)
    target_model = _load_single_model(
        target_name,
        torch_dtype=torch_dtype,
        quant_config=quant_config,
        device_map=device_cfg.get("target", "auto"),
    )

    _validate_vocab(draft_model, target_model, tokenizer, draft_name, target_name)

    logger.info("Both models loaded successfully.")
    pair = ModelPair(
        draft=draft_model,
        target=target_model,
        tokenizer=tokenizer,
        draft_name=draft_name,
        target_name=target_name,
    )
    logger.info("%s", pair)
    return pair


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_single_model(
    model_name: str,
    torch_dtype: torch.dtype,
    quant_config: Optional[BitsAndBytesConfig],
    device_map: str | dict,
) -> PreTrainedModel:
    kwargs: dict = {
        "trust_remote_code": True,
        "device_map": device_map,
    }
    if quant_config is not None:
        kwargs["quantization_config"] = quant_config
    else:
        kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return model


def _resolve_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unknown dtype '{dtype_str}'. Choose from {list(mapping)}")
    return mapping[dtype_str]


def _build_quant_config(
    load_in_4bit: bool,
    load_in_8bit: bool,
    compute_dtype: torch.dtype,
) -> Optional[BitsAndBytesConfig]:
    if load_in_4bit and load_in_8bit:
        raise ValueError("Cannot set both load_in_4bit and load_in_8bit.")
    if load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
    if load_in_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def _validate_vocab(
    draft: PreTrainedModel,
    target: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    draft_name: str,
    target_name: str,
) -> None:
    draft_vocab = draft.config.vocab_size
    target_vocab = target.config.vocab_size
    tok_vocab = tokenizer.vocab_size

    if draft_vocab != target_vocab:
        raise ValueError(
            f"Vocabulary size mismatch between draft ({draft_name}: {draft_vocab}) "
            f"and target ({target_name}: {target_vocab}). "
            "Speculative decoding requires identical vocabularies."
        )
    logger.info(
        "Vocabulary check passed: draft=%d, target=%d, tokenizer=%d",
        draft_vocab,
        target_vocab,
        tok_vocab,
    )


def _param_count(model: PreTrainedModel) -> float:
    """Return approximate parameter count in billions."""
    return sum(p.numel() for p in model.parameters()) / 1e9
