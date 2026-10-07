#!/usr/bin/env python3
"""Export OmniVoice 0.2.1 to OpenVINO FP16 IR.

PyTorch is the *export* frontend only (CPU torch already in the venv).
Saved IR is what AMC loads on OpenVINO device GPU. No PyTorch at runtime.

Two graphs, both compiled on GPU later:
  forward.xml  — embed + Qwen3 + audio heads  (iterative unmask step)
  decode.xml   — Higgs acoustic decoder (codes → 24 kHz wav)
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

# Weights and IR live next to the other AMC models (gitignored).
MODELS = Path("/home/light/Documents/projectxi1/models")
SRC = MODELS / "omnivoice"
DEST = MODELS / "omnivoice-fp16-ov"
OV_SRC = Path("/tmp/OmniVoice-0.2.1")

# Short example shapes for tracing. Runtime uses dynamic S / T.
TRACE_S = 96
TRACE_T = 32
CODEBOOKS = 8


def main() -> int:
    if not SRC.is_dir() or not (SRC / "model.safetensors").is_file():
        raise SystemExit(f"missing OmniVoice weights at {SRC}")
    if not OV_SRC.is_dir():
        raise SystemExit("missing /tmp/OmniVoice-0.2.1 (clone tag 0.2.1)")

    sys.path.insert(0, str(OV_SRC))
    import torch
    import openvino as ov
    from openvino import PartialShape
    from omnivoice import OmniVoice

    print(f"export torch {torch.__version__} (CPU frontend) -> OpenVINO FP16", flush=True)
    print(f"src {SRC}", flush=True)
    print(f"dst {DEST}", flush=True)

    DEST.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer, HiggsAudioV2TokenizerModel

    # train=True skips Higgs device_map= which needs accelerate.
    model = OmniVoice.from_pretrained(
        str(SRC),
        torch_dtype=torch.float32,
        local_files_only=True,
        attn_implementation="sdpa",
        train=True,
    )
    model.to("cpu")
    model.text_tokenizer = AutoTokenizer.from_pretrained(str(SRC), local_files_only=True)
    tok = HiggsAudioV2TokenizerModel.from_pretrained(
        str(SRC / "audio_tokenizer"),
        local_files_only=True,
    )
    tok.to("cpu")
    model.audio_tokenizer = tok
    model.eval()
    tok.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in tok.parameters():
        p.requires_grad_(False)

    class ForwardWrap(torch.nn.Module):
        """Same math as OmniVoice.forward, with use_cache=False so OV can trace."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.n_cb = int(inner.config.num_audio_codebook)
            self.v = int(inner.config.audio_vocab_size)

        def forward(self, input_ids, audio_mask, attention_mask):
            embeds = self.inner._prepare_embed_inputs(input_ids, audio_mask)
            hidden = self.inner.llm(
                inputs_embeds=embeds,
                attention_mask=attention_mask,
                return_dict=True,
                use_cache=False,
            )[0]
            bsz, seq, _ = hidden.shape
            logits = self.inner.audio_heads(hidden).view(bsz, seq, self.n_cb, self.v)
            return logits.permute(0, 2, 1, 3)

    class DecodeWrap(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, audio_codes):
            out = self.inner.decode(audio_codes, return_dict=True)
            return out.audio_values

    fwd = ForwardWrap(model)
    dec = DecodeWrap(tok)

    b = 2  # classifier-free guidance packs cond + uncond
    input_ids = torch.randint(0, 1024, (b, CODEBOOKS, TRACE_S), dtype=torch.long)
    audio_mask = torch.zeros(b, TRACE_S, dtype=torch.bool)
    audio_mask[:, TRACE_S // 2 :] = True
    attention_mask = torch.ones(b, 1, TRACE_S, TRACE_S, dtype=torch.bool)
    codes = torch.randint(0, 1024, (1, CODEBOOKS, TRACE_T), dtype=torch.long)

    print("convert forward (embed+llm+heads) ...", flush=True)
    with torch.inference_mode():
        ov_fwd = ov.convert_model(
            fwd,
            example_input=(input_ids, audio_mask, attention_mask),
        )
    ov_fwd.reshape(
        {
            ov_fwd.input(0): PartialShape([-1, CODEBOOKS, -1]),
            ov_fwd.input(1): PartialShape([-1, -1]),
            ov_fwd.input(2): PartialShape([-1, 1, -1, -1]),
        }
    )
    fwd_xml = DEST / "forward.xml"
    ov.save_model(ov_fwd, str(fwd_xml), compress_to_fp16=True)
    print(f"  wrote {fwd_xml} + {fwd_xml.with_suffix('.bin')}", flush=True)

    print("convert decode (acoustic decoder) ...", flush=True)
    with torch.inference_mode():
        ov_dec = ov.convert_model(dec, example_input=(codes,))
    ov_dec.reshape({ov_dec.input(0): PartialShape([1, CODEBOOKS, -1])})
    dec_xml = DEST / "decode.xml"
    ov.save_model(ov_dec, str(dec_xml), compress_to_fp16=True)
    print(f"  wrote {dec_xml} + {dec_xml.with_suffix('.bin')}", flush=True)

    for name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ):
        shutil.copy2(SRC / name, DEST / name)
    (DEST / "audio_tokenizer").mkdir(exist_ok=True)
    for name in ("config.json", "preprocessor_config.json"):
        shutil.copy2(SRC / "audio_tokenizer" / name, DEST / "audio_tokenizer" / name)

    meta = {
        "format": "openvino-fp16",
        "source": "k2-fsa/OmniVoice",
        "version": "0.2.1",
        "graphs": ["forward", "decode"],
        "codebooks": CODEBOOKS,
        "sample_rate": 24000,
        "runtime": "openvino GPU — no PyTorch",
    }
    (DEST / "ov_export.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("done FP16 IR", flush=True)
    for p in sorted(DEST.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(DEST)}  {p.stat().st_size}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
