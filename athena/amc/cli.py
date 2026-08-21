"""INT4 tutor CLI. First GPU load compiles the model (slow); later loads use ov_cache."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from athena.amc.config import AMCConfig
from athena.amc.controller import AMC
from athena.amc.exceptions import ModelPathError
from athena.amc.gpu import list_openvino_devices, resolve_device


DEFAULT_SECTION = {
    "id": "cli-s1",
    "title": "Fractions",
    "body": (
        "A fraction a/b is a parts of a whole split into b equal pieces. "
        "The top number is the numerator, the bottom is the denominator."
    ),
}


def _streamer(piece: str) -> bool:
    print(piece, end="", flush=True)
    return False


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _base_config(*, timeout: float) -> AMCConfig:
    root = _root()
    return AMCConfig(
        device="GPU",
        backend="openvino",
        llama_model_path=str(root / "models" / "llama-3.1-8b-instruct-int4-ov"),
        llama_int8_path=str(root / "models" / "llama-3.1-8b-instruct-int8-ov"),
        cache_dir=str(root / "models" / "ov_cache"),
        kv_cache_gb=4,
        kv_cache_gb_int8=2,
        allow_cpu_fallback=False,
        job_timeout_s=timeout,
    )


def _int4_config(*, tokens: int, timeout: float) -> AMCConfig:
    cfg = _base_config(timeout=timeout)
    cfg.max_new_tokens_tutor = tokens
    return cfg


def _int8_config(*, tokens: int, timeout: float) -> AMCConfig:
    cfg = _base_config(timeout=timeout)
    cfg.max_new_tokens_eval = tokens
    return cfg


def _print_boot(cfg: AMCConfig) -> None:
    devices = list_openvino_devices()
    print("xilo INT4 tutor")
    print(f"  devices:  {devices}")
    print(f"  resolved: {resolve_device(cfg)}")
    print(f"  model:    {cfg.llama_model_path}")
    print(f"  cache:    {Path(cfg.cache_dir) / 'int4'}")
    print("  first GPU compile can take several minutes; later runs reuse cache.")


def _ask(amc: AMC, question: str, *, stream: bool) -> str:
    t0 = time.perf_counter()
    reply = amc.tutor_ask(question, streamer=_streamer if stream else None)
    dt = time.perf_counter() - t0
    if stream:
        print()
    else:
        print(reply)
    print(f"  [{dt:.1f}s] {amc.status().resident_model}")
    return reply


def cmd_status(_: argparse.Namespace) -> int:
    cfg = AMCConfig()
    devices = list_openvino_devices()
    print("xilo v6 AMC")
    print(f"  backend:     {cfg.backend}")
    print(f"  configured:  {cfg.device}")
    print(f"  ov devices:  {devices or '(openvino not installed)'}")
    print(f"  resolved:    {resolve_device(cfg)}")
    print(f"  llama INT4:  {cfg.llama_model_path}  (KV {cfg.kv_cache_gb} GB)")
    print(f"  llama INT8:  {cfg.llama_int8_path}  (KV {cfg.kv_cache_gb_int8} GB)")
    print(f"  whisper:     {cfg.whisper_model_path}")
    print(f"  ov cache:    {cfg.cache_dir}")
    return 0


def cmd_tutor(args: argparse.Namespace) -> int:
    cfg = _int4_config(tokens=args.tokens, timeout=args.timeout)
    _print_boot(cfg)
    ir = Path(cfg.llama_model_path)
    if not (ir / "openvino_model.bin").is_file():
        print(f"missing INT4 IR at {ir}", file=sys.stderr)
        return 2
    section: object = DEFAULT_SECTION
    if args.section:
        section = json.loads(args.section)

    t0 = time.perf_counter()
    try:
        with AMC(cfg) as amc:
            print("loading INT4 on GPU…")
            amc.enter_event_t(section)
            print(f"  ready in {time.perf_counter() - t0:.1f}s  {amc.status()}")
            if args.prompt:
                _ask(amc, args.prompt, stream=not args.no_stream)
                return 0
            print("type a doubt (quit to exit)")
            while True:
                try:
                    line = input("student> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not line or line.lower() in {"quit", "exit", "q"}:
                    break
                _ask(amc, line, stream=not args.no_stream)
    except ModelPathError as exc:
        print(exc, file=sys.stderr)
        return 2
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    cfg = _int8_config(tokens=args.tokens, timeout=args.timeout)
    devices = list_openvino_devices()
    print("xilo INT8 evaluator")
    print(f"  devices:  {devices}")
    print(f"  resolved: {resolve_device(cfg)}")
    print(f"  model:    {cfg.llama_int8_path}")
    print(f"  cache:    {Path(cfg.cache_dir) / 'int8'}")
    print(f"  KV:       {cfg.kv_cache_gb_int8} GB")
    print("  first GPU compile can take several minutes; later runs reuse cache.")
    ir = Path(cfg.llama_int8_path)
    if not (ir / "openvino_model.bin").is_file():
        print(f"missing INT8 IR at {ir}", file=sys.stderr)
        return 2

    t0 = time.perf_counter()
    try:
        with AMC(cfg) as amc:
            print("loading INT8 on GPU…")
            amc.enter_event_e(behavioral_level=args.level)
            st = amc.status()
            print(f"  ready in {time.perf_counter() - t0:.1f}s  {st}")
            if st.resident_model and st.resident_model.value != "llama_int8":
                print(f"expected llama_int8 resident, got {st.resident_model}", file=sys.stderr)
                return 1
            t1 = time.perf_counter()
            hint = amc.evaluator_hint(
                args.question,
                args.answer,
                allow_approximate=args.approximate,
            )
            print(hint)
            print(f"  [{time.perf_counter() - t1:.1f}s] {amc.status().resident_model}")
    except ModelPathError as exc:
        print(exc, file=sys.stderr)
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m athena.amc")
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("status", help="print device/paths, do not load")
    s.set_defaults(func=cmd_status)

    t = sub.add_parser("tutor", help="INT4 Llama 3.1 tutor on GPU")
    t.add_argument("prompt", nargs="?", help="one-shot question; omit for a REPL")
    t.add_argument("--tokens", type=int, default=96, help="max new tokens (default 96)")
    t.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        help="seconds to wait including first GPU compile (default 900)",
    )
    t.add_argument("--section", default="", help="JSON section blob")
    t.add_argument("--no-stream", action="store_true")
    t.set_defaults(func=cmd_tutor)

    e = sub.add_parser("eval", help="INT8 Llama 3.1 evaluator hint on GPU")
    e.add_argument(
        "--question",
        default="What is 2+2?",
        help="quiz question",
    )
    e.add_argument("--answer", default="5", help="student answer")
    e.add_argument("--level", type=int, default=2, help="behavioral gear 1-3")
    e.add_argument("--tokens", type=int, default=64)
    e.add_argument("--timeout", type=float, default=900.0)
    e.add_argument(
        "--approximate",
        action="store_true",
        help="paragraph-style closeness (default: exact)",
    )
    e.set_defaults(func=cmd_eval)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        return cmd_status(argparse.Namespace())
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "cmd", None):
        parser.print_help()
        return 2
    if args.cmd == "status":
        return cmd_status(args)
    return args.func(args)
