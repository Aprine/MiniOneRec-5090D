import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--skip-model-load", action="store_true")
    args = parser.parse_args()

    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available to PyTorch.")

    print("cuda runtime:", torch.version.cuda)
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    print("bf16 supported:", torch.cuda.is_bf16_supported())

    if args.skip_model_load:
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print("loaded tokenizer vocab:", len(tokenizer))
    print("loaded model dtype:", next(model.parameters()).dtype)
    print("loaded model device:", next(model.parameters()).device)


if __name__ == "__main__":
    main()
