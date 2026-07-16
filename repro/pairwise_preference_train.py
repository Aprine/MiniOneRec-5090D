#!/usr/bin/env python
"""Reference-free pairwise SID preference tuning for the 5090D benchmark.

This is a lightweight alternative to GRPO: for each sequence recommendation
prompt, it trains the model to score the gold SID above a prefix-matched hard
negative SID. The default setting freezes the LLM and updates only SID token
rows through gradient masks.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import Tokenizer  # noqa: E402


SID_TOKEN_RE = re.compile(r"^<[a-z]_\d+>$")
SID_SEQUENCE_RE = re.compile(r"<[^>]+>")

INSTRUCTION = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Can you predict the next possible item that the user may expect?

"""


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_history(value) -> list[str]:
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


def split_sid_tokens(sid: str) -> list[str]:
    return SID_SEQUENCE_RE.findall(str(sid))


def sid_prefix(sid: str, depth: int) -> str:
    return "".join(split_sid_tokens(sid)[:depth])


def load_item_sids(info_file: Path) -> list[str]:
    sids: list[str] = []
    with info_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            sids.append(line.split("\t", 1)[0].strip())
    return sorted(set(sids))


class HardNegativeSampler:
    def __init__(self, item_sids: list[str], seed: int) -> None:
        self.item_sids = item_sids
        self.rng = random.Random(seed)
        self.by_prefix2: dict[str, list[str]] = defaultdict(list)
        self.by_prefix1: dict[str, list[str]] = defaultdict(list)
        for sid in item_sids:
            self.by_prefix2[sid_prefix(sid, 2)].append(sid)
            self.by_prefix1[sid_prefix(sid, 1)].append(sid)

    def _choose_from(self, candidates: list[str], banned: set[str]) -> str | None:
        filtered = [sid for sid in candidates if sid not in banned]
        if not filtered:
            return None
        return self.rng.choice(filtered)

    def sample(self, target_sid: str, history_sids: list[str]) -> str:
        banned = set(history_sids)
        banned.add(target_sid)
        for candidates in (
            self.by_prefix2.get(sid_prefix(target_sid, 2), []),
            self.by_prefix1.get(sid_prefix(target_sid, 1), []),
            self.item_sids,
        ):
            sampled = self._choose_from(candidates, banned)
            if sampled is not None:
                return sampled
        return self.rng.choice([sid for sid in self.item_sids if sid != target_sid])


def build_user_prompt(history_sids: list[str]) -> str:
    history = ", ".join(history_sids)
    user_input = (
        f"The user has interacted with items {history} in chronological order. "
        "Can you predict the next possible item that the user may expect?"
    )
    return f"""### User Input: 
{user_input}

### Response:
"""


def build_sequence(tokenizer: Tokenizer, prompt_ids: list[int], sid: str, max_len: int) -> dict[str, list[int]]:
    response_ids = tokenizer.encode(f"{sid}\n", bos=False, eos=True)
    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids[-max_len:],
        "labels": labels[-max_len:],
        "attention_mask": attention_mask[-max_len:],
    }


class PairwiseSidDataset(Dataset):
    def __init__(
        self,
        csv_file: Path,
        info_file: Path,
        tokenizer,
        category: str,
        sample: int,
        seed: int,
        max_len: int,
    ) -> None:
        self.tokenizer = Tokenizer(tokenizer)
        self.max_len = max_len
        item_sids = load_item_sids(info_file)
        negative_sampler = HardNegativeSampler(item_sids, seed=seed)

        df = pd.read_csv(csv_file)
        if sample > 0:
            df = df.sample(n=min(sample, len(df)), random_state=seed)
        df = df.reset_index(drop=True)

        self.examples: list[dict[str, dict[str, list[int]]]] = []
        instruction_ids = self.tokenizer.encode(INSTRUCTION, bos=True, eos=False)
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Build pairs from {csv_file.name}"):
            history_sids = parse_history(row.get("history_item_sid", ""))
            target_sid = str(row.get("item_sid", "")).strip()
            if not history_sids or not target_sid:
                continue
            negative_sid = negative_sampler.sample(target_sid, history_sids)
            prompt = build_user_prompt(history_sids)
            prompt_ids = instruction_ids + self.tokenizer.encode(prompt, bos=False, eos=False)
            self.examples.append(
                {
                    "chosen": build_sequence(self.tokenizer, prompt_ids, target_sid, max_len=max_len),
                    "rejected": build_sequence(self.tokenizer, prompt_ids, negative_sid, max_len=max_len),
                    "target_sid": target_sid,
                    "negative_sid": negative_sid,
                }
            )

        if not self.examples:
            raise ValueError(f"No pairwise examples built from {csv_file}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


def pad_sequences(sequences: list[dict[str, list[int]]], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(item["input_ids"]) for item in sequences)
    batch_size = len(sequences)
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

    for idx, item in enumerate(sequences):
        length = len(item["input_ids"])
        input_ids[idx, :length] = torch.tensor(item["input_ids"], dtype=torch.long)
        labels[idx, :length] = torch.tensor(item["labels"], dtype=torch.long)
        attention_mask[idx, :length] = torch.tensor(item["attention_mask"], dtype=torch.long)

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def collate_pairwise(batch: list[dict], pad_token_id: int) -> dict[str, torch.Tensor | int]:
    chosen = [item["chosen"] for item in batch]
    rejected = [item["rejected"] for item in batch]
    merged = pad_sequences(chosen + rejected, pad_token_id=pad_token_id)
    merged["pair_count"] = len(batch)
    return merged


def find_sid_token_ids(tokenizer) -> list[int]:
    token_ids = []
    for token, token_id in tokenizer.get_vocab().items():
        if SID_TOKEN_RE.match(token):
            token_ids.append(int(token_id))
    return sorted(set(token_ids))


def enable_sid_token_row_training(model, tokenizer) -> list[int]:
    sid_token_ids = find_sid_token_ids(tokenizer)
    if not sid_token_ids:
        raise ValueError("No SID tokens found in tokenizer vocabulary")

    for param in model.parameters():
        param.requires_grad = False

    def enable_rows(embedding, label: str) -> bool:
        if embedding is None or not hasattr(embedding, "weight"):
            return False
        embedding.weight.requires_grad = True

        def mask_grad(grad):
            row_mask = torch.ones(grad.shape[0], dtype=torch.bool, device=grad.device)
            ids = torch.tensor(sid_token_ids, dtype=torch.long, device=grad.device)
            row_mask[ids] = False
            grad[row_mask].zero_()
            return grad

        embedding.weight.register_hook(mask_grad)
        print(f"Enabled gradient-masked {label} SID rows: {len(sid_token_ids)}")
        return True

    input_embedding = model.get_input_embeddings()
    output_embedding = model.get_output_embeddings()
    input_enabled = enable_rows(input_embedding, "input embedding")
    output_enabled = False
    if (
        output_embedding is not None
        and input_embedding is not None
        and output_embedding.weight.data_ptr() != input_embedding.weight.data_ptr()
    ):
        output_enabled = enable_rows(output_embedding, "output embedding/lm_head")

    if not input_enabled and not output_enabled:
        raise ValueError("Could not enable SID token rows")
    return sid_token_ids


def sequence_log_probs(model, batch: dict[str, torch.Tensor], length_normalize: bool) -> torch.Tensor:
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = outputs.logits[:, :-1, :]
    labels = batch["labels"][:, 1:]
    mask = labels.ne(-100)
    safe_labels = labels.masked_fill(~mask, 0)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    token_log_probs = torch.gather(log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs * mask
    summed = token_log_probs.sum(dim=-1)
    if length_normalize:
        lengths = mask.sum(dim=-1).clamp_min(1)
        return summed / lengths
    return summed


@torch.no_grad()
def evaluate_pairwise(model, dataloader: DataLoader, device: torch.device, beta: float, length_normalize: bool) -> dict:
    model.eval()
    losses = []
    accuracies = []
    margins = []
    for batch in dataloader:
        pair_count = int(batch.pop("pair_count"))
        batch = {key: value.to(device) for key, value in batch.items()}
        logps = sequence_log_probs(model, batch, length_normalize=length_normalize)
        chosen_logps = logps[:pair_count]
        rejected_logps = logps[pair_count:]
        margin = chosen_logps - rejected_logps
        loss = -F.logsigmoid(beta * margin).mean()
        losses.append(float(loss.item()))
        accuracies.append(float((margin > 0).float().mean().item()))
        margins.append(float(margin.mean().item()))
    model.train()
    return {
        "eval_pair_loss": float(np.mean(losses)),
        "eval_pair_accuracy": float(np.mean(accuracies)),
        "eval_margin": float(np.mean(margins)),
    }


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = PairwiseSidDataset(
        csv_file=Path(args.train_file),
        info_file=Path(args.info_file),
        tokenizer=tokenizer,
        category=args.category,
        sample=args.sample,
        seed=args.seed,
        max_len=args.max_len,
    )
    eval_dataset = PairwiseSidDataset(
        csv_file=Path(args.eval_file),
        info_file=Path(args.info_file),
        tokenizer=tokenizer,
        category=args.category,
        sample=args.eval_sample,
        seed=args.seed + 1,
        max_len=args.max_len,
    )

    collate = lambda batch: collate_pairwise(batch, pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    model.to(device)
    model.config.use_cache = False
    sid_token_ids = enable_sid_token_row_training(model, tokenizer)

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_dir = output_dir / "final_checkpoint"

    log_history = []
    global_step = 0
    optimizer_step = 0
    running_loss = []
    running_acc = []
    start_time = time.time()
    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_micro_steps = args.num_epochs * len(train_loader)
    progress = tqdm(total=total_micro_steps, desc="D1 pairwise preference")
    for epoch in range(args.num_epochs):
        for micro_step, batch in enumerate(train_loader, start=1):
            pair_count = int(batch.pop("pair_count"))
            batch = {key: value.to(device) for key, value in batch.items()}
            logps = sequence_log_probs(model, batch, length_normalize=args.length_normalize)
            chosen_logps = logps[:pair_count]
            rejected_logps = logps[pair_count:]
            margin = chosen_logps - rejected_logps
            pair_loss = -F.logsigmoid(args.beta * margin).mean()
            sft_loss = -chosen_logps.mean()
            loss = pair_loss + args.sft_weight * sft_loss
            (loss / args.gradient_accumulation_steps).backward()

            running_loss.append(float(loss.item()))
            running_acc.append(float((margin.detach() > 0).float().mean().item()))
            global_step += 1

            should_step = (
                micro_step % args.gradient_accumulation_steps == 0
                or micro_step == len(train_loader)
            )
            if should_step:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

            if global_step % args.logging_steps == 0:
                record = {
                    "loss": float(np.mean(running_loss[-args.logging_steps :])),
                    "pair_accuracy": float(np.mean(running_acc[-args.logging_steps :])),
                    "learning_rate": args.learning_rate,
                    "epoch": epoch + micro_step / len(train_loader),
                    "step": global_step,
                }
                print(record)
                log_history.append(record)

            if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                metrics = evaluate_pairwise(
                    model,
                    eval_loader,
                    device=device,
                    beta=args.beta,
                    length_normalize=args.length_normalize,
                )
                metrics.update({"epoch": epoch + micro_step / len(train_loader), "step": global_step})
                print(metrics)
                log_history.append(metrics)

            progress.update(1)

    progress.close()
    runtime = time.time() - start_time
    final_metrics = evaluate_pairwise(
        model,
        eval_loader,
        device=device,
        beta=args.beta,
        length_normalize=args.length_normalize,
    )
    summary = {
        "train_runtime": runtime,
        "train_loss": float(np.mean(running_loss)) if running_loss else None,
        "train_pair_accuracy": float(np.mean(running_acc)) if running_acc else None,
        "optimizer_steps": optimizer_step,
        "global_step": global_step,
        "sid_token_rows": len(sid_token_ids),
    }
    summary.update(final_metrics)
    print(summary)
    log_history.append(summary)

    state = {
        "global_step": global_step,
        "optimizer_step": optimizer_step,
        "log_history": log_history,
        "train_runtime": runtime,
        "train_loss": summary["train_loss"],
        "eval_pair_loss": final_metrics["eval_pair_loss"],
        "eval_pair_accuracy": final_metrics["eval_pair_accuracy"],
    }
    with (output_dir / "trainer_state.json").open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)

    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"D1 final checkpoint: {final_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--info_file", required=True)
    parser.add_argument("--category", default="Industrial_and_Scientific")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sample", type=int, default=1024)
    parser.add_argument("--eval_sample", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--sft_weight", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=0.3)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--length_normalize", type=lambda value: str(value).lower() == "true", default=True)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
