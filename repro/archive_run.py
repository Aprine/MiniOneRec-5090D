import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path


def load_valid_items(item_info_path: Path) -> set[str]:
    items = set()
    with item_info_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split("\t")
            if parts and parts[0].strip():
                items.add(parts[0].strip())
    return items


def compute_metrics(result_path: Path, item_info_path: Path, topks: list[int]) -> dict:
    valid_items = load_valid_items(item_info_path)
    with result_path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)

    if not records:
        raise ValueError(f"No records found in {result_path}")

    beam_count = len(records[0].get("predict", []))
    valid_topks = [topk for topk in topks if topk <= beam_count]
    hr = {str(topk): 0.0 for topk in valid_topks}
    ndcg = {str(topk): 0.0 for topk in valid_topks}
    invalid_count = 0

    for record in records:
        predictions = [str(item).strip(" \n\"") for item in record.get("predict", [])]
        target = record.get("output", "")
        if isinstance(target, list):
            target = target[0] if target else ""
        target = str(target).strip(" \n\"")

        hit_rank = None
        for rank, prediction in enumerate(predictions):
            if prediction not in valid_items:
                invalid_count += 1
            if prediction == target and hit_rank is None:
                hit_rank = rank

        for topk in valid_topks:
            if hit_rank is not None and hit_rank < topk:
                hr[str(topk)] += 1.0
                ndcg[str(topk)] += 1.0 / math.log(hit_rank + 2)

    normalizer = len(records)
    ideal_dcg = 1.0 / math.log(2)
    return {
        "records": normalizer,
        "beam_count": beam_count,
        "topks": valid_topks,
        "invalid_prediction_count": invalid_count,
        "hr": {topk: value / normalizer for topk, value in hr.items()},
        "ndcg": {topk: value / normalizer / ideal_dcg for topk, value in ndcg.items()},
    }


def parse_train_log(train_log_path: Path) -> dict:
    if not train_log_path or not train_log_path.exists():
        return {}
    text = train_log_path.read_text(encoding="utf-8", errors="ignore")
    metrics = {}
    patterns = {
        "runtime_seconds": r"['\"]train_runtime['\"]\s*:\s*([-+0-9.eE]+)",
        "train_loss": r"['\"]train_loss['\"]\s*:\s*([-+0-9.eE]+)",
        "final_eval_loss": r"['\"]eval_loss['\"]\s*:\s*([-+0-9.eE]+)",
    }
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            metrics[key] = float(matches[-1])
    return metrics


def parse_trainer_state(trainer_state_path: Path) -> dict:
    if not trainer_state_path or not trainer_state_path.exists():
        return {}
    with trainer_state_path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    metrics = {}
    if state.get("global_step") is not None:
        metrics["steps"] = int(state["global_step"])
    for record in reversed(state.get("log_history", [])):
        if "eval_loss" in record:
            metrics["final_eval_loss"] = float(record["eval_loss"])
            break
    return metrics


def format_value(value) -> str:
    return "n/a" if value is None else str(value)


def write_markdown(payload: dict, output_path: Path) -> None:
    metrics = payload["metrics"]
    lines = [
        "# MiniOneRec 5090D Experiment Archive",
        "",
        f"- Created at: {payload['created_at']}",
        f"- Experiment: `{payload['experiment_name']}`",
        f"- Model: `{payload['model']}`",
        f"- Dataset: `{payload['dataset']}`",
        f"- Checkpoint: `{payload['checkpoint']}`",
        f"- Result file: `{payload['result_file']}`",
        "",
        "## Training Summary",
        "",
        f"- SFT runtime: {format_value(payload['training']['runtime_seconds'])} seconds",
        f"- SFT steps: {format_value(payload['training']['steps'])}",
        f"- Train loss: {format_value(payload['training']['train_loss'])}",
        f"- Final eval loss: {format_value(payload['training']['final_eval_loss'])}",
        f"- Trainable setting: {payload['training']['trainable_setting']}",
        "",
        "## Evaluation Metrics",
        "",
        f"- Records: {metrics['records']}",
        f"- Beam count: {metrics['beam_count']}",
        f"- Invalid prediction count: {metrics['invalid_prediction_count']}",
        "",
        "| K | HR@K | NDCG@K |",
        "|---:|---:|---:|",
    ]
    for topk in metrics["topks"]:
        key = str(topk)
        lines.append(f"| {topk} | {metrics['hr'][key]:.8f} | {metrics['ndcg'][key]:.8f} |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This is a single-RTX-5090D smoke baseline, not the full multi-GPU MiniOneRec setting.",
            "- SFT used frozen LLM weights plus trainable SID token embeddings.",
            "- Evaluation used the single-GPU script with constrained decoding.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--item-info", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", default="repro/archive")
    parser.add_argument("--experiment-name", default="baseline_freeze_sid_qwen25_3b_industrial")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--dataset", default="Industrial_and_Scientific")
    parser.add_argument("--runtime-seconds", type=float)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--train-loss", type=float)
    parser.add_argument("--final-eval-loss", type=float)
    parser.add_argument("--train-log")
    parser.add_argument("--trainer-state")
    parser.add_argument(
        "--trainable-setting",
        default="freeze_LLM=True; train added SID token embeddings only",
    )
    parser.add_argument("--topks", default="1,3,5,10,20")
    args = parser.parse_args()

    result_path = Path(args.result)
    item_info_path = Path(args.item_info)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    topks = [int(value.strip()) for value in args.topks.split(",") if value.strip()]
    parsed_training = {}
    if args.train_log:
        parsed_training.update(parse_train_log(Path(args.train_log)))
    if args.trainer_state:
        parsed_training.update(parse_trainer_state(Path(args.trainer_state)))

    runtime_seconds = args.runtime_seconds
    steps = args.steps
    train_loss = args.train_loss
    final_eval_loss = args.final_eval_loss
    if runtime_seconds is None:
        runtime_seconds = parsed_training.get("runtime_seconds")
    if steps is None:
        steps = parsed_training.get("steps")
    if train_loss is None:
        train_loss = parsed_training.get("train_loss")
    if final_eval_loss is None:
        final_eval_loss = parsed_training.get("final_eval_loss")

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_name": args.experiment_name,
        "model": args.model,
        "dataset": args.dataset,
        "checkpoint": args.checkpoint,
        "result_file": str(result_path),
        "training": {
            "runtime_seconds": runtime_seconds,
            "steps": steps,
            "train_loss": train_loss,
            "final_eval_loss": final_eval_loss,
            "trainable_setting": args.trainable_setting,
        },
        "metrics": compute_metrics(result_path, item_info_path, topks),
    }

    json_path = out_dir / f"{args.experiment_name}.json"
    md_path = out_dir / f"{args.experiment_name}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(payload, md_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
