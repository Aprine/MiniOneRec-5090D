import argparse
import ast
import csv
import json
from pathlib import Path
from statistics import mean


def parse_literal_list(value):
    if value is None or value == "":
        return []
    parsed = ast.literal_eval(value)
    if isinstance(parsed, list):
        return parsed
    return [parsed]


def percentile(values, q):
    if not values:
        return 0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * q))
    return ordered[idx]


def rewrite_split(src_path, dst_path, max_history):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    before_lengths = []
    after_lengths = []
    changed_rows = 0

    with src_path.open("r", encoding="utf-8", newline="") as src, dst_path.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            history_titles = parse_literal_list(row.get("history_item_title", ""))
            history_ids = parse_literal_list(row.get("history_item_id", ""))
            history_sids = parse_literal_list(row.get("history_item_sid", ""))
            before_len = len(history_sids)

            if before_len > max_history:
                history_titles = history_titles[-max_history:]
                history_ids = history_ids[-max_history:]
                history_sids = history_sids[-max_history:]
                row["history_item_title"] = repr(history_titles)
                row["history_item_id"] = repr(history_ids)
                row["history_item_sid"] = repr(history_sids)
                changed_rows += 1

            before_lengths.append(before_len)
            after_lengths.append(min(before_len, max_history))
            writer.writerow(row)

    return {
        "path": str(dst_path),
        "rows": len(before_lengths),
        "changed_rows": changed_rows,
        "mean_before": round(mean(before_lengths), 6) if before_lengths else 0,
        "mean_after": round(mean(after_lengths), 6) if after_lengths else 0,
        "max_before": max(before_lengths) if before_lengths else 0,
        "max_after": max(after_lengths) if after_lengths else 0,
        "p50_before": percentile(before_lengths, 0.5),
        "p75_before": percentile(before_lengths, 0.75),
        "p90_before": percentile(before_lengths, 0.9),
        "p95_before": percentile(before_lengths, 0.95),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="Industrial_and_Scientific")
    parser.add_argument("--variant", default="Industrial_and_Scientific_b1recent5")
    parser.add_argument("--data-root", default="data/Amazon")
    parser.add_argument("--suffix", default="_5_2016-10-2018-11")
    parser.add_argument("--max-history", type=int, default=5)
    parser.add_argument("--summary", default="repro/archive/b1_recent5_variant_summary.json")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    split_summaries = {}
    for split in ["train", "valid", "test"]:
        src = data_root / split / f"{args.category}{args.suffix}.csv"
        dst = data_root / split / f"{args.variant}{args.suffix}.csv"
        split_summaries[split] = rewrite_split(src, dst, args.max_history)

    summary = {
        "category": args.category,
        "variant": args.variant,
        "max_history": args.max_history,
        "splits": split_summaries,
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
