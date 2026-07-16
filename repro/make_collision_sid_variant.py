import argparse
import ast
import csv
import json
from collections import defaultdict
from pathlib import Path


def sid_to_text(tokens):
    return "".join(tokens)


def item_sort_key(item_id):
    try:
        return (0, int(item_id))
    except ValueError:
        return (1, item_id)


def load_index(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_collision_variant(index):
    sid_to_items = defaultdict(list)
    for item_id, sid_tokens in index.items():
        sid_to_items[tuple(sid_tokens)].append(str(item_id))

    new_index = {str(item_id): list(tokens) for item_id, tokens in index.items()}
    changed = {}
    collision_groups = []

    for sid_tokens, item_ids in sid_to_items.items():
        if len(item_ids) <= 1:
            continue
        sorted_items = sorted(item_ids, key=item_sort_key)
        group = {
            "original_sid": sid_to_text(sid_tokens),
            "items": [],
        }
        for rank, item_id in enumerate(sorted_items):
            suffix = f"<d_{rank}>"
            new_tokens = list(sid_tokens) + [suffix]
            new_index[item_id] = new_tokens
            changed[item_id] = sid_to_text(new_tokens)
            group["items"].append(
                {
                    "item_id": item_id,
                    "new_sid": sid_to_text(new_tokens),
                    "suffix": suffix,
                }
            )
        collision_groups.append(group)

    return new_index, changed, collision_groups


def parse_literal_list(value):
    if value is None or value == "":
        return []
    parsed = ast.literal_eval(value)
    if isinstance(parsed, list):
        return parsed
    return [parsed]


def rewrite_csv(src_path, dst_path, item_to_sid):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    changed_targets = 0
    changed_histories = 0

    with src_path.open("r", encoding="utf-8", newline="") as src, dst_path.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            item_id = str(row.get("item_id", ""))
            if item_id in item_to_sid:
                row["item_sid"] = item_to_sid[item_id]
                changed_targets += 1

            history_ids = [str(item) for item in parse_literal_list(row.get("history_item_id", ""))]
            history_sids = [str(item) for item in parse_literal_list(row.get("history_item_sid", ""))]
            rewritten_history = []
            history_changed = False
            for idx, old_sid in enumerate(history_sids):
                history_id = history_ids[idx] if idx < len(history_ids) else ""
                new_sid = item_to_sid.get(history_id, old_sid)
                rewritten_history.append(new_sid)
                history_changed = history_changed or new_sid != old_sid
            if history_changed:
                row["history_item_sid"] = repr(rewritten_history)
                changed_histories += 1

            writer.writerow(row)

    return {
        "path": str(dst_path),
        "changed_targets": changed_targets,
        "changed_history_rows": changed_histories,
    }


def rewrite_info(src_path, dst_path, item_to_sid):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    changed = 0
    lines = []
    with src_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                lines.append(line.rstrip("\n"))
                continue
            item_id = str(parts[-1])
            if item_id in item_to_sid:
                parts[0] = item_to_sid[item_id]
                changed += 1
            lines.append("\t".join(parts))
    dst_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"path": str(dst_path), "changed_items": changed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="Industrial_and_Scientific")
    parser.add_argument("--variant", default="Industrial_and_Scientific_c1collision")
    parser.add_argument("--data-root", default="data/Amazon")
    parser.add_argument("--suffix", default="_5_2016-10-2018-11")
    parser.add_argument("--summary", default="repro/archive/c1_collision_variant_summary.json")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    source_index = data_root / "index" / f"{args.category}.index.json"
    variant_index = data_root / "index" / f"{args.variant}.index.json"

    index = load_index(source_index)
    new_index, changed, collision_groups = build_collision_variant(index)
    variant_index.parent.mkdir(parents=True, exist_ok=True)
    variant_index.write_text(json.dumps(new_index, indent=2), encoding="utf-8")

    csv_summaries = {}
    for split in ["train", "valid", "test"]:
        src = data_root / split / f"{args.category}{args.suffix}.csv"
        dst = data_root / split / f"{args.variant}{args.suffix}.csv"
        csv_summaries[split] = rewrite_csv(src, dst, changed)

    info_summary = rewrite_info(
        data_root / "info" / f"{args.category}{args.suffix}.txt",
        data_root / "info" / f"{args.variant}{args.suffix}.txt",
        changed,
    )

    summary = {
        "category": args.category,
        "variant": args.variant,
        "index": str(variant_index),
        "collision_group_count": len(collision_groups),
        "changed_item_count": len(changed),
        "new_suffix_tokens": sorted({token[-1] for token in new_index.values() if token and token[-1].startswith("<d_")}),
        "collision_groups": collision_groups,
        "csv": csv_summaries,
        "info": info_summary,
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
