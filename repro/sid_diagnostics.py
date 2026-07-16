import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def gini(values: list[int]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(v for v in values if v >= 0)
    total = sum(sorted_values)
    if total == 0:
        return 0.0
    weighted = sum((idx + 1) * value for idx, value in enumerate(sorted_values))
    n = len(sorted_values)
    return (2 * weighted) / (n * total) - (n + 1) / n


def load_sid_index(path: Path) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_item_frequency(csv_paths: list[Path]) -> Counter:
    freq = Counter()
    for path in csv_paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                item_id = str(row.get("item_id", "")).strip()
                if item_id:
                    freq[item_id] += 1
    return freq


def summarize_layer(layer_counts: Counter) -> dict:
    return {
        "used_codes": len(layer_counts),
        "total_assignments": sum(layer_counts.values()),
        "gini": gini(list(layer_counts.values())),
        "top_codes": layer_counts.most_common(10),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--train")
    parser.add_argument("--valid")
    parser.add_argument("--test")
    parser.add_argument("--out", default="repro/archive/sid_diagnostics_industrial.json")
    args = parser.parse_args()

    indices = load_sid_index(Path(args.index))
    sequence_to_items = defaultdict(list)
    layer_counts = [Counter(), Counter(), Counter()]
    token_counts = Counter()
    prefix2_counts = Counter()

    for item_id, sid_tokens in indices.items():
        sequence = "".join(sid_tokens)
        sequence_to_items[sequence].append(item_id)
        for layer, token in enumerate(sid_tokens[:3]):
            layer_counts[layer][token] += 1
            token_counts[token] += 1
        if len(sid_tokens) >= 2:
            prefix2_counts["".join(sid_tokens[:2])] += 1

    collision_groups = {
        sequence: item_ids
        for sequence, item_ids in sequence_to_items.items()
        if len(item_ids) > 1
    }
    item_freq = collect_item_frequency(
        [Path(path) for path in [args.train, args.valid, args.test] if path]
    )

    freq_values = list(item_freq.values())
    payload = {
        "index_path": args.index,
        "item_count": len(indices),
        "unique_sid_sequences": len(sequence_to_items),
        "collision_sequence_count": len(collision_groups),
        "collision_item_count": sum(len(items) for items in collision_groups.values()),
        "collision_examples": list(collision_groups.items())[:20],
        "layers": {
            "a": summarize_layer(layer_counts[0]),
            "b": summarize_layer(layer_counts[1]),
            "c": summarize_layer(layer_counts[2]),
        },
        "prefix2": {
            "unique_prefixes": len(prefix2_counts),
            "gini": gini(list(prefix2_counts.values())),
            "top_prefixes": prefix2_counts.most_common(10),
        },
        "sid_token": {
            "unique_tokens": len(token_counts),
            "gini": gini(list(token_counts.values())),
            "top_tokens": token_counts.most_common(20),
        },
        "interaction_frequency": {
            "observed_items": len(item_freq),
            "total_targets": sum(freq_values),
            "gini": gini(freq_values),
            "top_items": item_freq.most_common(20),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(json.dumps({
        "item_count": payload["item_count"],
        "unique_sid_sequences": payload["unique_sid_sequences"],
        "collision_sequence_count": payload["collision_sequence_count"],
        "collision_item_count": payload["collision_item_count"],
        "sid_token_gini": payload["sid_token"]["gini"],
        "target_freq_gini": payload["interaction_frequency"]["gini"],
    }, indent=2))


if __name__ == "__main__":
    main()
