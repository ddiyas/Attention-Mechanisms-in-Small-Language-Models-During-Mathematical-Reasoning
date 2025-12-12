import pickle
import json


def load_attention_data(path="attention_analysis_data.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_manual_labels(path="manual_label.json"):
    with open(path, "r") as f:
        data = json.load(f)
    return {item["id"]: item for item in data["manual_labels"]}


def merge_data(attention_results, manual_labels):
    merged = []

    for idx, result in enumerate(attention_results):
        if idx in manual_labels:
            label_data = manual_labels[idx]
            result["is_correct"] = label_data["correct"]
            result["manual_note"] = label_data.get("note", "")
            result["failure_mode"] = label_data.get("failure_mode", None)
        else:
            result["is_correct"] = False
            result["manual_note"] = "Missing label"
            result["failure_mode"] = None

        merged.append(result)

    return merged


def print_accuracy_breakdown(merged_results):
    total = len(merged_results)
    correct = sum(1 for r in merged_results if r["is_correct"])

    by_difficulty = {}
    for r in merged_results:
        diff = r["difficulty"]
        if diff not in by_difficulty:
            by_difficulty[diff] = {"correct": 0, "total": 0}
        by_difficulty[diff]["total"] += 1
        if r["is_correct"]:
            by_difficulty[diff]["correct"] += 1

    print("=" * 60)
    print("CORRECTED ACCURACY BREAKDOWN")
    print("=" * 60)
    print(f"Overall: {correct}/{total} ({100*correct/total:.1f}%)\n")

    for diff in ["easy", "medium", "hard"]:
        if diff in by_difficulty:
            stats = by_difficulty[diff]
            pct = 100 * stats["correct"] / stats["total"]
            print(
                f"{diff.capitalize():8s}: {stats['correct']}/{stats['total']} ({pct:.1f}%)"
            )


if __name__ == "__main__":
    attention_data = load_attention_data()
    labels = load_manual_labels()

    merged = merge_data(attention_data, labels)

    with open("merged_results.pkl", "wb") as f:
        pickle.dump(merged, f)

    print_accuracy_breakdown(merged)
    print(f"\nSaved merged_results.pkl ({len(merged)} problems)")
