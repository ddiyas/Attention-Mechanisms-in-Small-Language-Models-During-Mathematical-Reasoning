import pickle
import numpy as np
from scipy.stats import entropy, ttest_ind


def load_merged_data(path="merged_results.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_attention_entropy(result):
    entropies = []
    for step in result["step_attentions"]:
        last_layer = step[-1].mean(axis=0)
        last_query = last_layer[-1]
        ent = entropy(last_query + 1e-10)
        entropies.append(ent)
    return np.mean(entropies)


def attention_to_numbers(result):
    tokens = result["all_tokens"]
    attentions = result["step_attentions"]

    num_indices = [
        i for i, tok in enumerate(tokens) if any(c.isdigit() for c in str(tok))
    ]

    if len(num_indices) == 0:
        return 0.0

    attn_to_nums = []
    for step_attn in attentions:
        last_layer = step_attn[-1].mean(axis=0)
        last_query = last_layer[-1]
        attn_sum = sum(last_query[i] for i in num_indices if i < len(last_query))
        attn_to_nums.append(attn_sum)

    return np.mean(attn_to_nums)


def attention_to_operators(result):
    tokens = result["all_tokens"]
    attentions = result["step_attentions"]

    operators = {"+", "-", "*", "/", "(", ")"}
    op_indices = [i for i, tok in enumerate(tokens) if str(tok).strip() in operators]

    if len(op_indices) == 0:
        return 0.0

    attn_to_ops = []
    for step_attn in attentions:
        last_layer = step_attn[-1].mean(axis=0)
        last_query = last_layer[-1]
        attn_sum = sum(last_query[i] for i in op_indices if i < len(last_query))
        attn_to_ops.append(attn_sum)

    return np.mean(attn_to_ops)


def gini_coefficient(attention_weights):
    sorted_weights = np.sort(attention_weights)
    n = len(sorted_weights)
    cumsum = np.cumsum(sorted_weights)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n


def layer_attention_evolution(result):
    last_step = result["step_attentions"][-1]
    num_layers = len(last_step)

    entropies_by_layer = []
    for layer_idx in range(num_layers):
        layer_attn = last_step[layer_idx].mean(axis=0)
        last_query = layer_attn[-1]
        ent = entropy(last_query + 1e-10)
        entropies_by_layer.append(ent)

    return entropies_by_layer


def compute_all_metrics(results):
    metrics = {
        "entropy": [],
        "attention_to_numbers": [],
        "attention_to_operators": [],
        "gini": [],
        "layer_evolution": [],
    }

    for r in results:
        if "step_attentions" not in r or len(r["step_attentions"]) == 0:
            continue

        metrics["entropy"].append(
            {
                "value": compute_attention_entropy(r),
                "difficulty": r["difficulty"],
                "correct": r["is_correct"],
            }
        )

        metrics["attention_to_numbers"].append(
            {
                "value": attention_to_numbers(r),
                "difficulty": r["difficulty"],
                "correct": r["is_correct"],
            }
        )

        metrics["attention_to_operators"].append(
            {
                "value": attention_to_operators(r),
                "difficulty": r["difficulty"],
                "correct": r["is_correct"],
            }
        )

        last_attn = r["step_attentions"][-1][-1].mean(axis=0)[-1]
        metrics["gini"].append(
            {"value": gini_coefficient(last_attn), "correct": r["is_correct"]}
        )

        metrics["layer_evolution"].append(
            {"values": layer_attention_evolution(r), "correct": r["is_correct"]}
        )

    return metrics


def print_statistics(metrics):
    print("\n" + "=" * 60)
    print("ATTENTION METRICS SUMMARY")
    print("=" * 60)

    # Entropy by difficulty
    print("\n--- Attention Entropy by Difficulty ---")
    for diff in ["easy", "medium", "hard"]:
        values = [m["value"] for m in metrics["entropy"] if m["difficulty"] == diff]
        if values:
            print(
                f"{diff.capitalize():8s}: {np.mean(values):.3f} ± {np.std(values):.3f}"
            )

    easy_ent = [m["value"] for m in metrics["entropy"] if m["difficulty"] == "easy"]
    hard_ent = [m["value"] for m in metrics["entropy"] if m["difficulty"] == "hard"]
    if len(easy_ent) > 0 and len(hard_ent) > 0:
        t_stat, p_val = ttest_ind(easy_ent, hard_ent)
        print(f"Easy vs Hard: t={t_stat:.3f}, p={p_val:.4f}")

    # Attention to numbers by difficulty
    print("\n--- Attention to Numbers by Difficulty ---")
    for diff in ["easy", "medium", "hard"]:
        values = [
            m["value"]
            for m in metrics["attention_to_numbers"]
            if m["difficulty"] == diff
        ]
        if values:
            print(
                f"{diff.capitalize():8s}: {np.mean(values):.3f} ± {np.std(values):.3f}"
            )

    easy_nums = [
        m["value"] for m in metrics["attention_to_numbers"] if m["difficulty"] == "easy"
    ]
    hard_nums = [
        m["value"] for m in metrics["attention_to_numbers"] if m["difficulty"] == "hard"
    ]
    if len(easy_nums) > 0 and len(hard_nums) > 0:
        t_stat, p_val = ttest_ind(easy_nums, hard_nums)
        print(f"Easy vs Hard: t={t_stat:.3f}, p={p_val:.4f}")

    # Attention to operators by difficulty
    print("\n--- Attention to Operators by Difficulty ---")
    for diff in ["easy", "medium", "hard"]:
        values = [
            m["value"]
            for m in metrics["attention_to_operators"]
            if m["difficulty"] == diff
        ]
        if values:
            print(
                f"{diff.capitalize():8s}: {np.mean(values):.3f} ± {np.std(values):.3f}"
            )

    # Correct vs Incorrect
    print("\n--- Correct vs Incorrect Solutions ---")
    correct_ent = [m["value"] for m in metrics["entropy"] if m["correct"]]
    incorrect_ent = [m["value"] for m in metrics["entropy"] if not m["correct"]]
    print(
        f"Entropy - Correct:   {np.mean(correct_ent):.3f} ± {np.std(correct_ent):.3f}"
    )
    print(
        f"Entropy - Incorrect: {np.mean(incorrect_ent):.3f} ± {np.std(incorrect_ent):.3f}"
    )

    correct_gini = [m["value"] for m in metrics["gini"] if m["correct"]]
    incorrect_gini = [m["value"] for m in metrics["gini"] if not m["correct"]]
    print(f"Gini - Correct:      {np.mean(correct_gini):.3f}")
    print(f"Gini - Incorrect:    {np.mean(incorrect_gini):.3f}")


if __name__ == "__main__":
    results = load_merged_data()
    metrics = compute_all_metrics(results)

    with open("attention_metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)

    print_statistics(metrics)
    print("\nSaved attention_metrics.pkl")
