import pickle
import numpy as np
import matplotlib.pyplot as plt


def load_metrics(path="attention_metrics.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def plot_entropy_by_difficulty(metrics):
    difficulties = ["easy", "medium", "hard"]
    data = {
        d: [m["value"] for m in metrics["entropy"] if m["difficulty"] == d]
        for d in difficulties
    }

    plt.figure(figsize=(10, 6))
    plt.boxplot([data[d] for d in difficulties], labels=difficulties)
    plt.ylabel("Attention Entropy", fontsize=12)
    plt.xlabel("Problem Difficulty", fontsize=12)
    plt.title("Attention Entropy vs Problem Difficulty", fontsize=14)
    plt.savefig("entropy_by_difficulty.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved entropy_by_difficulty.png")


def plot_attention_to_numbers(metrics):
    difficulties = ["easy", "medium", "hard"]
    data = {
        d: [m["value"] for m in metrics["attention_to_numbers"] if m["difficulty"] == d]
        for d in difficulties
    }

    plt.figure(figsize=(10, 6))
    plt.boxplot([data[d] for d in difficulties], labels=difficulties)
    plt.ylabel("Attention to Numerical Tokens", fontsize=12)
    plt.xlabel("Problem Difficulty", fontsize=12)
    plt.title("Focus on Numbers vs Problem Difficulty", fontsize=14)
    plt.savefig("attention_to_numbers.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved attention_to_numbers.png")


def plot_attention_to_operators(metrics):
    difficulties = ["easy", "medium", "hard"]
    data = {
        d: [
            m["value"]
            for m in metrics["attention_to_operators"]
            if m["difficulty"] == d
        ]
        for d in difficulties
    }

    plt.figure(figsize=(10, 6))
    plt.boxplot([data[d] for d in difficulties], labels=difficulties, showfliers=False)
    plt.ylabel("Attention to Operators", fontsize=12)
    plt.xlabel("Problem Difficulty", fontsize=12)
    plt.title("Focus on Operators vs Problem Difficulty", fontsize=14)
    plt.savefig("attention_to_operators.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved attention_to_operators.png")


def plot_correct_vs_incorrect_entropy(metrics):
    correct = [m["value"] for m in metrics["entropy"] if m["correct"]]
    incorrect = [m["value"] for m in metrics["entropy"] if not m["correct"]]

    plt.figure(figsize=(10, 6))
    plt.boxplot([correct, incorrect], labels=["Correct", "Incorrect"])
    plt.ylabel("Attention Entropy", fontsize=12)
    plt.title("Attention Entropy: Correct vs Incorrect Answers", fontsize=14)
    plt.savefig("correct_vs_incorrect_entropy.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved correct_vs_incorrect_entropy.png")


def plot_layer_evolution(metrics):
    correct_layers = [m["values"] for m in metrics["layer_evolution"] if m["correct"]]
    incorrect_layers = [
        m["values"] for m in metrics["layer_evolution"] if not m["correct"]
    ]

    correct_mean = np.mean(correct_layers, axis=0)
    incorrect_mean = np.mean(incorrect_layers, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(correct_mean, label="Correct", marker="o", linewidth=2)
    plt.plot(incorrect_mean, label="Incorrect", marker="s", linewidth=2)
    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel("Attention Entropy", fontsize=12)
    plt.title("Attention Entropy Evolution Across Layers", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.savefig("layer_evolution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved layer_evolution.png")


if __name__ == "__main__":
    metrics = load_metrics()

    plot_entropy_by_difficulty(metrics)
    plot_attention_to_numbers(metrics)
    plot_attention_to_operators(metrics)
    plot_correct_vs_incorrect_entropy(metrics)
    plot_layer_evolution(metrics)

    print("\nAll plots generated successfully")
