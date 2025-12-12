import pickle
import matplotlib.pyplot as plt
import seaborn as sns


def load_merged_data(path="merged_results.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_case_study(results, case_id):
    return results[case_id]


def plot_attention_trajectory(case, case_id, case_name):
    tokens = case["all_tokens"]
    attentions = case["step_attentions"]

    num_indices = [
        i for i, tok in enumerate(tokens) if any(c.isdigit() for c in str(tok))
    ]
    operators = {"+", "-", "*", "/", "(", ")"}
    op_indices = [i for i, tok in enumerate(tokens) if str(tok).strip() in operators]

    attn_to_nums = []
    attn_to_ops = []
    attn_to_prev = []

    for step_idx, step_attn in enumerate(attentions):
        last_layer = step_attn[-1].mean(axis=0)
        last_query = last_layer[-1]

        num_attn = sum(last_query[i] for i in num_indices if i < len(last_query))
        op_attn = sum(last_query[i] for i in op_indices if i < len(last_query))

        # Attention to recently generated tokens (last 5 positions before current)
        current_pos = len(last_query) - 1
        prev_start = max(0, current_pos - 5)
        prev_attn = sum(last_query[prev_start:current_pos])

        attn_to_nums.append(num_attn)
        attn_to_ops.append(op_attn)
        attn_to_prev.append(prev_attn)

    plt.figure(figsize=(12, 6))
    steps = range(len(attn_to_nums))
    plt.plot(steps, attn_to_nums, label="Numbers", marker="o", linewidth=2)
    plt.plot(steps, attn_to_ops, label="Operators", marker="s", linewidth=2)
    plt.plot(steps, attn_to_prev, label="Recent Context", marker="^", linewidth=2)

    plt.xlabel("Generation Step", fontsize=12)
    plt.ylabel("Attention Weight", fontsize=12)
    plt.title(f"Attention Trajectory - {case_name}\n{case['problem']}", fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"trajectory_case_{case_id}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved trajectory_case_{case_id}.png")


def plot_multistep_heatmap(case, case_id, case_name):
    attentions = case["step_attentions"]
    tokens = case["all_tokens"]

    num_steps = len(attentions)
    key_steps = [
        0,
        int(0.2 * num_steps),
        int(0.4 * num_steps),
        int(0.6 * num_steps),
        int(0.8 * num_steps),
        num_steps - 1,
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, step_idx in enumerate(key_steps):
        step_attn = attentions[step_idx][-1].mean(axis=0)

        # Show last 30 tokens for readability
        display_size = min(30, step_attn.shape[0])
        plot_attn = step_attn[-display_size:, -display_size:]

        sns.heatmap(
            plot_attn,
            cmap="viridis",
            square=True,
            cbar=(idx == 2),
            ax=axes[idx],
            vmin=0,
            vmax=0.3,
            cbar_kws={"label": "Attention"} if idx == 2 else None,
        )

        pct = int(100 * step_idx / (num_steps - 1))
        axes[idx].set_title(f"Step {step_idx} ({pct}%)", fontsize=11)
        axes[idx].set_xlabel("")
        axes[idx].set_ylabel("")

    plt.suptitle(
        f"Attention Evolution - {case_name}\n{case['problem']}", fontsize=13, y=0.995
    )
    plt.tight_layout()
    plt.savefig(f"heatmap_case_{case_id}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap_case_{case_id}.png")


def plot_critical_moment(case, case_id, case_name):
    attentions = case["step_attentions"]
    tokens = case["all_tokens"]

    # Last generation step (when writing final answer)
    last_attn = attentions[-1][-1].mean(axis=0)[-1]

    # Find tokens with >5% attention
    threshold = 0.05
    important_indices = [i for i, weight in enumerate(last_attn) if weight > threshold]

    if len(important_indices) > 15:
        # Too many, just show top 15
        sorted_indices = sorted(
            range(len(last_attn)), key=lambda i: last_attn[i], reverse=True
        )
        important_indices = sorted_indices[:15]

    important_tokens = [
        str(tokens[i]) if i < len(tokens) else f"[{i}]" for i in important_indices
    ]
    important_weights = [last_attn[i] for i in important_indices]

    plt.figure(figsize=(12, 6))
    bars = plt.barh(range(len(important_tokens)), important_weights)

    # Color code: numbers=blue, operators=red, others=gray
    for i, (token, bar) in enumerate(zip(important_tokens, bars)):
        if any(c.isdigit() for c in token):
            bar.set_color("steelblue")
        elif token.strip() in {"+", "-", "*", "/", "(", ")"}:
            bar.set_color("coral")
        else:
            bar.set_color("lightgray")

    plt.yticks(range(len(important_tokens)), important_tokens, fontsize=10)
    plt.xlabel("Attention Weight", fontsize=12)
    plt.title(
        f"Critical Moment: What model attends to when generating answer\n{case_name} - {case['problem']}",
        fontsize=12,
    )
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"critical_case_{case_id}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved critical_case_{case_id}.png")


def analyze_case_narrative(case, case_id):
    print("\n" + "=" * 70)
    print(f"CASE STUDY {case_id}")
    print("=" * 70)
    print(f"Problem: {case['problem']}")
    print(f"Expected: {case['expected_answer']}")
    print(f"Correct: {case['is_correct']}")
    print(f"Difficulty: {case['difficulty']}")
    if case.get("failure_mode"):
        print(f"Failure Mode: {case['failure_mode']}")
    print(f"\nGeneration ({len(case['step_attentions'])} steps):")
    print(case["generated_text"][:300])
    if len(case["generated_text"]) > 300:
        print("...")


def run_case_study(results, case_id, case_name):
    case = get_case_study(results, case_id)

    analyze_case_narrative(case, case_id)
    plot_attention_trajectory(case, case_id, case_name)
    plot_multistep_heatmap(case, case_id, case_name)
    plot_critical_moment(case, case_id, case_name)

    print(f"\nCase {case_id} complete\n")


if __name__ == "__main__":
    results = load_merged_data()

    print("Analyzing 3 case studies...")
    print("=" * 70)

    run_case_study(results, 5, "Correct: PEMDAS Success")
    run_case_study(results, 27, "Incorrect: PEMDAS Failure")
    run_case_study(results, 19, "Incorrect: Logic Error")

    print("\n" + "=" * 70)
    print("All case studies complete")
    print("Generated 9 visualizations (3 per case)")
