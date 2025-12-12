# %%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pickle
import torch.nn.functional as F
import re

# %%
model_name = "google/gemma-2b"

access_token = "ACCESS_TOKEN"

tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
model = AutoModelForCausalLM.from_pretrained(
    model_name, token=access_token, output_attentions=True, dtype=torch.float16
).to("cuda")

# %%
problems = [
    ("Calculate: 5 + 3", "8", "easy", "baseline"),
    ("What is 12 - 7?", "5", "easy", "baseline"),
    ("Compute 4 * 6", "24", "easy", "baseline"),
    ("Calculate: 2 + 3 * 4", "14", "medium", "order_ops"),
    ("What is 10 - 2 * 3?", "4", "medium", "order_ops"),
    ("Compute: 6 / 2 + 4", "7", "medium", "order_ops"),
    ("Calculate: 5 + 6 / 3", "7", "medium", "order_ops"),
    (
        "Calculate step by step: First add 7 and 5, then multiply the result by 3",
        "36",
        "medium",
        "multi_step",
    ),
    ("What is (8 + 4) - (3 * 2)?", "6", "medium", "multi_step"),
    ("First divide 20 by 4, then add 7 to the result", "12", "medium", "multi_step"),
    ("What is 0.5 + 0.25?", "0.75", "medium", "decimals"),
    ("Calculate: 1.5 * 4", "6", "medium", "decimals"),
    ("What is 7.5 - 2.25?", "5.25", "medium", "decimals"),
    ("Calculate: -5 + 8", "3", "medium", "negatives"),
    ("What is -3 * -4?", "12", "medium", "negatives"),
    ("Compute: 10 - (-5)", "15", "medium", "negatives"),
    (
        "John has 15 apples and gives away 6. How many does he have left?",
        "9",
        "medium",
        "word_problem",
    ),
    (
        "A book costs $12 and a pen costs $3. What is the total cost?",
        "15",
        "medium",
        "word_problem",
    ),
    (
        "If there are 24 students and 6 leave, how many remain?",
        "18",
        "medium",
        "word_problem",
    ),
    (
        "What is 8 / 2(2+2)?",
        "16",
        "hard",
        "ambiguous",
    ),
    ("Calculate: 0 * 5 + 3", "3", "hard", "zero_trick"),
    ("What is 1 - 0.9?", "0.1", "hard", "precision"),
    ("Compute: 50 / 0.5", "100", "hard", "fraction"),
    ("What is 2^3 + 1?", "9", "hard", "exponent"),
    ("What is 50% of 80?", "40", "medium", "percentage"),
    ("Calculate 25% of 60", "15", "medium", "percentage"),
    ("What is 10% of 150?", "15", "medium", "percentage"),
    ("Calculate: ((3 + 5) * 2) - 4 / 2", "14", "hard", "complex"),
    ("What is 100 - 20 * 3 + 5?", "45", "hard", "complex"),
    ("Compute: 15 / 3 + 2 * 4 - 1", "12", "hard", "complex"),
]


# %%
def generate_with_attention(problem, max_new_tokens=80, temperature=0, top_p=1.0):
    prompt = f"""
    You are a careful and precise math solver.

    INSTRUCTIONS:
    - Solve THIS SINGLE problem, step-by-step.
    - Show ONLY the minimal steps required to solve THIS problem.
    - Follow standard arithmetic order of operations (PEMDAS): Parentheses, Exponents, Multiplication/Division (left-to-right), Addition/Subtraction (left-to-right).
    - Do NOT invent or solve any additional problems.
    - Do NOT write code.
    - End the response with EXACTLY two lines:

    FINAL ANSWER: <answer>
    §END§

    Problem:
    {problem}
    """

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    generated = inputs["input_ids"]

    prompt_len = generated.shape[1]

    step_attentions = []
    step_tokens = []
    all_token_ids = generated[0].cpu().numpy().tolist()

    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(generated, output_attentions=True, return_dict=True)

        attentions_this_step = [a[0].detach().cpu().numpy() for a in out.attentions]
        step_attentions.append(attentions_this_step)

        logits = out.logits[:, -1, :]

        # if temperature > 0:
        #     logits = logits / temperature

        # if top_p < 1.0:
        #     sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        #     cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        #     sorted_indices_to_remove = cumulative_probs > top_p
        #     sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
        #         ..., :-1
        #     ].clone()
        #     sorted_indices_to_remove[..., 0] = 0

        #     indices_to_remove = sorted_indices[sorted_indices_to_remove]
        #     logits[:, indices_to_remove] = -float("Inf")

        # probs = F.softmax(logits, dim=-1)
        # if torch.sum(probs) == 0:
        #     next_token_id = torch.argmax(logits, dim=-1).unsqueeze(-1)
        # else:
        #     next_token_id = torch.multinomial(probs, num_samples=1)

        next_token_id = torch.argmax(logits, dim=-1).unsqueeze(-1)

        generated = torch.cat([generated, next_token_id], dim=1)
        token_id_int = int(next_token_id[0, 0].cpu().numpy())

        step_tokens.append(token_id_int)
        all_token_ids.append(token_id_int)

        gen_only_ids = generated[0].cpu().numpy().tolist()[prompt_len:]
        gen_only_text = tokenizer.decode(gen_only_ids, skip_special_tokens=True)
        tail = gen_only_text[-400:]

        if "§END§" in tail:
            break

        FINAL_ANS_RE = re.compile(r"final\s*answer\s*[:\-]?\s*([+-]?\d+(\.\d+)?)", re.I)

        if FINAL_ANS_RE.search(tail):
            break

        if (
            gen_only_text.lower().count("now solve this problem") > 0
            and len(gen_only_text) > 200
        ):
            break

        if token_id_int == tokenizer.eos_token_id:
            break

    full_output = tokenizer.decode(generated[0], skip_special_tokens=True)
    generated_only = tokenizer.decode(
        generated[0].cpu().numpy().tolist()[prompt_len:], skip_special_tokens=True
    )

    return {
        "prompt": prompt,
        "full_output": full_output,
        "generated_text": generated_only,
        "all_token_ids": all_token_ids,
        "all_tokens": tokenizer.convert_ids_to_tokens(all_token_ids),
        "step_attentions": step_attentions,
        "step_tokens": step_tokens,
    }


# %%
all_results = []

for problem, expected_answer, difficulty, category in problems:
    print(f"\n{'='*60}")
    print(f"Problem: {problem}")
    print(f"Expected: {expected_answer}")

    try:
        result = generate_with_attention(problem, max_new_tokens=80)

        full_output = result["generated_text"]
        print("RAW MODEL OUTPUT:")
        print(full_output)

        all_results.append(
            {
                "problem": problem,
                "expected_answer": expected_answer,
                "difficulty": difficulty,
                "failure_type": category,
                "model_answer": None,
                "is_correct": None,
                "generated_text": result["generated_text"],
                "all_tokens": result["all_tokens"],
                "all_token_ids": result["all_token_ids"],
                "step_attentions": result["step_attentions"],
            }
        )
    except Exception as e:
        print(f"Error during generation: {e}")
        all_results.append(
            {
                "problem": problem,
                "error": str(e),
                "is_correct": False,
            }
        )

    if len(all_results) % 5 == 0:
        torch.cuda.empty_cache()

# %%
with open("attention_analysis_data.pkl", "wb") as f:
    pickle.dump(all_results, f)

print("\nData saved to attention_analysis_data.pkl")
