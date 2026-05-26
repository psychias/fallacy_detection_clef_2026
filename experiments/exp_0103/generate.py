"""
exp_0103 — Generate synthetic pe (propaganda-everything) examples via LLM API.
Gate: Only proceed if exp_0082 NLI pe-F1 < 0.40.
"""
import json
import os
import sys
import time
from pathlib import Path

script_dir = Path(__file__).parent.resolve()
workspace = script_dir.parent.parent
sys.path.insert(0, str(workspace))

from shared.train_utils import load_config, write_status, write_traceback

# ── pe-specific generation prompts ────────────────────────────────
PE_SYSTEM = """You are an expert in argumentation theory and propaganda techniques.
Generate realistic argumentative text snippets that contain propaganda or 
emotional manipulation techniques (loaded language, fear appeals, bandwagon, 
false dilemma, etc.) rather than logical reasoning.

The text should be 2-4 sentences, resemble real-world political discourse or 
persuasive media, and clearly exemplify propaganda/emotional fallacies."""

PE_ENHANCED_LABELS = ["pe"]
PE_BASE_LABELS = ["pa"]  # pe maps to pathos in base track

SCHEMES_PROMPTS = {
    "pe": "Generate a short argumentative text (2-4 sentences) that uses propaganda "
          "or emotional manipulation. Include at least one of: loaded language, "
          "appeal to fear, bandwagon appeal, false dilemma, or emotional blackmail. "
          "Make it sound like real political/media discourse.",
    "pa": "Generate a short argumentative text (2-4 sentences) that primarily "
          "appeals to emotion (pathos) rather than logic. Use emotional language, "
          "fear, pity, or desire to persuade. Make it realistic.",
}


def generate_pe_examples(config):
    """Call OpenRouter API to generate pe-labeled synthetic data."""
    from shared.router_client import get_client, chat_completion

    client = get_client()
    gen_cfg = config.get("generation", {})
    model = gen_cfg.get("model", "openai/gpt-4o-mini")
    num_per_label = gen_cfg.get("num_per_label", 50)
    temperature = gen_cfg.get("temperature", 0.9)
    output_dir = workspace / gen_cfg.get("output_dir", "data_synth/pe_synth_v001_st3")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing outputs (resume support)
    output_file = output_dir / "generated.jsonl"
    existing_ids = set()
    if output_file.exists():
        with open(output_file, "r") as f:
            for line in f:
                obj = json.loads(line)
                existing_ids.add(obj.get("id", ""))
        print(f"[generate] Resuming: {len(existing_ids)} existing examples")

    generated = []
    for label, prompt_template in SCHEMES_PROMPTS.items():
        for i in range(num_per_label):
            gen_id = f"pe_synth_{label}_{i:04d}"
            if gen_id in existing_ids:
                continue

            try:
                messages = [
                    {"role": "system", "content": PE_SYSTEM},
                    {"role": "user", "content": prompt_template + f"\n\nVariation #{i+1}. Be creative and diverse."},
                ]
                resp = chat_completion(
                    client, messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=300,
                )
                text = resp["content"].strip()
                if len(text) < 20:
                    print(f"[generate] WARNING: short response for {gen_id}, skipping")
                    continue

                entry = {
                    "id": gen_id,
                    "text_raw": text,
                    "label_st3_enhanced": label if label in PE_ENHANCED_LABELS else None,
                    "label_st3_base": label if label in PE_BASE_LABELS else "pa",
                    "source": "pe_synth",
                    "model": model,
                }
                generated.append(entry)

                # Write incrementally
                with open(output_file, "a") as f:
                    f.write(json.dumps(entry) + "\n")

            except Exception as e:
                print(f"[generate] ERROR for {gen_id}: {e}")
                time.sleep(2)
                continue

            if (i + 1) % 10 == 0:
                print(f"[generate] {label}: {i+1}/{num_per_label}")
            time.sleep(0.5)  # Rate limit

    print(f"[generate] Generated {len(generated)} new examples")
    return generated


def check_gate(config):
    """Check if exp_0082 NLI pe-F1 < 0.40 (proceed only if pe is weak)."""
    exp_0082_dir = workspace / "experiments" / "exp_0082"
    metrics_path = exp_0082_dir / "metrics.json"

    if not metrics_path.exists():
        print("[generate] WARNING: exp_0082 metrics not found, proceeding anyway")
        return True

    with open(metrics_path) as f:
        metrics = json.load(f)

    per_class = metrics.get("per_class", {})
    pe_f1 = per_class.get("pe", {}).get("f1", 0.0)
    print(f"[generate] Gate check: exp_0082 pe-F1 = {pe_f1:.4f}")

    if pe_f1 >= 0.40:
        print("[generate] Gate BLOCKED: pe-F1 >= 0.40, NLI already handles pe well")
        return False
    return True


def main():
    exp_dir = script_dir
    config = load_config(exp_dir)
    write_status(exp_dir, "running")

    if not check_gate(config):
        write_status(exp_dir, "skipped")
        metrics = {"status": "skipped", "reason": "pe_f1_above_threshold"}
        with open(exp_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        return

    generated = generate_pe_examples(config)

    metrics = {
        "num_generated": len(generated),
        "labels": list(SCHEMES_PROMPTS.keys()),
        "status": "done",
    }
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    write_status(exp_dir, "done")
    print(f"[generate] DONE. {len(generated)} examples generated.")


if __name__ == "__main__":
    exp_dir = script_dir
    try:
        main()
    except Exception as exc:
        write_traceback(exp_dir, exc)
        write_status(exp_dir, "crashed")
        raise
