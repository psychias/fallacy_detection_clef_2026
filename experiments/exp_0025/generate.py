"""
Touché 2026 — Contrastive-pair synthetic data generation (v2).

Supports two modes:
  - "st1_st2": generates contrastive fallacious/legitimate pairs (for ST1 + ST2)
  - "st3": generates contrastive scheme pairs across basis axis (for ST3)

v2 changes:
  - passes_artifact_filter (regex) before LLM verifier
  - Asymmetric verification: fallacious + legitimate sides checked separately
  - text_base used for verification (self-contained); text_raw is parent-dependent
  - rng threaded to prompt builders for variable length/supports
  - 3× topic pool to absorb ~40-55% reject rate
"""

import json
import os
import shutil
import sys
import time
import random
import re
import tempfile
from pathlib import Path

script_dir = Path(__file__).parent.resolve()
workspace = script_dir.parent.parent
sys.path.insert(0, str(workspace))

from shared.router_client import get_client, chat_completion, estimate_cost
from shared.synth_contrastive import (
    FALLACY_CARDS,
    SCHEME_CARDS,
    build_st1_st2_contrastive_prompt,
    build_st3_contrastive_prompt,
    build_st2_verifier_prompt,
    build_st3_verifier_prompt,
    get_real_exemplars,
    get_real_scheme_exemplars,
    format_exemplar_block,
    parse_verifier_response,
    passes_st2_verification,
    passes_st3_verification,
    passes_artifact_filter,
    contains_stock_phrase,
    sample_topics,
)
from shared.synth_utils import append_to_jsonl
from shared.data_utils import load_jsonl, normalize_fallacy_type
from shared.train_utils import write_status, write_traceback, Heartbeat

# Disable Qwen3 thinking mode on verifier calls — chain-of-thought eats
# the entire max_tokens and the JSON answer never gets emitted.
# OpenRouter ignores extra_body reasoning param, so we append /no_think
# to the last user message (Qwen3 chat template respects this).
_VERIFY_EXTRA = {"reasoning": {"effort": "none"}}


def _disable_thinking(messages: list[dict]) -> list[dict]:
    """Append /no_think to the last user message to suppress Qwen3 thinking."""
    msgs = [dict(m) for m in messages]
    for m in reversed(msgs):
        if m.get("role") == "user":
            m["content"] = m["content"] + "\n/no_think"
            break
    return msgs


def parse_json_response(content: str) -> dict | None:
    """Parse a JSON response from the LLM, handling markdown fences and think blocks."""
    content = content.strip()
    # Strip <think>...</think> blocks (Qwen3 thinking mode)
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(l for l in lines
                           if not l.strip().startswith("```"))
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Try to extract JSON from response
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


def generate_st1_st2_pairs(client, config, train_data, output_path, heartbeat):
    """Generate contrastive pairs for ST1+ST2 (v2: artifact filter + asymmetric verify)."""
    models = config["models"]
    targets = config["targets"]
    temperature = config.get("temperature", 0.9)
    verify = config.get("verify", True)

    rng = random.Random(config.get("seed", 42))
    total_cost = 0.0
    total_generated = 0
    total_verified = 0
    total_rejected = 0
    total_artifact_rejected = 0

    for fallacy_type, target_count in targets.items():
        if target_count <= 0:
            continue

        exemplars = get_real_exemplars(train_data, fallacy_type, n=3, rng=rng)
        if not exemplars:
            raw_type = fallacy_type.replace("-", "")
            exemplars = get_real_exemplars(train_data, raw_type, n=3, rng=rng)

        # 3× pool for ~40-55% reject rate
        topics = sample_topics(target_count * 3, rng)
        generated = 0
        topic_idx = 0

        print(f"[synth] Generating {target_count} contrastive pairs for '{fallacy_type}' "
              f"(have {len(exemplars)} real exemplars)...")

        while generated < target_count and topic_idx < len(topics):
            topic, subreddit = topics[topic_idx]
            topic_idx += 1

            model = models[generated % len(models)]

            prompt = build_st1_st2_contrastive_prompt(
                fallacy_type, topic, subreddit, exemplars, rng=rng
            )

            try:
                resp = chat_completion(
                    client, _disable_thinking(prompt), model=model,
                    temperature=temperature, max_tokens=1536,
                    extra_body=_VERIFY_EXTRA,
                )
                total_cost += estimate_cost(
                    resp["usage"].get("prompt_tokens", 0),
                    resp["usage"].get("completion_tokens", 0),
                    model
                )

                pair = parse_json_response(resp["content"])
                if not pair or "fallacious" not in pair or "legitimate" not in pair:
                    print(f"[synth] WARNING: failed to parse pair for {fallacy_type}/{topic}")
                    continue

                fal = pair["fallacious"]
                leg = pair["legitimate"]
                thread_title = pair.get("thread_title", "")
                parent = pair.get("parent", "")

                # Use text_base (self-contained) for validation; text_raw for the entry
                fal_text_base = fal.get("text_base", fal.get("text_raw", ""))
                fal_text_raw = fal.get("text_raw", fal_text_base)
                leg_text_base = leg.get("text_base", leg.get("text_raw", ""))
                leg_text_raw = leg.get("text_raw", leg_text_base)

                if len(fal_text_base) < 30 or len(leg_text_base) < 30:
                    continue

                # 1. Cheap surface-artifact filter (regex, ~microsecond)
                art_ok, art_reason = passes_artifact_filter(fal_text_base, leg_text_base)
                if not art_ok:
                    total_artifact_rejected += 1
                    total_rejected += 1
                    print(f"[synth] Artifact reject ({fallacy_type}/{topic}): {art_reason}")
                    continue

                # 2. LLM verifier — both sides, asymmetric
                if verify:
                    # Verify fallacious side
                    fal_v_prompt = build_st2_verifier_prompt(
                        fal_text_base, fallacy_type, is_legitimate_side=False
                    )
                    fal_v_resp = chat_completion(
                        client, _disable_thinking(fal_v_prompt), model=models[0],
                        temperature=0.1, max_tokens=300,
                        extra_body=_VERIFY_EXTRA,
                    )
                    total_cost += estimate_cost(
                        fal_v_resp["usage"].get("prompt_tokens", 0),
                        fal_v_resp["usage"].get("completion_tokens", 0),
                        models[0]
                    )
                    fal_vresult = parse_verifier_response(fal_v_resp["content"])
                    if not passes_st2_verification(fal_vresult, is_legitimate_side=False):
                        total_rejected += 1
                        print(f"[synth] Rejected fal ({fallacy_type}/{topic}): {fal_vresult}")
                        continue

                    # Verify legitimate side
                    leg_v_prompt = build_st2_verifier_prompt(
                        leg_text_base, fallacy_type, is_legitimate_side=True
                    )
                    leg_v_resp = chat_completion(
                        client, _disable_thinking(leg_v_prompt), model=models[0],
                        temperature=0.1, max_tokens=300,
                        extra_body=_VERIFY_EXTRA,
                    )
                    total_cost += estimate_cost(
                        leg_v_resp["usage"].get("prompt_tokens", 0),
                        leg_v_resp["usage"].get("completion_tokens", 0),
                        models[0]
                    )
                    leg_vresult = parse_verifier_response(leg_v_resp["content"])
                    if not passes_st2_verification(leg_vresult, is_legitimate_side=True):
                        total_rejected += 1
                        print(f"[synth] Rejected leg ({fallacy_type}/{topic}): {leg_vresult}")
                        continue

                    total_verified += 1

                # Write fallacious example
                fal_entry = {
                    "id": f"synth_{fallacy_type}_{generated}_fal",
                    "text_raw": fal_text_raw,
                    "text_raw_parent": parent,
                    "text_raw_title": thread_title,
                    "text_base": fal_text_base,
                    "argument_base": {
                        "claim": fal.get("claim", ""),
                        "supports": fal.get("supports", []),
                    },
                    "fallacy_exists": 1,
                    "fallacy_type": fallacy_type,
                    "resembles_fallacy": fallacy_type,
                    "classification": {"argument_goal": "", "argument_basis": ""},
                    "source": "synth",
                    "synth_model": model,
                    "synth_version": config["synth_version"],
                }
                append_to_jsonl(str(output_path), fal_entry)

                # Write legitimate (non-fallacious) example
                leg_entry = {
                    "id": f"synth_{fallacy_type}_{generated}_leg",
                    "text_raw": leg_text_raw,
                    "text_raw_parent": parent,
                    "text_raw_title": thread_title,
                    "text_base": leg_text_base,
                    "argument_base": {
                        "claim": leg.get("claim", ""),
                        "supports": leg.get("supports", []),
                    },
                    "fallacy_exists": 0,
                    "fallacy_type": "",
                    "resembles_fallacy": fallacy_type,
                    "classification": {"argument_goal": "", "argument_basis": ""},
                    "source": "synth",
                    "synth_model": model,
                    "synth_version": config["synth_version"],
                }
                append_to_jsonl(str(output_path), leg_entry)

                generated += 1
                total_generated += 1
                heartbeat.beat(step=total_generated)

            except Exception as e:
                print(f"[synth] WARNING: generation failed ({fallacy_type}/{topic}): {e}")
                time.sleep(3)

            time.sleep(0.5)

        print(f"[synth] {fallacy_type}: {generated}/{target_count} pairs generated")

    print(f"[synth] ST1/ST2 summary: artifact_rejected={total_artifact_rejected}, "
          f"llm_rejected={total_rejected - total_artifact_rejected}, verified={total_verified}")
    return total_generated, total_cost, total_verified, total_rejected


def generate_st3_pairs(client, config, train_data, output_path, heartbeat):
    """Generate contrastive pairs for ST3 scheme classification (v2)."""
    models = config["models"]
    targets = config["targets"]
    temperature = config.get("temperature", 0.9)
    verify = config.get("verify", True)

    rng = random.Random(config.get("seed", 42))
    total_cost = 0.0
    total_generated = 0
    total_verified = 0
    total_rejected = 0
    total_artifact_rejected = 0

    for scheme_label, target_count in targets.items():
        if target_count <= 0:
            continue

        goal, basis = scheme_label.split("-")
        exemplars = get_real_scheme_exemplars(train_data, goal, basis, n=3, rng=rng)

        # 4× pool: dual-side verification drops ~40-50% of pairs
        topics = sample_topics(target_count * 4, rng)
        generated = 0
        topic_idx = 0

        contrast_label = SCHEME_CARDS[scheme_label]["contrast_with"]
        contrast_card = SCHEME_CARDS[contrast_label]
        print(f"[synth] Generating {target_count} contrastive pairs: "
              f"'{scheme_label}' vs '{contrast_label}' "
              f"(have {len(exemplars)} real exemplars)...")

        while generated < target_count and topic_idx < len(topics):
            topic, subreddit = topics[topic_idx]
            topic_idx += 1

            model = models[generated % len(models)]

            prompt = build_st3_contrastive_prompt(
                scheme_label, topic, subreddit, exemplars, rng=rng
            )

            try:
                resp = chat_completion(
                    client, _disable_thinking(prompt), model=model,
                    temperature=temperature, max_tokens=1536,
                    extra_body=_VERIFY_EXTRA,
                )
                total_cost += estimate_cost(
                    resp["usage"].get("prompt_tokens", 0),
                    resp["usage"].get("completion_tokens", 0),
                    model
                )

                pair = parse_json_response(resp["content"])
                if not pair or "scheme_a" not in pair or "scheme_b" not in pair:
                    print(f"[synth] WARNING: failed to parse pair for {scheme_label}/{topic}")
                    continue

                sa = pair["scheme_a"]
                sb = pair["scheme_b"]
                thread_title = pair.get("thread_title", "")
                parent = pair.get("parent", "")

                # Use text_base for validation
                sa_text_base = sa.get("text_base", sa.get("text_raw", ""))
                sa_text_raw = sa.get("text_raw", sa_text_base)
                sb_text_base = sb.get("text_base", sb.get("text_raw", ""))
                sb_text_raw = sb.get("text_raw", sb_text_base)

                if len(sa_text_base) < 30 or len(sb_text_base) < 30:
                    continue

                # Surface-artifact filter for ST3: only stock-phrase check.
                # The evidence-citation check is intentionally SKIPPED because
                # epistemic-external arguments legitimately cite sources.
                if contains_stock_phrase(sa_text_base) or contains_stock_phrase(sb_text_base):
                    total_artifact_rejected += 1
                    total_rejected += 1
                    print(f"[synth] Stock-phrase reject ({scheme_label}/{topic})")
                    continue

                # Verify BOTH sides — scheme_a (target) AND scheme_b (contrast).
                # If we only verify scheme_a, scheme_b's labels may not match its text.
                if verify:
                    sa_goal = sa.get("argument_goal", goal)
                    sa_basis = sa.get("argument_basis", basis)
                    sa_v_prompt = build_st3_verifier_prompt(sa_text_base, sa_goal, sa_basis)
                    sa_v_resp = chat_completion(
                        client, _disable_thinking(sa_v_prompt), model=models[0],
                        temperature=0.1, max_tokens=300,
                        extra_body=_VERIFY_EXTRA,
                    )
                    total_cost += estimate_cost(
                        sa_v_resp["usage"].get("prompt_tokens", 0),
                        sa_v_resp["usage"].get("completion_tokens", 0),
                        models[0]
                    )
                    sa_vresult = parse_verifier_response(sa_v_resp["content"])
                    if not passes_st3_verification(sa_vresult, basis=sa_basis):
                        total_rejected += 1
                        print(f"[synth] Rejected scheme_a ({scheme_label}/{topic}): {sa_vresult}")
                        continue

                    sb_goal = sb.get("argument_goal", contrast_card["goal"])
                    sb_basis = sb.get("argument_basis", contrast_card["basis"])
                    sb_v_prompt = build_st3_verifier_prompt(sb_text_base, sb_goal, sb_basis)
                    sb_v_resp = chat_completion(
                        client, _disable_thinking(sb_v_prompt), model=models[0],
                        temperature=0.1, max_tokens=300,
                        extra_body=_VERIFY_EXTRA,
                    )
                    total_cost += estimate_cost(
                        sb_v_resp["usage"].get("prompt_tokens", 0),
                        sb_v_resp["usage"].get("completion_tokens", 0),
                        models[0]
                    )
                    sb_vresult = parse_verifier_response(sb_v_resp["content"])
                    if not passes_st3_verification(sb_vresult, basis=sb_basis):
                        total_rejected += 1
                        print(f"[synth] Rejected scheme_b ({scheme_label}/{topic}): {sb_vresult}")
                        continue

                    total_verified += 1

                # Write scheme_a entry (target scheme)
                sa_entry = {
                    "id": f"synth_{scheme_label}_{generated}_a",
                    "text_raw": sa_text_raw,
                    "text_raw_parent": parent,
                    "text_raw_title": thread_title,
                    "text_base": sa_text_base,
                    "argument_base": {
                        "claim": sa.get("claim", ""),
                        "supports": sa.get("supports", []),
                    },
                    "fallacy_exists": 0,
                    "classification": {
                        "argument_goal": sa.get("argument_goal", goal),
                        "argument_basis": sa.get("argument_basis", basis),
                    },
                    "source": "synth",
                    "synth_model": model,
                    "synth_version": config["synth_version"],
                }
                append_to_jsonl(str(output_path), sa_entry)

                # Write scheme_b entry (contrast scheme)
                sb_entry = {
                    "id": f"synth_{scheme_label}_{generated}_b",
                    "text_raw": sb_text_raw,
                    "text_raw_parent": parent,
                    "text_raw_title": thread_title,
                    "text_base": sb_text_base,
                    "argument_base": {
                        "claim": sb.get("claim", ""),
                        "supports": sb.get("supports", []),
                    },
                    "fallacy_exists": 0,
                    "classification": {
                        "argument_goal": sb.get("argument_goal", contrast_card["goal"]),
                        "argument_basis": sb.get("argument_basis", contrast_card["basis"]),
                    },
                    "source": "synth",
                    "synth_model": model,
                    "synth_version": config["synth_version"],
                }
                append_to_jsonl(str(output_path), sb_entry)

                generated += 1
                total_generated += 1
                heartbeat.beat(step=total_generated)

            except Exception as e:
                print(f"[synth] WARNING: generation failed ({scheme_label}/{topic}): {e}")
                time.sleep(3)

            time.sleep(0.5)

        print(f"[synth] {scheme_label}: {generated}/{target_count} pairs generated")

    print(f"[synth] ST3 summary: artifact_rejected={total_artifact_rejected}, "
          f"llm_rejected={total_rejected - total_artifact_rejected}, verified={total_verified}")
    return total_generated, total_cost, total_verified, total_rejected


def main():
    exp_dir = script_dir
    write_status(exp_dir, "running")
    heartbeat = Heartbeat(exp_dir, interval_s=30)

    config_path = exp_dir / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    mode = config["mode"]  # "st1_st2" or "st3"
    synth_version = config["synth_version"]

    # Final Drive destination
    drive_output_dir = workspace / "data_synth" / synth_version
    drive_output_dir.mkdir(parents=True, exist_ok=True)
    drive_output_path = drive_output_dir / "data.jsonl"
    if drive_output_path.exists():
        drive_output_path.unlink()

    # Write to LOCAL temp dir during generation — Drive I/O causes hangs
    local_tmp = Path(tempfile.mkdtemp(prefix="synth_"))
    local_output_path = local_tmp / "data.jsonl"
    print(f"[synth] Writing to local: {local_tmp}")

    # Load training data for real exemplars
    train_path = workspace / "data" / "touchefallacy_2026_train.jsonl"
    train_data = load_jsonl(str(train_path))
    print(f"[synth] Loaded {len(train_data)} real train examples for anchoring")

    client = get_client()

    if mode == "st1_st2":
        gen, cost, verified, rejected = generate_st1_st2_pairs(
            client, config, train_data, local_output_path, heartbeat
        )
    elif mode == "st3":
        gen, cost, verified, rejected = generate_st3_pairs(
            client, config, train_data, local_output_path, heartbeat
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Count output
    n_lines = 0
    if local_output_path.exists():
        with open(local_output_path, encoding="utf-8") as f:
            n_lines = sum(1 for _ in f)

    # Copy local output to Drive
    if local_output_path.exists() and n_lines > 0:
        shutil.copy2(str(local_output_path), str(drive_output_path))
        print(f"[synth] Copied {n_lines} lines to {drive_output_path}")
    else:
        print("[synth] WARNING: no output to copy")

    # Clean up local temp
    shutil.rmtree(str(local_tmp), ignore_errors=True)

    summary = {
        "total_pairs_generated": gen,
        "total_examples_written": n_lines,
        "total_verified": verified,
        "total_rejected": rejected,
        "reject_rate": round(rejected / max(verified + rejected, 1), 3),
        "cost_usd": round(cost, 4),
        "models": config["models"],
        "mode": mode,
        "synth_version": synth_version,
    }
    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    write_status(exp_dir, "done")
    print(f"[synth] DONE. {gen} pairs ({n_lines} examples), "
          f"verified={verified}, rejected={rejected}, cost=${cost:.4f}")


if __name__ == "__main__":
    exp_dir = script_dir
    try:
        main()
    except Exception as exc:
        write_traceback(exp_dir, exc)
        write_status(exp_dir, "crashed")
        print(f"[synth] CRASHED: {exc}")
        sys.exit(1)
