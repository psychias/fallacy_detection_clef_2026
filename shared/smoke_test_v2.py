"""
Quick 30-pair smoke test for v2 synth pipeline.
Generates ~4 pairs per type for 8 fallacy types, checks artifact/citation rates.
"""

import json
import os
import sys
import time
import random
import re
from pathlib import Path

script_dir = Path(__file__).parent.resolve()
workspace = script_dir.parent

sys.path.insert(0, str(workspace))

from shared.router_client import get_client, chat_completion, estimate_cost
from shared.synth_contrastive import (
    FALLACY_CARDS,
    build_st1_st2_contrastive_prompt,
    build_st2_verifier_prompt,
    get_real_exemplars,
    parse_verifier_response,
    passes_st2_verification,
    passes_artifact_filter,
    cites_specific_evidence,
    contains_stock_phrase,
    sample_topics,
)
from shared.data_utils import load_jsonl

# ── JSON parser (same as generate_contrastive.py) ──

def parse_json_response(content: str) -> dict | None:
    content = content.strip()
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(l for l in lines if not l.strip().startswith("```"))
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


def main():
    train_path = workspace / "data" / "touchefallacy_2026_train.jsonl"
    train_data = load_jsonl(str(train_path))
    print(f"Loaded {len(train_data)} train examples")

    client = get_client()
    models = ["qwen/qwen3-32b", "google/gemma-3-27b-it"]
    rng = random.Random(99)

    # 4 pairs per type × 8 types = 32 target pairs
    targets = {
        "population": 4, "black-white": 4, "hasty_generalization": 4,
        "worse_problems": 4, "tradition": 4, "authority": 4,
        "natural": 4, "slippery_slope": 4,
    }

    results = []
    stats = {
        "total_attempted": 0,
        "total_generated": 0,
        "artifact_rejected": 0,
        "fal_verifier_rejected": 0,
        "leg_verifier_rejected": 0,
        "parse_failed": 0,
        "too_short": 0,
    }
    total_cost = 0.0

    for fallacy_type, target_count in targets.items():
        exemplars = get_real_exemplars(train_data, fallacy_type, n=3, rng=rng)
        if not exemplars:
            exemplars = get_real_exemplars(train_data, fallacy_type.replace("-", ""), n=3, rng=rng)

        topics = sample_topics(target_count * 3, rng)
        generated = 0
        topic_idx = 0

        while generated < target_count and topic_idx < len(topics):
            topic, subreddit = topics[topic_idx]
            topic_idx += 1
            stats["total_attempted"] += 1

            model = models[generated % len(models)]
            prompt = build_st1_st2_contrastive_prompt(
                fallacy_type, topic, subreddit, exemplars, rng=rng
            )

            try:
                resp = chat_completion(
                    client, prompt, model=model,
                    temperature=0.9, max_tokens=1536,
                )
                total_cost += estimate_cost(
                    resp["usage"].get("prompt_tokens", 0),
                    resp["usage"].get("completion_tokens", 0),
                    model,
                )

                pair = parse_json_response(resp["content"])
                if not pair or "fallacious" not in pair or "legitimate" not in pair:
                    stats["parse_failed"] += 1
                    print(f"  PARSE FAIL: {fallacy_type}/{topic}")
                    continue

                fal = pair["fallacious"]
                leg = pair["legitimate"]
                fal_text = fal.get("text_base", fal.get("text_raw", ""))
                leg_text = leg.get("text_base", leg.get("text_raw", ""))

                if len(fal_text) < 30 or len(leg_text) < 30:
                    stats["too_short"] += 1
                    continue

                # Artifact filter
                art_ok, art_reason = passes_artifact_filter(fal_text, leg_text)
                if not art_ok:
                    stats["artifact_rejected"] += 1
                    print(f"  ARTIFACT: {fallacy_type}/{topic}: {art_reason}")
                    continue

                # Verify fallacious side
                fal_vp = build_st2_verifier_prompt(fal_text, fallacy_type, is_legitimate_side=False)
                fal_vr = chat_completion(client, fal_vp, model=models[0], temperature=0.1, max_tokens=300)
                total_cost += estimate_cost(fal_vr["usage"].get("prompt_tokens", 0),
                                            fal_vr["usage"].get("completion_tokens", 0), models[0])
                fal_vresult = parse_verifier_response(fal_vr["content"])
                if not passes_st2_verification(fal_vresult, is_legitimate_side=False):
                    stats["fal_verifier_rejected"] += 1
                    print(f"  FAL REJECT: {fallacy_type}/{topic}: {fal_vresult}")
                    continue

                # Verify legitimate side
                leg_vp = build_st2_verifier_prompt(leg_text, fallacy_type, is_legitimate_side=True)
                leg_vr = chat_completion(client, leg_vp, model=models[0], temperature=0.1, max_tokens=300)
                total_cost += estimate_cost(leg_vr["usage"].get("prompt_tokens", 0),
                                            leg_vr["usage"].get("completion_tokens", 0), models[0])
                leg_vresult = parse_verifier_response(leg_vr["content"])
                if not passes_st2_verification(leg_vresult, is_legitimate_side=True):
                    stats["leg_verifier_rejected"] += 1
                    print(f"  LEG REJECT: {fallacy_type}/{topic}: {leg_vresult}")
                    continue

                # Accepted — record for analysis
                results.append({
                    "fallacy_type": fallacy_type,
                    "topic": topic,
                    "model": model,
                    "fal_text": fal_text,
                    "leg_text": leg_text,
                    "fal_word_count": len(fal_text.split()),
                    "leg_word_count": len(leg_text.split()),
                    "fal_cites_evidence": cites_specific_evidence(fal_text),
                    "leg_cites_evidence": cites_specific_evidence(leg_text),
                    "fal_stock_phrase": contains_stock_phrase(fal_text),
                    "leg_stock_phrase": contains_stock_phrase(leg_text),
                    "fal_supports": len(fal.get("supports", [])),
                    "leg_supports": len(leg.get("supports", [])),
                })
                generated += 1
                stats["total_generated"] += 1
                print(f"  OK: {fallacy_type}/{topic} "
                      f"(fal {len(fal_text.split())}w, leg {len(leg_text.split())}w)")

            except Exception as e:
                print(f"  ERROR: {fallacy_type}/{topic}: {e}")
                time.sleep(3)

            time.sleep(0.5)

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SMOKE TEST SUMMARY")
    print("=" * 60)
    print(f"Total attempted:        {stats['total_attempted']}")
    print(f"Total generated (pass): {stats['total_generated']}")
    print(f"Parse failures:         {stats['parse_failed']}")
    print(f"Too short:              {stats['too_short']}")
    print(f"Artifact rejected:      {stats['artifact_rejected']}")
    print(f"Fal verifier rejected:  {stats['fal_verifier_rejected']}")
    print(f"Leg verifier rejected:  {stats['leg_verifier_rejected']}")
    total_rejected = (stats['artifact_rejected'] + stats['fal_verifier_rejected']
                      + stats['leg_verifier_rejected'] + stats['parse_failed'] + stats['too_short'])
    total_decided = stats['total_generated'] + total_rejected
    if total_decided:
        print(f"Reject rate:            {total_rejected / total_decided:.1%}")
    print(f"Cost:                   ${total_cost:.4f}")

    if results:
        n = len(results)
        fal_cite = sum(1 for r in results if r["fal_cites_evidence"])
        leg_cite = sum(1 for r in results if r["leg_cites_evidence"])
        fal_stock = sum(1 for r in results if r["fal_stock_phrase"])
        leg_stock = sum(1 for r in results if r["leg_stock_phrase"])
        fal_words = [r["fal_word_count"] for r in results]
        leg_words = [r["leg_word_count"] for r in results]
        fal_sups = [r["fal_supports"] for r in results]
        leg_sups = [r["leg_supports"] for r in results]

        print(f"\n--- Quality checks on {n} accepted pairs ---")
        print(f"Fal cites evidence:  {fal_cite}/{n} = {fal_cite/n:.0%}")
        print(f"Leg cites evidence:  {leg_cite}/{n} = {leg_cite/n:.0%}  (TARGET: <10%)")
        print(f"Fal stock phrases:   {fal_stock}/{n} = {fal_stock/n:.0%}")
        print(f"Leg stock phrases:   {leg_stock}/{n} = {leg_stock/n:.0%}")
        print(f"Fal word count:      min={min(fal_words)}, median={sorted(fal_words)[n//2]}, max={max(fal_words)}")
        print(f"Leg word count:      min={min(leg_words)}, median={sorted(leg_words)[n//2]}, max={max(leg_words)}")
        print(f"Fal supports:        min={min(fal_sups)}, median={sorted(fal_sups)[n//2]}, max={max(fal_sups)}")
        print(f"Leg supports:        min={min(leg_sups)}, median={sorted(leg_sups)[n//2]}, max={max(leg_sups)}")

        # Dump 3 sample pairs for eyeballing
        print("\n--- Sample pairs ---")
        for r in results[:3]:
            print(f"\n[{r['fallacy_type']}] topic={r['topic']}")
            print(f"  FALLACIOUS ({r['fal_word_count']}w): {r['fal_text'][:200]}...")
            print(f"  LEGITIMATE ({r['leg_word_count']}w): {r['leg_text'][:200]}...")


if __name__ == "__main__":
    main()
