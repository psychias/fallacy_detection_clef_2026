"""
CoT Quality Audit for exp_0123 output.
Samples 20 random examples, displays input + label + reasoning chain
for manual scoring. Also computes automated quality signals.
"""
import json
import random
import sys
from pathlib import Path
from collections import Counter

random.seed(42)
ws = Path("G:/My Drive/fallacy_detection")

# Load CoT data (enhanced track - primary interest)
with open(ws / "data_synth/cot_v001_st3/data_enhanced.jsonl", encoding="utf-8") as f:
    enh_rows = [json.loads(l) for l in f]

with open(ws / "data_synth/cot_v001_st3/data_base.jsonl", encoding="utf-8") as f:
    base_rows = [json.loads(l) for l in f]

# Load original training data for cross-reference
with open(ws / "data/touchefallacy_2026_train.jsonl", encoding="utf-8") as f:
    orig = {json.loads(l)["id"]: json.loads(l) for l in f}

print(f"Enhanced CoT: {len(enh_rows)} examples")
print(f"Base CoT: {len(base_rows)} examples")

# Label distribution in CoT data
enh_labels = Counter(r["label"] for r in enh_rows)
print(f"\nLabel distribution: {dict(enh_labels)}")

# Sample 20 stratified: ensure pe examples are included
pe_rows = [r for r in enh_rows if r["label"] == "practical-external"]
non_pe_rows = [r for r in enh_rows if r["label"] != "practical-external"]
random.shuffle(non_pe_rows)

# Include all pe examples (should be ~8) + fill to 20 with others
sample = pe_rows[:5] + random.sample(non_pe_rows, 15)
random.shuffle(sample)

# Also check base track for a few
base_sample = random.sample(base_rows, 5)

print(f"\n{'='*80}")
print(f"AUDIT: {len(sample)} enhanced + {len(base_sample)} base examples")
print(f"{'='*80}")

# Automated quality signals
def analyze_cot(row):
    """Return quality signals for a CoT reasoning chain."""
    cot = row.get("reasoning_chain", "")
    label = row["label"]
    
    signals = {}
    # Length
    signals["cot_length"] = len(cot)
    signals["cot_words"] = len(cot.split())
    
    # Does it mention the label components?
    goal, basis = label.split("-")
    signals["mentions_goal"] = goal.lower() in cot.lower()
    signals["mentions_basis"] = basis.lower() in cot.lower()
    signals["mentions_full_label"] = label.lower() in cot.lower() or (
        goal.lower() in cot.lower() and basis.lower() in cot.lower()
    )
    
    # Structure signals
    signals["has_numbered_steps"] = any(f"{i}." in cot for i in range(1, 6))
    signals["has_because"] = "because" in cot.lower()
    signals["has_therefore"] = "therefore" in cot.lower() or "thus" in cot.lower()
    
    # Contrast signal — does it explain why NOT the other labels?
    other_goals = ["epistemic", "practical"]
    other_bases = ["internal", "external"]
    signals["contrasts_goal"] = any(
        f"not {g}" in cot.lower() or f"isn't {g}" in cot.lower() or f"rather than {g}" in cot.lower()
        for g in other_goals if g != goal
    )
    signals["contrasts_basis"] = any(
        f"not {b}" in cot.lower() or f"isn't {b}" in cot.lower() or f"rather than {b}" in cot.lower()
        for b in other_bases if b != basis
    )
    
    # Red flags
    signals["too_short"] = len(cot.split()) < 30
    signals["generic_filler"] = any(phrase in cot.lower() for phrase in [
        "this is a clear example",
        "it's obvious that",
        "clearly falls into",
    ])
    
    return signals


# Print detailed audit for each sample
for i, row in enumerate(sample):
    print(f"\n{'─'*80}")
    print(f"SAMPLE {i+1}/20 | ID: {row['id']} | LABEL: {row['label']}")
    print(f"{'─'*80}")
    
    # Truncate text for readability
    text = row["text"]
    if len(text) > 400:
        text = text[:400] + "..."
    print(f"\nINPUT: {text}")
    print(f"\nCOT ({len(row['reasoning_chain'].split())} words):")
    cot = row["reasoning_chain"]
    if len(cot) > 800:
        cot = cot[:800] + "..."
    print(cot)
    
    sigs = analyze_cot(row)
    flags = []
    if sigs["mentions_full_label"]: flags.append("LABEL_MENTIONED")
    if sigs["has_numbered_steps"]: flags.append("STRUCTURED")
    if sigs["has_because"]: flags.append("CAUSAL")
    if sigs["contrasts_goal"]: flags.append("CONTRASTS_GOAL")
    if sigs["contrasts_basis"]: flags.append("CONTRASTS_BASIS")
    if sigs["too_short"]: flags.append("⚠️ TOO_SHORT")
    if sigs["generic_filler"]: flags.append("⚠️ GENERIC")
    print(f"\nSIGNALS: {' | '.join(flags)}")

# Aggregate statistics
print(f"\n{'='*80}")
print("AGGREGATE QUALITY SIGNALS (all {len(enh_rows)} enhanced)")
print(f"{'='*80}")

all_signals = [analyze_cot(r) for r in enh_rows]
n = len(all_signals)

print(f"Mean CoT length: {sum(s['cot_words'] for s in all_signals)/n:.0f} words")
print(f"Mentions full label: {sum(s['mentions_full_label'] for s in all_signals)/n:.1%}")
print(f"Mentions goal component: {sum(s['mentions_goal'] for s in all_signals)/n:.1%}")
print(f"Mentions basis component: {sum(s['mentions_basis'] for s in all_signals)/n:.1%}")
print(f"Has numbered steps: {sum(s['has_numbered_steps'] for s in all_signals)/n:.1%}")
print(f"Has causal language: {sum(s['has_because'] for s in all_signals)/n:.1%}")
print(f"Has conclusion language: {sum(s['has_therefore'] for s in all_signals)/n:.1%}")
print(f"Contrasts alternative goal: {sum(s['contrasts_goal'] for s in all_signals)/n:.1%}")
print(f"Contrasts alternative basis: {sum(s['contrasts_basis'] for s in all_signals)/n:.1%}")
print(f"Too short (<30 words): {sum(s['too_short'] for s in all_signals)/n:.1%}")
print(f"Generic filler: {sum(s['generic_filler'] for s in all_signals)/n:.1%}")

# Per-class breakdown
print(f"\nPer-class CoT quality:")
for label in sorted(enh_labels):
    label_sigs = [s for r, s in zip(enh_rows, all_signals) if r["label"] == label]
    nl = len(label_sigs)
    print(f"  {label} (n={nl}):")
    print(f"    mean words: {sum(s['cot_words'] for s in label_sigs)/nl:.0f}")
    print(f"    mentions_label: {sum(s['mentions_full_label'] for s in label_sigs)/nl:.1%}")
    print(f"    structured: {sum(s['has_numbered_steps'] for s in label_sigs)/nl:.1%}")
    print(f"    contrasts_goal: {sum(s['contrasts_goal'] for s in label_sigs)/nl:.1%}")
    print(f"    contrasts_basis: {sum(s['contrasts_basis'] for s in label_sigs)/nl:.1%}")

# Base track quick check
print(f"\n{'='*80}")
print(f"BASE TRACK SAMPLE ({len(base_sample)} examples)")
print(f"{'='*80}")
for i, row in enumerate(base_sample):
    print(f"\n--- Base {i+1} | {row['id']} | {row['label']} ---")
    cot = row.get("reasoning_chain", "NO COT")
    print(f"CoT ({len(cot.split())} words): {cot[:500]}{'...' if len(cot) > 500 else ''}")
    sigs = analyze_cot(row)
    print(f"  label_mentioned={sigs['mentions_full_label']} structured={sigs['has_numbered_steps']}")
