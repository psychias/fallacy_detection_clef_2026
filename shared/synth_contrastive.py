"""
shared/synth_contrastive.py — Contrastive-pair synthetic data generation.

PATCH NOTES (v2):
- Per-type legitimate counterparts (no more universal "grounded in evidence")
- Explicit no-statistics rule on legitimate side (kills the +72% citation gap)
- Stock-phrase blocklist in system prompt + post-gen regex filter
- Variable target length + variable supports count (kills "always 2 supports, 40-70 words")
- Two-field generation: text_raw is parent-dependent, text_base is self-contained
- Verifier checks for surface artifacts (citations, stock phrases) and drops /no_think
"""

import json
import random
import re
from pathlib import Path

# ── Fallacy cards: definition + load-bearing signals + legitimate counterpart ─

FALLACY_CARDS = {
    "authority": {
        "type": "authority",
        "definition": "Citing an authority figure as evidence without proper justification — the authority is irrelevant, unqualified on the topic, or the claim relies solely on their status.",
        "load_bearing_signals": "name-dropping a figure, credential flex irrelevant to the claim, vague gestures at 'experts' without specifying field or work.",
        "legitimate_counterpart": "an argument that mentions a relevant qualified person, but where the reasoning rests on what that person actually demonstrated, observed, or argued — not just on their status.",
    },
    "black-white": {
        "type": "black-white",
        "definition": "Presenting only two options (usually one extreme) when more alternatives exist. Forces a false dilemma.",
        "load_bearing_signals": "framing the choice as binary when it isn't, collapsing a spectrum to two endpoints, ruling out middle positions implicitly.",
        "legitimate_counterpart": "an argument about a situation where the binary really is binary, OR where the speaker acknowledges a spectrum and argues for one end on its merits.",
    },
    "hasty_generalization": {
        "type": "hasty_generalization",
        "definition": "Drawing a broad conclusion from a sample that is too small, biased, or unrepresentative.",
        "load_bearing_signals": "single-anecdote-to-rule moves, n=1 reasoning generalized to populations, small-sample claims about whole groups.",
        "legitimate_counterpart": "a generalization explicitly qualified by sample size or scope ('in my experience over years', 'across the people I've worked with'), or that limits its conclusion to the observed cases rather than projecting universally.",
    },
    "natural": {
        "type": "natural",
        "definition": "Arguing something is good, right, or better simply because it is natural, or bad because it is artificial/unnatural.",
        "load_bearing_signals": "treating natural as inherently safe/healthy/correct, treating synthetic as inherently bad, ancestral-practice-as-proof.",
        "legitimate_counterpart": "an argument about a natural product or process where the reasoning is the actual mechanism or observed outcome, not its naturalness.",
    },
    "population": {
        "type": "population",
        "definition": "Arguing something is true or good because many people believe it or do it (ad populum / bandwagon).",
        "load_bearing_signals": "popularity treated as proof of correctness, mass adoption invoked as validation, dissent dismissed because it is rare.",
        "legitimate_counterpart": "an argument that mentions wide adoption but uses it to establish something popularity actually evidences (preference, market fit, social norm) — not truth or moral rightness.",
    },
    "slippery_slope": {
        "type": "slippery_slope",
        "definition": "Claiming one event will inevitably lead to a chain of extreme consequences without adequate justification for each causal link.",
        "load_bearing_signals": "unjustified causal chain, jumping from a small change to an extreme outcome, treating each step as automatic.",
        "legitimate_counterpart": "a causal-chain argument where each step has a stated mechanism or precedent, and the conclusion is proportionate to what's been justified.",
    },
    "tradition": {
        "type": "tradition",
        "definition": "Arguing something is correct, good, or should continue simply because it has been done that way for a long time.",
        "load_bearing_signals": "age of practice as sole justification, appeals to how things have always been, treating change as wrong by default.",
        "legitimate_counterpart": "an argument that a long-standing practice works, where the reasoning is the specific outcomes it produces — or the specific costs of changing it — rather than its age.",
    },
    "worse_problems": {
        "type": "worse_problems",
        "definition": "Dismissing a problem or concern by pointing to a worse problem (relative privation / 'whataboutism').",
        "load_bearing_signals": "dismissal of the topic via comparison to bigger issues, refusing engagement on the grounds that something else matters more.",
        "legitimate_counterpart": "an argument that engages with the original problem, acknowledges it is real, and makes an explicit prioritization case rather than dismissing.",
    },
}

# ── Scheme cards for ST3 ──────────────────────────────────────────────

SCHEME_CARDS = {
    "practical-external": {
        "goal": "practical",
        "basis": "external",
        "definition": "Advocating for or against a COURSE OF ACTION based on what AUTHORITIES, EXPERTS, or POPULAR OPINION say about it.",
        "load_bearing_signals": "appealing to authorities/experts/tradition to justify an action, deferring the action question to a source.",
        "contrast_with": "practical-internal",
        "contrast_description": "Same practical goal, but justified by analyzing the action's own properties (consequences, causes, values) rather than citing sources.",
    },
    "practical-internal": {
        "goal": "practical",
        "basis": "internal",
        "definition": "Advocating for or against a COURSE OF ACTION by analyzing the action's own PROPERTIES — consequences, causes, costs, values.",
        "load_bearing_signals": "consequence analysis, cost-benefit, causal reasoning about the action itself.",
        "contrast_with": "practical-external",
        "contrast_description": "Same practical goal, but justified by citing authorities or external sources rather than analyzing properties.",
    },
    "epistemic-external": {
        "goal": "epistemic",
        "basis": "external",
        "definition": "Establishing a FACTUAL CLAIM or JUDGMENT based on AUTHORITIES, STUDIES, or POPULAR BELIEF.",
        "load_bearing_signals": "deferring the truth-question to a source — an expert, a study, a consensus.",
        "contrast_with": "epistemic-internal",
        "contrast_description": "Same epistemic goal, but justified by direct evidence, logic, or properties of the subject rather than citing sources.",
    },
    "epistemic-internal": {
        "goal": "epistemic",
        "basis": "internal",
        "definition": "Establishing a FACTUAL CLAIM or JUDGMENT based on direct EVIDENCE, LOGIC, or PROPERTIES of the subject.",
        "load_bearing_signals": "direct evidence, logical deduction, observable properties of the subject.",
        "contrast_with": "epistemic-external",
        "contrast_description": "Same epistemic goal, but justified by citing authorities or studies rather than analyzing properties directly.",
    },
}

# ── Topic seeds (unchanged) ───────────────────────────────────────────

TOPIC_SEEDS = [
    # Politics / policy
    ("universal basic income", "r/politics"),
    ("immigration policy reform", "r/news"),
    ("gun control legislation", "r/neutralpolitics"),
    ("electoral college vs popular vote", "r/PoliticalDiscussion"),
    ("minimum wage increase to $20/hr", "r/economics"),
    ("mandatory voting", "r/AskReddit"),
    ("police funding and reform", "r/news"),
    ("free speech on social media", "r/technology"),
    ("drug decriminalization", "r/politics"),
    ("universal healthcare", "r/healthcare"),
    ("ranked choice voting", "r/PoliticalDiscussion"),
    ("death penalty abolition", "r/law"),
    ("AI replacing creative jobs", "r/technology"),
    ("self-driving cars safety", "r/SelfDrivingCars"),
    ("social media effects on teens", "r/parenting"),
    ("right to repair electronics", "r/technology"),
    ("cryptocurrency as currency", "r/CryptoCurrency"),
    ("remote work productivity", "r/jobs"),
    ("smartphone addiction", "r/nosurf"),
    ("nuclear power for climate", "r/energy"),
    ("gene editing in humans", "r/science"),
    ("surveillance cameras everywhere", "r/privacy"),
    ("TikTok ban", "r/technology"),
    ("electric vehicles vs gas cars", "r/cars"),
    ("intermittent fasting benefits", "r/nutrition"),
    ("mental health medication stigma", "r/mentalhealth"),
    ("organic vs conventional food", "r/nutrition"),
    ("exercise vs medication for depression", "r/science"),
    ("fluoride in drinking water", "r/askscience"),
    ("vegan diet for health", "r/nutrition"),
    ("alternative medicine effectiveness", "r/medicine"),
    ("screen time limits for kids", "r/parenting"),
    ("raw milk safety", "r/food"),
    ("sunscreen chemicals safety", "r/SkincareAddiction"),
    ("college degree worth the debt", "r/personalfinance"),
    ("homeschooling vs public school", "r/education"),
    ("standardized testing value", "r/education"),
    ("coding bootcamps vs CS degree", "r/cscareerquestions"),
    ("phones in classrooms", "r/Teachers"),
    ("gap year before college", "r/college"),
    ("trade schools vs university", "r/careerguidance"),
    ("grade inflation in universities", "r/professors"),
    ("carbon tax effectiveness", "r/environment"),
    ("plastic straw bans impact", "r/environment"),
    ("lab grown meat adoption", "r/Futurology"),
    ("local food vs imported food", "r/sustainability"),
    ("wind farms vs solar farms", "r/energy"),
    ("reusable vs disposable products", "r/ZeroWaste"),
    ("tipping culture in restaurants", "r/unpopularopinion"),
    ("work-life balance in America", "r/antiwork"),
    ("marriage age trends", "r/sociology"),
    ("participation trophies for kids", "r/parenting"),
    ("cancel culture impact", "r/TrueOffMyChest"),
    ("gender pay gap causes", "r/TwoXChromosomes"),
    ("four day work week", "r/WorkReform"),
    ("fast fashion environmental cost", "r/fashion"),
    ("suburban sprawl vs urban living", "r/urbanplanning"),
    ("influencer marketing to kids", "r/parenting"),
    ("esports as real sports", "r/gaming"),
    ("youth sports specialization", "r/sports"),
    ("performance enhancing drugs in sports", "r/sports"),
    ("video game violence effects", "r/gaming"),
    ("loot boxes as gambling", "r/gaming"),
    ("paying college athletes", "r/CollegeBasketball"),
    ("renting vs buying a home", "r/personalfinance"),
    ("stock market index funds vs active", "r/investing"),
    ("student loan forgiveness", "r/StudentLoans"),
    ("wealth tax feasibility", "r/economics"),
    ("trickle-down economics", "r/economics"),
    ("retiring early (FIRE movement)", "r/financialindependence"),
    ("age gaps in relationships", "r/relationship_advice"),
    ("co-parenting strategies", "r/Parenting"),
    ("social media in relationships", "r/relationships"),
    ("long distance relationships viability", "r/LongDistance"),
    ("space exploration funding priorities", "r/space"),
    ("animal testing in research", "r/science"),
    ("nuclear waste storage solutions", "r/energy"),
    ("GMO food safety", "r/science"),
    ("cold plunge health claims", "r/fitness"),
    ("supplements industry regulation", "r/nutrition"),
    ("meal prep vs cooking fresh", "r/MealPrepSunday"),
    ("coffee health effects", "r/coffee"),
    ("standing desks workplace benefits", "r/ergonomics"),
    ("meditation for stress reduction", "r/Meditation"),
    ("homeownership as investment", "r/RealEstate"),
    ("pets in rental apartments", "r/renting"),
    ("tipping delivery drivers", "r/doordash"),
    ("paper books vs e-readers", "r/books"),
    ("daylight saving time abolition", "r/unpopularopinion"),
    ("jury duty system reform", "r/law"),
    ("speed limit enforcement cameras", "r/driving"),
    ("self checkout vs cashiers", "r/mildlyinfuriating"),
    ("dress codes in schools", "r/Teachers"),
    ("noise ordinances in neighborhoods", "r/legaladvice"),
]


# ── NEW: surface-form artifact filters ────────────────────────────────

# Stock phrases the model defaults to. These appeared 5–15× in v1 synth
# and 0× in 938 real examples, so they're learnable shortcuts a classifier
# will overfit to. Reject any pair where either text contains any of these.
STOCK_PHRASES = [
    # population / bandwagon
    r"\bmillions of people can'?t be wrong\b",
    r"\bpeople can'?t be wrong\b",
    r"\beveryone knows\b",
    r"\bmost people agree\b",
    # black-white
    r"\bno middle ground\b",
    r"\bno in between\b",
    r"\bno gray area\b",
    # slippery slope
    r"\bbefore you know it\b",
    r"\bnext thing you know\b",
    r"\bwhere does it end\b",
    r"\bit'?s a slippery slope\b",
    # tradition
    r"\bwe'?ve always done it\b",
    r"\bthat'?s how it'?s always been\b",
    # generic LLM-cosplay openers
    r"\byeah i agree\b",
    r"\bi agree it'?s\b",
    # natural
    r"\bnatural is better\b",
    r"\bchemicals are bad\b",
]
_STOCK_RE = re.compile("|".join(STOCK_PHRASES), re.IGNORECASE)

# Specific-evidence markers. The legitimate side citing fake stats was the
# +72% gap artifact. Reject the pair if the LEGITIMATE side hits any of these.
EVIDENCE_PATTERNS = [
    r"\b\d{1,3}\s*%",                              # "65%", "10 %"
    r"\b\d+\s*(percent|per\s*cent)\b",             # "65 percent"
    r"\b(19|20)\d{2}\b",                           # year tags 1900-2099
    r"\b(?:gallup|pew|reuters|nielsen|ipsos|yougov|harris|cdc|who|nih|fda)\b",
    r"\baccording to (?:a |the )?(?:study|report|poll|survey|paper)\b",
    r"\b(?:a|the|recent|new) (?:study|report|poll|survey|paper) (?:found|shows|showed|says|said)\b",
    r"\bresearch shows\b",
    r"\bstudies show\b",
    r"\b\d+\s*(?:million|billion|thousand)\s+(?:people|users|americans|kids|adults)\b",
]
_EVIDENCE_RE = re.compile("|".join(EVIDENCE_PATTERNS), re.IGNORECASE)


def contains_stock_phrase(text: str) -> bool:
    """True if text uses any of the canonical fallacy formulations."""
    return bool(_STOCK_RE.search(text or ""))


def cites_specific_evidence(text: str) -> bool:
    """True if text cites a specific stat, year-tagged study, or named institution."""
    return bool(_EVIDENCE_RE.search(text or ""))


def passes_artifact_filter(fallacious_text: str, legitimate_text: str) -> tuple[bool, str]:
    """Surface-form filter applied AFTER generation, BEFORE the LLM verifier.
    Returns (passed, reason). Reason is empty string when passed."""
    if contains_stock_phrase(fallacious_text):
        return False, "fallacious_text contains stock phrase"
    if contains_stock_phrase(legitimate_text):
        return False, "legitimate_text contains stock phrase"
    if cites_specific_evidence(legitimate_text):
        return False, "legitimate_text cites specific evidence (would create shortcut)"
    return True, ""


# ── NEW: distributional samplers ──────────────────────────────────────

def sample_target_length(rng: random.Random) -> tuple[int, str]:
    """Sample a target word count to break the 40-70 word collapse.
    Real text_raw runs 6 to 660 words (p10=13, p90=93). We sample buckets
    that cover that range, weighted to roughly match the empirical distribution."""
    bucket = rng.choices(
        ["short", "medium", "long", "very_long"],
        weights=[30, 50, 15, 5],
        k=1,
    )[0]
    if bucket == "short":
        return rng.randint(15, 35), "short — terse, can be a single sentence"
    if bucket == "medium":
        return rng.randint(35, 80), "medium — a normal Reddit reply"
    if bucket == "long":
        return rng.randint(80, 150), "long — a more developed comment"
    return rng.randint(150, 250), "very long — a rant or detailed reply"


def sample_n_supports(rng: random.Random) -> int:
    """Sample number of supports to break the 91% length-2 collapse.
    Real distribution: 1=25%, 2=34%, 3=23%, 4+=15% (rest are 0)."""
    return rng.choices([1, 2, 3, 4], weights=[25, 34, 23, 18], k=1)[0]


def sample_topics(n: int, rng: random.Random) -> list[tuple[str, str]]:
    pool = list(TOPIC_SEEDS)
    rng.shuffle(pool)
    if n <= len(pool):
        return pool[:n]
    result = []
    while len(result) < n:
        rng.shuffle(pool)
        result.extend(pool[:min(n - len(result), len(pool))])
    return result


def get_real_exemplars(train_data: list[dict], fallacy_type: str,
                       n: int = 3, rng: random.Random = None) -> list[dict]:
    if rng is None:
        rng = random.Random(42)
    examples = [e for e in train_data if e.get("fallacy_type") == fallacy_type
                and e.get("fallacy_exists") == 1]
    rng.shuffle(examples)
    return examples[:n]


def get_real_scheme_exemplars(train_data: list[dict], goal: str, basis: str,
                               n: int = 3, rng: random.Random = None) -> list[dict]:
    if rng is None:
        rng = random.Random(42)
    examples = [e for e in train_data
                if e.get("fallacy_exists") == 0
                and e.get("classification", {}).get("argument_goal") == goal
                and e.get("classification", {}).get("argument_basis") == basis]
    rng.shuffle(examples)
    return examples[:n]


def format_exemplar_block(exemplars: list[dict]) -> str:
    lines = []
    for i, ex in enumerate(exemplars, 1):
        text = ex.get("text_base", ex.get("text_raw", ""))[:300]
        arg = ex.get("argument_base", {})
        claim = arg.get("claim", "N/A") if isinstance(arg, dict) else "N/A"
        supports = arg.get("supports", []) if isinstance(arg, dict) else []
        sup_str = "; ".join(s[:100] for s in supports[:2]) if supports else "N/A"
        lines.append(f"{i}. text: \"{text}\"\n   claim: \"{claim}\"\n   supports: \"{sup_str}\"")
    return "\n".join(lines)


# ── Prompt builders ───────────────────────────────────────────────────

# Shared system-prompt blocklist text. These are the v1 stock formulations
# we observed leaking through. The model needs to be told explicitly because
# the FALLACY_CARDS load-bearing signals were being used as a phrase bank.
_BLOCKLIST_TEXT = (
    "Do NOT use any of these stock formulations (fresh wording only): "
    "'millions of people can't be wrong', 'people can't be wrong', "
    "'everyone knows', 'no middle ground', 'no in between', "
    "'before you know it', 'next thing you know', 'where does it end', "
    "'it's a slippery slope', 'we've always done it', "
    "'natural is better', 'chemicals are bad', "
    "'yeah i agree', 'i agree it's'. "
    "If your draft contains any of these phrases, rewrite the draft."
)


def build_st1_st2_contrastive_prompt(
    fallacy_type: str,
    topic: str,
    subreddit: str,
    exemplars: list[dict],
    rng: random.Random | None = None,
) -> list[dict]:
    """Build contrastive pair prompt for ST1+ST2. Now samples target length
    and number of supports per call to break v1's length/structure collapse."""
    if rng is None:
        rng = random.Random()
    card = FALLACY_CARDS[fallacy_type]
    exemplar_block = format_exemplar_block(exemplars)
    target_words, length_desc = sample_target_length(rng)
    n_supports_fal = sample_n_supports(rng)
    n_supports_leg = sample_n_supports(rng)

    system = (
        "You generate training data for a fallacy detection model. The data must read "
        "as authentic Reddit replies: informal register, hedges (\"idk\", \"tbh\", \"iirc\"), "
        "occasional typos, no debate-club phrasing, no signposting. Never name the "
        "fallacy inside the generated text.\n\n"
        "The 'load-bearing signals' you'll be shown describe the LOGICAL PATTERN of the "
        "fallacy, not phrases to use verbatim. Express the same reasoning move with "
        "fresh wording. " + _BLOCKLIST_TEXT + "\n\n"
        "You will produce a CONTRASTIVE PAIR: one reply that commits the target fallacy, "
        "and one reply using structurally similar reasoning that does NOT commit it. "
        "Both replies must address the same topic with claims of similar polarity, so "
        "the only meaningful difference is the soundness of the inference.\n\n"
        "CRITICAL: Do NOT cite specific percentages, year-tagged studies (e.g. '2023 "
        "study'), polls, named institutions (Gallup, Pew, CDC, etc.), or made-up "
        "statistics in EITHER reply. Both replies must reason from lived experience, "
        "anecdote, observation, or general claims. The difference between them is the "
        "soundness of the inference, not the presence of evidence."
    )

    sup_fal = ', '.join(['"..."'] * n_supports_fal)
    sup_leg = ', '.join(['"..."'] * n_supports_leg)

    user = (
        f"FALLACY CARD\n"
        f"type: {card['type']}\n"
        f"definition: {card['definition']}\n"
        f"load-bearing signals: {card['load_bearing_signals']}\n"
        f"legitimate counterpart: {card['legitimate_counterpart']}\n\n"
        f"EXEMPLARS (real, from train split)\n{exemplar_block}\n\n"
        f"topic_seed: \"{topic}\"\n"
        f"subreddit: \"{subreddit}\"\n"
        f"target length for each reply: ~{target_words} words ({length_desc})\n"
        f"# of supports — fallacious: {n_supports_fal}, legitimate: {n_supports_leg}\n\n"
        "Write a plausible thread title and a parent comment stating a moderate position "
        "on the topic. Then write two replies to the parent.\n\n"
        "For each reply, produce TWO text fields:\n"
        "  - text_raw: a Reddit reply that genuinely depends on the parent — uses "
        "    pronouns ('that's', 'this', 'they'), doesn't restate the topic, can be "
        "    elliptical or incomplete. Imagine someone scrolling and replying without "
        "    re-explaining context.\n"
        "  - text_base: a self-contained rewrite of text_raw that integrates the topic "
        "    and parent context, so a reader who hasn't seen the thread still gets it.\n\n"
        f"- \"fallacious\": commits {card['type']}. The fallacy must be load-bearing — "
        "  if you removed it, the argument would collapse.\n"
        f"- \"legitimate\": {card['legitimate_counterpart']}\n\n"
        "Return ONLY this JSON (use exactly the support counts specified above):\n\n"
        "{\n"
        "  \"thread_title\": \"...\",\n"
        "  \"parent\": \"...\",\n"
        "  \"fallacious\": {\n"
        "    \"text_raw\": \"...\",\n"
        "    \"text_base\": \"...\",\n"
        "    \"claim\": \"...\",\n"
        f"    \"supports\": [{sup_fal}]\n"
        "  },\n"
        "  \"legitimate\": {\n"
        "    \"text_raw\": \"...\",\n"
        "    \"text_base\": \"...\",\n"
        "    \"claim\": \"...\",\n"
        f"    \"supports\": [{sup_leg}]\n"
        "  }\n"
        "}"
    )

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_st3_contrastive_prompt(
    scheme_label: str,
    topic: str,
    subreddit: str,
    exemplars: list[dict],
    rng: random.Random | None = None,
) -> list[dict]:
    """ST3 contrast pair. Same length/support sampling as ST1/ST2."""
    if rng is None:
        rng = random.Random()
    card = SCHEME_CARDS[scheme_label]
    contrast_card = SCHEME_CARDS[card["contrast_with"]]
    exemplar_block = format_exemplar_block(exemplars)
    target_words, length_desc = sample_target_length(rng)
    n_supports_a = sample_n_supports(rng)
    n_supports_b = sample_n_supports(rng)

    system = (
        "You generate training data for an argumentation scheme classifier. The data must "
        "read as authentic Reddit replies: informal register, hedges, occasional typos, "
        "natural conversation. No academic phrasing.\n\n"
        + _BLOCKLIST_TEXT + "\n\n"
        "You will produce a CONTRASTIVE PAIR: two non-fallacious replies to the same "
        "parent comment, on the same topic, with similar conclusions — but using DIFFERENT "
        "argumentation schemes. The structural difference should be clear but natural."
    )

    sup_a = ', '.join(['"..."'] * n_supports_a)
    sup_b = ', '.join(['"..."'] * n_supports_b)

    user = (
        f"SCHEME A: {scheme_label}\n"
        f"goal: {card['goal']} ({'advocate for/against an action' if card['goal'] == 'practical' else 'establish a factual claim'})\n"
        f"basis: {card['basis']} ({'cite authorities/sources' if card['basis'] == 'external' else 'analyze properties/consequences'})\n"
        f"definition: {card['definition']}\n"
        f"signals: {card['load_bearing_signals']}\n\n"
        f"SCHEME B (contrast): {card['contrast_with']}\n"
        f"definition: {contrast_card['definition']}\n"
        f"signals: {contrast_card['load_bearing_signals']}\n"
        f"key difference: {card['contrast_description']}\n\n"
        f"EXEMPLARS of Scheme A (real, from train split)\n{exemplar_block}\n\n"
        f"topic_seed: \"{topic}\"\n"
        f"subreddit: \"{subreddit}\"\n"
        f"target length for each reply: ~{target_words} words ({length_desc})\n"
        f"# of supports — scheme_a: {n_supports_a}, scheme_b: {n_supports_b}\n\n"
        "Write a plausible thread title and parent comment. Then write two NON-FALLACIOUS "
        "replies reaching similar conclusions but using different schemes.\n\n"
        "For each reply produce TWO text fields:\n"
        "  - text_raw: a Reddit reply that depends on the parent (pronouns, ellipses, "
        "    no topic restatement)\n"
        "  - text_base: a self-contained rewrite\n\n"
        f"- \"scheme_a\": uses {scheme_label} reasoning ({card['basis']} basis)\n"
        f"- \"scheme_b\": uses {card['contrast_with']} reasoning ({contrast_card['basis']} basis)\n\n"
        "Return ONLY this JSON (use exactly the support counts specified above):\n\n"
        "{\n"
        "  \"thread_title\": \"...\",\n"
        "  \"parent\": \"...\",\n"
        "  \"scheme_a\": {\n"
        f"    \"scheme\": \"{scheme_label}\",\n"
        f"    \"argument_goal\": \"{card['goal']}\",\n"
        f"    \"argument_basis\": \"{card['basis']}\",\n"
        "    \"text_raw\": \"...\",\n"
        "    \"text_base\": \"...\",\n"
        "    \"claim\": \"...\",\n"
        f"    \"supports\": [{sup_a}]\n"
        "  },\n"
        "  \"scheme_b\": {\n"
        f"    \"scheme\": \"{card['contrast_with']}\",\n"
        f"    \"argument_goal\": \"{contrast_card['goal']}\",\n"
        f"    \"argument_basis\": \"{contrast_card['basis']}\",\n"
        "    \"text_raw\": \"...\",\n"
        "    \"text_base\": \"...\",\n"
        "    \"claim\": \"...\",\n"
        f"    \"supports\": [{sup_b}]\n"
        "  }\n"
        "}"
    )

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ── Verifier prompts ──────────────────────────────────────────────────
# Note: /no_think dropped. The verifier is the cheap step (one call per
# sample) and we want it to actually deliberate. We also added explicit
# checks for the surface artifacts that hurt v1 transfer.

def build_st2_verifier_prompt(text: str, fallacy_type: str,
                               is_legitimate_side: bool = False) -> list[dict]:
    """Verifier: does this text commit (or not commit) the claimed fallacy
    as a load-bearing inference, AND is it free of the surface artifacts
    that would create shortcuts in a downstream classifier?"""
    role = "legitimate counterpart (should NOT commit the fallacy)" if is_legitimate_side \
        else "fallacious example (should commit the fallacy load-bearingly)"
    return [
        {"role": "system", "content": (
            "You are a fallacy detection expert evaluating synthetic training data. "
            "You will be shown a short text, a claimed fallacy type, and which "
            "side of the contrastive pair it represents. Decide whether the text "
            "is high-quality synthetic data — which means it must (a) match its "
            "claimed role and (b) avoid surface artifacts that would create "
            "shortcuts for a downstream classifier. Answer ONLY with a JSON object."
        )},
        {"role": "user", "content": (
            f"Text: \"{text}\"\n\n"
            f"Claimed fallacy: {fallacy_type}\n"
            f"Role in pair: {role}\n\n"
            "Evaluate:\n"
            f"1. commits_fallacy_correctly — does this text {'NOT commit' if is_legitimate_side else 'commit'} "
            f"the fallacy as expected for its role? (yes/no)\n"
            f"2. fallacy_is_load_bearing — {'N/A, answer yes' if is_legitimate_side else 'would removing the fallacious move collapse the argument?'} (yes/no)\n"
            "3. natural_register — does it read like genuine Reddit writing, not a textbook example? (yes/no)\n"
            "4. cites_fake_evidence — does it cite specific percentages, year-tagged studies, polls, or named institutions like Gallup/Pew/CDC? (yes/no — yes is BAD)\n"
            "5. uses_stock_phrasing — does it contain canonical fallacy phrases like 'millions can't be wrong', 'no middle ground', 'before you know it', 'where does it end', 'we've always done it'? (yes/no — yes is BAD)\n\n"
            'Answer as: {"commits_fallacy_correctly": bool, "fallacy_is_load_bearing": bool, "natural_register": bool, "cites_fake_evidence": bool, "uses_stock_phrasing": bool}'
        )},
    ]


def build_st3_verifier_prompt(text: str, goal: str, basis: str) -> list[dict]:
    """ST3 verifier — also checks for surface artifacts."""
    scheme = f"{goal}-{basis}"
    return [
        {"role": "system", "content": (
            "You are an argumentation scheme expert evaluating synthetic training data. "
            "Decide whether the text matches its claimed scheme AND is free of surface "
            "artifacts that would create shortcuts. Answer ONLY with JSON."
        )},
        {"role": "user", "content": (
            f"Text: \"{text}\"\n\n"
            f"Claimed scheme: {scheme}\n"
            f"- goal={goal}: {'advocates for/against a course of action' if goal == 'practical' else 'establishes a factual claim/judgment'}\n"
            f"- basis={basis}: {'cites authorities, experts, or popular opinion' if basis == 'external' else 'analyzes properties, consequences, or direct evidence'}\n\n"
            "Evaluate:\n"
            f"1. correct_goal — does this argument have a {goal} goal? (yes/no)\n"
            f"2. correct_basis — does it primarily use {basis} basis? (yes/no)\n"
            "3. natural_register — does it read like genuine Reddit writing? (yes/no)\n"
            "4. cites_fake_evidence — specific %, year-tagged studies, named pollsters? (yes/no — yes is BAD unless this is epistemic-external)\n"
            "5. uses_stock_phrasing — canonical phrases the model defaults to? (yes/no — yes is BAD)\n\n"
            'Answer as: {"correct_goal": bool, "correct_basis": bool, "natural_register": bool, "cites_fake_evidence": bool, "uses_stock_phrasing": bool}'
        )},
    ]


def parse_verifier_response(content: str) -> dict:
    """Parse verifier JSON, handling code fences and stray text."""
    content = content.strip()
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(l for l in lines if not l.strip().startswith("```"))
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r'\{[^{}]+\}', content)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {}


def passes_st2_verification(result: dict, is_legitimate_side: bool = False) -> bool:
    """ST2 pass criteria. Note the asymmetry: load_bearing only matters for the
    fallacious side, but the artifact checks apply to both."""
    role_ok = result.get("commits_fallacy_correctly", False) is True
    load_bearing_ok = is_legitimate_side or (result.get("fallacy_is_load_bearing", False) is True)
    natural_ok = result.get("natural_register", False) is True
    no_evidence = result.get("cites_fake_evidence", True) is False
    no_stock = result.get("uses_stock_phrasing", True) is False
    return role_ok and load_bearing_ok and natural_ok and no_evidence and no_stock


def passes_st3_verification(result: dict, basis: str = "") -> bool:
    """ST3 pass criteria. epistemic-external arguments legitimately cite
    sources, so we relax the evidence check for that scheme only."""
    goal_ok = result.get("correct_goal", False) is True
    basis_ok = result.get("correct_basis", False) is True
    natural_ok = result.get("natural_register", False) is True
    no_stock = result.get("uses_stock_phrasing", True) is False
    if basis == "external":
        # citing sources is the whole point of an external-basis argument
        return goal_ok and basis_ok and natural_ok and no_stock
    no_evidence = result.get("cites_fake_evidence", True) is False
    return goal_ok and basis_ok and natural_ok and no_evidence and no_stock