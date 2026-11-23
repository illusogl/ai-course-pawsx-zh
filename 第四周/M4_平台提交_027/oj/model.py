"""Symmetric features and portable offline inference."""
import json
import math
import re
import unicodedata
from difflib import SequenceMatcher
from collections import Counter, defaultdict

USAGE = {"tokens_in": 0, "tokens_out": 0, "cost": 0.0, "tool_calls": 0}
LATIN_NAME = re.compile(r"\b[A-Z][a-z]{2,}(?:[ -][A-Z][a-z]{2,})*\b")
DIRECTIONS = re.compile(r"东北|东南|西北|西南|东|南|西|北")
NEGATIONS = re.compile(r"没有|未曾|不会|不能|并非|不是|不|无|未|没|否")
NUMBERS = re.compile(r"\d+(?:[.,]\d+)*")
# Integer bitsets compute the exact LCS length without a quadratic Python loop.
def lcs_length(left, right):
    masks = defaultdict(int)
    for index, char in enumerate(right):
        masks[char] |= 1 << index
    state = 0
    for char in left:
        union = state | masks[char]
        state = union & ~(union - ((state << 1) | 1))
    return state.bit_count()


def normalize(text):
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


def grams(text, size):
    return {text[index:index + size] for index in range(len(text) - size + 1)} or {text}


def features(pair):
    if not isinstance(pair, list) or len(pair) != 2 or not all(isinstance(s, str) for s in pair):
        raise ValueError("text_pair must contain two strings")
    left, right = map(normalize, pair)
    if not left or not right:
        raise ValueError("empty sentence")
    short, long = sorted((len(left), len(right)))
    values = []
    for size in (1, 2, 3):
        a, b = grams(left, size), grams(right, size)
        shared = len(a & b)
        values.extend((shared / len(a | b), shared / min(len(a), len(b))))
    lcs = lcs_length(left, right)
    common = sum((Counter(left) & Counter(right)).values())
    values.extend((short / long, lcs / short, lcs / long, (common - lcs) / short,
                   int(Counter(NUMBERS.findall(left)) != Counter(NUMBERS.findall(right))),
                   int(Counter(NEGATIONS.findall(left)) != Counter(NEGATIONS.findall(right)))))
    # Average both matching directions because repeated spans can change the greedy alignment.
    alignments = [SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks()
                  for a, b in ((left, right), (right, left))]
    block_lengths = [sorted((block.size for block in blocks if block.size), reverse=True)
                     for blocks in alignments]
    matched_span_ratio = sum(sum(lengths) for lengths in block_lengths) / (short + long)
    longest_span_ratio = sum(lengths[0] if lengths else 0 for lengths in block_lengths) / (2 * short)
    left_numbers, right_numbers = NUMBERS.findall(left), NUMBERS.findall(right)
    left_directions, right_directions = DIRECTIONS.findall(left), DIRECTIONS.findall(right)
    values.extend((
        lcs / max(common, 1), matched_span_ratio, longest_span_ratio,
        (2 * lcs) / (short + long) - matched_span_ratio,
        int(left_numbers != right_numbers and Counter(left_numbers) == Counter(right_numbers)),
        int(left_directions != right_directions and Counter(left_directions) == Counter(right_directions)),
        int(bool(LATIN_NAME.search(pair[0])) != bool(LATIN_NAME.search(pair[1]))),
        values[1] * values[9], values[2] * values[9],
        values[2] * values[10], values[2] * values[11],
    ))
    return values


BASE_FEATURE_NAMES = [
    "unigram_jaccard", "unigram_containment", "bigram_jaccard", "bigram_containment",
    "trigram_jaccard", "trigram_containment", "length_ratio", "lcs_short_ratio",
    "lcs_long_ratio", "character_order_gap", "numeric_mismatch", "negation_mismatch",
]

FEATURE_NAMES = BASE_FEATURE_NAMES + [
    "lcs_shared_ratio", "aligned_span_ratio", "longest_span_ratio", "alignment_gap",
    "number_order_change", "direction_order_change", "latin_name_asymmetry",
    "unigram_containment_order", "bigram_overlap_order",
    "bigram_overlap_numbers", "bigram_overlap_negations",
]


def standardize(values, means, scales):
    return [(value - mean) / scale for value, mean, scale in zip(values, means, scales)]


def decision_score(model, values):
    vector = standardize(values, model["means"], model["scales"])
    if model.get("kind") == "tanh_mlp":
        vector = [math.tanh(bias + sum(w * x for w, x in zip(weights, vector)))
                  for weights, bias in zip(model["hidden_weights"], model["hidden_bias"])]
    return model["bias"] + sum(w * x for w, x in zip(model["weights"], vector))


def classify(model, values):
    return int(decision_score(model, values) >= model.get("threshold", 0.0))


def read_jsonl(path):
    with path.open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
