"""Reference-free OCR quality metrics.

We have no ground-truth transcription of 1940s newsprint, so quality is
estimated three ways:
  * dict_hit_rate  — fraction of word-tokens that are real English words
  * fluency        — mean word frequency (zipf) of recognised words
  * consensus      — agreement with the OTHER models on the same page
Plus raw volume (words/chars) and speed (handled by the caller).
"""
from __future__ import annotations
import re
from wordfreq import zipf_frequency

WORD_RE = re.compile(r"[A-Za-z']{2,}")
# Words AND numbers — dates, addresses, prices are first-class search targets
# (2.8% of reference tokens contain digits; WORD_RE never sees them).
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'/\-]*")
DIGIT_RE = re.compile(r"\d")
# Proper-noun-ish word in original case: "Anaheim", "Pellegrino"
CAP_RE = re.compile(r"^[A-Z][a-z']{2,}$")
# Line-break hyphenation: "Laugh-\nlin" -> "Laughlin". Newsprint hyphenates
# constantly; Claude/Gemini transcribe it literally ("exactly as printed"),
# GPT/GLM rejoin. Without this, the consensus truth holds fragments
# (laugh+lin) and penalizes models that correctly join — and an FTS5 query
# for "Laughlin" misses the page. Join before tokenizing, everywhere.
DEHYPH_RE = re.compile(r"(\w)-[ \t]*\n[ \t]*(\w)")


def dehyphenate(text: str) -> str:
    return DEHYPH_RE.sub(r"\1\2", text or "")


def words(text: str) -> list[str]:
    return WORD_RE.findall(dehyphenate(text))


def text_metrics(text: str) -> dict:
    toks = [w.lower().strip("'") for w in words(text)]
    n = len(toks)
    if n == 0:
        return {"chars": len(text or ""), "words": 0,
                "dict_hit_rate": 0.0, "fluency": 0.0, "alpha_ratio": 0.0}
    zipfs = [zipf_frequency(w, "en") for w in toks]
    known = [z for z in zipfs if z > 0]
    nonspace = [c for c in (text or "") if not c.isspace()]
    alpha = sum(c.isalpha() for c in nonspace)
    return {
        "chars": len(text or ""),
        "words": n,
        "dict_hit_rate": round(len(known) / n, 4),
        "fluency": round(sum(known) / len(known), 3) if known else 0.0,
        "alpha_ratio": round(alpha / len(nonspace), 4) if nonspace else 0.0,
        "max_repeat_run": max_repeat_run(text),
    }


def _norm_words(text: str) -> list[str]:
    return [w.lower() for w in words(text)]


def word_error_rate(hyp: str, ref: str) -> float:
    """Levenshtein edit distance over words, normalized by reference length.

    Standard OCR accuracy metric vs a ground-truth `ref`. 0.0 = perfect,
    higher = worse. Capped DP keeps it fast on full newspaper pages.
    """
    a, b = _norm_words(hyp), _norm_words(ref)
    if not b:
        return 0.0 if not a else 1.0
    # classic DP edit distance (rows over ref, cols over hyp)
    prev = list(range(len(a) + 1))
    for i, rw in enumerate(b, 1):
        cur = [i] + [0] * len(a)
        for j, hw in enumerate(a, 1):
            cost = 0 if rw == hw else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return round(prev[len(a)] / len(b), 4)


def char_similarity(hyp: str, ref: str) -> float:
    """difflib ratio on characters — 1.0 = identical, 0.0 = nothing in common.

    autojunk=False: the heuristic treats characters appearing >1% of the time as
    'junk', which wrecks the ratio on long natural-language text. Disable it.
    """
    from difflib import SequenceMatcher
    if not ref and not hyp:
        return 1.0
    return round(SequenceMatcher(None, hyp or "", ref or "", autojunk=False).ratio(), 4)


def bag_of_words_f1(hyp: str, ref: str) -> dict:
    """Order-independent word-overlap. Fair when models differ in column reading
    order: compares word MULTISETS (so repeats are capped) → precision/recall/F1.
    recall = how much of the reference's text the model captured.
    """
    from collections import Counter
    h, r = Counter(_norm_words(hyp)), Counter(_norm_words(ref))
    if not r:
        return {"bow_precision": 0.0, "bow_recall": 0.0, "bow_f1": 0.0}
    overlap = sum((h & r).values())
    prec = overlap / max(1, sum(h.values()))
    rec = overlap / sum(r.values())
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return {"bow_precision": round(prec, 4), "bow_recall": round(rec, 4),
            "bow_f1": round(f1, 4)}


def accuracy_vs_reference(hyp: str, ref: str) -> dict:
    wer = word_error_rate(hyp, ref)
    out = {
        "wer": wer,
        "word_accuracy": round(max(0.0, 1.0 - wer), 4),
        "char_sim": char_similarity(hyp, ref),
    }
    out.update(bag_of_words_f1(hyp, ref))
    # Search-oriented metrics (None when the ref lacks the token class)
    out["digit_recall"] = digit_recall(hyp, ref)
    out["rare_word_recall"] = rare_word_recall(hyp, ref)
    out.update(phrase_recovery(hyp, ref))
    return out


def max_repeat_run(text: str) -> int:
    """Longest consecutive repetition of the same 1-3 word phrase.

    Repetition loops repeat a token/phrase dozens of times CONSECUTIVELY;
    page-level dict-rate misses a loop confined to one tile of a dense page
    (observed: 45x "shouldn't" inside an otherwise-clean 10k-word page).
    Natural prose stays under ~5 even on tabular pages."""
    toks = [w.lower() for w in TOKEN_RE.findall(text or "")]
    best = 1
    for n in (1, 2, 3):
        run = 1
        for i in range(n, len(toks) - n + 1, n):
            if toks[i:i + n] == toks[i - n:i]:
                run += 1
                best = max(best, run)
            else:
                run = 1
    return best


def digit_tokens(text: str) -> list[str]:
    """Tokens containing a digit ('1949', '$1.25' → '1.25', '123'), lowered."""
    return [t.lower() for t in TOKEN_RE.findall(dehyphenate(text))
            if DIGIT_RE.search(t)]


def digit_recall(hyp: str, ref: str) -> float | None:
    """Multiset recall of digit-bearing tokens (dates/addresses/prices).
    None if the reference has no digit tokens (don't average zeros in)."""
    from collections import Counter
    h, r = Counter(digit_tokens(hyp)), Counter(digit_tokens(ref))
    if not r:
        return None
    return round(sum((h & r).values()) / sum(r.values()), 4)


def rare_word_recall(hyp: str, ref: str, zipf_max: float = 3.5) -> float | None:
    """Recall restricted to RARE reference words (zipf < zipf_max) — proper
    nouns, place names, surnames: the words people actually search for. BOW F1
    is dominated by 'the/and'; this is the number that predicts query hit rate.
    None if the reference has no rare words."""
    from collections import Counter
    rare_ref = Counter(w.lower() for w in words(ref)
                       if zipf_frequency(w.lower(), "en") < zipf_max)
    if not rare_ref:
        return None
    h = Counter(w.lower() for w in words(hyp))
    return round(sum((h & rare_ref).values()) / sum(rare_ref.values()), 4)


def proper_phrases(text: str, max_len: int = 2) -> set[tuple[str, ...]]:
    """Adjacent capitalized-word bigrams from ORIGINAL-case text ('John Smith',
    'Center Street') — proxies for real phrase queries. Lowercased tuples."""
    toks = TOKEN_RE.findall(dehyphenate(text))
    out = set()
    for i in range(len(toks) - 1):
        if CAP_RE.match(toks[i]) and CAP_RE.match(toks[i + 1]):
            out.add((toks[i].lower(), toks[i + 1].lower()))
    return out


def phrase_recovery(hyp: str, ref: str) -> dict:
    """Fraction of the reference's proper-noun bigrams that survive as ADJACENT
    tokens in the hypothesis. FTS5 phrase queries ('"john smith"') need
    adjacency — bag-of-words scoring is blind to tiling scrambling it."""
    ref_ph = proper_phrases(ref)
    if not ref_ph:
        return {"phrases": 0, "phrase_recovery": None}
    toks = [t.lower() for t in TOKEN_RE.findall(dehyphenate(hyp))]
    hyp_bigrams = set(zip(toks, toks[1:]))
    hit = sum(1 for p in ref_ph if p in hyp_bigrams)
    return {"phrases": len(ref_ph),
            "phrase_recovery": round(hit / len(ref_ph), 4)}


def _wordset(text: str) -> set[str]:
    return {w.lower() for w in words(text)}


def consensus_scores(model_texts: dict[str, str]) -> dict[str, float]:
    """For each model, mean Jaccard word-overlap vs every other model.

    High score = this model agrees with the crowd (likely correct).
    Low score on one model = outlier (possible OCR failure or hallucination).
    """
    names = [n for n, t in model_texts.items() if t]
    sets = {n: _wordset(model_texts[n]) for n in names}
    out = {n: 0.0 for n in model_texts}
    if len(names) < 2:
        return out
    for n in names:
        scores = []
        for m in names:
            if m == n:
                continue
            a, b = sets[n], sets[m]
            union = a | b
            scores.append(len(a & b) / len(union) if union else 0.0)
        out[n] = round(sum(scores) / len(scores), 4) if scores else 0.0
    return out
