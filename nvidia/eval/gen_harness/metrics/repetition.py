"""Repetition metrics — catch the degeneracy modes ELBO doesn't see."""

from collections import Counter


def _ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def rep_n(completion_list, n=4):
    """Fraction of n-grams that are repeats (any earlier occurrence).

    Welleck et al. 2019 neural text degeneration paper.
    Higher = more repetition.
    """
    scores = []
    for toks in completion_list:
        grams = _ngrams(toks, n)
        if len(grams) < 2:
            continue
        seen = set()
        repeats = 0
        for g in grams:
            if g in seen:
                repeats += 1
            else:
                seen.add(g)
        scores.append(repeats / len(grams))
    return sum(scores) / len(scores) if scores else 0.0


def seq_rep_n(completion_list, n=4):
    """1 - unique_n_grams / total_n_grams per sample, averaged.

    Equivalent of distinct-n flipped — compatible reporting with the Welleck recipe.
    """
    scores = []
    for toks in completion_list:
        grams = _ngrams(toks, n)
        if not grams:
            continue
        scores.append(1.0 - len(set(grams)) / len(grams))
    return sum(scores) / len(scores) if scores else 0.0


def top_word_share(completion_list, k=10):
    """Share of tokens taken by the top-k most common tokens across the corpus.

    Detects stopword-collapse: if top-10 tokens account for >50% of output,
    the model is degenerating.
    """
    counts = Counter()
    for toks in completion_list:
        counts.update(toks)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    top = sum(c for _, c in counts.most_common(k))
    return top / total
