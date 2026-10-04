"""
Confidence and Communication Analysis Module (Module 4)
Analyzes explainable linguistic indicators: filler words, uncertainty language,
hedging phrases, grammar structure, clarity, and communication metrics.

DISCLAIMER: Text-based linguistic estimation only.
These indicators reflect writing and phrasing patterns; they do not measure
voice acoustics, facial emotion, or psychological traits.
"""

import re
from .text_processing import clean_text, split_sentences, tokenize

FILLER_PATTERNS = [
    r'\bum\b', r'\buh\b', r'\blike\b', r'\byou know\b', r'\bactually\b',
    r'\bbasically\b', r'\bsort of\b', r'\bsorta\b', r'\bkinda\b', r'\bkind of\b',
    r'\bi mean\b', r'\bso yeah\b', r'\bliterally\b'
]

UNCERTAINTY_PATTERNS = [
    r'\bmaybe\b', r'\bi think\b', r'\bi guess\b', r'\bi\'m not sure\b',
    r'\bim not sure\b', r'\bprobably\b', r'\bperhaps\b', r'\bmight be\b',
    r'\bnot really sure\b', r'\bnot certain\b', r'\bcould be\b', r'\bi suppose\b'
]

HEDGING_PATTERNS = [
    r'\bsomething like\b', r'\bsomething related to\b', r'\bkind of like\b',
    r'\bmore or less\b', r'\bor whatever\b', r'\band stuff\b'
]

def count_pattern_matches(patterns: list[str], text: str) -> int:
    """Count regex pattern occurrences in lowercase text."""
    text_lower = text.lower()
    total = 0
    for pat in patterns:
        matches = re.findall(pat, text_lower)
        total += len(matches)
    return total

def evaluate_grammar_and_structure(text: str) -> float:
    """
    Evaluate basic sentence structure and punctuation quality.
    Returns a score between 0.0 and 1.0.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return 0.0

    sentences = split_sentences(cleaned)
    if not sentences:
        return 0.2

    score = 0.5

    # Check sentence beginnings for capitalization
    cap_count = sum(1 for s in sentences if s[0].isupper())
    cap_ratio = cap_count / len(sentences)
    score += 0.25 * cap_ratio

    # Check terminal punctuation (. ! ?)
    punct_count = sum(1 for s in sentences if s[-1] in '.!?')
    punct_ratio = punct_count / len(sentences)
    score += 0.25 * punct_ratio

    # Check sentence length appropriateness (5 - 40 words per sentence)
    tokens = tokenize(cleaned)
    avg_words = len(tokens) / len(sentences)
    if 6 <= avg_words <= 35:
        score += 0.1
    elif avg_words < 4:
        score -= 0.15

    return max(0.1, min(1.0, round(score, 2)))

def evaluate_clarity(text: str, filler_count: int, uncertainty_count: int) -> float:
    """
    Evaluate clarity and conciseness.
    Returns a score between 0.0 and 1.0.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return 0.0

    tokens = tokenize(cleaned)
    if len(tokens) == 0:
        return 0.0

    base = 0.85

    # Penalize filler and uncertainty density
    penalty = (filler_count * 0.08) + (uncertainty_count * 0.10)
    base -= penalty

    # Bonus for clean, moderate length answers
    if 25 <= len(tokens) <= 150:
        base += 0.05

    # Penalize extremely short answers (< 6 words)
    if len(tokens) < 6:
        base -= 0.30

    return max(0.15, min(0.98, round(base, 2)))

def analyze_answer_linguistics(answer_text: str) -> dict:
    """
    Analyze linguistic confidence and communication features for a single answer.
    Directly aligns with the MockMind dataset Confidence_Features schema.
    """
    cleaned = clean_text(answer_text)
    if not cleaned:
        return {
            'filler_word_count': 0,
            'uncertainty_word_count': 0,
            'hedging_word_count': 0,
            'grammar_quality': 0.0,
            'clarity_score': 0.0,
            'confidence_score': 0.0,
            'communication_score': 0.0,
            'word_count': 0,
            'indicators_summary': 'No response provided.'
        }

    tokens = tokenize(cleaned)
    word_count = len(tokens)

    filler_count = count_pattern_matches(FILLER_PATTERNS, cleaned)
    uncertainty_count = count_pattern_matches(UNCERTAINTY_PATTERNS, cleaned)
    hedging_count = count_pattern_matches(HEDGING_PATTERNS, cleaned)

    grammar_quality = evaluate_grammar_and_structure(cleaned)
    clarity_score = evaluate_clarity(cleaned, filler_count, uncertainty_count)

    # Base confidence calculation (0 - 100 scale)
    base_conf = 88.0
    conf_penalty = (filler_count * 8.0) + (uncertainty_count * 11.0) + (hedging_count * 7.0)
    conf_score = base_conf - conf_penalty

    # Structural clarity boost
    if grammar_quality >= 0.8 and word_count >= 15:
        conf_score += 4.0

    # Minimum floor for empty/very brief responses
    if word_count < 6:
        conf_score = min(conf_score, 40.0)

    conf_score = max(15.0, min(96.0, round(conf_score, 1)))

    # Communication score (0 - 100 scale): combination of clarity, structure, and articulation
    comm_score = (clarity_score * 45.0) + (grammar_quality * 35.0) + (min(1.0, word_count / 30.0) * 20.0)
    comm_score = max(20.0, min(95.0, round(comm_score, 1)))

    # Generate helpful human-readable indicator summary
    insights = []
    if filler_count > 0:
        insights.append(f"{filler_count} filler word(s) detected")
    if uncertainty_count > 0:
        insights.append(f"{uncertainty_count} uncertainty marker(s) detected")
    if hedging_count > 0:
        insights.append(f"{hedging_count} hedging phrase(s) detected")
    if not insights:
        insights.append("Clear and assertive phrasing")

    return {
        'filler_word_count': filler_count,
        'uncertainty_word_count': uncertainty_count,
        'hedging_word_count': hedging_count,
        'grammar_quality': grammar_quality,
        'clarity_score': clarity_score,
        'confidence_score': conf_score,
        'communication_score': comm_score,
        'word_count': word_count,
        'indicators_summary': '; '.join(insights)
    }

def analyze_performance(user_answers_data: list) -> dict:
    """
    Placeholder matching original function signature for backward compatibility.
    Aggregates linguistic communication metrics across a list of answers.
    """
    if not user_answers_data:
        return {'avg_confidence': 0.0, 'avg_communication': 0.0}

    conf_scores = []
    comm_scores = []
    for item in user_answers_data:
        text = item if isinstance(item, str) else item.get('user_answer', '')
        res = analyze_answer_linguistics(text)
        conf_scores.append(res['confidence_score'])
        comm_scores.append(res['communication_score'])

    return {
        'avg_confidence': round(sum(conf_scores) / len(conf_scores), 1) if conf_scores else 0.0,
        'avg_communication': round(sum(comm_scores) / len(comm_scores), 1) if comm_scores else 0.0
    }
