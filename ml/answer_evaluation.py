"""
Answer Evaluation Module (Module 3)
Evaluates candidate answers against interview questions and ideal answers.
Computes technical score, relevance score, semantic similarity, concept coverage,
covered/missing concepts, quality indicators, and constructive feedback.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .text_processing import (
    clean_text,
    tokenize,
    calculate_token_overlap,
    match_concept_in_text
)
from .gemini_evaluator import is_gemini_configured, evaluate_with_gemini

def parse_keywords(keywords_raw: str | list | None) -> list[str]:
    """Parse keywords string (semicolon or comma separated) into a clean list."""
    if not keywords_raw:
        return []
    if isinstance(keywords_raw, list):
        return [k.strip() for k in keywords_raw if k and k.strip()]

    raw_str = str(keywords_raw)
    delimiters = [';', ',']
    for d in delimiters:
        if d in raw_str:
            return [k.strip() for k in raw_str.split(d) if k.strip()]
    return [raw_str.strip()] if raw_str.strip() else []

def compute_semantic_similarity(user_text: str, ideal_text: str) -> float:
    """
    Compute semantic similarity between candidate answer and ideal answer
    using TF-IDF N-grams cosine similarity and token overlap.
    Returns a calibrated score between 0.0 and 100.0.
    """
    clean_user = clean_text(user_text)
    clean_ideal = clean_text(ideal_text)

    if not clean_user or not clean_ideal:
        return 0.0

    tokens_user = tokenize(clean_user, remove_stopwords=True)
    if len(tokens_user) == 0:
        return 0.0

    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform([clean_ideal, clean_user])
        cos_sim = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
    except Exception:
        cos_sim = 0.0

    overlap = calculate_token_overlap(clean_user, clean_ideal)

    # Calibrate short-answer similarity: raw cosine on 2 short sentences maxes around 0.50-0.60
    norm_score = (cos_sim / 0.52) * 80.0 + (overlap * 30.0)
    score = min(100.0, max(0.0, round(norm_score, 1)))
    return score

def compute_relevance_score(user_text: str, question_text: str, ideal_text: str) -> float:
    """
    Evaluate whether the candidate's answer directly addresses the interview question.
    Returns a score between 0.0 and 100.0.
    """
    clean_user = clean_text(user_text)
    clean_q = clean_text(question_text)
    clean_ideal = clean_text(ideal_text)

    if not clean_user or not clean_q:
        return 0.0

    tokens_user = tokenize(clean_user, remove_stopwords=True)
    if len(tokens_user) == 0:
        return 0.0

    try:
        vec_q = TfidfVectorizer(ngram_range=(1, 2))
        q_matrix = vec_q.fit_transform([clean_q, clean_user])
        q_cos = float(cosine_similarity(q_matrix[0:1], q_matrix[1:2])[0][0])
    except Exception:
        q_cos = 0.0

    try:
        vec_ideal = TfidfVectorizer(ngram_range=(1, 2))
        ideal_matrix = vec_ideal.fit_transform([clean_ideal, clean_user])
        ideal_cos = float(cosine_similarity(ideal_matrix[0:1], ideal_matrix[1:2])[0][0])
    except Exception:
        ideal_cos = 0.0

    q_overlap = calculate_token_overlap(clean_user, clean_q)

    # If the user literally just copied the question word for word
    if clean_user.lower() == clean_q.lower():
        return 20.0

    # Weighted topical alignment
    raw_rel = (q_cos / 0.45) * 45.0 + (ideal_cos / 0.52) * 45.0 + (q_overlap * 20.0)
    score = min(100.0, max(0.0, round(raw_rel, 1)))
    return score

def evaluate_answer(
    question: str,
    ideal_answer: str,
    user_answer: str,
    category: str = "General",
    topic: str = "General",
    difficulty: str = "Medium",
    keywords: str | list | None = None
) -> dict:
    """
    Main answer evaluation entry point.
    Returns comprehensive evaluation dictionary.
    """
    clean_user = clean_text(user_answer)
    clean_q = clean_text(question)
    clean_ideal = clean_text(ideal_answer)

    expected_keywords = parse_keywords(keywords)

    # Check for empty or non-informative answer
    tokens = tokenize(clean_user, remove_stopwords=True)
    if not clean_user or len(tokens) == 0:
        return {
            'technical_score': 0.0,
            'relevance_score': 0.0,
            'semantic_similarity': 0.0,
            'concept_coverage': 0.0,
            'covered_concepts': [],
            'missing_concepts': expected_keywords,
            'answer_quality_indicators': {
                'word_count': 0,
                'status': 'Empty / No substantive answer provided'
            },
            'feedback': 'No substantive answer was provided. To receive credit, explain the core concepts and underlying logic.',
            'evaluation_source': 'Local ML Engine (Empty Input)'
        }

    # Handle extremely short non-answers like "idk", "no", "yes", "none"
    trivial_answers = {'idk', 'dont know', "don't know", 'no idea', 'skip', 'na', 'pass', 'nothing'}
    if clean_user.lower().strip() in trivial_answers or (len(tokens) == 1 and clean_user.lower().strip() not in {'recursion', 'stack', 'queue', 'tree', 'graph', 'lifo', 'fifo'}):
        return {
            'technical_score': 5.0,
            'relevance_score': 5.0,
            'semantic_similarity': 5.0,
            'concept_coverage': 0.0,
            'covered_concepts': [],
            'missing_concepts': expected_keywords,
            'answer_quality_indicators': {
                'word_count': len(tokens),
                'status': 'Trivial / Incomplete answer'
            },
            'feedback': 'The response is too brief to evaluate understanding. Please provide a complete explanation with examples.',
            'evaluation_source': 'Local ML Engine (Trivial Input)'
        }

    # Concept Coverage Analysis
    covered_concepts = []
    missing_concepts = []
    for concept in expected_keywords:
        if match_concept_in_text(concept, clean_user):
            covered_concepts.append(concept)
        else:
            missing_concepts.append(concept)

    if expected_keywords:
        concept_coverage = round((len(covered_concepts) / len(expected_keywords)) * 100.0, 1)
    else:
        concept_coverage = 75.0 if len(tokens) >= 15 else 40.0

    # Semantic Similarity & Relevance
    raw_semantic_sim = compute_semantic_similarity(clean_user, clean_ideal)
    raw_relevance = compute_relevance_score(clean_user, clean_q, clean_ideal)

    # Concept-Aware Paraphrase Adjustment:
    # If the candidate used technical synonyms that covered the required concepts,
    # ensure that semantic similarity and relevance properly credit that conceptual knowledge.
    if concept_coverage > 0:
        semantic_sim = max(raw_semantic_sim, round((0.65 * concept_coverage) + (0.35 * raw_semantic_sim), 1))
        relevance = max(raw_relevance, round((0.60 * concept_coverage) + (0.40 * raw_relevance), 1))
    else:
        semantic_sim = raw_semantic_sim
        relevance = raw_relevance

    # Technical Score Calculation based on category & difficulty
    cat_upper = category.upper() if category else ""
    if "HR" in cat_upper:
        tech_score = (0.35 * concept_coverage) + (0.35 * semantic_sim) + (0.30 * relevance)
    else:
        tech_score = (0.50 * concept_coverage) + (0.30 * semantic_sim) + (0.20 * relevance)

    # Difficulty adjustment
    diff_lower = difficulty.lower() if difficulty else "medium"
    if diff_lower == "hard":
        if missing_concepts and len(missing_concepts) > len(covered_concepts):
            tech_score *= 0.90
    elif diff_lower == "easy":
        if covered_concepts and len(covered_concepts) == len(expected_keywords):
            tech_score = min(100.0, tech_score * 1.04)

    tech_score = max(0.0, min(100.0, round(float(tech_score), 1)))
    relevance = max(0.0, min(100.0, round(float(relevance), 1)))

    # Construct constructive feedback
    covered_str = ", ".join(covered_concepts) if covered_concepts else "none of the target concepts"
    missing_str = ", ".join(missing_concepts) if missing_concepts else "none"

    if tech_score >= 82:
        feedback = f"Excellent answer! Demonstrates strong technical understanding. Effectively covered: {covered_str}."
    elif tech_score >= 60:
        if missing_concepts:
            feedback = f"Solid answer with good foundational points ({covered_str}). To elevate your response, elaborate further on: {missing_str}."
        else:
            feedback = f"Good explanation covering all key concepts ({covered_str}). Aim to add practical code or architecture examples."
    elif tech_score >= 40:
        feedback = f"Partially correct answer. You addressed {covered_str}, but omitted crucial concepts: {missing_str}. Review the ideal mechanism to improve precision."
    else:
        feedback = f"Needs improvement. The response misses core elements: {missing_str}. Focus on directly answering the question and explaining the underlying concepts."

    evaluation_result = {
        'technical_score': tech_score,
        'relevance_score': relevance,
        'semantic_similarity': semantic_sim,
        'concept_coverage': concept_coverage,
        'covered_concepts': covered_concepts,
        'missing_concepts': missing_concepts,
        'answer_quality_indicators': {
            'word_count': len(tokens),
            'concepts_covered_count': len(covered_concepts),
            'concepts_total': len(expected_keywords)
        },
        'feedback': feedback,
        'evaluation_source': 'Local NLP/ML Engine'
    }

    # Optional Gemini AI Enhancer
    if is_gemini_configured():
        ai_res = evaluate_with_gemini(clean_q, clean_ideal, clean_user, category, topic)
        if ai_res and ai_res.get('feedback'):
            evaluation_result['feedback'] = ai_res['feedback']
            evaluation_result['evaluation_source'] = 'MockMind AI Engine (Gemini Powered)'
            if 'technical_score' in ai_res and ai_res['technical_score'] > 0:
                blended_tech = (0.50 * tech_score) + (0.50 * ai_res['technical_score'])
                evaluation_result['technical_score'] = round(blended_tech, 1)

    return evaluation_result
