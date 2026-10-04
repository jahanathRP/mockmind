"""
Performance Scoring Module (Module 4/5)
Aggregates question-level scores into session-level metrics and skill classification.
Calculates overall performance, skill level (Strong/Good/Needs Improvement),
strong areas, and areas needing improvement.
"""

def calculate_session_performance(evaluated_answers: list[dict]) -> dict:
    """
    Calculate session-wide performance summary from a list of evaluated answers.
    """
    if not evaluated_answers:
        return {
            'overall_score': 0.0,
            'avg_technical_score': 0.0,
            'avg_relevance_score': 0.0,
            'avg_concept_coverage': 0.0,
            'avg_communication_score': 0.0,
            'avg_confidence_score': 0.0,
            'questions_answered': 0,
            'questions_skipped': 0,
            'skill_level': 'Needs Improvement',
            'strong_areas': [],
            'areas_needing_improvement': []
        }

    total_questions = len(evaluated_answers)
    technical_scores = []
    relevance_scores = []
    concept_coverages = []
    communication_scores = []
    confidence_scores = []

    topic_scores = {}
    answered_count = 0
    skipped_count = 0

    for ans in evaluated_answers:
        user_text = (ans.get('user_answer') or '').strip()
        t_score = float(ans.get('technical_score', 0.0))
        r_score = float(ans.get('relevance_score', 0.0))
        cov_score = float(ans.get('concept_coverage', 0.0))
        comm_score = float(ans.get('communication_score', 0.0))
        conf_score = float(ans.get('confidence_score', 0.0))
        topic = ans.get('topic') or ans.get('category') or 'General'

        if not user_text or user_text.lower() in {'skip', 'pass', 'idk', 'na'}:
            skipped_count += 1
        else:
            answered_count += 1

        technical_scores.append(t_score)
        relevance_scores.append(r_score)
        concept_coverages.append(cov_score)
        communication_scores.append(comm_score)
        confidence_scores.append(conf_score)

        if topic not in topic_scores:
            topic_scores[topic] = []
        topic_scores[topic].append(t_score)

    avg_tech = round(sum(technical_scores) / total_questions, 1)
    avg_rel = round(sum(relevance_scores) / total_questions, 1)
    avg_cov = round(sum(concept_coverages) / total_questions, 1)
    avg_comm = round(sum(communication_scores) / total_questions, 1)
    avg_conf = round(sum(confidence_scores) / total_questions, 1)

    # Harmonious composite overall performance score
    # 40% Technical, 20% Relevance, 15% Concept Coverage, 15% Communication, 10% Confidence
    overall = (0.40 * avg_tech) + (0.20 * avg_rel) + (0.15 * avg_cov) + (0.15 * avg_comm) + (0.10 * avg_conf)
    overall_score = round(max(0.0, min(100.0, overall)), 1)

    # Skill Level Classification matching Performance_Scoring sheet in Excel dataset:
    # Strong: >= 80, Good: 65 - 79, Needs Improvement: < 65
    if overall_score >= 80.0:
        skill_level = 'Strong'
    elif overall_score >= 65.0:
        skill_level = 'Good'
    else:
        skill_level = 'Needs Improvement'

    # Identify strong areas and improvement areas
    strong_areas = []
    areas_needing_improvement = []

    for topic, scores in topic_scores.items():
        avg_topic_score = sum(scores) / len(scores)
        if avg_topic_score >= 75.0:
            strong_areas.append(topic)
        elif avg_topic_score < 60.0:
            areas_needing_improvement.append(topic)

    return {
        'overall_score': overall_score,
        'avg_technical_score': avg_tech,
        'avg_relevance_score': avg_rel,
        'avg_concept_coverage': avg_cov,
        'avg_communication_score': avg_comm,
        'avg_confidence_score': avg_conf,
        'questions_answered': answered_count,
        'questions_skipped': skipped_count,
        'skill_level': skill_level,
        'strong_areas': list(set(strong_areas)),
        'areas_needing_improvement': list(set(areas_needing_improvement))
    }

def calculate_scores():
    """Placeholder matching original function signature for backward compatibility."""
    return calculate_session_performance([])
