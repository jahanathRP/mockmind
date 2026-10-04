"""
Personalized Feedback Engine Module (Module 5)
Generates actionable, category-specific feedback, strengths, weaknesses,
and next practice recommendations based on evaluation metrics and the
MockMind Feedback_Rules knowledge base.
"""

CATEGORY_RECOMMENDATIONS = {
    'HR': [
        "Use the STAR technique (Situation, Task, Action, Result) to structure behavioral answers concisely.",
        "Ground your answers with specific, measurable achievements from your academic or project experience.",
        "Demonstrate high self-awareness by pairing every mentioned weakness with active steps you take to overcome it."
    ],
    'Java': [
        "Reinforce Object-Oriented principles: encapsulation, polymorphism, inheritance, and abstraction.",
        "Deepen your understanding of JVM internals: Heap vs Stack memory, Garbage Collection algorithms, and class loading.",
        "Familiarize yourself with Java Collections framework time complexities and thread-safe alternatives (ConcurrentHashMap vs HashMap)."
    ],
    'DSA': [
        "Always state the Time Complexity and Space Complexity upfront using Big-O notation.",
        "Explain edge cases before describing the main algorithm (e.g., empty inputs, negative numbers, cycles).",
        "Practice visualizing data structure transformations (e.g., pointer manipulations in Linked Lists, recursion stack frames)."
    ],
    'Machine Learning': [
        "Always analyze the bias-variance trade-off when discussing model performance and overfitting.",
        "Highlight practical preprocessing considerations such as handling missing values, encoding, and feature scaling.",
        "Select and justify evaluation metrics based on class distribution (e.g., Precision/Recall/PR-AUC for imbalanced datasets)."
    ]
}

CATEGORY_PRACTICE_TOPICS = {
    'HR': ['Behavioral Questions', 'Conflict Resolution', 'Career Progression', 'Project Leadership'],
    'Java': ['Multithreading & Concurrency', 'Java Streams & Lambdas', 'Custom Exceptions', 'Spring Boot Fundamentals'],
    'DSA': ['Dynamic Programming', 'Binary Tree Traversals', 'Graph Traversal (BFS/DFS)', 'Two-Pointer Technique'],
    'Machine Learning': ['Gradient Boosting & Ensembles', 'Regularization (L1/L2/Dropout)', 'Hyperparameter Tuning', 'Transformer Architectures']
}

def generate_personalized_feedback(performance_data: dict, evaluated_answers: list[dict], category: str = "General") -> dict:
    """
    Generate comprehensive, personalized feedback adhering to Feedback_Rules and category coaching.
    """
    overall_score = performance_data.get('overall_score', 0.0)
    avg_tech = performance_data.get('avg_technical_score', 0.0)
    avg_rel = performance_data.get('avg_relevance_score', 0.0)
    avg_comm = performance_data.get('avg_communication_score', 0.0)
    avg_conf = performance_data.get('avg_confidence_score', 0.0)
    skill_level = performance_data.get('skill_level', 'Needs Improvement')
    strong_areas = performance_data.get('strong_areas', [])
    weak_areas = performance_data.get('areas_needing_improvement', [])

    # Apply dataset Feedback_Rules
    rule_suggestions = []
    if avg_tech < 50.0:
        rule_suggestions.append("Technical Correctness: Review the core concepts, learn the exact definitions, and practice explaining them with concrete examples.")
    if avg_rel < 50.0:
        rule_suggestions.append("Answer Relevance: Focus on directly answering the prompt without introducing tangential or unrelated topics.")
    if avg_comm < 50.0:
        rule_suggestions.append("Communication Structure: Structure each response logically: Definition -> Mechanism/How it Works -> Example -> Trade-offs.")
    if avg_conf < 50.0:
        rule_suggestions.append("Confidence & Phrasing: Practice speaking or writing decisively, minimizing filler phrases ('like', 'um') and uncertainty markers ('maybe', 'i think').")

    if overall_score >= 80.0:
        rule_suggestions.append("Advanced Progression: Outstanding performance! You are ready to tackle Hard difficulty questions and system design scenarios.")
    elif overall_score < 50.0:
        rule_suggestions.append("Foundational Revision: Dedicate time to revise foundational concepts and practice 5 beginner-level questions before advancing.")

    # Strengths Identification
    strengths = []
    if avg_tech >= 75.0:
        strengths.append("High technical precision and conceptual accuracy across answered questions.")
    if avg_rel >= 80.0:
        strengths.append("Consistently stayed on-topic and answered the direct scope of the interview questions.")
    if avg_comm >= 75.0:
        strengths.append("Well-structured phrasing with clear sentence boundaries and articulate vocabulary.")
    if avg_conf >= 75.0:
        strengths.append("Confident and assertive tone with minimal hesitation or filler phrasing.")
    if strong_areas:
        strengths.append(f"Demonstrated mastery in key topics: {', '.join(strong_areas)}.")

    if not strengths:
        strengths.append("Demonstrated willingness to tackle challenging interview scenarios under mock conditions.")

    # Weaknesses Identification
    weaknesses = []
    all_missing_concepts = []
    for ans in evaluated_answers:
        missing = ans.get('missing_concepts', [])
        all_missing_concepts.extend(missing)

    unique_missing = list(dict.fromkeys(all_missing_concepts))[:4]
    if unique_missing:
        weaknesses.append(f"Missed expected core concepts or terminology in several questions: {', '.join(unique_missing)}.")

    if avg_tech < 60.0:
        weaknesses.append("Answers lacked technical depth and essential domain mechanisms.")
    if avg_conf < 60.0:
        weaknesses.append("Frequent use of uncertainty language or hedging expressions undermined answer impact.")
    if weak_areas:
        weaknesses.append(f"Needs dedicated review in: {', '.join(weak_areas)}.")

    if not weaknesses:
        weaknesses.append("Minor opportunities to provide deeper edge-case analysis and real-world system examples.")

    # Domain Coaching
    cat_norm = 'HR' if 'HR' in category.upper() else ('Java' if 'JAVA' in category.upper() else ('DSA' if 'DSA' in category.upper() else ('Machine Learning' if 'MACHINE' in category.upper() or 'ML' in category.upper() else 'General')))
    category_tips = CATEGORY_RECOMMENDATIONS.get(cat_norm, CATEGORY_RECOMMENDATIONS['HR'])

    # Recommended Topics
    recommended_topics = weak_areas if weak_areas else CATEGORY_PRACTICE_TOPICS.get(cat_norm, ['System Design', 'Core Fundamentals'])

    # Overall Feedback Summary
    if skill_level == 'Strong':
        summary_text = (
            f"You demonstrated an impressive performance in this {category} interview session with an overall score of {overall_score}%. "
            f"Your answers showed high conceptual clarity and solid technical grounding. Continue pushing your limits with advanced scenarios."
        )
    elif skill_level == 'Good':
        summary_text = (
            f"Good effort! You achieved an overall score of {overall_score}% in {category}. "
            f"You possess a solid foundational grasp of the material. Refining your technical precision and reducing filler expressions will help you stand out to top interviewers."
        )
    else:
        summary_text = (
            f"You completed the {category} session with an overall score of {overall_score}%. "
            f"There are several foundational gaps in your technical explanations and concept coverage. Follow the targeted study recommendations below to boost your scores."
        )

    return {
        'overall_feedback': summary_text,
        'strengths': strengths,
        'weaknesses': weaknesses,
        'improvement_suggestions': rule_suggestions,
        'category_coaching': category_tips,
        'recommended_topics': recommended_topics,
        'suggested_next_difficulty': 'Hard' if overall_score >= 80 else ('Medium' if overall_score >= 55 else 'Easy')
    }

def generate_feedback(performance_data: dict) -> dict:
    """Placeholder matching original function signature for backward compatibility."""
    return generate_personalized_feedback(performance_data, [], "General")
