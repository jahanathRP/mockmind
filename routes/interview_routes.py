import json
import random
from datetime import datetime, timezone
from flask import Blueprint, render_template, request, redirect, url_for, session, flash, abort
from flask_login import login_required, current_user
from database.models import db, InterviewSession, Question, Answer
from ml.answer_evaluation import evaluate_answer
from ml.confidence_analysis import analyze_answer_linguistics
from ml.performance_scoring import calculate_session_performance
from ml.feedback_engine import generate_personalized_feedback

interview_bp = Blueprint('interview', __name__)

def evaluate_and_persist_session(interview_session: InterviewSession):
    """
    Evaluate all answers in an interview session and compute session-level
    aggregates and personalized feedback, saving all results to SQLite.
    """
    answers = Answer.query.filter_by(session_id=interview_session.id).all()
    evaluated_list = []

    for ans in answers:
        q = ans.question
        # If not evaluated yet or needs refresh
        if ans.technical_score is None:
            eval_res = evaluate_answer(
                question=q.question,
                ideal_answer=q.ideal_answer,
                user_answer=ans.user_answer,
                category=q.category,
                topic=q.topic,
                difficulty=q.difficulty,
                keywords=q.keywords
            )
            ling_res = analyze_answer_linguistics(ans.user_answer)

            ans.technical_score = eval_res['technical_score']
            ans.relevance_score = eval_res['relevance_score']
            ans.semantic_score = eval_res['semantic_similarity']
            ans.concept_coverage = eval_res['concept_coverage']
            ans.confidence_score = ling_res['confidence_score']
            ans.communication_score = ling_res['communication_score']
            ans.overall_score = round((0.55 * ans.technical_score) + (0.45 * ans.communication_score), 1)
            ans.covered_concepts = json.dumps(eval_res['covered_concepts'])
            ans.missing_concepts = json.dumps(eval_res['missing_concepts'])
            ans.feedback = eval_res['feedback']

        evaluated_list.append({
            'user_answer': ans.user_answer,
            'technical_score': ans.technical_score or 0.0,
            'relevance_score': ans.relevance_score or 0.0,
            'concept_coverage': ans.concept_coverage or 0.0,
            'communication_score': ans.communication_score or 0.0,
            'confidence_score': ans.confidence_score or 0.0,
            'topic': q.topic,
            'category': q.category,
            'missing_concepts': ans.get_missing_concepts(),
            'covered_concepts': ans.get_covered_concepts()
        })

    # Session performance aggregation
    perf = calculate_session_performance(evaluated_list)
    feedback = generate_personalized_feedback(perf, evaluated_list, interview_session.category)

    interview_session.overall_score = perf['overall_score']
    interview_session.technical_score = perf['avg_technical_score']
    interview_session.relevance_score = perf['avg_relevance_score']
    interview_session.concept_coverage = perf['avg_concept_coverage']
    interview_session.communication_score = perf['avg_communication_score']
    interview_session.confidence_score = perf['avg_confidence_score']
    interview_session.skill_level = perf['skill_level']
    interview_session.strengths = json.dumps(feedback['strengths'])
    interview_session.weaknesses = json.dumps(feedback['weaknesses'])
    interview_session.recommended_topics = json.dumps(feedback['recommended_topics'])
    interview_session.overall_feedback = feedback['overall_feedback']

    db.session.commit()
    return perf, feedback

@interview_bp.route('/setup', methods=['GET', 'POST'])
@login_required
def setup():
    if request.method == 'POST':
        category = request.form.get('category')
        difficulty = request.form.get('difficulty')
        num_questions = int(request.form.get('num_questions', 3))

        # Normalize category
        cat_filter = ['Machine Learning', 'ML'] if category in ['Machine Learning', 'ML'] else [category]
        available_questions = Question.query.filter(
            Question.category.in_(cat_filter),
            Question.difficulty == difficulty
        ).all()

        available_count = len(available_questions)
        if available_count == 0:
            flash(f'No questions found for {category} ({difficulty}). Please choose another category or difficulty level.', 'warning')
            return redirect(url_for('interview.setup'))

        # Adjust count if user requested more questions than currently exist in database
        selected_count = min(num_questions, available_count)
        if selected_count < num_questions:
            flash(f'Note: Selected {selected_count} available questions for {category} ({difficulty}).', 'info')

        new_session = InterviewSession(
            user_id=current_user.id,
            category=category,
            difficulty=difficulty,
            total_questions=selected_count
        )
        db.session.add(new_session)
        db.session.commit()

        # Select random questions
        selected_questions = random.sample(available_questions, selected_count)
        session['current_interview_id'] = new_session.id
        session['question_ids'] = [q.id for q in selected_questions]
        session['current_q_index'] = 0

        return redirect(url_for('interview.start_interview'))

    return render_template('interview_setup.html')

@interview_bp.route('/interview', methods=['GET', 'POST'])
@login_required
def start_interview():
    if 'current_interview_id' not in session or 'question_ids' not in session:
        return redirect(url_for('dashboard.dashboard'))

    interview_session = db.session.get(InterviewSession, session['current_interview_id'])
    if not interview_session or interview_session.user_id != current_user.id:
        session.pop('current_interview_id', None)
        return redirect(url_for('dashboard.dashboard'))

    if interview_session.status == 'Completed':
        return redirect(url_for('interview.summary', session_id=interview_session.id))

    q_index = session.get('current_q_index', 0)
    q_ids = session.get('question_ids', [])

    if q_index >= len(q_ids):
        # Interview finished: run evaluation & persist
        interview_session.status = 'Completed'
        interview_session.completed_at = datetime.now(timezone.utc)
        db.session.commit()

        evaluate_and_persist_session(interview_session)

        # Clear active session state
        session.pop('current_interview_id', None)
        session.pop('question_ids', None)
        session.pop('current_q_index', None)
        return redirect(url_for('interview.completed', session_id=interview_session.id))

    current_q_id = q_ids[q_index]
    question = db.session.get(Question, current_q_id)

    if request.method == 'POST':
        action = request.form.get('action', 'submit')
        if action == 'skip':
            user_answer = "Skipped by candidate."
        else:
            user_answer = request.form.get('answer', '').strip()
            if not user_answer:
                user_answer = "No response provided."

        # Real-time question evaluation
        eval_res = evaluate_answer(
            question=question.question,
            ideal_answer=question.ideal_answer,
            user_answer=user_answer,
            category=question.category,
            topic=question.topic,
            difficulty=question.difficulty,
            keywords=question.keywords
        )
        ling_res = analyze_answer_linguistics(user_answer)

        answer = Answer(
            session_id=interview_session.id,
            question_id=question.id,
            user_id=current_user.id,
            user_answer=user_answer,
            technical_score=eval_res['technical_score'],
            relevance_score=eval_res['relevance_score'],
            semantic_score=eval_res['semantic_similarity'],
            concept_coverage=eval_res['concept_coverage'],
            confidence_score=ling_res['confidence_score'],
            communication_score=ling_res['communication_score'],
            overall_score=round((0.55 * eval_res['technical_score']) + (0.45 * ling_res['communication_score']), 1),
            covered_concepts=json.dumps(eval_res['covered_concepts']),
            missing_concepts=json.dumps(eval_res['missing_concepts']),
            feedback=eval_res['feedback']
        )
        db.session.add(answer)
        db.session.commit()

        session['current_q_index'] = q_index + 1
        return redirect(url_for('interview.start_interview'))

    return render_template(
        'interview.html',
        question=question,
        current_q=q_index + 1,
        total_q=len(q_ids)
    )

@interview_bp.route('/completed/<int:session_id>')
@login_required
def completed(session_id):
    interview_session = db.session.get(InterviewSession, session_id)
    if not interview_session:
        abort(404)
    if interview_session.user_id != current_user.id:
        flash("Unauthorized access", "danger")
        return redirect(url_for('dashboard.dashboard'))
    return render_template('interview_completed.html', session=interview_session)

@interview_bp.route('/summary/<int:session_id>')
@login_required
def summary(session_id):
    interview_session = db.session.get(InterviewSession, session_id)
    if not interview_session:
        abort(404)
    if interview_session.user_id != current_user.id:
        flash("Unauthorized access", "danger")
        return redirect(url_for('dashboard.dashboard'))

    # If this session has not yet been evaluated (e.g. from earlier runs), evaluate it now
    if interview_session.overall_score is None:
        evaluate_and_persist_session(interview_session)

    answers = Answer.query.filter_by(session_id=session_id).all()
    evaluated_list = [
        {
            'user_answer': a.user_answer,
            'technical_score': a.technical_score or 0.0,
            'relevance_score': a.relevance_score or 0.0,
            'concept_coverage': a.concept_coverage or 0.0,
            'communication_score': a.communication_score or 0.0,
            'confidence_score': a.confidence_score or 0.0,
            'topic': a.question.topic if a.question else 'General',
            'category': a.question.category if a.question else interview_session.category,
            'missing_concepts': a.get_missing_concepts(),
            'covered_concepts': a.get_covered_concepts()
        }
        for a in answers
    ]

    perf = calculate_session_performance(evaluated_list)
    feedback = generate_personalized_feedback(perf, evaluated_list, interview_session.category)

    return render_template(
        'summary.html',
        session=interview_session,
        answers=answers,
        performance=perf,
        feedback=feedback
    )
