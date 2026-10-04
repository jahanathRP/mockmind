from flask import Blueprint, render_template
from flask_login import login_required, current_user
from database.models import InterviewSession

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/dashboard')
@login_required
def dashboard():
    sessions = InterviewSession.query.filter_by(user_id=current_user.id).order_by(InterviewSession.started_at.desc()).all()
    completed_sessions = [s for s in sessions if s.status == 'Completed']

    # Compute user aggregate performance
    scores = [s.overall_score for s in completed_sessions if s.overall_score is not None]
    avg_score = round(sum(scores) / len(scores), 1) if scores else 0.0

    # Determine most practiced category
    cat_counts = {}
    for s in completed_sessions:
        cat_counts[s.category] = cat_counts.get(s.category, 0) + 1
    top_category = max(cat_counts, key=cat_counts.get) if cat_counts else 'None'

    return render_template(
        'dashboard.html',
        name=current_user.name,
        total_interviews=len(completed_sessions),
        avg_score=avg_score,
        top_category=top_category,
        recent_sessions=sessions[:8]
    )
