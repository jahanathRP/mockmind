from __future__ import annotations
import json
from datetime import datetime, timezone
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from sqlalchemy import text

db = SQLAlchemy()

def get_utc_now():
    return datetime.now(timezone.utc)

class User(db.Model, UserMixin):
    __tablename__ = 'USERS'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=get_utc_now)
    sessions = db.relationship('InterviewSession', backref='user', lazy=True)

class InterviewSession(db.Model):
    __tablename__ = 'INTERVIEW_SESSIONS'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('USERS.id'), nullable=False)
    category = db.Column(db.String(50), nullable=False)
    difficulty = db.Column(db.String(50), nullable=False)
    total_questions = db.Column(db.Integer, nullable=False)
    started_at = db.Column(db.DateTime, default=get_utc_now)
    completed_at = db.Column(db.DateTime, nullable=True)
    status = db.Column(db.String(50), default='In Progress') # In Progress, Completed
    
    # Session Level ML Performance Aggregates
    overall_score = db.Column(db.Float, nullable=True)
    technical_score = db.Column(db.Float, nullable=True)
    relevance_score = db.Column(db.Float, nullable=True)
    concept_coverage = db.Column(db.Float, nullable=True)
    communication_score = db.Column(db.Float, nullable=True)
    confidence_score = db.Column(db.Float, nullable=True)
    skill_level = db.Column(db.String(50), nullable=True) # Strong, Good, Needs Improvement
    
    strengths = db.Column(db.Text, nullable=True) # JSON list
    weaknesses = db.Column(db.Text, nullable=True) # JSON list
    recommended_topics = db.Column(db.Text, nullable=True) # JSON list
    overall_feedback = db.Column(db.Text, nullable=True)

    answers = db.relationship('Answer', backref='session', lazy=True)

    def get_strengths(self) -> list[str]:
        if not self.strengths:
            return []
        try:
            return json.loads(self.strengths)
        except Exception:
            return [s.strip() for s in self.strengths.split(';') if s.strip()]

    def get_weaknesses(self) -> list[str]:
        if not self.weaknesses:
            return []
        try:
            return json.loads(self.weaknesses)
        except Exception:
            return [w.strip() for w in self.weaknesses.split(';') if w.strip()]

    def get_recommended_topics(self) -> list[str]:
        if not self.recommended_topics:
            return []
        try:
            return json.loads(self.recommended_topics)
        except Exception:
            return [t.strip() for t in self.recommended_topics.split(';') if t.strip()]

class Question(db.Model):
    __tablename__ = 'QUESTIONS'
    id = db.Column(db.Integer, primary_key=True)
    question_id = db.Column(db.String(20), unique=True, nullable=False)
    question = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(50), nullable=False)
    topic = db.Column(db.String(100))
    difficulty = db.Column(db.String(50), nullable=False)
    ideal_answer = db.Column(db.Text)
    keywords = db.Column(db.Text)

class Answer(db.Model):
    __tablename__ = 'ANSWERS'
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('INTERVIEW_SESSIONS.id'), nullable=False)
    question_id = db.Column(db.Integer, db.ForeignKey('QUESTIONS.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('USERS.id'), nullable=False)
    user_answer = db.Column(db.Text, nullable=False)
    answered_at = db.Column(db.DateTime, default=get_utc_now)
    
    # ML Evaluation Metrics
    technical_score = db.Column(db.Float, nullable=True)
    relevance_score = db.Column(db.Float, nullable=True)
    semantic_score = db.Column(db.Float, nullable=True)
    concept_coverage = db.Column(db.Float, nullable=True)
    confidence_score = db.Column(db.Float, nullable=True)
    communication_score = db.Column(db.Float, nullable=True)
    overall_score = db.Column(db.Float, nullable=True)
    feedback = db.Column(db.Text, nullable=True)

    covered_concepts = db.Column(db.Text, nullable=True) # JSON list
    missing_concepts = db.Column(db.Text, nullable=True) # JSON list

    question = db.relationship('Question', backref='given_answers')

    def get_covered_concepts(self) -> list[str]:
        if not self.covered_concepts:
            return []
        try:
            return json.loads(self.covered_concepts)
        except Exception:
            return [c.strip() for c in self.covered_concepts.split(';') if c.strip()]

    def get_missing_concepts(self) -> list[str]:
        if not self.missing_concepts:
            return []
        try:
            return json.loads(self.missing_concepts)
        except Exception:
            return [m.strip() for m in self.missing_concepts.split(';') if m.strip()]

def migrate_database_schema():
    """
    Safely alter tables to add any missing ML columns without
    dropping or resetting existing user data. Supports SQLite and external DBs.
    """
    try:
        with db.engine.connect() as conn:
            if db.engine.dialect.name == 'sqlite':
                # Check ANSWERS table columns
                answers_cols = [row[1] for row in conn.execute(text("PRAGMA table_info(ANSWERS);")).fetchall()]
                new_answer_cols = {
                    'semantic_score': 'FLOAT',
                    'concept_coverage': 'FLOAT',
                    'covered_concepts': 'TEXT',
                    'missing_concepts': 'TEXT'
                }
                for col_name, col_type in new_answer_cols.items():
                    if col_name not in answers_cols:
                        conn.execute(text(f"ALTER TABLE ANSWERS ADD COLUMN {col_name} {col_type};"))
                        conn.commit()

                # Check INTERVIEW_SESSIONS table columns
                sessions_cols = [row[1] for row in conn.execute(text("PRAGMA table_info(INTERVIEW_SESSIONS);")).fetchall()]
                new_session_cols = {
                    'overall_score': 'FLOAT',
                    'technical_score': 'FLOAT',
                    'relevance_score': 'FLOAT',
                    'concept_coverage': 'FLOAT',
                    'communication_score': 'FLOAT',
                    'confidence_score': 'FLOAT',
                    'skill_level': 'VARCHAR(50)',
                    'strengths': 'TEXT',
                    'weaknesses': 'TEXT',
                    'recommended_topics': 'TEXT',
                    'overall_feedback': 'TEXT'
                }
                for col_name, col_type in new_session_cols.items():
                    if col_name not in sessions_cols:
                        conn.execute(text(f"ALTER TABLE INTERVIEW_SESSIONS ADD COLUMN {col_name} {col_type};"))
                        conn.commit()
    except Exception as e:
        print(f"[Schema Migration Notice] Non-fatal migration check: {e}")
