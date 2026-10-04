"""
Comprehensive Test Suite for MockMind
Validates:
1. ML Answer Evaluation Pipeline
2. Linguistic Confidence and Communication Analysis
3. Session Performance Scoring & Feedback Engine
4. Question Bank Integrity & Category Normalization
5. End-to-End Flask User Journey (Auth, Setup, Answering, Results)
"""

import os
import unittest
import json
from app import create_app
from database.models import db, User, Question, InterviewSession, Answer
from ml.answer_evaluation import evaluate_answer
from ml.confidence_analysis import analyze_answer_linguistics
from ml.performance_scoring import calculate_session_performance
from ml.feedback_engine import generate_personalized_feedback

class MockMindMLUnitTests(unittest.TestCase):
    """Test individual ML and linguistic evaluation functions."""

    def test_case_1_correct_answer(self):
        """Relevant, complete, technically sound answer."""
        q = "What is the time complexity of binary search?"
        ideal = "Binary search works on a sorted array by halving the search space at each step, achieving O(log n) time."
        user = "Binary search requires a sorted array and repeatedly halves the search space, giving an O(log n) time complexity."
        res = evaluate_answer(q, ideal, user, "DSA", "Searching", "Easy", "sorted;halves;O(log n)")

        self.assertGreaterEqual(res['technical_score'], 75.0)
        self.assertGreaterEqual(res['concept_coverage'], 80.0)
        self.assertIn('sorted', res['covered_concepts'])
        self.assertIn('halves', res['covered_concepts'])
        self.assertEqual(len(res['missing_concepts']), 0)
        print(f"[PASS] Test Case 1 (Correct Answer): Tech Score = {res['technical_score']}")

    def test_case_2_partially_correct_answer(self):
        """Partially correct answer missing critical concepts."""
        q = "What is the time complexity of binary search?"
        ideal = "Binary search works on a sorted array by halving the search space at each step, achieving O(log n) time."
        user = "Binary search is an algorithm that searches through an array very quickly."
        res = evaluate_answer(q, ideal, user, "DSA", "Searching", "Easy", "sorted;halves;O(log n)")

        self.assertLess(res['technical_score'], 60.0)
        self.assertLess(res['concept_coverage'], 50.0)
        self.assertIn('O(log n)', res['missing_concepts'])
        print(f"[PASS] Test Case 2 (Partially Correct Answer): Tech Score = {res['technical_score']}")

    def test_case_3_incorrect_off_topic_answer(self):
        """Completely unrelated / off-topic answer."""
        q = "What is inheritance in Java?"
        ideal = "Inheritance allows a child class to inherit fields and methods from a parent class, promoting code reuse."
        user = "I enjoy eating pepperoni pizza on Friday nights while listening to jazz."
        res = evaluate_answer(q, ideal, user, "Java", "OOP", "Easy", "class;parent;child;reuse")

        self.assertLess(res['technical_score'], 35.0)
        self.assertLess(res['relevance_score'], 30.0)
        self.assertEqual(res['concept_coverage'], 0.0)
        print(f"[PASS] Test Case 3 (Off-topic Answer): Tech Score = {res['technical_score']}, Relevance = {res['relevance_score']}")

    def test_case_4_empty_answer(self):
        """Empty or whitespace-only answer."""
        q = "What is overfitting in machine learning?"
        ideal = "Overfitting happens when a model learns training noise and fails to generalize to unseen test data."
        user = "   "
        res = evaluate_answer(q, ideal, user, "Machine Learning", "Evaluation", "Easy", "training;unseen;generalization")

        self.assertEqual(res['technical_score'], 0.0)
        self.assertEqual(res['concept_coverage'], 0.0)
        self.assertEqual(len(res['missing_concepts']), 3)
        print("[PASS] Test Case 4 (Empty Answer): Tech Score = 0.0")

    def test_case_5_trivial_short_answer(self):
        """Trivial one-word non-answer like 'idk'."""
        q = "Explain method overloading and overriding."
        ideal = "Overloading is same method name with different parameters at compile time. Overriding is redefining a parent method in a subclass at runtime."
        user = "idk"
        res = evaluate_answer(q, ideal, user, "Java", "OOP", "Medium", "overloading;overriding;parameters;subclass")

        self.assertLessEqual(res['technical_score'], 10.0)
        print(f"[PASS] Test Case 5 (Trivial 'idk' Answer): Tech Score = {res['technical_score']}")

    def test_case_6_synonyms_and_different_wording(self):
        """Correct answer using alternative terminology/synonyms."""
        q = "What is a stack and where is it used?"
        ideal = "A stack is a LIFO data structure used in function calls and recursion."
        user = "A stack follows last in first out principle and is heavily utilized for managing recursive call stack frames."
        res = evaluate_answer(q, ideal, user, "DSA", "Stack", "Easy", "LIFO;recursion;function calls")

        self.assertGreaterEqual(res['technical_score'], 70.0)
        self.assertGreaterEqual(res['concept_coverage'], 65.0)
        print(f"[PASS] Test Case 6 (Synonyms / Alternate Phrasing): Tech Score = {res['technical_score']}")

    def test_confidence_and_linguistics(self):
        """Test linguistic indicators: fillers, uncertainty, clarity."""
        confident_text = "Inheritance allows a subclass to acquire properties and methods from a superclass, fostering code reusability."
        hesitant_text = "Um, like, I guess maybe inheritance is sort of when one class gets things from another, I'm not really sure."

        conf_res = analyze_answer_linguistics(confident_text)
        hes_res = analyze_answer_linguistics(hesitant_text)

        self.assertGreaterEqual(conf_res['confidence_score'], 80.0)
        self.assertLessEqual(hes_res['confidence_score'], 55.0)
        self.assertGreater(hes_res['filler_word_count'], 1)
        self.assertGreater(hes_res['uncertainty_word_count'], 1)
        print(f"[PASS] Confidence Analysis: Confident = {conf_res['confidence_score']}, Hesitant = {hes_res['confidence_score']}")

    def test_performance_scoring_and_rules(self):
        """Test session aggregation, skill classification, and feedback rules."""
        mock_answers = [
            {'user_answer': 'Good ans', 'technical_score': 85.0, 'relevance_score': 90.0, 'concept_coverage': 90.0, 'communication_score': 85.0, 'confidence_score': 90.0, 'topic': 'OOP', 'category': 'Java', 'missing_concepts': []},
            {'user_answer': 'Decent ans', 'technical_score': 75.0, 'relevance_score': 80.0, 'concept_coverage': 80.0, 'communication_score': 80.0, 'confidence_score': 85.0, 'topic': 'Collections', 'category': 'Java', 'missing_concepts': ['segment']}
        ]
        perf = calculate_session_performance(mock_answers)
        feedback = generate_personalized_feedback(perf, mock_answers, "Java")

        self.assertEqual(perf['skill_level'], 'Strong')
        self.assertGreaterEqual(perf['overall_score'], 80.0)
        self.assertGreater(len(feedback['strengths']), 0)
        self.assertGreater(len(feedback['category_coaching']), 0)
        print(f"[PASS] Performance & Feedback: Skill Level = {perf['skill_level']}, Overall = {perf['overall_score']}")

class MockMindFlaskIntegrationTests(unittest.TestCase):
    """Test database, routing, authentication, and full user journey."""

    def setUp(self):
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.app.config['WTF_CSRF_ENABLED'] = False
        self.client = self.app.test_client()

    def test_database_questions_coverage(self):
        """Verify all categories have Easy, Medium, and Hard questions."""
        with self.app.app_context():
            for cat in ['HR', 'Java', 'DSA', 'Machine Learning']:
                for diff in ['Easy', 'Medium', 'Hard']:
                    count = Question.query.filter_by(category=cat, difficulty=diff).count()
                    self.assertGreater(count, 0, f"Missing questions for {cat} ({diff})")
            print("[PASS] Database Questions Coverage: All 4 categories x 3 difficulty levels verified!")

    def test_user_registration_and_login_flow(self):
        """Test user registration, duplicate email handling, and login."""
        with self.app.app_context():
            test_email = "test_candidate_auth@mockmind.ai"
            existing = User.query.filter_by(email=test_email).first()
            if existing:
                Answer.query.filter_by(user_id=existing.id).delete()
                InterviewSession.query.filter_by(user_id=existing.id).delete()
                User.query.filter_by(id=existing.id).delete()
                db.session.commit()

        # 1. Register with mismatched passwords
        res = self.client.post('/register', data={
            'name': 'Auth Test Candidate',
            'email': 'test_candidate_auth@mockmind.ai',
            'password': 'Password123!',
            'confirm_password': 'DifferentPassword'
        }, follow_redirects=True)
        self.assertIn(b'Passwords must match', res.data)

        # 2. Register with valid details
        res = self.client.post('/register', data={
            'name': 'Auth Test Candidate',
            'email': 'test_candidate_auth@mockmind.ai',
            'password': 'Password123!',
            'confirm_password': 'Password123!'
        }, follow_redirects=True)
        self.assertIn(b'Registration successful', res.data)

        # 3. Duplicate email test
        res = self.client.post('/register', data={
            'name': 'Auth Test Candidate',
            'email': 'test_candidate_auth@mockmind.ai',
            'password': 'Password123!',
            'confirm_password': 'Password123!'
        }, follow_redirects=True)
        self.assertIn(b'Email address already exists', res.data)

        # 4. Login with wrong password
        res = self.client.post('/login', data={
            'email': 'test_candidate_auth@mockmind.ai',
            'password': 'WrongPassword'
        }, follow_redirects=True)
        self.assertIn(b'Login Unsuccessful', res.data)

        # 5. Login with correct password
        res = self.client.post('/login', data={
            'email': 'test_candidate_auth@mockmind.ai',
            'password': 'Password123!'
        }, follow_redirects=True)
        self.assertIn(b'Dashboard', res.data)
        print("[PASS] Authentication Flow: Registration, duplicate checks, and login verified!")

    def test_full_interview_journey_and_results(self):
        """Simulate complete interview lifecycle from setup to results dashboard."""
        test_email = 'journey_user@mockmind.ai'
        password = 'Password123!'

        with self.app.app_context():
            existing = User.query.filter_by(email=test_email).first()
            if not existing:
                from werkzeug.security import generate_password_hash
                u = User(name='Journey User', email=test_email, password_hash=generate_password_hash(password))
                db.session.add(u)
                db.session.commit()

        # Login using client session
        login_res = self.client.post('/login', data={
            'email': test_email,
            'password': password
        }, follow_redirects=True)
        self.assertIn(b'Dashboard', login_res.data)

        # Setup interview: Java, Easy, 1 question
        setup_res = self.client.post('/setup', data={
            'category': 'Java',
            'difficulty': 'Easy',
            'num_questions': '1'
        }, follow_redirects=True)
        self.assertIn(b'Question 1 of 1', setup_res.data)

        # Fetch the selected question to provide a relevant answer
        with self.client.session_transaction() as sess:
            q_id = sess['question_ids'][0]
        with self.app.app_context():
            q_obj = db.session.get(Question, q_id)
            answer_text = q_obj.ideal_answer

        ans_res = self.client.post('/interview', data={
            'answer': answer_text,
            'action': 'submit'
        }, follow_redirects=True)

        # Should redirect to completed page
        self.assertIn(b'Interview Completed Successfully', ans_res.data)

        # Fetch session ID from database
        with self.app.app_context():
            user = User.query.filter_by(email=test_email).first()
            s = InterviewSession.query.filter_by(user_id=user.id).order_by(InterviewSession.started_at.desc()).first()
            session_id = s.id
            self.assertIsNotNone(s.overall_score)
            self.assertIsNotNone(s.skill_level)
            print(f"[PASS] Session {session_id} Evaluated: Score = {s.overall_score}%, Level = {s.skill_level}")

        # View Summary / Results Dashboard
        summary_res = self.client.get(f'/summary/{session_id}')
        self.assertIn(b'Interview Results Dashboard', summary_res.data)
        self.assertIn(b'Personalized Feedback', summary_res.data)
        self.assertIn(b'Detailed Question Analysis', summary_res.data)
        self.assertIn(b'Covered Concepts', summary_res.data)
        print("[PASS] Full Interview Journey: Setup -> Answering -> ML Evaluation -> Results Dashboard verified!")

if __name__ == '__main__':
    unittest.main(verbosity=2)
