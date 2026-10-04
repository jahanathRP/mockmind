"""
Optional AI Evaluation Module via Gemini API
Enhances answer feedback when an API key is available in environment variables.
Falls back seamlessly to local evaluation if unavailable, unconfigured, or failing.
"""

from __future__ import annotations
import os
import json
import urllib.request
import urllib.error

def is_gemini_configured() -> bool:
    """Check if Gemini or Google API key is configured in the environment."""
    return bool(os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY'))

def evaluate_with_gemini(question: str, ideal_answer: str, user_answer: str, category: str, topic: str) -> dict | None:
    """
    Attempt to evaluate the candidate answer using the Gemini API.
    Returns structured dict or None if API is unavailable or fails.
    """
    api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    if not api_key or not user_answer.strip():
        return None

    prompt = f"""
You are a senior technical interviewer evaluating a candidate's answer for the following question.

Category: {category}
Topic: {topic}
Question: {question}
Ideal Reference Answer: {ideal_answer}
Candidate's Answer: {user_answer}

Provide an objective, constructive evaluation. Output ONLY a valid JSON object matching this exact schema:
{{
    "technical_score": <number between 0 and 100>,
    "relevance_score": <number between 0 and 100>,
    "concept_coverage": <number between 0 and 100>,
    "missing_concepts": [<list of specific missing concepts or keywords>],
    "covered_concepts": [<list of concepts the candidate addressed>],
    "feedback": "<2-3 constructive sentences explaining strengths and what was missed>"
}}
"""

    model_name = os.environ.get('GEMINI_MODEL', 'gemini-1.5-flash')
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json"
        }
    }

    try:
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            url,
            data=data,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            result = json.loads(response.read().decode('utf-8'))
            candidates = result.get('candidates', [])
            if candidates:
                content = candidates[0].get('content', {}).get('parts', [{}])[0].get('text', '')
                parsed = json.loads(content)
                return {
                    'technical_score': float(parsed.get('technical_score', 0)),
                    'relevance_score': float(parsed.get('relevance_score', 0)),
                    'concept_coverage': float(parsed.get('concept_coverage', 0)),
                    'missing_concepts': parsed.get('missing_concepts', []),
                    'covered_concepts': parsed.get('covered_concepts', []),
                    'feedback': parsed.get('feedback', '')
                }
    except Exception as e:
        print(f"[Gemini API Notice] Fallback to local evaluation engine: {e}")
        return None

    return None
