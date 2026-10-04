# MockMind – AI & Machine Learning Powered Interview Simulator

**MockMind** is an intelligent interview preparation platform designed for students to practice technical and HR interviews, submit answers, receive explainable AI/ML evaluation, understand their strengths and weaknesses, and systematically improve interview performance.

---

## 1. Project Overview

MockMind allows candidates to select from multiple domains (**HR**, **Java**, **Data Structures & Algorithms (DSA)**, and **Machine Learning**), choose their preferred difficulty level (**Easy**, **Medium**, **Hard**), and answer realistic interview questions. 

Upon completing an interview, candidate responses are evaluated by an internal **Machine Learning & Natural Language Processing (NLP)** evaluation engine that scores technical correctness, semantic similarity, concept coverage, and linguistic communication markers. The system generates transparent scorecards and personalized feedback grounded in candidate responses.

---

## 2. Key Features

- **User Authentication**: Secure registration and login with encrypted password hashing (`Werkzeug`).
- **Interview Setup Flow**: Filter by category (`HR`, `Java`, `DSA`, `Machine Learning`), difficulty (`Easy`, `Medium`, `Hard`), and question count (1 to 5 questions).
- **One-Question-at-a-Time Interface**: Progress tracker, live word & character counter, and "Skip Question" option.
- **ML Answer Evaluation**:
  - Preprocessing with technical term preservation (`O(log n)`, `LIFO`, `try/catch`, etc.).
  - TF-IDF N-gram cosine similarity and token overlap against ideal reference answers.
  - Keyword & concept analysis: detects `covered_concepts` and `missing_concepts` with synonym and inflection handling.
  - Topical relevance analysis: penalizes off-topic or empty responses.
  - Transparent composite technical score calculation.
- **Linguistic Confidence & Communication Analysis**:
  - Detects filler words (`um`, `uh`, `like`, `basically`, `actually`).
  - Detects uncertainty markers (`maybe`, `i think`, `i guess`, `probably`).
  - Evaluates sentence structure, terminal punctuation, and clarity.
  - Generates confidence and communication indicators with a clear text-based limitation disclaimer.
- **Performance Scoring & Feedback Engine**:
  - Aggregates session scores: Overall Score, Technical Score, Relevance, Concept Coverage, Linguistic Clarity.
  - Skill level classification (`Strong` >= 80%, `Good` 65-79%, `Needs Improvement` < 65%).
  - Identifies strong topic areas and weak areas needing revision.
  - Generates actionable category coaching (e.g. STAR method for HR, Big-O analysis for DSA, JVM internals for Java, bias-variance for ML).
- **Interactive Results Dashboard**:
  - Executive scorecards with progress bars and skill badges.
  - Side-by-side comparison of candidate answer vs ideal reference answer.
  - Green badges for covered concepts, amber badges for missing concepts.
  - Question-level constructive feedback and printable summary.
- **Interactive User Dashboard**:
  - Tracks total completed interviews, average score across sessions, and top practiced category.
  - Persistent recent session list with instant access to past results.
- **Hybrid AI Architecture**:
  - Runs 100% locally and deterministically using scikit-learn, numpy, and python-dateutil.
  - Optional Gemini API integration (`GEMINI_API_KEY` or `GOOGLE_API_KEY`) for AI feedback enrichment, with seamless fallback to local evaluation if absent.

---

## 3. Technology Stack

- **Backend Framework**: Python 3.14 / Flask 3.1
- **Database & ORM**: SQLite with SQLAlchemy (`database.db`)
- **Authentication**: Flask-Login, Werkzeug Security
- **Machine Learning & NLP**:
  - `scikit-learn` (TF-IDF Vectorizer, Cosine Similarity)
  - `numpy` & `pandas` (Matrix operations and dataset parsing)
- **Frontend**: Server-side rendered Jinja2 templates, Vanilla HTML5 & modern Vanilla CSS (glassmorphism, score dials, responsive grid)

---

## 4. Machine Learning & NLP Evaluation Architecture

```
   [ Candidate Answer ]  +  [ Ideal Reference Answer ]  +  [ Expected Keywords ]
            |                               |                           |
   [ Text Preprocessing ]                   |                           |
   (Preserves O(log n), etc.)               |                           |
            |                               |                           |
   +--------+--------+                      |                           |
   |                 |                      |                           |
[TF-IDF N-Grams]  [Token Overlap]           |                           |
   |                 |                      |                           |
   +--------+--------+                      |                           |
            |                               |                           |
   [ Cosine Similarity ]                    |                           |
            |                               |                           |
            v                               v                           v
  [ Semantic Similarity ] <----+   [ Concept Matcher ] <----+   [ Relevance Score ]
            |                  |   (Synonyms & Variations)  |           |
            +------------------+----------------------------+-----------+
                                            |
                                            v
                            [ Technical Score Calculation ]
                                            |
                                            v
                        [ Linguistic Confidence Analyzer ]
                        (Fillers, Uncertainty, Clarity)
                                            |
                                            v
                        [ Performance Scoring & Feedback ]
                        (Strong/Good/Needs Improvement)
```

### Scoring Formula:
- **Concept Coverage (%)**: $\frac{\text{Count of Covered Concepts}}{\text{Total Expected Concepts}} \times 100$
- **Semantic Similarity (%)**: Calibrated TF-IDF cosine similarity ($65\%$) + token overlap ($35\%$).
- **Topical Relevance (%)**: Alignment between user response, prompt tokens, and domain context.
- **Technical Score (%)**:
  - Technical Domains (Java, DSA, ML): $50\% \text{ Concept Coverage} + 30\% \text{ Semantic Sim} + 20\% \text{ Relevance}$
  - HR Domain: $35\% \text{ Concept Coverage} + 35\% \text{ Semantic Sim} + 30\% \text{ Relevance}$
- **Overall Performance Score (%)**:
  $40\% \text{ Technical} + 20\% \text{ Relevance} + 15\% \text{ Coverage} + 15\% \text{ Communication} + 10\% \text{ Confidence}$

---

## 5. Question Dataset Structure

Dataset file: `mockmind/data/MockMind_complete_ml_dataset.xlsx`
Contains 5 interconnected sheets:
1. `Questions`: Core question bank (`question_id`, `question`, `category`, `topic`, `difficulty`, `ideal_answer`, `keywords`).
2. `Answer_Evaluation`: Human-evaluated benchmark training pairs with technical and relevance ground truths.
3. `Confidence_Features`: Linguistic indicator training set (filler count, uncertainty count, hedging phrases, grammar quality).
4. `Performance_Scoring`: Benchmark scores and skill level classifications.
5. `Feedback_Rules`: Structured feedback rules based on threshold triggers.

*All 4 categories (`HR`, `Java`, `DSA`, `Machine Learning`) feature questions across `Easy`, `Medium`, and `Hard` difficulties.*

---

## 6. Installation and Setup

### Prerequisites
- Python 3.8+ (tested up to Python 3.14)
- Git (optional)

### Setup Steps
1. Navigate to the project directory:
   ```bash
   cd "d:\projects\2nd year 3rd sem ML project\ai interview simulator\mockmind"
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Configure Gemini API:
   Set the environment variable if you wish to enable Google Gemini AI feedback enrichment:
   ```powershell
   $env:GEMINI_API_KEY="your_api_key_here"
   ```
   *Note: If no key is set, MockMind runs completely and reliably in local ML mode.*

4. Run the application:
   ```bash
   python app.py
   ```
5. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

---

## 7. Running the Automated Tests

MockMind includes an automated test suite verifying ML evaluation, confidence metrics, session scoring, database coverage, and end-to-end Flask user flows:

```bash
python test_suite.py
```

### Verified Test Cases:
- **Test Case 1**: Relevant and correct answer (verifies high score & covered concepts).
- **Test Case 2**: Partially correct answer (verifies partial credit & missing concepts).
- **Test Case 3**: Incorrect / off-topic answer (verifies low score penalty).
- **Test Case 4**: Empty answer (verifies 0 score & helpful prompt).
- **Test Case 5**: Trivial answer like "idk" (verifies baseline non-answer handling).
- **Test Case 6**: Synonyms and alternative phrasing (verifies conceptual synonym recognition).
- **Confidence Analysis**: Tests filler and uncertainty penalty calculations.
- **Performance Scoring**: Verifies session-wide metrics and rule triggers.
- **End-to-End User Journey**: Registration -> Login -> Interview Setup -> Answering -> Automatic Evaluation -> Results Dashboard.

---

## 8. Hackathon Demonstration Flow

To demonstrate MockMind to judges:
1. **Landing Page**: Open `http://127.0.0.1:5000` to show the modern landing hero.
2. **Login / Register**: Log in with candidate credentials (e.g. `jahanath1913@gmail.com` or create a new user).
3. **Dashboard**: Show the overview metrics (Completed sessions, Average score, Top category) and recent sessions.
4. **Interview Setup**: Click "+ Start New Interview". Select **Machine Learning** (or **DSA** / **Java** / **HR**), choose **Medium** or **Hard**, and select question count.
5. **Interview Simulation**:
   - Show question prompt and difficulty badge.
   - Demonstrate the live word and character counter.
   - Enter an answer (e.g. explain the bias-variance tradeoff or binary search).
   - Click "Submit & Finish Interview".
6. **Results Dashboard**:
   - Highlight the **Skill Level badge** (`Strong`, `Good`, or `Needs Improvement`).
   - Review the 5 core scorecards (Overall, Technical, Relevance, Concept Coverage, Linguistic Clarity).
   - Show the **Personalized Feedback** section: Identified Strengths, Targeted Weaknesses, and Actionable Rule Recommendations.
   - Inspect the **Question Drilldown**: point out the green **Covered Concepts** badges and amber **Missing Concepts** badges compared against the Ideal Reference Answer.
7. **Return to Dashboard**: Demonstrate that the session is stored in SQLite and stats update immediately.

---

## 9. Known Limitations & Future Improvements

- **Speech / Audio Input**: Currently, candidate responses are entered via text. Future enhancements will integrate Web Speech API or local Whisper models for speech-to-text and speech tempo analysis.
- **Video / Eye-Contact Tracking**: MockMind intentionally uses explainable text-based linguistic indicators rather than pseudo-scientific facial emotion detection.
- **Code Execution Sandbox**: For programming questions, adding an isolated code runner (e.g., Pyodide or Docker sandbox) will allow running unit tests on submitted code.
