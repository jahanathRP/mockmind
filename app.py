import os
from flask import Flask
from flask_login import LoginManager
from database.models import db, User, Question, migrate_database_schema
import pandas as pd

SUPPLEMENTARY_QUESTIONS = [
    {
        'question_id': 'Q019',
        'question': 'How do you handle tight project deadlines and high-pressure situations?',
        'category': 'HR',
        'topic': 'Stress Management',
        'difficulty': 'Medium',
        'ideal_answer': 'I handle tight deadlines by prioritizing tasks using an urgency-impact matrix, breaking deliverables into smaller milestones, maintaining transparent communication with stakeholders, and staying calm under pressure.',
        'keywords': 'prioritize;milestones;communication;deadlines;calm'
    },
    {
        'question_id': 'Q020',
        'question': 'Describe a situation where you had a significant technical disagreement with a teammate. How did you resolve it?',
        'category': 'HR',
        'topic': 'Conflict Resolution',
        'difficulty': 'Hard',
        'ideal_answer': 'When facing a technical disagreement, I focus on objective criteria rather than personal opinions. I listen carefully to the other perspective, compare trade-offs using benchmarks or prototypes, and seek alignment on shared project goals or consult team leads when necessary.',
        'keywords': 'disagreement;listen;trade-offs;compromise;resolution;benchmarks'
    },
    {
        'question_id': 'Q021',
        'question': 'Tell me about a project that failed or missed its key milestones. What went wrong, and what did you learn?',
        'category': 'HR',
        'topic': 'Resilience & Growth',
        'difficulty': 'Hard',
        'ideal_answer': 'In a previous project, scope creep and unclear requirements caused a deadline to slip. I took ownership of the issue, organized a retrospective to identify communication bottlenecks, adopted agile sprint planning, and implemented strict requirement freezing for future milestones.',
        'keywords': 'scope;retrospective;lessons;ownership;communication;planning'
    },
    {
        'question_id': 'Q022',
        'question': 'What is the difference between HashMap and ConcurrentHashMap in Java?',
        'category': 'Java',
        'topic': 'Collections & Concurrency',
        'difficulty': 'Medium',
        'ideal_answer': 'HashMap is not thread-safe and allows null keys/values. ConcurrentHashMap is thread-safe and achieves concurrency without locking the whole table, using CAS operations and synchronized segment bucket locks, and disallows null keys or values.',
        'keywords': 'thread-safe;ConcurrentHashMap;HashMap;locking;bucket;null'
    },
    {
        'question_id': 'Q023',
        'question': 'Explain the Java Memory Model and how Garbage Collection works across Heap generations.',
        'category': 'Java',
        'topic': 'JVM Architecture',
        'difficulty': 'Hard',
        'ideal_answer': 'The JVM heap is divided into Young Generation (Eden and Survivor spaces) and Old/Tenured Generation. Short-lived objects are allocated in Eden and collected quickly via Minor GC. Surviving objects are promoted to the Old Generation, which is collected by Major/Full GC algorithms like G1 or ZGC.',
        'keywords': 'heap;young generation;old generation;eden;survivor;garbage collection;minor gc;major gc'
    },
    {
        'question_id': 'Q024',
        'question': 'What are Java Streams, and how do intermediate operations differ from terminal operations?',
        'category': 'Java',
        'topic': 'Functional Programming',
        'difficulty': 'Hard',
        'ideal_answer': 'Java Streams support functional-style declarative operations on collections. Intermediate operations like filter() and map() are lazy and return a new Stream without executing. Terminal operations like collect(), count(), or forEach() trigger pipeline execution and produce a result or side effect.',
        'keywords': 'streams;lazy evaluation;intermediate;terminal;filter;map;collect;pipeline'
    },
    {
        'question_id': 'Q025',
        'question': 'How do you detect a cycle in a singly linked list, and what is the optimal time and space complexity?',
        'category': 'DSA',
        'topic': 'Linked Lists',
        'difficulty': 'Medium',
        'ideal_answer': 'A cycle is detected using Floyd\'s Tortoise and Hare algorithm with two pointers: slow moves 1 step and fast moves 2 steps. If they meet, a cycle exists. Time complexity is O(n) and space complexity is O(1).',
        'keywords': 'floyd;two pointers;slow;fast;cycle;O(n);O(1)'
    },
    {
        'question_id': 'Q026',
        'question': 'Compare QuickSort and MergeSort in terms of time complexity, space complexity, and stability.',
        'category': 'DSA',
        'topic': 'Sorting Algorithms',
        'difficulty': 'Medium',
        'ideal_answer': 'MergeSort has guaranteed O(n log n) time complexity and is stable, but requires O(n) auxiliary space. QuickSort has average O(n log n) time and O(log n) space (in-place), but worst case O(n^2) and is not stable by default.',
        'keywords': 'quicksort;mergesort;O(n log n);space;stability;in-place;worst case'
    },
    {
        'question_id': 'Q027',
        'question': 'Explain Dijkstra\'s algorithm for finding shortest paths and its time complexity with a min-heap priority queue.',
        'category': 'DSA',
        'topic': 'Graph Algorithms',
        'difficulty': 'Hard',
        'ideal_answer': 'Dijkstra\'s algorithm finds single-source shortest paths in graphs with non-negative edge weights. It greedily selects the unvisited node with minimum tentative distance using a min-heap priority queue and relaxes its outgoing edges. With an adjacency list and min-heap, its time complexity is O((V + E) log V).',
        'keywords': 'dijkstra;shortest path;priority queue;min-heap;greedy;relaxation;non-negative;O((V + E) log V)'
    },
    {
        'question_id': 'Q028',
        'question': 'Explain Dynamic Programming using the 0/1 Knapsack problem. What is its recurrence relation and complexity?',
        'category': 'DSA',
        'topic': 'Dynamic Programming',
        'difficulty': 'Hard',
        'ideal_answer': 'The 0/1 Knapsack problem demonstrates DP through optimal substructure and overlapping subproblems. For each item i and capacity w: dp[i][w] = max(dp[i-1][w], val[i-1] + dp[i-1][w - wt[i-1]]). Time complexity is O(N * W) and space complexity is O(N * W) or O(W) with state compression.',
        'keywords': 'optimal substructure;overlapping subproblems;dp table;recurrence;capacity;O(N * W);knapsack'
    },
    {
        'question_id': 'Q029',
        'question': 'Explain the Bias-Variance tradeoff and how it relates to underfitting and overfitting in machine learning.',
        'category': 'Machine Learning',
        'topic': 'Model Diagnostics',
        'difficulty': 'Medium',
        'ideal_answer': 'Bias represents error from oversimplified assumptions leading to underfitting. Variance represents sensitivity to small fluctuations in training data leading to overfitting. Total error is Bias^2 + Variance + Irreducible Error. The goal is finding optimal model complexity minimizing generalization error.',
        'keywords': 'bias;variance;underfitting;overfitting;tradeoff;generalization error;complexity'
    },
    {
        'question_id': 'Q030',
        'question': 'Explain the mathematical intuition behind the Attention Mechanism and Scaled Dot-Product Attention in Transformers.',
        'category': 'Machine Learning',
        'topic': 'Deep Learning',
        'difficulty': 'Hard',
        'ideal_answer': 'Attention computes dynamic relevance between tokens using Query (Q), Key (K), and Value (V) matrices. Scaled Dot-Product Attention computes Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V. The scaling factor sqrt(d_k) prevents dot products from growing large where softmax gradients vanish.',
        'keywords': 'query;key;value;scaled dot-product;softmax;sqrt(d_k);transformer;attention weights'
    },
    {
        'question_id': 'Q031',
        'question': 'What is the vanishing gradient problem in deep neural networks, and what techniques are used to mitigate it?',
        'category': 'Machine Learning',
        'topic': 'Deep Learning',
        'difficulty': 'Hard',
        'ideal_answer': 'The vanishing gradient problem occurs during backpropagation through deep layers when gradients become exponentially small (especially with sigmoid or tanh activations), stopping early layers from training. It is mitigated using non-saturating activations like ReLU/LeakyReLU, He/Xavier initialization, Batch Normalization, and Residual skip connections (ResNets).',
        'keywords': 'vanishing gradient;backpropagation;relu;residual connections;batch normalization;initialization;resnets'
    },
    {
        'question_id': 'Q032',
        'question': 'Explain Gradient Boosting and how XGBoost optimizes performance and prevents overfitting.',
        'category': 'Machine Learning',
        'topic': 'Ensemble Learning',
        'difficulty': 'Hard',
        'ideal_answer': 'Gradient Boosting builds an ensemble of weak decision trees sequentially, with each new tree fitting the pseudo-residuals (negative gradient of loss) of previous trees. XGBoost optimizes this by using second-order Taylor expansion (gradients and hessians), exact/approximate split finding, L1/L2 leaf weight regularization, and column subsampling.',
        'keywords': 'gradient boosting;residuals;xgboost;second-order;hessians;regularization;pruning;trees'
    }
]

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'mockmind_super_secret_key_change_in_production')
    
    database_url = os.environ.get('DATABASE_URL')
    if database_url:
        if database_url.startswith('postgres://'):
            database_url = database_url.replace('postgres://', 'postgresql://', 1)
        app.config['SQLALCHEMY_DATABASE_URI'] = database_url
    else:
        db_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'database')
        os.makedirs(db_dir, exist_ok=True)
        db_path = os.path.join(db_dir, 'database.db').replace('\\', '/')
        app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    db.init_app(app)

    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id):
        return db.session.get(User, int(user_id))

    from routes.auth_routes import auth_bp
    from routes.dashboard_routes import dashboard_bp
    from routes.interview_routes import interview_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(interview_bp)

    with app.app_context():
        db.create_all()
        migrate_database_schema()
        sync_question_dataset()

    return app

def sync_question_dataset():
    """
    Import questions from dataset and ensure all categories and difficulties
    (Easy, Medium, Hard) have sufficient questions.
    Normalizes category naming ('ML' -> 'Machine Learning').
    """
    # 1. First-time import from Excel if DB has no questions
    if Question.query.first() is None:
        excel_path = os.path.join(os.path.dirname(__file__), 'data', 'MockMind_complete_ml_dataset.xlsx')
        if os.path.exists(excel_path):
            df = pd.read_excel(excel_path, sheet_name='Questions')
            for _, row in df.iterrows():
                category = 'Machine Learning' if row['category'] == 'ML' else row['category']
                q = Question(
                    question_id=str(row['question_id']),
                    question=row['question'],
                    category=category,
                    topic=row['topic'],
                    difficulty=row['difficulty'],
                    ideal_answer=row['ideal_answer'],
                    keywords=str(row['keywords'])
                )
                db.session.add(q)
            db.session.commit()
            print("Imported base questions from Excel to DB.")

    # 2. Normalize existing 'ML' categories in DB to 'Machine Learning'
    ml_questions = Question.query.filter_by(category='ML').all()
    for q in ml_questions:
        q.category = 'Machine Learning'
    if ml_questions:
        db.session.commit()

    # 3. Synchronize supplementary Hard & Medium questions across categories
    existing_ids = {q.question_id for q in Question.query.all()}
    added_count = 0
    for q_data in SUPPLEMENTARY_QUESTIONS:
        if q_data['question_id'] not in existing_ids:
            q = Question(
                question_id=q_data['question_id'],
                question=q_data['question'],
                category=q_data['category'],
                topic=q_data['topic'],
                difficulty=q_data['difficulty'],
                ideal_answer=q_data['ideal_answer'],
                keywords=q_data['keywords']
            )
            db.session.add(q)
            added_count += 1

    if added_count > 0:
        db.session.commit()
        print(f"Synchronized {added_count} supplementary interview questions (Hard & Medium).")

app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() in ('true', '1', 't')
    app.run(host='0.0.0.0', port=port, debug=debug)
