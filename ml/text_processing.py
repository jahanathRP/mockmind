"""
Text Processing Utilities for MockMind
Handles tokenization, technical term preservation, sentence segmentation,
and concept synonym matching.
"""

import re
import string

# Stop words tailored for technical interview analysis
# Note: We deliberately exclude critical programming/logic terms like 'class', 'is', 'not', 'and', 'or', 'in'
STOP_WORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'any',
    'are', 'aren\'t', 'as', 'at', 'be', 'because', 'been', 'before', 'being',
    'below', 'between', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn\'t',
    'did', 'didn\'t', 'do', 'does', 'doesn\'t', 'doing', 'don\'t', 'down', 'during',
    'each', 'few', 'for', 'from', 'further', 'had', 'hadn\'t', 'has', 'hasn\'t',
    'have', 'haven\'t', 'having', 'he', 'he\'d', 'he\'ll', 'he\'s', 'her', 'here',
    'here\'s', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'how\'s', 'i',
    'i\'d', 'i\'ll', 'i\'m', 'i\'ve', 'if', 'into', 'its', 'it\'s', 'itself',
    'let\'s', 'me', 'more', 'most', 'my', 'myself', 'no', 'nor', 'of', 'off', 'on',
    'once', 'only', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over',
    'own', 'same', 'she', 'she\'d', 'she\'ll', 'she\'s', 'should', 'shouldn\'t',
    'so', 'some', 'such', 'than', 'that', 'that\'s', 'the', 'their', 'theirs',
    'them', 'themselves', 'then', 'there', 'there\'s', 'these', 'they', 'they\'d',
    'they\'ll', 'they\'re', 'they\'ve', 'this', 'those', 'through', 'to', 'too',
    'under', 'until', 'up', 'very', 'was', 'wasn\'t', 'we', 'we\'d', 'we\'ll',
    'we\'re', 'we\'ve', 'were', 'weren\'t', 'what', 'what\'s', 'when', 'when\'s',
    'where', 'where\'s', 'which', 'while', 'who', 'who\'s', 'whom', 'why', 'why\'s',
    'with', 'won\'t', 'would', 'wouldn\'t', 'you', 'you\'d', 'you\'ll', 'you\'re',
    'you\'ve', 'your', 'yours', 'yourself', 'yourselves'
}

# Technical concept synonym and variation mapping
CONCEPT_SYNONYMS = {
    'o(log n)': ['logarithmic', 'o(log n)', 'ologn', 'log n', 'logarithmic time', 'o(logn)'],
    'o(n)': ['linear', 'o(n)', 'on', 'linear time'],
    'o(1)': ['constant', 'o(1)', 'constant time'],
    'o(n log n)': ['linearithmic', 'n log n', 'o(nlogn)', 'o(n log n)'],
    'sorted': ['sort', 'sorted', 'ordered', 'order', 'sorting', 'ascending', 'descending'],
    'halves': ['half', 'halves', 'halving', 'divide in two', 'divide and conquer', 'divided by half', 'divides'],
    'lifo': ['lifo', 'last in first out', 'last in, first out', 'stack'],
    'fifo': ['fifo', 'first in first out', 'first in, first out', 'queue'],
    'recursion': ['recursive', 'recursion', 'recursively', 'call stack', 'function calls'],
    'function calls': ['call stack', 'function call', 'activation record', 'stack frames'],
    'class': ['class', 'classes', 'object', 'blueprint'],
    'parent': ['parent', 'superclass', 'base class', 'ancestor'],
    'child': ['child', 'subclass', 'derived class'],
    'reuse': ['reuse', 'reusable', 'reusability', 'reusing', 'inherit code'],
    'overloading': ['overload', 'overloading', 'same name different parameters', 'compile-time polymorphism', 'static polymorphism'],
    'overriding': ['override', 'overriding', 'runtime polymorphism', 'dynamic method dispatch', 'same signature in subclass'],
    'interface': ['interface', 'interfaces', 'pure contract', 'contract', 'multiple inheritance'],
    'abstract class': ['abstract class', 'abstraction', 'partial implementation'],
    'try': ['try', 'try block', 'exception handling'],
    'catch': ['catch', 'catch block', 'handle error', 'exception handling'],
    'finally': ['finally', 'cleanup', 'always executes'],
    'runtime error': ['runtime exception', 'runtime error', 'exceptions', 'unhandled exception'],
    'contiguous': ['contiguous', 'continuous', 'sequential memory', 'index-based'],
    'nodes': ['node', 'nodes', 'pointers', 'pointer', 'reference', 'references'],
    'insertion': ['insert', 'insertion', 'adding elements', 'add'],
    'bfs': ['breadth first search', 'bfs', 'level order', 'queue'],
    'dfs': ['depth first search', 'dfs', 'recursive search', 'stack traversal'],
    'queue': ['queue', 'fifo', 'line'],
    'stack': ['stack', 'lifo', 'push and pop'],
    'level': ['level', 'level by level', 'breadth'],
    'training': ['train', 'training', 'fit', 'fitting', 'trained'],
    'unseen': ['unseen', 'test data', 'validation data', 'new data', 'generalization'],
    'generalization': ['generalize', 'generalizing', 'generalization', 'overfitting'],
    'classification': ['classify', 'classifier', 'classification', 'discrete', 'labels', 'categories'],
    'classes': ['class labels', 'classes', 'categories', 'categorical'],
    'regression': ['continuous values', 'regression', 'regressor', 'continuous output', 'numerical'],
    'continuous': ['continuous', 'real numbers', 'numeric values', 'numbers'],
    'precision': ['precision', 'positive predictive value', 'tp / (tp + fp)'],
    'recall': ['recall', 'sensitivity', 'true positive rate', 'tp / (tp + fn)'],
    'true positives': ['true positive', 'true positives', 'tp'],
    'tp': ['true positive', 'true positives', 'tp'],
    'tn': ['true negative', 'true negatives', 'tn'],
    'fp': ['false positive', 'false positives', 'fp', 'type 1 error'],
    'fn': ['false negative', 'false negatives', 'fn', 'type 2 error'],
    'normalization': ['min-max', 'normalize', 'normalization', 'scaling [0,1]', 'rescaling'],
    'standardization': ['z-score', 'standardize', 'standardization', 'mean 0 std 1', 'unit variance'],
    'scale': ['scaling', 'scale', 'feature scale', 'magnitude'],
    'education': ['college', 'degree', 'study', 'education', 'university', 'btech', 'engineering', 'academics'],
    'skills': ['skill', 'skills', 'programming', 'languages', 'frameworks', 'technologies', 'tools', 'proficient'],
    'projects': ['project', 'projects', 'built', 'developed', 'application', 'portfolio', 'created'],
    'goals': ['goal', 'goals', 'aspire', 'aim', 'future', 'career objective', 'growth'],
    'strengths': ['strength', 'strengths', 'strong at', 'good at', 'passionate', 'adaptable', 'quick learner'],
    'weakness': ['weakness', 'weaknesses', 'working on', 'improve', 'challenge', 'learning curve'],
    'self-awareness': ['improve', 'improving', 'conscious', 'feedback', 'reflect'],
    'experience': ['experience', 'internship', 'hands-on', 'practical', 'background'],
    'role fit': ['fit', 'company', 'role', 'contribute', 'value', 'alignment'],
    'growth': ['grow', 'growth', 'develop', 'advancement', 'learning'],
    'learning': ['learn', 'learning', 'upskill', 'knowledge', 'technologies']
}

def clean_text(text: str) -> str:
    """Normalize text while preserving technical symbols and case structures."""
    if not text:
        return ""
    text = text.replace('\r\n', ' ').replace('\n', ' ').replace('\t', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text: str, remove_stopwords: bool = False) -> list[str]:
    """Tokenize text into lowercase alphanumeric and technical tokens."""
    if not text:
        return []
    raw_tokens = re.findall(r'[A-Za-z0-9_]+(?:\([A-Za-z0-9\s]+\))?|[A-Za-z0-9]+', text.lower())
    if remove_stopwords:
        return [t for t in raw_tokens if t not in STOP_WORDS and len(t) > 1]
    return raw_tokens

def split_sentences(text: str) -> list[str]:
    """Split text into sentences based on punctuation."""
    if not text:
        return []
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def calculate_token_overlap(text1: str, text2: str) -> float:
    """Calculate token overlap (Jaccard similarity) between two texts."""
    tokens1 = set(tokenize(text1, remove_stopwords=True))
    tokens2 = set(tokenize(text2, remove_stopwords=True))
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return len(intersection) / len(union)

def match_concept_in_text(concept: str, text: str) -> bool:
    """
    Check if a key concept or any of its known synonyms/variations
    appear in the candidate text.
    """
    concept_lower = concept.lower().strip()
    text_lower = text.lower()

    if concept_lower in text_lower:
        return True

    synonyms = CONCEPT_SYNONYMS.get(concept_lower, [concept_lower])
    for syn in synonyms:
        if ' ' in syn or '(' in syn:
            if syn in text_lower:
                return True
        else:
            if re.search(r'\b' + re.escape(syn) + r'\b', text_lower):
                return True

    concept_words = tokenize(concept_lower, remove_stopwords=True)
    if concept_words:
        matched_words = [w for w in concept_words if re.search(r'\b' + re.escape(w) + r'\b', text_lower)]
        if len(matched_words) == len(concept_words):
            return True

    return False
