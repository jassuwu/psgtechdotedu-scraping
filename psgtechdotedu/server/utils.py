import json
from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.stem import PorterStemmer
import pandas as pd
import networkx as nx
import re
from urllib.parse import urlparse, unquote

vectorizer = TfidfVectorizer()
stemmer = PorterStemmer()

def load_data():
    with open('../data/inverted_index.json', 'r') as f:
        inverted_index = json.load(f)

    docsDF = pd.read_json('../data/psgtech.json')
    docsDF.set_index(keys='url', inplace=True)

    links_df = pd.read_csv('../data/links.csv')
    G = nx.from_pandas_edgelist(links_df, 'from', 'to', create_using=nx.DiGraph())
    pagerank = nx.pagerank(G)

    return inverted_index, docsDF, pagerank


def process_query(query):
    punctuation_removed_query = re.sub(r'[^\w\s]', '', query)
    clean_query = re.sub(r'\s+',' ',punctuation_removed_query.strip())
    processed_query = [stemmer.stem(token.lower()) for token in word_tokenize(clean_query)]
    return processed_query

def find_top_n_relevant_docs(query_terms, inverted_index, docsDF, pagerank, N, alpha=0.7):
    doc_vectors = defaultdict(list)

    for term in query_terms:
        if term in inverted_index:
            for doc_id, weight in inverted_index[term]:
                doc_vectors[doc_id].append(weight)

    max_len = max(len(vec) for vec in doc_vectors.values())

    for doc_id, vec in doc_vectors.items():
        if len(vec) < max_len:
            doc_vectors[doc_id] = vec + [0] * (max_len - len(vec))

    doc_matrix = np.array([vec for vec in doc_vectors.values()])
    query_vector = np.ones((1, len(query_terms)))
    cosine_similarity_scores = cosine_similarity(query_vector, doc_matrix)

    relevance_df = pd.DataFrame(list(doc_vectors.keys()), columns=['docID'])
    relevance_df['cos_sim_score'] = cosine_similarity_scores.flatten()

    pagerank_scores = np.array([pagerank[url] if url in pagerank else 0 for url in relevance_df['docID']])
    scores = alpha * relevance_df['cos_sim_score'].values + (1-alpha) * pagerank_scores

    relevance_df['score'] = scores
    sorted_df = relevance_df.sort_values(by='score', ascending=False)

    results = docsDF.loc[sorted_df['docID'].values[:N].tolist()][['title']].reset_index()
    results['last_segment'] = results['url'].apply(extract_url_segments_to_title)
    results['title'] = results.apply(lambda row: f"{row['last_segment']} | {row['title']}" if row['last_segment'] else row['title'], axis=1)
    results['cos_sim_score'] = sorted_df['cos_sim_score'].values[:N]
    results['pagerank_score'] = pagerank_scores[:N]
    results['alpha'] = alpha
    return results[['url', 'title', 'cos_sim_score', 'pagerank_score', 'alpha']].to_dict(orient='records')


def extract_url_segments_to_title(url):
    parsed = urlparse(url)
    last_segment = unquote(parsed.path.split('/')[-1])
    last_segment = ' '.join(word.capitalize() for word in re.split(r'[/.\-_]', last_segment) if word.lower() not in {'html', 'php'})
    return last_segment

def grid_search(results, feedback):

    def calculate_scores(alpha, cos_sim_scores, pagerank_scores):
        return alpha * cos_sim_scores + (1 - alpha) * pagerank_scores

    def calculate_loss(scores, feedback):
         return np.sum((scores - feedback) ** 2)
    
    cos_sim_scores = np.array([r['cos_sim_score'] for r in results])
    pagerank_scores = np.array([r['pagerank_score'] for r in results])
    
    best_alpha = None
    best_loss = float('inf')

    for alpha in np.linspace(0, 1, 1001):
        scores = calculate_scores(alpha, cos_sim_scores, pagerank_scores)
        loss = calculate_loss(scores, feedback)
        if loss < best_loss:
            best_alpha = alpha
            best_loss = loss

    return best_alpha
