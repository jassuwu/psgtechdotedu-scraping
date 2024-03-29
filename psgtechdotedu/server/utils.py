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
    df = pd.read_csv('../data/td_matrix.csv')
    df.set_index(keys='Unnamed: 0', inplace=True)
    docsDF = pd.read_json('../data/psgtech.json')
    docsDF.set_index(keys='url', inplace=True)

    # Load the links data and create the graph
    links_df = pd.read_csv('../data/links.csv')
    G = nx.from_pandas_edgelist(links_df, 'from', 'to', create_using=nx.DiGraph())
    pagerank = nx.pagerank(G)

    return df, docsDF, pagerank


def process_query(query, columns):
    punctuation_removed_query = re.sub(r'[^\w\s]', '', query)
    clean_query = re.sub(r'\s+',' ',punctuation_removed_query.strip())
    processed_query = [stemmer.stem(token.lower()) for token in word_tokenize(clean_query)]
    query_vector = vectorizer.fit_transform([' '.join(processed_query)])
    qf = pd.DataFrame(query_vector.toarray(), columns=vectorizer.get_feature_names_out())
    full_vector = [qf[col][0] if col in vectorizer.get_feature_names_out() else 0 for col in columns]
    return np.array(full_vector)

def find_top_n_relevant_docs(query_weights, tdMatrixDF, docsDF, pagerank, N, alpha=0.7):
    cosine_similarity_scores = cosine_similarity(query_weights.reshape(1, -1), tdMatrixDF.T.values)
    pagerank_scores = np.array([pagerank[url] if url in pagerank else 0 for url in tdMatrixDF.columns])
    scores = alpha * cosine_similarity_scores.flatten() + (1-alpha) * pagerank_scores
    df = pd.DataFrame({'docID': tdMatrixDF.columns, 'cos_sim_score': cosine_similarity_scores.flatten(), 'pagerank_score': pagerank_scores, 'score': scores})
    sorted_df = df.sort_values(by='score', ascending=False)
    results = docsDF.loc[sorted_df['docID'].values[:N].tolist()][['title']].reset_index()
    results['last_segment'] = results['url'].apply(extract_url_segments_to_title)
    results['title'] = results.apply(lambda row: f"{row['last_segment']} | {row['title']}" if row['last_segment'] else row['title'], axis=1)
    results['cos_sim_score'] = sorted_df['cos_sim_score'].values[:N]
    results['pagerank_score'] = sorted_df['pagerank_score'].values[:N]
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
