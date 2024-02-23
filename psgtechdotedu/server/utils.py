import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.stem import PorterStemmer
import pandas as pd
import re
from urllib.parse import urlparse, unquote

vectorizer = TfidfVectorizer()
stemmer = PorterStemmer()

def load_data():
    df = pd.read_csv('../data/td_matrix.csv')
    df.set_index(keys='Unnamed: 0', inplace=True)
    docsDF = pd.read_json('../data/psgtech.json')
    docsDF.set_index(keys='url', inplace=True)
    return df, docsDF

def process_query(query, columns):
    punctuation_removed_query = re.sub(r'[^\w\s]', '', query)
    clean_query = re.sub(r'\s+',' ',punctuation_removed_query.strip())
    processed_query = [stemmer.stem(token.lower()) for token in word_tokenize(clean_query)]
    query_vector = vectorizer.fit_transform([' '.join(processed_query)])
    qf = pd.DataFrame(query_vector.toarray(), columns=vectorizer.get_feature_names_out())
    full_vector = [qf[col][0] if col in vectorizer.get_feature_names_out() else 0 for col in columns]
    return np.array(full_vector)

def find_top_n_relevant_docs(query_weights, tdMatrixDF, docsDF, N):
    cosine_similarity_scores = cosine_similarity(query_weights.reshape(1, -1), tdMatrixDF.T.values)
    df = pd.DataFrame({'docID': tdMatrixDF.columns, 'cosineSimilarity': cosine_similarity_scores.flatten()})
    sorted_df = df.sort_values(by='cosineSimilarity', ascending=False)
    results = docsDF.loc[sorted_df['docID'].values[:N].tolist()][['title']].reset_index()
    results['last_segment'] = results['url'].apply(extract_url_segments_to_title)
    results['title'] = results.apply(lambda row: f"{row['last_segment']} | {row['title']}" if row['last_segment'] else row['title'], axis=1)
    return results[['url', 'title']].to_dict(orient='records')

def extract_url_segments_to_title(url):
    parsed = urlparse(url)
    last_segment = unquote(parsed.path.split('/')[-1])
    last_segment = ' '.join(word.capitalize() for word in re.split(r'[/.\-_]', last_segment) if word.lower() not in {'html', 'php'})
    return last_segment
