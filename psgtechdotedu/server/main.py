from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel
from typing import List

from utils import grid_search, load_data, process_query, find_top_n_relevant_docs

import nltk
nltk.download('punkt')

inverted_index, docsDF, pagerank = load_data()

alpha = 0.7

app = FastAPI()

class Result(BaseModel):
    url: str
    title: str
    cos_sim_score: float
    pagerank_score: float
    alpha: float

class Feedback(BaseModel):
    query: str
    feedback: int  # 1 for upvote, -1 for downvote

@app.get("/", status_code=200)
def index():
    """Return a greeting message."""
    return "sup, you've reached the psgtechdotedu backend."

@app.get("/ping", status_code=200)
@app.get("/health", status_code=200)
def health():
    """Return a health status message."""
    return {"message": "we healthy! or should i say pong!"}

@app.get("/search", status_code=201, response_model=List[Result])
async def get_query_results(q: str = Query(..., min_length=1, description="the search query")):
    """Return a list of relevant documents for the given query."""
    try:
        query_terms = process_query(q)
        results = find_top_n_relevant_docs(query_terms, inverted_index, docsDF, pagerank, 50, alpha)
        return results
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.post("/feedback", status_code=200)
async def receive_feedback(feedback_data: Feedback):
    """Update the global alpha value with the user feedback and respond with success or failure."""    
    global alpha
    query = feedback_data.query
    feedback = feedback_data.feedback
    query_terms = process_query(query)
    results = find_top_n_relevant_docs(query_terms, inverted_index, docsDF, pagerank, 50, alpha)
    alpha = grid_search(results, feedback, alpha)
    return {"message": f"Feedback received and alpha updated to {alpha}"}