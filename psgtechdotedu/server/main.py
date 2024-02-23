from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel
from typing import List

from utils import load_data, process_query, find_top_n_relevant_docs

df, docsDF = load_data()

app = FastAPI()

class Result(BaseModel):
    url: str
    title: str

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
        full_vector = process_query(q, df.index)
        results = find_top_n_relevant_docs(full_vector, df, docsDF, 50)
        return results
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
