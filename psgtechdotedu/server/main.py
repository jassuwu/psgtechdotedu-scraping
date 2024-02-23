from fastapi import FastAPI, HTTPException

from utils import load_data, process_query, find_top_n_relevant_docs

df, docsDF = load_data()

app = FastAPI()

@app.get("/search")
async def get_query_results(q: str):
    try:
        full_vector = process_query(q, df.index)
        results = find_top_n_relevant_docs(full_vector, df, docsDF, 10)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))