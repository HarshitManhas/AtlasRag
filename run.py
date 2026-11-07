import uvicorn

if __name__ == "__main__":
    uvicorn.run("atlas_rag.main:app", app_dir="src", host="127.0.0.1", port=8000, reload=True)
