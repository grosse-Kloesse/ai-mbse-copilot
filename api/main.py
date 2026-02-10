from fastapi import FastAPI

app = FastAPI(title="AI Maintenance Copilot")

@app.get("/health")
def health():
    return {"status": "ok"}