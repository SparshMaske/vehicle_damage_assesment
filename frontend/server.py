import os
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI(title="Frontend Proxy Server")

BACKEND_URL = "http://127.0.0.1:8000/predict"

# Setup static files directory
current_dir = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=current_dir), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open(os.path.join(current_dir, "index.html"), "r") as f:
        return f.read()

@app.post("/predict")
async def proxy_predict(file: UploadFile = File(...)):
    """Proxies the file upload to the main backend API to bypass CORS issues."""
    content = await file.read()
    files = {"file": (file.filename, content, file.content_type)}
    try:
        response = requests.post(BACKEND_URL, files=files)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        detail = str(e)
        if hasattr(e, 'response') and e.response is not None:
            try:
                detail = e.response.json().get("detail", detail)
            except:
                pass
        raise HTTPException(status_code=500, detail=detail)

if __name__ == "__main__":
    print(f"Starting frontend server on http://localhost:3000")
    print(f"Proxying API requests to {BACKEND_URL}")
    uvicorn.run(app, host="0.0.0.0", port=3000)
