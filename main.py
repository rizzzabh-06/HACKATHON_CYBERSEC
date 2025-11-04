from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import os
from dotenv import load_dotenv
from src.rag_system import RAGSystem
from pydantic import BaseModel

# Load environment variables
load_dotenv()

app = FastAPI(title="RAG API")
rag_system = RAGSystem()

class Query(BaseModel):
    question: str

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload one or more files to the RAG system"""
    try:
        uploaded_files = []
        for file in files:
            file_path = rag_system.save_uploaded_file(file)
            uploaded_files.append(file_path)
        
        return JSONResponse(
            content={
                "message": "Files uploaded and processed successfully",
                "files": uploaded_files
            },
            status_code=200
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_rag(query: Query):
    """Query the RAG system"""
    try:
        response = rag_system.query(query.question)
        return JSONResponse(
            content={"response": response},
            status_code=200
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-directory")
async def process_directory(directory_path: str):
    """Process all files in a directory"""
    try:
        if not os.path.exists(directory_path):
            raise HTTPException(status_code=404, detail="Directory not found")
            
        rag_system.process_and_add_directory(directory_path)
        return JSONResponse(
            content={"message": f"Directory {directory_path} processed successfully"},
            status_code=200
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)