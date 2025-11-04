from typing import List
import docx2txt
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def process_file(file_path: str) -> str:
    """
    Process different file types and extract text content
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.txt':
        with open(file_path, 'r') as file:
            text = file.read()
    elif file_extension == '.docx':
        text = docx2txt.process(file_path)
    elif file_extension == '.pdf':
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")
    
    return text

def split_text(text: str) -> List[str]:
    """
    Split text into chunks using Langchain's text splitter
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks