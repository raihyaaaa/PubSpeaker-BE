"""File upload and handling utilities."""

import os
import uuid
from fastapi import UploadFile

from config import AUDIO_DIR


def save_uploaded_file(upload: UploadFile) -> str:
    """
    Save an uploaded file to the audio directory.
    
    Args:
        upload: FastAPI UploadFile object
        
    Returns:
        Absolute path to the saved file
    """
    os.makedirs(AUDIO_DIR, exist_ok=True)
    
    extension = os.path.splitext(upload.filename)[1] or ".wav"
    filename = f"{uuid.uuid4().hex}{extension}"
    output_path = os.path.join(AUDIO_DIR, filename)
    
    with open(output_path, "wb") as f:
        f.write(upload.file.read())
        
    return output_path
