"""Transcription service using Whisper ASR."""

import whisper
from typing import Dict, List

from config import DEVICE, WHISPER_MODEL_SIZE


class TranscriptionService:
    """Service for audio transcription using Whisper."""
    
    def __init__(self):
        """Initialize Whisper model."""
        print(f"Loading Whisper model ({WHISPER_MODEL_SIZE}) on device: {DEVICE}")
        self.model = whisper.load_model(WHISPER_MODEL_SIZE, device=DEVICE)
        print("Whisper model loaded successfully")
    
    def transcribe(self, audio_path: str) -> Dict[str, any]:
        """
        Transcribe audio file to text with word-level timestamps.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with 'text', 'segments', and 'words' keys
        """
        print(f"Transcribing audio: {audio_path}")
        result = self.model.transcribe(audio_path, word_timestamps=True)
        text = result.get("text", "").strip()
        segments = result.get("segments", [])
        
        # Extract real word-level timestamps from Whisper
        words = self._extract_word_timestamps(segments)
        
        return {
            "text": text,
            "segments": segments,
            "words": words
        }
    
    def _extract_word_timestamps(self, segments: List[Dict]) -> List[Dict]:
        """
        Extract word-level timestamps from Whisper segments.
        
        Uses Whisper's native word_timestamps when available (from
        word_timestamps=True). Falls back to even distribution if the
        segment does not contain per-word timing data.
        
        Args:
            segments: List of Whisper segments
            
        Returns:
            List of word dictionaries with 'word', 'start', 'end' keys
        """
        words = []
        
        for segment in segments:
            # Prefer native word-level timestamps from Whisper
            segment_words = segment.get("words")
            if segment_words:
                for w in segment_words:
                    words.append({
                        "word": w.get("word", "").strip(),
                        "start": round(w.get("start", 0.0), 3),
                        "end": round(w.get("end", 0.0), 3)
                    })
                continue
            
            # Fallback: distribute segment duration evenly across tokens
            segment_text = segment.get("text", "").strip()
            start_time = segment.get("start", 0.0)
            end_time = segment.get("end", 0.0)
            tokens = segment_text.split()
            
            if len(tokens) == 0:
                continue
            
            duration = max(end_time - start_time, 0.01)
            time_per_word = duration / len(tokens)
            
            for i, word in enumerate(tokens):
                word_start = start_time + i * time_per_word
                word_end = start_time + (i + 1) * time_per_word
                
                words.append({
                    "word": word,
                    "start": round(word_start, 3),
                    "end": round(word_end, 3)
                })
        
        return words
