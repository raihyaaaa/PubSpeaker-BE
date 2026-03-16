"""FastAPI application for speech analysis and pronunciation feedback."""

import json
import os
import time
from datetime import datetime
from config import TMP_DIR, AUDIO_DIR, TTS_DIR, RESPONSES_DIR
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from models import AnalyzeRequest
from services import (
    TranscriptionService,
    PronunciationService,
    GrammarService,
    FeedbackService,
    TTSService,
    TranscriptImprovementService
)
from services.pronunciation_alignment import PronunciationAlignmentService
from utils import save_uploaded_file, annotate_transcript, is_junk_token
from utils.phonetics import arpabet_to_ipa
from utils.text import annotate_corrected

# Low-stress function words AND common content words that are naturally
# reduced in connected speech.  Flagging these as "mispronounced" is
# unhelpful — native speakers routinely reduce them (e.g. "the" → /ðə/,
# "for" → /fɚ/, "and" → /ən/, "although" → /ɔlˈðoʊ/).
_FUNCTION_WORDS = frozenset({
    # Determiners / articles
    "the", "a", "an",
    # Conjunctions
    "and", "or", "but", "nor", "so", "yet",
    # Prepositions
    "in", "on", "at", "to", "for", "of", "by", "from", "with",
    "into", "over", "under", "through", "between", "before",
    "out", "off", "down", "away", "back", "here", "there",
    "around", "across", "along", "among", "against", "during",
    "without", "within", "upon", "toward", "towards", "beyond",
    "aside", "after", "until", "since", "above", "below",
    # Copulas / auxiliaries
    "is", "am", "are", "was", "were", "be", "been", "being",
    "do", "does", "did",
    "has", "have", "had",
    # Pronouns
    "it", "its",
    "he", "she", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "our", "their",
    "this", "that", "these", "those",
    "i",  # lowercase on purpose
    # Negation / quantifiers / determiners
    "not", "no",
    "some", "any", "all", "each", "every", "much", "many",
    "more", "most", "other", "another", "such",
    # Adverbs commonly reduced in connected speech
    "if", "then", "as", "up",
    "because", "very", "like", "just", "also", "even", "still",
    "about", "only", "really", "sometimes", "already",
    "maybe", "actually", "usually", "when", "will",
    "what", "where", "how", "who", "which", "than", "too",
    # Modals
    "can", "could", "would", "should", "may", "might",
    # Common verbs naturally reduced
    "get", "got", "going", "go", "come", "came",
    # Common content words with highly variable natural pronunciation
    # that MFA frequently flags as deviations even for native speakers
    "although", "however", "overall", "offer", "color", "colour",
    "lives", "while", "new", "one", "now", "well", "know",
    "used", "being", "want", "own", "world", "year", "make",
    "people", "right", "think", "good", "said", "its",
})

# Ensure all required directories exist
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TTS_DIR, exist_ok=True)
os.makedirs(RESPONSES_DIR, exist_ok=True)
print(f"Ensured directories exist: {TMP_DIR}, {AUDIO_DIR}, {TTS_DIR}, {RESPONSES_DIR}")


def _save_response(prefix: str, data: dict, audio_filename: str | None = None) -> str:
    """Save an endpoint response as a timestamped JSON file and return the path."""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Use the original audio filename (without extension) if available
    if audio_filename:
        name = os.path.splitext(os.path.basename(audio_filename))[0]
        # Sanitise: keep only alphanumerics, hyphens, underscores, spaces
        name = "".join(c for c in name if c.isalnum() or c in " -_").strip()
        filename = f"{prefix}_{name}_{ts}.json"
    else:
        filename = f"{prefix}_{ts}.json"
    path = os.path.join(RESPONSES_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    print(f"Saved response → {path}")
    return path

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
print("Initializing services...")
transcription_service = TranscriptionService()
pronunciation_service = PronunciationService()  # Legacy dictionary-based
pronunciation_alignment_service = PronunciationAlignmentService()  # NEW: Phoneme-level
grammar_service = GrammarService()
feedback_service = FeedbackService()
tts_service = TTSService(pronunciation_service)
improvement_service = TranscriptImprovementService()
print("All services initialized successfully\n")


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """
    Transcribe uploaded audio to text with word-level timestamps.
    
    Args:
        audio: Audio file upload
        
    Returns:
        Dictionary with transcript, segments, word timestamps, and audio_path
    """
    audio_path = save_uploaded_file(audio)
    
    try:
        result = transcription_service.transcribe(audio_path)
        # Include audio_path in response so it can be used for pronunciation analysis
        result["audio_path"] = audio_path
        _save_response("transcribe", result, audio.filename)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    """
    Analyze speech for pronunciation, grammar, and pace issues.
    
    Args:
        request: Analysis request with transcript or audio path
        
    Returns:
        Dictionary with analysis results, annotations, suggestions, and feedback
    """
    # Get transcript and words (either from request or transcribe audio)
    if request.transcript is None:
        if not request.audio_path:
            return JSONResponse(
                status_code=400,
                content={"error": "Provide either audio_path or transcript"}
            )
        
        transcription_result = transcription_service.transcribe(request.audio_path)
        transcript = transcription_result["text"]
        words = transcription_result["words"]
    else:
        transcript = request.transcript
        words = request.words or []
        
        # Generate word list if not provided
        if not words:
            tokens = transcript.split()
            words = [
                {"word": token, "start": None, "end": None}
                for token in tokens
            ]
    
    _timings = {}
    _t0 = time.perf_counter()

    # Analyze pronunciation using phoneme-level forced alignment
    if request.audio_path:
        # Use advanced phoneme alignment (requires audio)
        pronunciation_analysis = pronunciation_alignment_service.analyze_pronunciation(
            request.audio_path,
            transcript,
            words
        )
    else:
        # Fallback when no audio available
        pronunciation_analysis = {
            "available": False,
            "deviations": [],
            "confidence_scores": {},
            "summary": "Audio required for phoneme-level pronunciation analysis",
            "disclaimer": pronunciation_alignment_service._get_ethical_disclaimer()
        }

    _timings['1_pronunciation_alignment'] = time.perf_counter() - _t0
    _t1 = time.perf_counter()

    # Backward-compatible: expose mispronunciations list for the UI.
    # Prefer MFA-based deviations when available; otherwise fall back to legacy
    # dictionary-based heuristics using the transcript tokens.
    mispronunciations = []
    seen_mispron = set()  # dedup by (word, start, end)
    deviations = pronunciation_analysis.get("deviations") or []
    if deviations:
        for d in deviations:
            word = d.get("word")
            if not word or is_junk_token(word):
                continue

            # Skip common function words — they are naturally reduced in
            # connected speech and flagging them is not useful.
            if word.strip().lower() in _FUNCTION_WORDS:
                continue

            # Dedup: skip exact duplicate entries
            dedup_key = (word.lower(), round(d.get("start", 0), 3), round(d.get("end", 0), 3))
            if dedup_key in seen_mispron:
                continue
            seen_mispron.add(dedup_key)

            # Use IPA format if available, otherwise convert from ARPABET
            expected_ipa = d.get("expected_ipa") or arpabet_to_ipa(d.get("expected"))
            actual_ipa = d.get("actual_ipa") or arpabet_to_ipa(d.get("actual"))
            expected_disp = expected_ipa or d.get("expected") or ""
            actual_disp = actual_ipa or d.get("actual") or ""

            # Human-readable pronunciation guides
            expected_readable = d.get("expected_readable") or ""
            actual_readable = d.get("actual_readable") or ""

            mispronunciations.append(
                {
                    "word": word,
                    "start": d.get("start", 0),
                    "end": d.get("end", 0),
                    # UI expects this field and currently filters it.
                    "suggested_intended": word,
                    "reason": (
                        f"{d.get('severity', 'deviation')} pronunciation deviation "
                        f"(similarity {d.get('similarity', 'n/a')}): expected {expected_disp} "
                        f"vs detected {actual_disp}"
                    ),
                    "expected": d.get("expected"),
                    "actual": d.get("actual"),
                    "expected_ipa": expected_ipa,
                    "actual_ipa": actual_ipa,
                    "expected_readable": expected_readable,
                    "actual_readable": actual_readable,
                    "similarity": d.get("similarity"),
                    "severity": d.get("severity"),
                }
            )
    else:
        # Legacy heuristic detection from transcript; useful when MFA isn't installed.
        try:
            mispronunciations = pronunciation_service.analyze_words(words)
        except Exception:
            mispronunciations = []
    
    _timings['2_mispron_filter'] = time.perf_counter() - _t1
    _t2 = time.perf_counter()

    # Generate grammar corrections (used for annotation and issue extraction)
    corrections = grammar_service.generate_corrections(transcript)
    
    # Don't use the suggestion if it's a message (not actual corrected text)
    corrected_best = corrections[0] if corrections else transcript
    
    # User-facing suggestions for the Suggestions tab (tips only)
    suggestions = grammar_service.generate_user_suggestions(transcript, corrections)

    # Build the corrected paragraph with <green> highlights on changed words.
    # corrected_best is the best grammar-corrected version of the transcript.
    corrected_transcript = corrected_best if corrected_best != transcript else None
    corrected_annotated = (
        annotate_corrected(transcript, corrected_best)
        if corrected_transcript
        else None
    )

    # Build a dedicated grammar_corrected object so the frontend can
    # consume correction data without touching the suggestions array.
    grammar_corrected = {
        "corrected_transcript": corrected_transcript,
        "corrected_annotated": corrected_annotated,
        "corrections": (
            grammar_service._extract_sentence_corrections(transcript, corrected_best)
            if corrected_transcript
            else []
        ),
    }

    # Merge sentence-level corrections INTO the suggestions array so the
    # frontend Suggestions tab can display them (it reads from suggestions).
    if grammar_corrected["corrections"]:
        suggestions = grammar_corrected["corrections"] + suggestions

    # Generate annotated transcript
    # When MFA is available, the mispronunciations list already reflects stricter thresholds.
    # Pass dicts with {word, start, end} so only the SPECIFIC occurrence is highlighted,
    # not every instance of the same word in the transcript.
    mispron_for_annotation = [
        {"word": m.get("word"), "start": m.get("start", 0), "end": m.get("end", 0)}
        for m in (mispronunciations or [])
        if m.get("word") and not is_junk_token(m.get("word"))
    ]

    # annotate_transcript returns (annotated_str, grammar_issue_words).
    # The grammar_issues list is derived from the SAME diff that produces
    # the yellow highlighting, so Feedback and Summary are always in sync.
    annotated, grammar_issues = annotate_transcript(
        transcript,
        corrected_best,
        mispron_for_annotation,
        words_with_timestamps=words,
    )

    # Generate explanations
    explanations = []
    if pronunciation_analysis.get("transcript_suspect"):
        explanations.append(
            "⚠ Many pronunciation deviations detected — the transcript may contain "
            "errors. Consider re-recording with clearer audio or editing the transcript."
        )
    if pronunciation_analysis.get("deviations"):
        # Use the filtered mispronunciation count (after removing function
        # words, minor deviations, etc.) so the explanation matches what the
        # UI actually shows.
        shown_count = len(mispronunciations)
        if shown_count > 0:
            explanations.append(
                f"Pronunciation: {shown_count} word(s) with notable deviations"
            )
    if grammar_issues:
        unique_issues = set(grammar_issues)
        explanations.append(
            f"Grammar: Found issues with: {', '.join(unique_issues)}"
        )
    
    _timings['3_grammar_correction'] = time.perf_counter() - _t2
    _t3 = time.perf_counter()

    # Generate dynamic feedback
    feedback = feedback_service.generate(
        transcript,
        pronunciation_analysis,  # NEW: Pass full analysis dict
        grammar_issues,
        words
    )

    _timings['4_feedback'] = time.perf_counter() - _t3
    _t4 = time.perf_counter()

    # Generate improved transcript versions (local LLM rewrite)
    base_for_improve = corrected_best if corrected_transcript else transcript
    improved_versions = improvement_service.generate_improved_versions(
        base_for_improve,
        original_transcript=transcript,
    )

    _timings['5_coedit_improvement'] = time.perf_counter() - _t4

    # Sync pronunciation_analysis.deviations with the filtered mispronunciations
    # list so that any frontend reading from either field sees the same data.
    # The APK reads from pronunciation_analysis.deviations directly.
    if pronunciation_analysis.get("deviations") and mispronunciations:
        filtered_keys = {
            (m["word"], m.get("start", 0), m.get("end", 0))
            for m in mispronunciations
        }
        pronunciation_analysis["deviations"] = [
            d for d in pronunciation_analysis["deviations"]
            if (d.get("word"), d.get("start", 0), d.get("end", 0)) in filtered_keys
        ]
    elif pronunciation_analysis.get("deviations") and not mispronunciations:
        pronunciation_analysis["deviations"] = []

    response = {
        "transcript": transcript,
        "pronunciation_analysis": pronunciation_analysis,  # Filtered to match mispronunciations
        "mispronunciations": mispronunciations,  # Backward-compatible for UI
        "annotated": annotated,
        "corrected_transcript": corrected_transcript,  # Plain corrected text (None if no fixes)
        "corrected_annotated": corrected_annotated,    # Corrected text with <green> on changed words
        "suggestions": suggestions,                     # Tips only (corrections are in corrected_*)
        "grammar_corrected": grammar_corrected,         # Dedicated grammar correction data
        "improved": improved_versions,                  # Rewritten versions for clarity & impact
        "explanations": explanations,
        "feedback": feedback
    }
    _timings['total'] = time.perf_counter() - _t0

    # Log timing breakdown
    print("\n=== ANALYZE TIMING BREAKDOWN ===")
    for k, v in _timings.items():
        print(f"  {k}: {v:.2f}s")
    print()

    # Derive original audio filename from the audio_path
    audio_filename = os.path.basename(request.audio_path) if request.audio_path else None
    _save_response("analyze", response, audio_filename)
    return response


@app.get("/pronounce/{word}")
async def pronounce(word: str):
    """
    Get pronunciation audio and phonemes for a word.
    
    Args:
        word: The word to get pronunciation for
        
    Returns:
        Dictionary with word, audio URL, and canonical phonemes
    """
    result = tts_service.get_or_create_audio(word)
    return result


@app.get("/tts/{filename}")
async def serve_tts(filename: str):
    """
    Serve TTS audio file.
    
    Args:
        filename: The audio filename
        
    Returns:
        Audio file response
    """
    audio_path = tts_service.get_audio_path(filename)
    
    if not audio_path:
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(audio_path, media_type="audio/mpeg")


