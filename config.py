"""Configuration constants for PubSpeaker application."""

import os

# Directory paths
# TMP_DIR = "/tmp/capstone_local"
TMP_DIR = "C:\\Users\\Raily Almeron\\AppData\\Local\\PubSpeaker\\tmp"  # Windows path for temporary files
AUDIO_DIR = os.path.join(TMP_DIR, "audio")
TTS_DIR = os.path.join(TMP_DIR, "tts")
RESPONSES_DIR = os.path.join(TMP_DIR, "responses")

# Model settings
DEVICE = "cuda"  
WHISPER_MODEL_SIZE = "small"

# Pronunciation analysis thresholds
SIMILARITY_CUTOFF = 0.9  # For difflib.get_close_matches (high = strict)
MAX_EDIT_DISTANCE = 2
MIN_WORD_LENGTH_FOR_MISPRONUNCIATION = 4
MIN_WORD_LENGTH_FOR_CONCATENATION = 8

# Speaking pace thresholds (words per minute)
PACE_VERY_SLOW = 100
PACE_GOOD_LOWER = 130
PACE_MODERATE_LOWER = 160

# Word inflections for morphological analysis
COMMON_INFLECTIONS = {'s', 'es', 'ed', 'ing', 'er', 'est', 'ly', 'd'}

# Grammar correction settings
DEFAULT_SUGGESTION_COUNT = 3

# Transcript improvement (CoEdIT model fine-tuned for text editing)
IMPROVED_TRANSCRIPT_MODEL = "grammarly/coedit-large"
IMPROVED_TRANSCRIPT_COUNT = 1  # Number of improved versions to generate

# MFA (Montreal Forced Aligner) settings
# MFA_PATH = os.environ.get('MFA_PATH', None)  # Set to None to auto-detect
# MFA_TMP_DIR = os.path.join(os.path.expanduser('~'), 'mfa_tmp')  # No spaces in path - MFA breaks with spaces
MFA_PATH = "C:\\conda_envs\\mfa_env\\Scripts\\mfa.exe"  # Windows path to MFA executable
MFA_TMP_DIR = "C:\\Users\\Raily Almeron\\AppData\\Local\\PubSpeaker\\mfa_tmp"
# Pronunciation detection thresholds
# Higher values = more sensitive (catches more deviations)
# Lower values = less sensitive (only catches clear mispronunciations)
PRONUNCIATION_DEVIATION_THRESHOLD = 0.90  # Report deviations below this similarity
PRONUNCIATION_SEVERITY_MINOR = 0.80       # >= this is minor
PRONUNCIATION_SEVERITY_MODERATE = 0.65    # >= this is moderate, < minor
PRONUNCIATION_SEVERITY_NOTABLE = 0.65     # < this is notable
