"""
Phoneme-level pronunciation alignment service using Montreal Forced Aligner.

ETHICAL FRAMEWORK:
- Reports "pronunciation deviations", NOT "errors"
- Uses probabilistic scores, not binary judgments
- Emphasizes clarity and communication effectiveness
- Acknowledges linguistic diversity and dialectal variation
"""

import os
import subprocess
import json
import tempfile
import shutil
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging

from config import (
    TMP_DIR, 
    MFA_PATH,
    MFA_TMP_DIR,
    PRONUNCIATION_DEVIATION_THRESHOLD,
    PRONUNCIATION_SEVERITY_MINOR,
    PRONUNCIATION_SEVERITY_MODERATE,
    PRONUNCIATION_SEVERITY_NOTABLE
)
from services.pronunciation import PronunciationService
from utils.phonetics import arpabet_to_ipa, arpabet_to_readable
from utils.text import is_junk_token

logger = logging.getLogger(__name__)


class PronunciationAlignmentService:
    """
    Phoneme-level forced alignment for pronunciation analysis.
    
    Uses Montreal Forced Aligner (MFA) to:
    1. Align audio to phonemes with timestamps
    2. Generate confidence scores for each phoneme
    3. Compare with canonical pronunciations
    4. Report deviations probabilistically
    """
    
    def __init__(self):
        """Initialize MFA alignment service."""
        self.pronunciation_service = PronunciationService()
        # Cache alignment results by (audio_hash, transcript) to ensure
        # deterministic mispronunciation counts across repeated analyses.
        self._alignment_cache: Dict[str, Dict] = {}
        
        print("\n=== MFA Initialization ===")
        self.mfa_path = self._find_mfa_executable()
        
        # Check if MFA is executable and models are available
        self.mfa_available = False
        if self.mfa_path:
            print(f"✓ MFA executable found: {self.mfa_path}")
            if self._check_mfa_installation():
                print("✓ MFA is executable and working")
                if self._check_mfa_models():
                    self.mfa_available = True
                    print("✓ MFA models installed (english_us_arpa)")
                    print("✓ MFA is FULLY READY for pronunciation analysis\n")
                    logger.info(f"MFA fully configured and ready at: {self.mfa_path}")
                else:
                    print("✗ MFA models not found")
                    print("  Install with: mfa model download acoustic english_us_arpa")
                    print("                mfa model download dictionary english_us_arpa\n")
                    logger.warning("MFA found but required models not available")
            else:
                print(f"✗ MFA not executable or not working properly\n")
                logger.warning(f"MFA found at {self.mfa_path} but not working properly")
        else:
            print("✗ MFA executable not found")
            print("  Install with: conda install -c conda-forge montreal-forced-aligner\n")
            logger.warning("MFA not found. Phoneme alignment unavailable")
    
    def _check_mfa_models(self) -> bool:
        """Check if required MFA models are downloaded."""
        if not self.mfa_path:
            return False
        
        try:
            # Check if english_us_arpa acoustic model is available
            print("  Checking acoustic models...")
            result = subprocess.run(
                [self.mfa_path, 'model', 'list', 'acoustic'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                print(f"  Failed to list acoustic models: {result.stderr}")
                return False
            
            if 'english_us_arpa' not in result.stdout:
                print(f"  Acoustic models available: {result.stdout.strip()}")
                print("  'english_us_arpa' not found in acoustic models")
                return False
            
            print("  ✓ Acoustic model 'english_us_arpa' found")
            
            # Check dictionary too
            print("  Checking dictionary models...")
            result = subprocess.run(
                [self.mfa_path, 'model', 'list', 'dictionary'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                print(f"  Failed to list dictionary models: {result.stderr}")
                return False
            
            if 'english_us_arpa' not in result.stdout:
                print(f"  Dictionary models available: {result.stdout.strip()}")
                print("  'english_us_arpa' not found in dictionary models")
                return False
            
            print("  ✓ Dictionary model 'english_us_arpa' found")
            return True
                
        except subprocess.TimeoutExpired:
            print("  Model check timed out")
            return False
        except Exception as e:
            print(f"  Model check error: {e}")
            logger.warning(f"Failed to check MFA models: {e}")
            return False
        
    def _find_mfa_executable(self) -> Optional[str]:
        """Find MFA executable path."""
        # 1. Check config
        if MFA_PATH and os.path.isfile(MFA_PATH):
            return MFA_PATH
        
        # 2. Check PATH
        mfa_in_path = shutil.which('mfa')
        if mfa_in_path:
            filename, file_ext = os.path.splitext(mfa_in_path)
            mfa_in_path = filename + '.exe'
            return mfa_in_path
        
        # 3. Check common conda locations
        common_paths = [
            '/opt/homebrew/Caskroom/miniconda/base/envs/mfa/bin/mfa',
            '/opt/homebrew/Caskroom/miniconda/base/bin/mfa',
            os.path.expanduser('~/miniconda3/envs/mfa/bin/mfa'),
            os.path.expanduser('~/anaconda3/envs/mfa/bin/mfa'),
            os.path.expanduser('~/opt/miniconda3/envs/mfa/bin/mfa'),
        ]
        
        for path in common_paths:
            if os.path.isfile(path):
                return path
        
        return None
    
    def _check_mfa_installation(self) -> bool:
        """Check if MFA is installed and working, including third-party deps."""
        if not self.mfa_path:
            return False
            
        try:
            result = subprocess.run(
                [self.mfa_path, 'version'],
                capture_output=True,
                text=True,
                check=True
            )
            print(result.returncode)
            if result.returncode == 0:
                logger.info(f"MFA version: {result.stdout.strip()}")
                
                # Check that fstcompile (OpenFST) is available
                mfa_bin_dir = os.path.dirname(self.mfa_path)
                mfa_env_dir = os.path.dirname(mfa_bin_dir)

                candidate_paths = [
                    os.path.join(mfa_env_dir, 'fstcompile'),
                    os.path.join(mfa_env_dir, 'fstcompile.exe'),
                    os.path.join(mfa_env_dir, 'Library', 'bin', 'fstcompile'),
                    os.path.join(mfa_env_dir, 'Library', 'bin', 'fstcompile.exe'),
                ]
                # fstcompile_path = os.path.join(mfa_bin_dir, 'fstcompile')
                fstcompile_path = next((p for p in candidate_paths if os.path.isfile(p)), None)
                # if not os.path.isfile(fstcompile_path):
                #     print("  ✗ fstcompile (OpenFST) not found in MFA bin directory")
                #     # Also check via which in the same env
                #     fst_check = subprocess.run(
                #         [os.path.join(mfa_bin_dir, 'python'), '-c',
                #          'import shutil; print(shutil.which("fstcompile"))'],
                #         capture_output=True, text=True, timeout=5
                #     )
                #     if 'None' in fst_check.stdout or not fst_check.stdout.strip():
                #         print("  ✗ fstcompile (OpenFST) not found")
                #         print("    Install with: conda install -n mfa -c conda-forge openfst")
                #         return False
                # else:
                #     print("  ✓ fstcompile (OpenFST) found in MFA bin directory")
                if not fstcompile_path:
                    fst_path = shutil.which('fstcompile') or shutil.which('fstcompile.exe')
                    if not fst_path:
                        print("  ✗ fstcompile (OpenFST) not found in MFA environment")
                        print("    Install with: conda install -n mfa -c conda-forge openfst")
                        for p in candidate_paths:
                            print(f"    Checked: {p} (exists: {os.path.isfile(p)})")
                        return False
                print("  ✓ fstcompile (OpenFST) found")
                return True
        except Exception as e:
            logger.warning(f"MFA check failed: {e}")
        return False
    
    def analyze_pronunciation(
        self,
        audio_path: str,
        transcript: str,
        words: List[Dict]
    ) -> Dict:
        """
        Analyze pronunciation using phoneme-level forced alignment.
        
        Args:
            audio_path: Path to audio file
            transcript: Reference transcript
            words: Word-level timestamps from Whisper
            
        Returns:
            {
                'available': bool,  # Whether MFA is available
                'deviations': List[Dict],  # Phoneme deviations found
                'confidence_scores': Dict,  # Per-word confidence
                'summary': str,  # Human-readable summary
                'disclaimer': str  # Ethical disclaimer
            }
        """
        if not self.mfa_available:
            logger.info("MFA not available, using fallback analysis")
            return self._fallback_analysis(words)
        
        try:
            # ── Cache lookup: reuse previous alignment for same audio+transcript ──
            import hashlib
            audio_hash = self._hash_file(audio_path)
            cache_key = f"{audio_hash}:{hashlib.sha256(transcript.encode()).hexdigest()[:16]}"
            if cache_key in self._alignment_cache:
                logger.info("Using cached MFA alignment result")
                return self._alignment_cache[cache_key]

            # Run MFA alignment
            logger.info(f"Starting MFA alignment for audio: {audio_path}")
            alignment = self._run_mfa_alignment(audio_path, transcript)
            
            if not alignment:
                print("MFA alignment failed, falling back")
                logger.warning("MFA alignment failed, falling back")
                return self._fallback_analysis(words)
            
            # Analyze phoneme deviations
            deviations = self._analyze_phoneme_deviations(alignment, transcript)
            
            # ── Transcript re-verification ──
            # If too many words show deviations, the transcript itself is likely
            # wrong (Whisper error propagation).  Flag the result so the caller
            # knows the transcript may need correction.
            total_words = len(alignment.get('words', []))
            deviation_ratio = len(deviations) / max(total_words, 1)
            transcript_suspect = deviation_ratio > 0.5  # >50% words deviated
            
            if transcript_suspect:
                logger.warning(
                    f"Transcript may be inaccurate: {len(deviations)}/{total_words} "
                    f"words ({deviation_ratio:.0%}) show pronunciation deviations. "
                    f"This likely indicates transcription errors, not pronunciation issues."
                )
                print(
                    f"[MFA] ⚠ High deviation rate ({deviation_ratio:.0%}). "
                    f"Transcript may contain errors from Whisper."
                )
            
            # Generate confidence scores
            confidence = self._calculate_confidence_scores(alignment)
            
            # Create summary
            summary = self._generate_summary(deviations, confidence)
            
            if transcript_suspect:
                summary = (
                    f"⚠ High deviation rate ({deviation_ratio:.0%}) suggests possible "
                    f"transcription errors — review the transcript for accuracy. "
                ) + summary
            
            result = {
                'available': True,
                'deviations': deviations,
                'confidence_scores': confidence,
                'summary': summary,
                'disclaimer': self._get_ethical_disclaimer(),
                'transcript_suspect': transcript_suspect,
                'deviation_ratio': round(deviation_ratio, 3)
            }

            # Store in cache for deterministic repeated lookups
            self._alignment_cache[cache_key] = result

            return result
            
        except Exception as e:
            logger.error(f"Pronunciation alignment error: {e}")
            return self._fallback_analysis(words)

    @staticmethod
    def _hash_file(path: str) -> str:
        """Return a short SHA-256 hex digest of a file's contents."""
        import hashlib
        h = hashlib.sha256()
        try:
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    h.update(chunk)
        except OSError:
            return "unknown"
        return h.hexdigest()[:16]

    def _run_mfa_alignment(
        self,
        audio_path: str,
        transcript: str
    ) -> Optional[Dict]:
        """
        Run Montreal Forced Aligner on audio + transcript.
        
        Returns:
            Dictionary with phoneme alignments and confidence scores
        """
        # Use MFA_TMP_DIR (in home dir) instead of /tmp
        # macOS /tmp -> /private/tmp symlink causes MFA path validation failures
        os.makedirs(MFA_TMP_DIR, exist_ok=True)
        
        tmpdir = None
        output_dir = None
        try:
            # Generate unique directory name
            import random
            import string
            random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
            tmpdir = os.path.join(MFA_TMP_DIR, f'corpus_{random_suffix}')
            tmpdir_path = Path(tmpdir)
            
            # Create directory using os.makedirs (more reliable than tempfile for MFA)
            os.makedirs(tmpdir, exist_ok=False)
            
            print(f"[MFA] Created temp directory: {tmpdir_path}")
            
            # Verify directory exists and is writable
            if not tmpdir_path.exists():
                print(f"[MFA] ERROR: Failed to create temp directory: {tmpdir_path}")
                logger.error(f"Failed to create temp directory: {tmpdir_path}")
                return None
            
            # Prepare input files
            audio_name = Path(audio_path).stem
            audio_dest = tmpdir_path / f"{audio_name}.wav"
            transcript_dest = tmpdir_path / f"{audio_name}.txt"
            
            print(f"[MFA] Audio source: {audio_path}")
            print(f"[MFA] Audio destination: {audio_dest}")
            
            # Copy audio (MFA expects .wav)
            try:
                if not audio_path.endswith('.wav'):
                    # Convert to WAV if needed (requires ffmpeg)
                    print(f"[MFA] Converting {Path(audio_path).suffix} to WAV...")
                    self._convert_to_wav(audio_path, str(audio_dest))
                else:
                    import shutil as sh
                    sh.copy(audio_path, audio_dest)
                
                print(f"[MFA] Audio file prepared: {audio_dest} (exists: {audio_dest.exists()})")
                
                # Write transcript
                transcript_clean = transcript.strip()
                transcript_dest.write_text(transcript_clean)
                print(f"[MFA] Transcript file prepared: {transcript_dest} (exists: {transcript_dest.exists()})")
                print(f"[MFA] Transcript content: {transcript_clean[:100]}...")
                
                # Verify files exist
                if not audio_dest.exists():
                    print(f"[MFA] ERROR: Audio file not found after preparation: {audio_dest}")
                    logger.error(f"Audio file not found: {audio_dest}")
                    return None
                if not transcript_dest.exists():
                    print(f"[MFA] ERROR: Transcript file not found: {transcript_dest}")
                    logger.error(f"Transcript file not found: {transcript_dest}")
                    return None
                    
            except Exception as e:
                print(f"[MFA] ERROR preparing input files: {e}")
                logger.error(f"Failed to prepare MFA input files: {e}")
                return None
            
            # Create output directory OUTSIDE corpus directory
            # MFA requires corpus to only contain .wav and .txt files, no subdirectories
            output_dir = Path(MFA_TMP_DIR) / f"output_{random_suffix}"
            output_dir.mkdir(exist_ok=True)
            print(f"[MFA] Output directory created: {output_dir}")
            
            # Verify corpus directory structure (should only have .wav and .txt files)
            corpus_files = list(tmpdir_path.iterdir())
            print(f"[MFA] Corpus directory contents: {[f.name for f in corpus_files]}")
            
            corpus_abs = str(Path(tmpdir).resolve())
            output_abs = str(Path(output_dir).resolve())
            
            print(f"[MFA] Corpus path: {corpus_abs}")
            print(f"[MFA] Output path: {output_abs}")
            
            mfa_tmp_data = os.path.join(MFA_TMP_DIR, 'mfa_data')
            os.makedirs(mfa_tmp_data, exist_ok=True)
            
            cmd = [
                'mfa', 'align',
                '--clean',
                '--single_speaker',
                '--num_jobs', '1',
                # '--temporary_directory', mfa_tmp_data,
                # '--beam', '100',
                # '--retry_beam', '400',
                corpus_abs,
                'english_us_arpa',
                'english_us_arpa',
                output_abs
            ]
            
            print(f"[MFA] Running: {' '.join(cmd)} in env {os.environ.get('CONDA_PREFIX', 'N/A')}")
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 min — allows for ~3 min audio on CPU
                    env=os.environ.copy()
                )
                
                if result.returncode != 0:
                    print(f"[MFA] FAILED with return code {result.returncode}")
                    if result.stdout:
                        print(f"[MFA] STDOUT:\n{result.stdout}")
                    if result.stderr:
                        print(f"[MFA] STDERR:\n{result.stderr}")
                    return None
                    
                print("[MFA] Alignment completed successfully")
                
            except subprocess.TimeoutExpired:
                print("[MFA] Alignment timed out after 120 seconds")
                return None
            except Exception as e:
                print(f"[MFA] Execution error: {e}")
                return None
            
            # Parse TextGrid output
            textgrid_path = output_dir / f"{audio_name}.TextGrid"
            print(f"[MFA] Looking for TextGrid at: {textgrid_path} (exists: {textgrid_path.exists()})")
            
            if textgrid_path.exists():
                print(f"[MFA] Found TextGrid output: {textgrid_path}")
                result = self._parse_textgrid(textgrid_path)
                if result:
                    print(f"[MFA] Successfully parsed TextGrid: {len(result.get('words', []))} words, {len(result.get('phonemes', []))} phonemes")
                else:
                    print("[MFA] Failed to parse TextGrid")
                return result
            else:
                print(f"[MFA] ERROR: No TextGrid output found")
                if output_dir.exists():
                    output_contents = list(output_dir.iterdir())
                    print(f"[MFA] Output dir contents: {[f.name for f in output_contents]}")
                return None
                
        except Exception as e:
            logger.error(f"MFA alignment error: {e}", exc_info=True)
            return None
            
        finally:
            # Clean up temporary directories
            if tmpdir and os.path.exists(tmpdir):
                try:
                    shutil.rmtree(tmpdir)
                    logger.info(f"Cleaned up temp directory: {tmpdir}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temp directory: {e}")
            
            # Also clean up output directory
            if output_dir and Path(output_dir).exists():
                try:
                    shutil.rmtree(output_dir)
                    logger.info(f"Cleaned up output directory: {output_dir}")
                except Exception as e:
                    logger.warning(f"Failed to clean up output directory: {e}")
    
    def _convert_to_wav(self, input_path: str, output_path: str):
        """Convert audio to WAV format using ffmpeg."""
        try:
            print(f"[MFA] Converting audio: {input_path} -> {output_path}")
            result = subprocess.run(
                [
                    'ffmpeg', '-i', input_path,
                    '-ar', '16000',  # 16kHz sample rate
                    '-ac', '1',      # Mono
                    '-y',            # Overwrite
                    output_path
                ],
                capture_output=True,
                check=True,
                timeout=60  # 60s — allows for longer audio files
            )
            print(f"[MFA] Audio conversion successful")
        except subprocess.TimeoutExpired:
            print(f"[MFA] ERROR: FFmpeg conversion timed out")
            logger.error(f"FFmpeg conversion timed out")
            raise
        except subprocess.CalledProcessError as e:
            print(f"[MFA] ERROR: FFmpeg conversion failed: {e.stderr.decode() if e.stderr else str(e)}")
            logger.error(f"FFmpeg conversion failed: {e}")
            raise
        except FileNotFoundError:
            print(f"[MFA] ERROR: ffmpeg not found. Please install: brew install ffmpeg")
            logger.error(f"ffmpeg not found")
            raise
    
    def _parse_textgrid(self, textgrid_path: Path) -> Dict:
        """
        Parse MFA TextGrid output into structured phoneme data.
        
        Returns:
            {
                'words': [{'word': str, 'start': float, 'end': float, 'phonemes': [...]}],
                'phonemes': [{'phone': str, 'start': float, 'end': float, 'confidence': float}]
            }
        """
        try:
            import textgrid  # Requires praat-textgrids package
        except ImportError:
            print("[MFA] ERROR: 'textgrid' package not found. Install with: pip install praat-textgrids")
            logger.error("textgrid package not installed. Install with: pip install praat-textgrids")
            return None
        
        try:
            print(f"[MFA] Parsing TextGrid file: {textgrid_path}")
            tg = textgrid.TextGrid.fromFile(str(textgrid_path))
            
            print(f"[MFA] TextGrid has {len(tg.tiers)} tiers")
            for tier in tg.tiers:
                print(f"[MFA]   Tier: {tier.name} ({len(tier)} intervals)")
            
            words = []
            phonemes = []
            
            # Extract word tier
            for tier in tg.tiers:
                if tier.name == 'words':
                    for interval in tier:
                        if interval.mark.strip():
                            words.append({
                                'word': interval.mark.strip(),
                                'start': interval.minTime,
                                'end': interval.maxTime
                            })
                
                # Extract phone tier
                elif tier.name == 'phones':
                    for interval in tier:
                        if interval.mark.strip():
                            phonemes.append({
                                'phone': interval.mark.strip(),
                                'start': interval.minTime,
                                'end': interval.maxTime,
                                'duration': interval.maxTime - interval.minTime
                            })
            
            print(f"[MFA] Parsed {len(words)} words and {len(phonemes)} phonemes")
            return {'words': words, 'phonemes': phonemes}
            
        except Exception as e:
            print(f"[MFA] ERROR parsing TextGrid: {e}")
            logger.error(f"TextGrid parsing error: {e}", exc_info=True)
            return None
    
    def _analyze_phoneme_deviations(
        self,
        alignment: Dict,
        transcript: str
    ) -> List[Dict]:
        """
        Compare aligned phonemes with canonical pronunciations.
        
        Returns:
            List of deviations with word, expected vs actual phonemes, confidence
        """
        deviations = []
        seen_keys = set()  # (word, start, end) for dedup
        
        for word_data in alignment.get('words', []):
            word = word_data['word'].lower()

            # Skip ASR junk tokens like <unk>
            if is_junk_token(word):
                continue
            
            # Dedup: skip if we already processed this exact (word, start, end)
            dedup_key = (word, round(word_data['start'], 3), round(word_data['end'], 3))
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)
            
            # Get canonical phonemes
            canonical = self.pronunciation_service.get_canonical_phonemes(word)
            
            # Get actual phonemes from alignment
            actual = self._get_phonemes_for_word(
                word_data,
                alignment['phonemes']
            )
            
            if not actual:
                # MFA produced only noise/spn for this word — it could not
                # align the audio to any known phonemes.  This is strong
                # evidence that the word was mispronounced badly enough that
                # MFA gave up, OR that Whisper transcribed it as a non-
                # existent word (OOV).  Flag it.
                if canonical:
                    expected_ipa = arpabet_to_ipa(canonical, wrap_slashes=True)
                    expected_readable = arpabet_to_readable(canonical)
                    deviations.append({
                        'word': word,
                        'expected': ' '.join(canonical),
                        'actual': '(unrecognized)',
                        'expected_ipa': expected_ipa or ' '.join(canonical),
                        'actual_ipa': '(unrecognized)',
                        'expected_readable': expected_readable or ' '.join(canonical),
                        'actual_readable': '(unrecognized)',
                        'similarity': 0.0,
                        'severity': 'notable',
                        'start': word_data['start'],
                        'end': word_data['end']
                    })
                continue
            
            if not canonical:
                # No canonical phonemes (not in CMU dict, G2P also failed).
                # The word itself is likely a Whisper mis-transcription of a
                # real word the speaker tried to say.  Flag it.
                actual_ipa = arpabet_to_ipa(actual, wrap_slashes=True)
                actual_readable = arpabet_to_readable(actual)
                deviations.append({
                    'word': word,
                    'expected': '(unknown word)',
                    'actual': ' '.join(actual),
                    'expected_ipa': '(unknown word)',
                    'actual_ipa': actual_ipa or ' '.join(actual),
                    'expected_readable': '(unknown word)',
                    'actual_readable': actual_readable or ' '.join(actual),
                    'similarity': 0.0,
                    'severity': 'notable',
                    'start': word_data['start'],
                    'end': word_data['end']
                })
                continue
            
            # Compare phoneme sequences
            similarity = self._phoneme_similarity(canonical, actual)
            
            # Apply duration-based penalty: MFA forces dictionary phone
            # labels even when pronunciation differs, producing artificially
            # high similarity.  Abnormally short phone durations reveal
            # that MFA struggled to align the audio → penalize similarity.
            duration_factor = self._duration_penalty(word_data, alignment['phonemes'])
            adjusted_similarity = similarity * duration_factor
            
            # Report only if adjusted similarity below threshold.
            if adjusted_similarity < PRONUNCIATION_DEVIATION_THRESHOLD:
                # Convert to IPA for display
                expected_ipa = arpabet_to_ipa(canonical, wrap_slashes=True)
                actual_ipa = arpabet_to_ipa(actual, wrap_slashes=True)

                # Convert to human-readable guide (e.g. "AW·fuhn")
                expected_readable = arpabet_to_readable(canonical)
                actual_readable = arpabet_to_readable(actual)
                
                deviations.append({
                    'word': word,
                    'expected': ' '.join(canonical),  # ARPABET format
                    'actual': ' '.join(actual),        # ARPABET format
                    'expected_ipa': expected_ipa or ' '.join(canonical),  # IPA format
                    'actual_ipa': actual_ipa or ' '.join(actual),          # IPA format
                    'expected_readable': expected_readable or ' '.join(canonical),
                    'actual_readable': actual_readable or ' '.join(actual),
                    'similarity': round(adjusted_similarity, 3),
                    'severity': self._classify_severity(adjusted_similarity),
                    'start': word_data['start'],
                    'end': word_data['end']
                })
        
        return deviations
    
    # MFA labels that represent noise / silence, NOT real phonemes.
    # When a word's time range only contains these, MFA failed to align it.
    _MFA_NOISE_LABELS = frozenset({'spn', 'sil', 'sp', ''})

    def _duration_penalty(
        self,
        word_data: Dict,
        all_phonemes: List[Dict]
    ) -> float:
        """
        Compute a duration-based penalty factor for a word's alignment.

        MFA forces dictionary phone labels even when pronunciation differs.
        The raw phoneme-similarity will be artificially high.  However, the
        *durations* that MFA assigns to each phone reveal alignment quality:
        phones that don't match the actual audio get squeezed into very short
        durations (<20 ms).

        Returns:
            A factor in [0.0, 1.0] that is multiplied with phoneme similarity.
            1.0 = normal durations (no penalty), lower = poor alignment.
        """
        word_start = word_data['start']
        word_end = word_data['end']

        phones = []
        for phone in all_phonemes:
            if phone['start'] >= word_start and phone['end'] <= word_end:
                label = phone['phone'].strip().lower()
                if label in self._MFA_NOISE_LABELS:
                    continue
                phones.append(phone)

        if not phones or len(phones) < 2:
            return 1.0  # Not enough data to judge

        durations = [p['duration'] for p in phones]
        short_count = sum(1 for d in durations if d < 0.020)
        short_ratio = short_count / len(durations)

        if short_ratio > 0.30:
            return 0.70   # >30% phones very short — strong penalty
        elif short_ratio > 0.15:
            return 0.85   # >15% phones very short — moderate penalty
        elif min(durations) < 0.010:
            return 0.90   # At least one phone extremely short
        return 1.0

    def _get_phonemes_for_word(
        self,
        word_data: Dict,
        all_phonemes: List[Dict]
    ) -> List[str]:
        """Extract phonemes that fall within word's time range.

        Filters out MFA noise labels (``spn``, ``sil``, ``sp``) so that
        words whose time range only contains noise are reported as having
        NO actual phonemes (empty list), which causes the caller to skip
        them instead of flagging a misleading 0.0 similarity deviation.
        """
        word_start = word_data['start']
        word_end = word_data['end']
        
        phonemes = []
        for phone in all_phonemes:
            # Phoneme overlaps with word
            if phone['start'] >= word_start and phone['end'] <= word_end:
                label = phone['phone'].strip().lower()
                # Skip noise / silence labels — they are not real phonemes
                if label in self._MFA_NOISE_LABELS:
                    continue
                phonemes.append(phone['phone'])
        
        return phonemes
    
    def _phoneme_similarity(
        self,
        canonical: List[str],
        actual: List[str]
    ) -> float:
        """
        Calculate phoneme sequence similarity using phoneme-level Levenshtein distance.
        
        Operates on phoneme tokens (not characters) so that each phoneme
        substitution/insertion/deletion counts as exactly 1 edit, regardless
        of whether the ARPABET symbol is 1 or 2 characters long.
        
        Returns:
            Similarity score 0.0-1.0 (1.0 = perfect match)
        """
        # Strip stress digits (e.g. 'AH0' → 'AH') for fairer comparison
        import re
        canon_tokens = [re.sub(r'\d+$', '', p) for p in canonical]
        actual_tokens = [re.sub(r'\d+$', '', p) for p in actual]
        
        max_len = max(len(canon_tokens), len(actual_tokens))
        if max_len == 0:
            return 1.0
        
        # Phoneme-level Levenshtein via dynamic programming
        n, m = len(canon_tokens), len(actual_tokens)
        d = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            d[i][0] = i
        for j in range(m + 1):
            d[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if canon_tokens[i - 1] == actual_tokens[j - 1] else 1
                d[i][j] = min(
                    d[i - 1][j] + 1,       # deletion
                    d[i][j - 1] + 1,       # insertion
                    d[i - 1][j - 1] + cost  # substitution
                )
        
        edit_dist = d[n][m]
        similarity = 1.0 - (edit_dist / max_len)
        
        return max(0.0, similarity)
    
    def _classify_severity(self, similarity: float) -> str:
        """Classify deviation severity based on similarity score."""
        if similarity >= PRONUNCIATION_SEVERITY_MINOR:
            return 'minor'
        elif similarity >= PRONUNCIATION_SEVERITY_MODERATE:
            return 'moderate'
        else:
            return 'notable'
    
    def _calculate_confidence_scores(self, alignment: Dict) -> Dict[str, float]:
        """
        Calculate per-word pronunciation confidence scores.
        
        Based on phoneme durations and acoustic model confidence.
        """
        scores = {}
        
        for word_data in alignment.get('words', []):
            word = word_data['word'].lower()

            # Skip junk tokens like <unk>
            if is_junk_token(word):
                continue

            phonemes = self._get_phonemes_for_word(
                word_data,
                alignment['phonemes']
            )
            
            if phonemes:
                # Heuristic: longer phoneme durations = clearer pronunciation
                avg_duration = sum(
                    p['duration'] for p in alignment['phonemes']
                    if p['phone'] in phonemes
                ) / len(phonemes)
                
                # Normalize to 0-1 scale (0.1s = clear)
                confidence = min(1.0, avg_duration / 0.1)
                scores[word] = round(confidence, 3)
        
        return scores
    
    def _generate_summary(
        self,
        deviations: List[Dict],
        confidence: Dict[str, float]
    ) -> str:
        """Generate human-readable summary of pronunciation analysis."""
        if not deviations:
            return "Your pronunciation shows strong clarity across all analyzed words."
        
        # Group by severity
        notable = [d for d in deviations if d['severity'] == 'notable']
        moderate = [d for d in deviations if d['severity'] == 'moderate']
        
        parts = []
        
        if notable:
            words = ', '.join([d['word'] for d in notable[:3]])
            parts.append(f"Notable deviations detected in: {words}")
        
        if moderate:
            words = ', '.join([d['word'] for d in moderate[:3]])
            parts.append(f"Moderate deviations in: {words}")
        
        # Calculate overall clarity score
        avg_confidence = sum(confidence.values()) / len(confidence) if confidence else 0.5
        clarity_pct = int(avg_confidence * 100)
        
        parts.append(f"Overall speech clarity: {clarity_pct}%")
        
        return ' '.join(parts)
    
    def _fallback_analysis(self, words: List[Dict]) -> Dict:
        """Fallback when MFA unavailable - return basic structure."""
        return {
            'available': False,
            'deviations': [],
            'confidence_scores': {},
            'summary': 'Pronunciation analysis unavailable (MFA not installed)',
            'disclaimer': self._get_ethical_disclaimer(),
            'installation_guide': 'Install MFA: pip install montreal-forced-aligner && mfa model download acoustic english_us_arpa'
        }
    
    def _get_ethical_disclaimer(self) -> str:
        """Return ethical framing disclaimer."""
        return (
            "Pronunciation analysis reports deviations from General American English phonemes "
            "for clarity assessment, not correctness judgments. Dialectal variations, accents, "
            "and non-native speech patterns are linguistically valid. This tool is for "
            "communication effectiveness feedback only."
        )
