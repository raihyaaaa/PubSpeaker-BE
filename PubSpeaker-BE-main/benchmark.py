#!/usr/bin/env python3
"""PubSpeaker benchmark script.

Measures speed and quality proxies for core models and operations:
- Grammar correction (latency + test-case accuracy)
- Transcript improvement (latency + content-retention proxy)
- Pronunciation lexicon coverage (canonical phoneme lookup)
- Optional transcription benchmark (latency + WER/CER)
- Optional MFA alignment benchmark (latency + optional precision/recall)

Examples:
  conda activate mfa
  python benchmark.py
  python benchmark.py --audio sample.wav --reference-text "your exact transcript"
  python benchmark.py --audio sample.wav --transcript-text "hello world"
  python benchmark.py --audio sample.wav --expected-deviations "hello,world"
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import IMPROVED_TRANSCRIPT_MODEL, WHISPER_MODEL_SIZE


def _percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _latency_summary(latencies: Sequence[float]) -> Dict[str, float]:
    if not latencies:
        return {
            "count": 0,
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
        }
    return {
        "count": len(latencies),
        "mean_ms": round(statistics.mean(latencies) * 1000, 2),
        "median_ms": round(statistics.median(latencies) * 1000, 2),
        "p95_ms": round(_percentile(latencies, 0.95) * 1000, 2),
        "min_ms": round(min(latencies) * 1000, 2),
        "max_ms": round(max(latencies) * 1000, 2),
    }


def _word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,
                    d[i][j - 1] + 1,
                    d[i - 1][j - 1] + 1,
                )

    return d[len(ref_words)][len(hyp_words)] / max(len(ref_words), 1)


def _char_error_rate(reference: str, hypothesis: str) -> float:
    ref = reference.lower().replace(" ", "")
    hyp = hypothesis.lower().replace(" ", "")

    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j

    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,
                    d[i][j - 1] + 1,
                    d[i - 1][j - 1] + 1,
                )

    return d[len(ref)][len(hyp)] / max(len(ref), 1)


def _precision_recall_f1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _memory_mb() -> Optional[float]:
    return None


@dataclass
class Case:
    text: str
    expected_contains: List[str]


GRAMMAR_CASES: List[Case] = [
    Case("he go to school every day", ["goes"]),
    Case("she don't like pizza", ["doesn't"]),
    Case("I has a dog", ["have"]),
    Case("their going to the park", ["they're"]),
    Case("your the best student", ["you're"]),
    Case("I seen the movie yesterday", ["saw", "have seen"]),
]


IMPROVEMENT_TEXTS = [
    "I want to improve my speaking so I can explain ideas clearly in meetings.",
    "The project had delays and confusion so the final presentation was not very strong.",
    "When I talk fast, my message becomes unclear and people ask me to repeat.",
]


PRONUNCIATION_WORDS = [
    "hello", "world", "computer", "tomorrow", "beautiful", "pronunciation",
    "algorithm", "squirrel", "through", "thought", "people", "language",
    "communication", "confidence", "presentation", "covfefe", "supercalifragilistic",
]


def run_grammar_benchmark(repeats: int) -> Dict:
    from services.grammar import GrammarService

    t0 = time.perf_counter()
    svc = GrammarService()
    init_s = time.perf_counter() - t0

    latencies: List[float] = []
    passes = 0
    details = []

    for case in GRAMMAR_CASES:
        best = case.text
        for _ in range(max(repeats, 1)):
            s = time.perf_counter()
            out = svc.generate_corrections(case.text, n=1)
            latencies.append(time.perf_counter() - s)
            best = out[0] if out else case.text

        ok = any(k.lower() in best.lower() for k in case.expected_contains)
        passes += int(ok)
        details.append({
            "input": case.text,
            "output": best,
            "expected_contains": case.expected_contains,
            "passed": ok,
        })

    return {
        "status": "ok",
        "model": "vennify/t5-base-grammar-correction",
        "init_time_s": round(init_s, 3),
        "latency": _latency_summary(latencies),
        "accuracy_proxy": {
            "pass_count": passes,
            "total": len(GRAMMAR_CASES),
            "pass_rate": round(passes / max(len(GRAMMAR_CASES), 1), 4),
        },
        "cases": details,
    }


def run_improvement_benchmark(repeats: int) -> Dict:
    from services.improvement import TranscriptImprovementService

    t0 = time.perf_counter()
    svc = TranscriptImprovementService()
    init_s = time.perf_counter() - t0

    latencies: List[float] = []
    changed = 0
    retention_ratios: List[float] = []
    details = []

    for text in IMPROVEMENT_TEXTS:
        latest = text
        for _ in range(max(repeats, 1)):
            s = time.perf_counter()
            out = svc.generate_improved_versions(text, n=1)
            latencies.append(time.perf_counter() - s)
            latest = out[0]["improved_transcript"] if out else text

        changed_flag = latest.strip().lower() != text.strip().lower()
        changed += int(changed_flag)

        orig_words = set(w.strip(".,!?;:").lower() for w in text.split() if len(w) > 3)
        new_words = set(w.strip(".,!?;:").lower() for w in latest.split())
        retention = len(orig_words & new_words) / max(len(orig_words), 1)
        retention_ratios.append(retention)

        details.append({
            "input": text,
            "output": latest,
            "changed": changed_flag,
            "content_retention": round(retention, 4),
        })

    return {
        "status": "ok",
        "model": IMPROVED_TRANSCRIPT_MODEL,
        "init_time_s": round(init_s, 3),
        "latency": _latency_summary(latencies),
        "quality_proxy": {
            "changed_rate": round(changed / max(len(IMPROVEMENT_TEXTS), 1), 4),
            "avg_content_retention": round(statistics.mean(retention_ratios), 4),
        },
        "cases": details,
    }


def run_pronunciation_coverage_benchmark(repeats: int) -> Dict:
    from services.pronunciation import PronunciationService

    t0 = time.perf_counter()
    svc = PronunciationService()
    init_s = time.perf_counter() - t0

    latencies: List[float] = []
    found = 0
    missing: List[str] = []

    for word in PRONUNCIATION_WORDS:
        ph = None
        for _ in range(max(repeats, 1)):
            s = time.perf_counter()
            ph = svc.get_canonical_phonemes(word)
            latencies.append(time.perf_counter() - s)
        if ph:
            found += 1
        else:
            missing.append(word)

    return {
        "status": "ok",
        "component": "PronunciationService canonical lookup",
        "init_time_s": round(init_s, 3),
        "latency": _latency_summary(latencies),
        "accuracy_proxy": {
            "coverage": round(found / max(len(PRONUNCIATION_WORDS), 1), 4),
            "found": found,
            "total": len(PRONUNCIATION_WORDS),
            "missing_examples": missing[:10],
        },
    }


def run_transcription_benchmark(audio: str, repeats: int, reference_text: Optional[str]) -> Dict:
    from services.transcription import TranscriptionService

    t0 = time.perf_counter()
    svc = TranscriptionService()
    init_s = time.perf_counter() - t0

    latencies: List[float] = []
    last_result = None

    for _ in range(max(repeats, 1)):
        s = time.perf_counter()
        last_result = svc.transcribe(audio)
        latencies.append(time.perf_counter() - s)

    transcript = (last_result or {}).get("text", "")
    words = (last_result or {}).get("words", [])

    metrics = {}
    if reference_text:
        metrics["wer"] = round(_word_error_rate(reference_text, transcript), 4)
        metrics["cer"] = round(_char_error_rate(reference_text, transcript), 4)

    duration_s = 0.0
    if words and words[0].get("start") is not None and words[-1].get("end") is not None:
        duration_s = max(words[-1]["end"] - words[0]["start"], 1e-9)
    words_per_second = (len(words) / duration_s) if duration_s > 0 else 0.0

    return {
        "status": "ok",
        "model": f"openai-whisper/{WHISPER_MODEL_SIZE}",
        "init_time_s": round(init_s, 3),
        "latency": _latency_summary(latencies),
        "output": {
            "transcript_preview": transcript[:180],
            "word_count": len(words),
            "detected_audio_duration_s": round(duration_s, 3),
            "words_per_second": round(words_per_second, 3),
        },
        "accuracy": metrics,
    }


def run_alignment_benchmark(
    audio: str,
    transcript: str,
    words: List[Dict],
    repeats: int,
    expected_deviations: Optional[Sequence[str]],
) -> Dict:
    from services.pronunciation_alignment import PronunciationAlignmentService

    t0 = time.perf_counter()
    svc = PronunciationAlignmentService()
    init_s = time.perf_counter() - t0

    if not svc.mfa_available:
        return {
            "status": "skipped",
            "reason": "MFA is not available/configured in this environment",
            "init_time_s": round(init_s, 3),
        }

    latencies: List[float] = []
    last_result = None

    for _ in range(max(repeats, 1)):
        s = time.perf_counter()
        last_result = svc.analyze_pronunciation(audio, transcript, words)
        latencies.append(time.perf_counter() - s)

    deviations = (last_result or {}).get("deviations", [])
    predicted_words = {d.get("word", "").strip().lower() for d in deviations if d.get("word")}

    accuracy = {}
    if expected_deviations:
        expected = {w.strip().lower() for w in expected_deviations if w.strip()}
        tp = len(predicted_words & expected)
        fp = len(predicted_words - expected)
        fn = len(expected - predicted_words)
        accuracy = {
            "expected_words": sorted(expected),
            "predicted_words": sorted(predicted_words),
            **_precision_recall_f1(tp, fp, fn),
        }

    return {
        "status": "ok",
        "component": "PronunciationAlignmentService",
        "init_time_s": round(init_s, 3),
        "latency": _latency_summary(latencies),
        "output": {
            "deviation_count": len(deviations),
            "predicted_deviation_words": sorted(predicted_words),
        },
        "accuracy": accuracy,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark PubSpeaker models and operations")
    parser.add_argument("--audio", type=str, default=None, help="Path to audio file for transcription/alignment benchmarks")
    parser.add_argument("--reference-text", type=str, default=None, help="Reference text for WER/CER when using --audio")
    parser.add_argument("--transcript-text", type=str, default=None, help="Transcript text for alignment benchmark")
    parser.add_argument("--expected-deviations", type=str, default=None, help="Comma-separated expected mispronounced words for alignment scoring")
    parser.add_argument("--repeats", type=int, default=1, help="Number of repeats for each measured operation")
    parser.add_argument("--skip-grammar", action="store_true")
    parser.add_argument("--skip-improvement", action="store_true")
    parser.add_argument("--skip-pronunciation", action="store_true")
    parser.add_argument("--with-alignment", action="store_true", help="Run MFA alignment benchmark (requires --audio)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    if args.audio and not os.path.isfile(args.audio):
        print(f"[error] Audio file not found: {args.audio}")
        return 1

    started = time.perf_counter()
    results: Dict[str, Dict] = {}

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "models": {
            "whisper": WHISPER_MODEL_SIZE,
            "improvement": IMPROVED_TRANSCRIPT_MODEL,
            "grammar": "vennify/t5-base-grammar-correction",
        },
        "memory_start_mb": _memory_mb(),
    }

    print("\n=== PubSpeaker Benchmark ===")

    if not args.skip_grammar:
        print("\n[run] Grammar benchmark")
        try:
            results["grammar"] = run_grammar_benchmark(args.repeats)
        except Exception as e:
            results["grammar"] = {"status": "error", "error": str(e)}

    if not args.skip_improvement:
        print("\n[run] Transcript improvement benchmark")
        try:
            results["improvement"] = run_improvement_benchmark(args.repeats)
        except Exception as e:
            results["improvement"] = {"status": "error", "error": str(e)}

    if not args.skip_pronunciation:
        print("\n[run] Pronunciation coverage benchmark")
        try:
            results["pronunciation_coverage"] = run_pronunciation_coverage_benchmark(args.repeats)
        except Exception as e:
            results["pronunciation_coverage"] = {"status": "error", "error": str(e)}

    transcribe_result = None
    if args.audio:
        print("\n[run] Transcription benchmark")
        try:
            transcribe_result = run_transcription_benchmark(args.audio, args.repeats, args.reference_text)
            results["transcription"] = transcribe_result
        except Exception as e:
            results["transcription"] = {"status": "error", "error": str(e)}

    if args.with_alignment:
        if not args.audio:
            results["alignment"] = {
                "status": "skipped",
                "reason": "--with-alignment requires --audio",
            }
        else:
            print("\n[run] MFA alignment benchmark")
            transcript = args.transcript_text
            words: List[Dict] = []
            if transcribe_result and transcribe_result.get("status") == "ok":
                # Re-run once for words if needed for alignment input
                from services.transcription import TranscriptionService

                svc = TranscriptionService()
                tr = svc.transcribe(args.audio)
                words = tr.get("words", [])
                if not transcript:
                    transcript = tr.get("text", "")

            if not transcript:
                results["alignment"] = {
                    "status": "skipped",
                    "reason": "No transcript available. Pass --transcript-text or run with transcribable audio.",
                }
            else:
                try:
                    expected = args.expected_deviations.split(",") if args.expected_deviations else None
                    results["alignment"] = run_alignment_benchmark(
                        args.audio,
                        transcript,
                        words,
                        args.repeats,
                        expected,
                    )
                except Exception as e:
                    results["alignment"] = {"status": "error", "error": str(e)}

    total_s = time.perf_counter() - started
    report = {
        "metadata": {
            **metadata,
            "memory_end_mb": _memory_mb(),
            "total_runtime_s": round(total_s, 3),
        },
        "results": results,
    }

    if args.output:
        out_path = args.output
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"benchmark_results_{stamp}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n=== Benchmark Summary ===")
    for name, payload in results.items():
        status = payload.get("status", "unknown")
        print(f"- {name}: {status}")
        if status == "ok" and "latency" in payload:
            p95 = payload["latency"].get("p95_ms")
            mean = payload["latency"].get("mean_ms")
            print(f"  mean={mean}ms p95={p95}ms")
    print(f"\nSaved report: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
