"""Transcript improvement service using grammarly/coedit-large.

Generates multiple rewritten versions of a transcript optimised for
clarity, impact, and public-speaking delivery — entirely offline.

Long transcripts are split into sentences and rewritten individually
to stay within the model's effective context window.
"""

import re
import difflib
from typing import List, Dict

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from config import DEVICE, IMPROVED_TRANSCRIPT_MODEL, IMPROVED_TRANSCRIPT_COUNT

# CoEdIT prompt prefixes — each style produces a different edit flavour.
# See https://huggingface.co/grammarly/coedit-large for supported tasks.
_PROMPT_STYLES = [
    "Paraphrase this sentence: {text}",
    "Simplify this sentence: {text}",
    "Rewrite this sentence to be more clear: {text}",
]


class TranscriptImprovementService:
    """Rewrite transcripts for clarity & impact using grammarly/coedit-large."""

    def __init__(self):
        model_name = IMPROVED_TRANSCRIPT_MODEL
        print(f"Loading transcript improvement model ({model_name})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)
        self.model.eval()
        print("Transcript improvement model loaded successfully")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_improved_versions(
        self,
        transcript: str,
        n: int = IMPROVED_TRANSCRIPT_COUNT,
        original_transcript: str | None = None,
    ) -> List[Dict]:
        """Return *n* improved versions of *transcript*.

        Args:
            transcript: The (possibly grammar-corrected) text to rewrite.
            n: Number of versions to generate.
            original_transcript: The raw user transcript.  Annotations are
                diffed against this so every change from the original is
                highlighted.
        """
        if not transcript or len(transcript.split()) < 4:
            return []

        annotation_base = transcript

        # Split into sentences for better model output quality.
        sentences = self._split_sentences(transcript)

        versions: List[Dict] = []
        seen_lower: set = set()

        for i in range(n):
            prompt_tpl = _PROMPT_STYLES[i % len(_PROMPT_STYLES)]
            temp = self._temperature_for_index(i, n)

            rewritten_sents: List[str] = []
            for sent in sentences:
                improved_sent = self._rewrite_sentence(sent, prompt_tpl, temp)
                # Quality gate: reject garbage and fall back to original sentence
                if improved_sent and self._is_acceptable(sent, improved_sent):
                    rewritten_sents.append(improved_sent)
                else:
                    rewritten_sents.append(sent)

            text = " ".join(rewritten_sents)

            normalised = text.lower().strip()
            if normalised in seen_lower or normalised == transcript.lower().strip():
                continue
            seen_lower.add(normalised)

            annotated = self._annotate_changes(annotation_base, text)

            versions.append({
                "improved_transcript": text,
                "improved_annotated": annotated if annotated else text,
            })

            if len(versions) >= n:
                break

        return versions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences, preserving punctuation."""
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        return [p.strip() for p in parts if p.strip()]

    def _rewrite_sentence(
        self,
        sentence: str,
        prompt_template: str,
        temperature: float = 0.7,
    ) -> str | None:
        """Rewrite a single sentence."""
        # Very short fragments — don't bother rewriting
        if len(sentence.split()) < 3:
            return sentence

        prompt = prompt_template.format(text=sentence)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=256,
            truncation=True,
        ).to(DEVICE)

        with torch.no_grad():
            gen_kwargs = dict(
                **inputs,
                max_new_tokens=min(len(sentence.split()) * 3, 256),
                no_repeat_ngram_size=3,
            )
            # Use sampling for diversity across versions; beam search
            # for the first (most faithful) version.
            if temperature > 0.55:
                gen_kwargs.update(
                    do_sample=True,
                    temperature=max(temperature, 0.01),
                    top_p=0.9,
                    top_k=40,
                )
            else:
                gen_kwargs.update(
                    num_beams=4,
                    early_stopping=True,
                )
            outputs = self.model.generate(**gen_kwargs)

        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        if not result or len(result) < 5:
            return None

        # Remove trailing incomplete sentence fragments
        if result and result[-1] not in '.!?':
            last_punc = max(result.rfind('.'), result.rfind('!'), result.rfind('?'))
            if last_punc > len(result) * 0.5:
                result = result[:last_punc + 1]

        return result

    @staticmethod
    def _is_acceptable(original: str, rewritten: str) -> bool:
        """Reject low-quality rewrites."""
        orig_words = original.lower().split()
        rew_words = rewritten.lower().split()

        # Reject if too short (lost more than half the content)
        if len(rew_words) < len(orig_words) * 0.4:
            return False

        # Reject if too long (ballooned to 3x+)
        if len(rew_words) > len(orig_words) * 3:
            return False

        # Reject repetitive output (same word appears 4+ times)
        from collections import Counter
        counts = Counter(rew_words)
        for word, count in counts.items():
            if count >= 4 and word not in ('the', 'a', 'an', 'and', 'to', 'of', 'in', 'is', 'i'):
                return False

        # Reject if it barely shares any words with original (hallucination)
        overlap = set(orig_words) & set(rew_words)
        if len(overlap) < len(orig_words) * 0.3:
            return False

        # Reject if new numbers/digits appear that weren't in the original
        import re as _re
        orig_nums = set(_re.findall(r'\d+', original))
        rew_nums = set(_re.findall(r'\d+', rewritten))
        if rew_nums - orig_nums:
            return False

        # Reject if key content words (nouns, verbs > 3 chars) are dropped.
        # Allow up to 30% loss of content words.
        _stop = {'the', 'a', 'an', 'and', 'or', 'but', 'to', 'of', 'in',
                 'on', 'at', 'is', 'was', 'are', 'were', 'be', 'been',
                 'for', 'with', 'that', 'this', 'it', 'its', 'not', 'no',
                 'so', 'if', 'as', 'by', 'he', 'she', 'we', 'they', 'i',
                 'my', 'his', 'her', 'our', 'your', 'me', 'him', 'them',
                 'who', 'which', 'what', 'how', 'very', 'just', 'than'}
        orig_content = {w.strip('.,!?;:') for w in orig_words
                        if len(w) > 3 and w.strip('.,!?;:') not in _stop}
        rew_set = set(rew_words)
        if orig_content:
            kept = sum(1 for w in orig_content if w in rew_set)
            if kept < len(orig_content) * 0.7:
                return False

        return True

    @staticmethod
    def _temperature_for_index(i: int, n: int) -> float:
        if n <= 1:
            return 0.7
        return round(0.5 + i * (0.5 / (n - 1)), 2)

    @staticmethod
    def _annotate_changes(original: str, improved: str) -> str | None:
        """Wrap each changed/new word individually with <blue> tags.

        Uses word-level difflib alignment.  Each new or replaced word
        gets its own tag so the highlighting is granular, not lumped.
        """
        orig_words = original.split()
        impr_words = improved.split()

        matcher = difflib.SequenceMatcher(
            a=[w.lower() for w in orig_words],
            b=[w.lower() for w in impr_words],
        )

        annotated_parts: List[str] = []
        has_changes = False

        for tag, _, _, j1, j2 in matcher.get_opcodes():
            segment = impr_words[j1:j2]
            if tag == "equal":
                annotated_parts.extend(segment)
            elif tag in ("replace", "insert"):
                has_changes = True
                for word in segment:
                    annotated_parts.append(f"<blue>{word}</blue>")
            # 'delete' — words removed from original; nothing to append

        if not has_changes:
            return improved

        return " ".join(annotated_parts)
