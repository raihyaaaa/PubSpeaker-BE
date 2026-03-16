"""Grammar correction and semantic validation service."""

import re
import difflib
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import List, Tuple, Optional

from config import DEFAULT_SUGGESTION_COUNT

_T5_MODEL_NAME = "vennify/t5-base-grammar-correction"


# ── Verb conjugation: base → 3rd-person singular ─────────────────────
_BASE_TO_CONJ = {
    'love': 'loves', 'make': 'makes', 'say': 'says', 'go': 'goes',
    'walk': 'walks', 'run': 'runs', 'take': 'takes', 'give': 'gives',
    'get': 'gets', 'want': 'wants', 'need': 'needs', 'like': 'likes',
    'come': 'comes', 'see': 'sees', 'know': 'knows', 'think': 'thinks',
    'tell': 'tells', 'work': 'works', 'call': 'calls', 'look': 'looks',
    'play': 'plays', 'move': 'moves', 'live': 'lives', 'hear': 'hears',
    'feel': 'feels', 'eat': 'eats', 'drink': 'drinks', 'sleep': 'sleeps',
    'purr': 'purrs', 'care': 'cares', 'watch': 'watches', 'chase': 'chases',
    'survive': 'survives', 'help': 'helps', 'start': 'starts',
    'keep': 'keeps', 'find': 'finds', 'show': 'shows', 'turn': 'turns',
    'leave': 'leaves', 'seem': 'seems', 'put': 'puts', 'bring': 'brings',
    'hold': 'holds', 'stand': 'stands', 'fall': 'falls', 'die': 'dies',
    'sit': 'sits', 'talk': 'talks', 'pay': 'pays', 'read': 'reads',
    'speak': 'speaks', 'grow': 'grows', 'lead': 'leads', 'begin': 'begins',
    'write': 'writes', 'allow': 'allows', 'create': 'creates',
    'happen': 'happens', 'appear': 'appears', 'include': 'includes',
    'believe': 'believes', 'cause': 'causes', 'digest': 'digests',
    'confuse': 'confuses', 'have': 'has', 'do': 'does',
    'try': 'tries', 'carry': 'carries', 'study': 'studies',
    'fly': 'flies', 'cry': 'cries', 'spend': 'spends', 'send': 'sends',
    'break': 'breaks', 'drive': 'drives', 'sing': 'sings',
    'teach': 'teaches', 'reach': 'reaches', 'catch': 'catches',
    'touch': 'touches', 'wash': 'washes', 'wish': 'wishes',
    'push': 'pushes', 'pass': 'passes', 'miss': 'misses',
    'mean': 'means', 'add': 'adds', 'change': 'changes',
    'stop': 'stops', 'open': 'opens', 'close': 'closes',
    'ask': 'asks', 'act': 'acts', 'end': 'ends',
    'pick': 'picks', 'drop': 'drops', 'kill': 'kills',
    'cut': 'cuts', 'bite': 'bites', 'hide': 'hides',
    'climb': 'climbs', 'jump': 'jumps', 'swim': 'swims',
    'bark': 'barks', 'scratch': 'scratches', 'lick': 'licks',
    'win': 'wins', 'lose': 'loses', 'build': 'builds',
    'buy': 'buys', 'sell': 'sells', 'draw': 'draws',
    'pull': 'pulls', 'raise': 'raises', 'set': 'sets',
    'wear': 'wears', 'cost': 'costs', 'hit': 'hits',
    'hurt': 'hurts', 'hang': 'hangs', 'feed': 'feeds',
}
_CONJ_TO_BASE = {v: k for k, v in _BASE_TO_CONJ.items()}

_PLURAL_SUBJECTS = frozenset({
    'people', 'they', 'we', 'you', 'cats', 'dogs', 'children',
    'men', 'women', 'students', 'animals', 'things', 'others',
    'parents', 'friends', 'kids', 'boys', 'girls', 'players',
    'workers', 'teachers', 'members', 'users', 'owners',
})

_SINGULAR_SUBJECTS = frozenset({
    'it', 'he', 'she', 'everyone', 'everybody', 'nobody',
    'somebody', 'someone', 'everything', 'nothing', 'something',
    'anyone', 'anything',
})

_CAUSATIVE_WORDS = frozenset({
    'let', 'make', 'help', 'watch', 'see', 'hear', 'feel',
    'would', 'could', 'should', 'might', 'to', 'will', 'can', 'may',
    'shall', 'must', 'did', 'does', 'do', "won't", "wouldn't",
    "couldn't", "shouldn't", "can't",
})

_INTERPOSING_ADVERBS = frozenset({
    'usually', 'always', 'often', 'sometimes', 'never', 'also', 'really',
    'just', 'still', 'even', 'actually', 'generally', 'normally', 'already',
    'only', 'probably', 'certainly', 'definitely', 'typically', 'rarely',
    'seldom', 'frequently', 'merely', 'simply', 'basically', 'hardly',
})

_COUNTABLE_SINGULARS = frozenset({
    'cat', 'dog', 'car', 'house', 'tree', 'book', 'word', 'thing',
    'place', 'time', 'day', 'week', 'month', 'year', 'hour', 'minute',
    'second', 'person', 'animal', 'bird', 'table', 'chair',
    'room', 'door', 'window', 'box', 'bag', 'bottle', 'cup',
    'plate', 'picture', 'song', 'game', 'toy', 'problem', 'question',
    'answer', 'idea', 'story', 'example', 'point', 'part', 'accident',
    'life', 'reason', 'way', 'type', 'kind', 'language', 'country',
    'city', 'school', 'student', 'teacher', 'friend', 'parent',
    'boy', 'girl', 'eye', 'ear', 'leg', 'arm', 'hand',
    'apple', 'orange', 'banana', 'mistake', 'change', 'difference',
    'step', 'piece', 'glass', 'side', 'page', 'line', 'level',
    'foot', 'child', 'man', 'woman', 'mouse', 'goose', 'tooth',
    'flower', 'river', 'mountain', 'island', 'star',
    'color', 'letter', 'number', 'group', 'team', 'family',
    'class', 'test', 'lesson', 'event', 'movie', 'show',
    'store', 'shop', 'restaurant', 'market', 'building',
    'road', 'street', 'park', 'garden',
    'phone', 'computer', 'screen', 'camera', 'button',
    'name', 'age', 'mile', 'inch', 'pound', 'dollar',
})

_IRREGULAR_PLURALS = {
    'person': 'people', 'child': 'children', 'man': 'men', 'woman': 'women',
    'mouse': 'mice', 'goose': 'geese', 'tooth': 'teeth', 'foot': 'feet',
    'life': 'lives', 'knife': 'knives', 'wife': 'wives', 'leaf': 'leaves',
}

_UNCOUNTABLE_NOUNS = frozenset({
    'food', 'milk', 'furniture', 'information', 'advice',
    'equipment', 'homework', 'knowledge', 'bread', 'rice',
    'luggage', 'traffic', 'music', 'hair', 'money',
    'water', 'air', 'sugar', 'salt', 'butter', 'cheese',
})

_COMMON_ADJECTIVES = frozenset({
    'happy', 'sad', 'angry', 'scared', 'sick', 'tired', 'hungry', 'thirsty',
    'cold', 'hot', 'cute', 'soft', 'hard', 'big', 'small', 'tall', 'short',
    'old', 'young', 'new', 'good', 'bad', 'nice', 'fine', 'ready', 'full',
    'empty', 'dirty', 'quiet', 'loud', 'fast', 'slow', 'easy',
    'lazy', 'busy', 'wrong', 'right', 'normal',
    'sure', 'safe', 'afraid', 'alone', 'alive', 'awake', 'asleep',
    'sorry', 'glad', 'proud', 'calm', 'weak', 'strong', 'wet', 'dry',
    'important', 'different', 'beautiful', 'wonderful', 'terrible',
    'amazing', 'awesome', 'great', 'perfect', 'horrible', 'pretty',
    'careful', 'careless', 'helpful', 'useful', 'polite', 'rude',
    'lucky', 'unlucky', 'nervous', 'anxious', 'confident',
    'comfortable', 'uncomfortable', 'familiar', 'serious',
    'honest', 'selfish', 'generous', 'patient', 'impatient',
    'responsible', 'gentle', 'kind', 'mean', 'strict', 'fair',
    'clean', 'messy', 'neat', 'loose', 'tight', 'rough', 'smooth',
    'thick', 'thin', 'heavy', 'light', 'dark', 'bright', 'clear',
    'rich', 'poor', 'cheap', 'expensive', 'free', 'fresh', 'rotten',
    'delicious', 'suspicious', 'ridiculous', 'gorgeous', 'curious',
    'dangerous', 'famous', 'jealous', 'obvious', 'precious',
})


def _pluralize_noun(word: str) -> str:
    """Pluralize a singular English noun, handling irregulars."""
    low = word.lower()
    if low in _IRREGULAR_PLURALS:
        p = _IRREGULAR_PLURALS[low]
        return p.capitalize() if word[0].isupper() else p
    if low.endswith(('sh', 'ch', 'ss', 'x', 'z')):
        return word + 'es'
    if low.endswith('y') and len(low) > 1 and low[-2] not in 'aeiou':
        return word[:-1] + 'ies'
    return word + 's'


class GrammarService:
    """
    Service for grammar correction using a model-based approach:
    1. Regex patterns for common ESL errors
    2. Semantic validation for nonsensical sentences
    3. T5 neural grammar correction (vennify/t5-base-grammar-correction)

    Errors are identified by diffing the original transcript against the
    corrected version — no external rule engine required.
    """

    def __init__(self):
        """Initialize T5 grammar correction model."""
        # Load T5 grammar correction model
        print(f"Loading T5 grammar model ({_T5_MODEL_NAME})...")
        try:
            self._t5_tokenizer = T5Tokenizer.from_pretrained(_T5_MODEL_NAME)
            self._t5_model = T5ForConditionalGeneration.from_pretrained(_T5_MODEL_NAME)
            self._t5_model.eval()
            print("T5 grammar model loaded.")
        except Exception as e:
            print(f"T5 grammar model error: {e}")
            self._t5_tokenizer = None
            self._t5_model = None
    
    def generate_corrections(
        self, 
        transcript: str, 
        n: int = DEFAULT_SUGGESTION_COUNT
    ) -> List[str]:
        """
        Generate grammar corrections using a **chained** model-based approach.

        Pipeline:
            transcript → regex → semantic → T5 → output

        Args:
            transcript: Original transcript text
            n: Number of suggestions to generate
            
        Returns:
            List of corrected text suggestions (best first)
        """
        # Layer 1: targeted safe regex corrections for common ESL patterns
        step1 = self._apply_regex_corrections(transcript)

        # Layer 2: semantic corrections
        semantic_issues = self._validate_semantics(step1)
        step2 = step1
        if semantic_issues:
            sem_result = self._apply_semantic_corrections(step1, semantic_issues)
            if sem_result and sem_result != step1:
                step2 = sem_result

        # Layer 3: T5 neural grammar correction (main correction engine)
        step3 = self._apply_t5_corrections(step2)

        # Layer 4: Post-T5 cleanup for errors T5 typically misses
        step3 = self._post_t5_cleanup(step3)

        # ── Post-processing: revert changes to ASR artifacts ────
        # Whisper sometimes produces hyphenated gibberish (e.g.
        # "early-year-dence") that the pipeline may silently alter to
        # an equally wrong form (e.g. "early-year-dance").  We restore
        # the original tokens so the corrected text doesn't pretend to
        # fix something it cannot.
        step3 = self._revert_asr_artifacts(transcript, step3)

        # Collect unique corrections that differ from the original
        candidates = []
        for version in [step3, step2, step1]:
            if version.lower().strip() != transcript.lower().strip():
                if version not in candidates:
                    candidates.append(version)

        return candidates[:n] if candidates else []

    def generate_user_suggestions(
        self,
        transcript: str,
        corrections: Optional[List[str]] = None,
        n: int = DEFAULT_SUGGESTION_COUNT,
    ) -> List[str]:
        """Generate user-facing suggestions even when grammar is correct.

        This is intentionally lightweight (no LLM): it combines any available
        corrected versions plus a few actionable speaking/writing tips based on
        transcript heuristics (fillers, repetition, very long sentences).

        Instead of showing the entire corrected paragraph (which is unhelpful),
        we extract sentence-level corrections: "X → Y" for each changed sentence.
        """
        suggestions: List[str] = []

        # Corrections are now returned as corrected_transcript / corrected_annotated
        # in the API response, so we only generate tips here.

        text = transcript or ""
        lowered = text.lower()

        # Filler words
        filler_words = ["um", "uh", "erm", "like", "you know"]
        filler_count = sum(lowered.split().count(w) for w in filler_words)
        if filler_count >= 2:
            suggestions.append("Tip: Reduce filler words (e.g., 'um', 'uh', 'like') by pausing silently instead.")

        # Repeated words (simple heuristic)
        tokens = [t for t in re.findall(r"[A-Za-z']+", lowered) if t]
        repeats = 0
        for i in range(1, len(tokens)):
            if tokens[i] == tokens[i - 1]:
                repeats += 1
        if repeats >= 2:
            suggestions.append("Tip: Avoid repeating the same word back-to-back; replace the second one or pause.")

        # Very long sentences
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        if sentences:
            avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
            if avg_len >= 20:
                suggestions.append("Tip: Break long sentences into 2 shorter ones to sound clearer and more confident.")

        # If we still have nothing, provide a gentle, concrete default
        if not suggestions:
            suggestions.append("Tip: Add a short pause after each key point to improve clarity.")

        # Deduplicate and cap
        return self._deduplicate_suggestions(suggestions)[:n]

    def _extract_sentence_corrections(
        self,
        original: str,
        corrected: str,
    ) -> List[str]:
        """Compare *original* and *corrected* sentence-by-sentence.

        Returns a list of human-readable correction strings like:
            ``'"resides not in grand" → "resides not in grand"'``

        Only sentences that actually changed are included.
        """
        # Split both texts into sentences
        orig_sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', original) if s.strip()]
        corr_sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', corrected) if s.strip()]

        # If no sentence boundaries, treat each as a single "sentence"
        if not orig_sents:
            orig_sents = [original]
        if not corr_sents:
            corr_sents = [corrected]

        corrections: List[str] = []
        matcher = difflib.SequenceMatcher(
            a=[s.lower() for s in orig_sents],
            b=[s.lower() for s in corr_sents],
        )
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            elif tag == "replace":
                for k in range(min(i2 - i1, j2 - j1)):
                    orig_s = orig_sents[i1 + k]
                    corr_s = corr_sents[j1 + k]
                    if orig_s.lower().strip() == corr_s.lower().strip():
                        continue
                    # Skip stylistic rephrasings — only keep genuine grammar
                    # fixes.  With T5 corrections the overlap can be lower
                    # (e.g. word-order fix "me and my friend go" → "my friend
                    # and I went") but they're still valid corrections, so
                    # we use a generous 0.5 threshold.
                    sim = difflib.SequenceMatcher(
                        a=orig_s.lower().split(),
                        b=corr_s.lower().split(),
                    ).ratio()
                    if sim < 0.5:
                        # Too different → a rephrase, not a correction
                        continue
                    corrections.append(f'"{orig_s}" → "{corr_s}"')
            elif tag == "delete":
                for k in range(i1, i2):
                    corrections.append(f'Remove: "{orig_sents[k]}"')
            elif tag == "insert":
                for k in range(j1, j2):
                    corrections.append(f'Add: "{corr_sents[k]}"')

        return corrections
    
    def _apply_regex_corrections(self, text: str) -> str:
        """Apply safe, targeted regex corrections for common ESL errors
        that LanguageTool frequently misses.

        Each pattern is carefully scoped with word boundaries to avoid
        false positives.  Only patterns that are unambiguously wrong in
        standard English are included.  T5 handles the rest.
        """
        corrected = text

        # ── Capitalize day-of-week & month names ────────────────
        # "last saturday" → "last Saturday", "on monday" → "on Monday"
        _days = r'(monday|tuesday|wednesday|thursday|friday|saturday|sunday)'
        _months = (r'(january|february|march|april|may|june|july'
                   r'|august|september|october|november|december)')
        corrected = re.sub(
            _days,
            lambda m: m.group(1).capitalize(),
            corrected,
        )
        corrected = re.sub(
            r'(?<![A-Za-z])' + _months + r'(?![A-Za-z])',
            lambda m: m.group(1).capitalize(),
            corrected,
        )

        # ── Irregular past tenses ────────────────────────────────
        # "I seen" / "and seen" / "we seen" → "... saw"
        # (does NOT touch "have seen" / "has seen" / "had seen")
        corrected = re.sub(
            r'\b(?:(?:[Ii]|[Ww]e|[Tt]hey|[Hh]e|[Ss]he)\s+seen'
            r'|and\s+seen)\b',
            lambda m: m.group(0).rsplit('seen', 1)[0] + 'saw',
            corrected,
        )

        # "Yesterday I go" / "Last week I go" → "... went"
        # Past-time marker + subject + base "go" (not "goes"/"going"/"gone")
        corrected = re.sub(
            r'(?i)\b(yesterday|last\s+(?:week|month|year|night|time)|earlier'
            r'|ago|the\s+other\s+day)\b'
            r'(.{0,30}?)'
            r'\b(I|we|they|he|she|it)\s+go\b',
            lambda m: m.group(1) + m.group(2) + m.group(3) + ' went',
            corrected,
        )

        # "Yesterday I buy" / "Last week she buy" → "... bought"
        # Requires subject pronoun before "buy" to avoid breaking
        # "for buy" (which should become "to buy", not "for bought").
        corrected = re.sub(
            r'(?i)\b(yesterday|last\s+(?:week|month|year|night|time)|earlier'
            r'|ago|the\s+other\s+day)\b'
            r'(.{0,40}?)'
            r'\b(I|we|they|he|she|it)\s+buy\b',
            lambda m: m.group(1) + m.group(2) + m.group(3) + ' bought',
            corrected,
        )

        # "buyed" → "bought" (common ESL over-regularisation)
        corrected = re.sub(r'\b[Bb]uyed\b',
                           lambda m: 'Bought' if m.group(0)[0] == 'B' else 'bought',
                           corrected)

        # ── Plural nouns after "some/many/few/several" ───────────
        # "some cat" → "some cats" (whitelist approach to avoid false positives)
        _count_quantifiers = r'(?:some|many|few|several|multiple|various|numerous|two|three|four|five|six|seven|eight|nine|ten)'
        corrected = re.sub(
            r'(?i)\b(' + _count_quantifiers + r')\s+([a-z]+)\b',
            lambda m: m.group(1) + ' ' + (
                _pluralize_noun(m.group(2))
                if m.group(2).lower() in _COUNTABLE_SINGULARS
                and not m.group(2).lower().endswith('s')
                else m.group(2)
            ),
            corrected,
        )

        # ── Plural subject + "was" → "were" ─────────────────────
        # "children was" / "childs was" / "people was" / "they was" → "... were"
        corrected = re.sub(
            r'\b(children|childs|people|men|women|they|we)\s+was\b',
            lambda m: m.group(1) + ' were',
            corrected,
            flags=re.IGNORECASE,
        )

        # "him and me decided" → "he and I decided"
        corrected = re.sub(
            r'\b[Hh]im\s+and\s+me\b',
            lambda m: ('Him' if m.group(0)[0] == 'H' else 'him').replace('him', 'he') + ' and I',
            corrected,
        )
        # "Me and him" → "He and I" (at sentence start or after period)
        corrected = re.sub(
            r'(?:^|(?<=[.!?]\s))[Mm]e\s+and\s+him\b',
            'He and I',
            corrected,
        )

        # ── Object pronoun "Me/me" used as subject → "I" ────────
        # "Me like banana" → "I like banana"
        # Matches at sentence start, after punctuation, or after conjunctions.
        _me_subj_verbs = (
            r'like|want|need|eat|go|have|think|know|see|hear|feel'
            r'|love|hate|wish|hope|try|ask|tell|say|make|take|give'
            r'|get|buy|sell|do|did|was|am|can|could|will|would|should|must'
        )
        corrected = re.sub(
            r'((?:^|[.!?,;:]\s*)\s*|'
            r'(?:and|but|so|because|or|then|if|when|while|since|although)\s+)'
            r'[Mm]e\s+(' + _me_subj_verbs + r')\b',
            lambda m: m.group(1) + 'I ' + m.group(2),
            corrected,
            flags=re.MULTILINE,
        )

        # ── Article corrections ──────────────────────────────────
        # "a orange/apple/elephant/..." → "an orange/apple/..."
        corrected = re.sub(
            r'\b([Aa])\s+(?=[aeiouAEIOU]\w)',
            lambda m: ('An ' if m.group(1)[0] == 'A' else 'an '),
            corrected,
        )

        # ── "much" vs "many" with countable nouns ────────────────
        # "much people/things/words" → "many people/things/words"
        corrected = re.sub(
            r'\b([Mm])uch\s+(people|things|words|times|books|cats|dogs|cars|children|students|places)\b',
            lambda m: ('M' if m.group(1) == 'M' else 'm') + 'any ' + m.group(2),
            corrected,
        )

        # ── "There was" + plural noun → "There were" ────────────
        corrected = re.sub(
            r'\b([Tt]here)\s+was\s+(many|much|several|numerous|lots|a\s+lot\s+of)\b',
            lambda m: m.group(1) + ' were ' + m.group(2),
            corrected,
        )
        corrected = re.sub(
            r'\b([Tt]here)\s+was\s+(\w+s)\s+(walking|running|sitting|standing|waiting|playing|making|doing|going|coming)\b',
            lambda m: m.group(1) + ' were ' + m.group(2) + ' ' + m.group(3),
            corrected,
        )

        # ── Double negatives ─────────────────────────────────────
        # "didn't want no" → "didn't want any"
        corrected = re.sub(
            r"\bdidn'?t\s+want\s+no\b",
            "didn't want any",
            corrected,
            flags=re.IGNORECASE,
        )
        # "didn't ... from nobody" → "didn't ... from anybody"
        corrected = re.sub(
            r"\bfrom\s+nobody\b",
            "from anybody",
            corrected,
        )

        # ── "quick" (adjective) used as adverb → "quickly" ──────
        # "run away quick" → "run away quickly"
        corrected = re.sub(
            r'\b(ran?|run|walk(?:ed)?|mov(?:ed?|ing)|drove|left|went)\s+(away\s+)?quick\b',
            lambda m: m.group(1) + ' ' + (m.group(2) or '') + 'quickly',
            corrected,
        )

        # ── Action verb + "good" (adjective) → "well" (adverb) ──
        # "speaking good" → "speaking well", "play good" → "play well"
        # NOT linking verbs: "look good", "feel good", "taste good" are correct.
        _well_verbs = (
            r'speak|speaking|speaks|spoke|'
            r'sing|singing|sings|sang|'
            r'play|playing|plays|played|'
            r'perform|performing|performs|performed|'
            r'work|working|works|worked|'
            r'communicate|communicating|communicates|communicated|'
            r'do|doing|does|did|done|'
            r'behave|behaving|behaves|behaved|'
            r'sleep|sleeping|sleeps|slept|'
            r'write|writing|writes|wrote|read|reading|reads'
        )
        corrected = re.sub(
            r'\b(' + _well_verbs + r')\s+good\b',
            lambda m: m.group(1) + ' well',
            corrected, flags=re.IGNORECASE,
        )

        # ── Adjective as adverb after verb/gerund → -ly ──────────
        # "speaking regular" → "speaking regularly"
        # Only common adj→adv pairs; excludes linking verbs.
        _adj_to_adv = {
            'regular': 'regularly', 'careful': 'carefully',
            'proper': 'properly', 'frequent': 'frequently',
            'occasional': 'occasionally', 'constant': 'constantly',
            'fluent': 'fluently', 'confident': 'confidently',
            'consistent': 'consistently', 'independent': 'independently',
            'comfortable': 'comfortably', 'effective': 'effectively',
            'accurate': 'accurately', 'professional': 'professionally',
            'silent': 'silently',
        }
        _adj_adv_pat = '|'.join(sorted(_adj_to_adv.keys(), key=len, reverse=True))
        _linking_gerunds = frozenset({
            'looking', 'feeling', 'tasting', 'smelling', 'sounding',
            'seeming', 'appearing', 'becoming', 'being', 'remaining',
        })
        corrected = re.sub(
            r'\b(\w+ing)\s+(' + _adj_adv_pat + r')\b',
            lambda m: (
                m.group(1) + ' ' + _adj_to_adv.get(m.group(2).lower(), m.group(2))
                if m.group(1).lower() not in _linking_gerunds
                else m.group(0)
            ),
            corrected, flags=re.IGNORECASE,
        )

        # ── "seating" (to seat someone) vs "sitting" ─────────────
        # "seating on a bench" → "sitting on a bench"
        corrected = re.sub(
            r'\bseating\s+(on|in|at|down)\b',
            lambda m: 'sitting ' + m.group(1),
            corrected,
        )

        # ── "cut" used as "cat" (common ASR error) ───────────────
        # Only when article precedes: "a cut seating" → "a cat sitting"
        corrected = re.sub(
            r'\b(a|an|the)\s+cut\s+(sit|seat|was|is|ran|run|walk)',
            lambda m: m.group(1) + ' cat ' + m.group(2),
            corrected,
        )

        # ── "for" + base verb → "to" + base verb ────────────────
        # Very common ESL error: "for buy" → "to buy", "for eat" → "to eat"
        _common_base_verbs = (
            r'buy|sell|eat|drink|find|get|see|watch|make|take|give|help'
            r'|read|write|learn|study|play|work|cook|clean|wash|fix'
            r'|bring|carry|send|meet|visit|ask|tell|show|build|drive'
        )
        corrected = re.sub(
            r'\bfor\s+(' + _common_base_verbs + r')\b',
            r'to \1',
            corrected,
            flags=re.IGNORECASE,
        )

        # ── "like/want/need + bare verb" → "… + to + verb" ───────
        # Very common ESL error: "like eat" → "like to eat"
        _inf_verbs = (
            r'like|want|need|love|hate|prefer|wish|hope|try'
            r'|plan|decide|agree|offer|refuse|learn|promise|used'
        )
        _base_verbs2 = (
            r'eat|go|come|see|get|make|take|buy|sell|drink|play|work'
            r'|cook|clean|wash|bring|run|walk|swim|sleep|talk|read'
            r'|write|sit|stand|drive|fly|sing|dance|fight|wait|shop'
            r'|watch|find|give|tell|ask|send|help|do|have|be'
        )
        corrected = re.sub(
            r'\b(' + _inf_verbs + r')\s+(' + _base_verbs2 + r')\b',
            r'\1 to \2',
            corrected,
            flags=re.IGNORECASE,
        )

        # ── preposition + base verb → gerund ─────────────────────
        # "after eat" → "after eating", "before go" → "before going"
        # "without bring" → "without bringing"
        _gerund_preps = r'(?:after|before|without|by|while|instead\s+of)'
        corrected = re.sub(
            r'\b(' + _gerund_preps + r')\s+(eat|go|come|see|get|make|take|buy|sell'
            r'|drink|play|work|cook|clean|wash|bring|run|walk|swim|sleep|talk'
            r'|read|write|sit|stand|drive|fly|sing|dance|fight|wait|shop|watch)\b',
            lambda m: m.group(1) + ' ' + (
                m.group(2) + 'ing' if not m.group(2).endswith('e')
                else m.group(2)[:-1] + 'ing'
            ),
            corrected,
            flags=re.IGNORECASE,
        )

        # ── modal + base adjective (missing "be") ────────────────
        # "I will more careful" → "I will be more careful"
        # "she can happy" → "she can be happy"
        _modals = r'(?:will|would|can|could|should|shall|must|might|may)'
        _adj_list = '|'.join(sorted(_COMMON_ADJECTIVES, key=len, reverse=True))
        corrected = re.sub(
            r'\b(' + _modals + r')\s+(more\s+)?(' + _adj_list + r')\b',
            lambda m: m.group(1) + ' be ' + (m.group(2) or '') + m.group(3),
            corrected,
            flags=re.IGNORECASE,
        )

        # ── noun/pronoun + adjective (missing copula "was/is") ───
        # "My mom angry" → "My mom was angry"
        # "clothes all dirty" → "clothes were all dirty"
        _possessive = r'(?:[Mm]y|[Hh]is|[Hh]er|[Oo]ur|[Tt]heir|[Ii]ts)'
        corrected = re.sub(
            r'\b(' + _possessive + r'\s+\w+)\s+(angry|happy|sad|dirty|clean'
            r'|sick|tired|hungry|scared|afraid|sorry|glad|proud|upset|busy'
            r'|ready|worried|confused|surprised|excited|bored|annoyed)\b',
            lambda m: m.group(1) + ' was ' + m.group(2),
            corrected,
        )

        # ── -ed/-ing participial adjective confusion ─────────────
        # "We was very exciting" → "We were very excited"
        # "I am boring" → "I am bored" (when subject is a person)
        _person_subjects = r'(?:I|we|they|he|she|you|people|everyone|everybody|my\s+\w+)'
        corrected = re.sub(
            r'\b(' + _person_subjects + r')\s+'
            r'(?:was|were|am|is|are|felt|feel|got|get|become|became|seem|seemed)\s+'
            r'(?:very\s+|really\s+|so\s+|quite\s+)?'
            r'(exciting|boring|confusing|surprising|amusing|annoying|disappointing'
            r'|embarrassing|exhausting|frightening|interesting|overwhelming'
            r'|relaxing|satisfying|shocking|tiring|worrying|thrilling)\b',
            lambda m: m.group(0)[:-3] + 'ed',
            corrected,
            flags=re.IGNORECASE,
        )

        # ── "forget" + base verb (missing "to") ──────────────────
        # "I forget bring" → "I forget to bring"
        corrected = re.sub(
            r'\b(forget|forgot|remember|remembered)\s+'
            r'(bring|take|buy|get|make|eat|drink|go|come|see|tell|ask|send'
            r'|give|pay|call|close|open|lock|turn|check|do|put|clean|wash)\b',
            lambda m: m.group(1) + ' to ' + m.group(2),
            corrected,
            flags=re.IGNORECASE,
        )

        # ── "complete" used as adverb → "completely" ──────────────
        # "I get complete wet" → "I get completely wet"
        corrected = re.sub(
            r'\b(complete)\s+(' + _adj_list + r')\b',
            r'completely \2',
            corrected,
            flags=re.IGNORECASE,
        )

        # ══ COMPREHENSIVE ESL GRAMMAR RULES ══════════════════════

        # ── Irregular plural corrections ─────────────────────────
        _irreg = [
            (r'\b[Mm]ouses\b', 'mice'), (r'\b[Cc]hilds\b', 'children'),
            (r'\b[Pp]eoples\b(?!\s+of\b)', 'people'),
            (r'\b[Tt]ooths\b', 'teeth'), (r'\b[Ff]oots\b', 'feet'),
            (r'\b[Gg]ooses\b', 'geese'), (r'\b[Ww]omans\b', 'women'),
        ]
        for pat, repl in _irreg:
            corrected = re.sub(
                pat,
                lambda m, r=repl: r.capitalize() if m.group(0)[0].isupper() else r,
                corrected,
            )

        # ── Uncountable nouns: strip wrong plural -s ─────────────
        for w in _UNCOUNTABLE_NOUNS:
            corrected = re.sub(
                r'\b(' + re.escape(w) + r')s\b', r'\1',
                corrected, flags=re.IGNORECASE,
            )

        # ── Pronoun–be verb agreement ────────────────────────────
        # "it are" → "it is", "they is" → "they are"
        corrected = re.sub(
            r'\b(it|he|she)\s+are\b',
            lambda m: m.group(1) + ' is', corrected, flags=re.IGNORECASE,
        )
        corrected = re.sub(
            r'\b(they|we|people)\s+is\b',
            lambda m: m.group(1) + ' are', corrected, flags=re.IGNORECASE,
        )

        # ── Stative verbs in continuous → simple form ────────────
        # "people are believing" → "people believe"
        # "he is knowing" → "he knows"
        # Stative verbs express states, not actions — they should NOT
        # take the progressive/continuous form in standard English.
        _stative_ing = {
            'believing': 'believe', 'knowing': 'know',
            'wanting': 'want', 'needing': 'need',
            'liking': 'like', 'loving': 'love', 'hating': 'hate',
            'understanding': 'understand', 'remembering': 'remember',
            'belonging': 'belong', 'owning': 'own', 'meaning': 'mean',
            'deserving': 'deserve', 'doubting': 'doubt',
            'preferring': 'prefer', 'recognizing': 'recognize',
        }
        _stative_pat = '|'.join(_stative_ing.keys())

        def _fix_stative(m):
            subj = m.group(1)
            verb_ing = m.group(3).lower()
            base = _stative_ing.get(verb_ing, verb_ing)
            if subj.lower() in ('he', 'she', 'it'):
                conj = _BASE_TO_CONJ.get(base, base + 's')
                return subj + ' ' + conj
            return subj + ' ' + base

        corrected = re.sub(
            r'\b(I|he|she|it|we|you|they|people|everyone|everybody)'
            r'\s+(is|are|am)\s+(' + _stative_pat + r')\b',
            _fix_stative,
            corrected, flags=re.IGNORECASE,
        )

        # ── have → has (singular subjects) ───────────────────────
        corrected = re.sub(
            r'\b(it|he|she)\s+have\b',
            lambda m: m.group(1) + ' has', corrected, flags=re.IGNORECASE,
        )
        # "a/an/this/that cat (usually) have" → "has"
        _advs = (r'(?:(?:usually|always|often|sometimes|never|also|really'
                 r'|just|still|even|actually)\s+)*')
        corrected = re.sub(
            r'\b(a|an|this|that|each|every)\s+(\w+)\s+' + _advs + r'have\b',
            lambda m: m.group(0)[:-4] + 'has',
            corrected, flags=re.IGNORECASE,
        )
        # Broader: singular countable noun + have → has
        corrected = re.sub(
            r'\b([a-z]+)\s+have\b',
            lambda m: (m.group(1) + ' has'
                       if m.group(1).lower() in _COUNTABLE_SINGULARS
                       and not m.group(1).lower().endswith('s')
                       else m.group(0)),
            corrected, flags=re.IGNORECASE,
        )

        # ── Adjective used as adverb: "very good" → "very well" ──
        corrected = re.sub(
            r'\bvery good\b(?=\s*[.,;:!?]|\s+(?:and|but|because|so|which|when)\b|\s*$)',
            'very well', corrected,
        )

        # ── "not literal true" → "not literally true" ────────────
        corrected = re.sub(
            r'\b(not)\s+(literal|actual|real|exact|basic|general|near'
            r'|sure|mere|usual|natural|normal|practical|regular'
            r'|rough|simple|typical|virtual)\s+'
            r'(true|false|correct|right|wrong|same|different'
            r'|impossible|possible|sure|certain)\b',
            lambda m: m.group(1) + ' ' + m.group(2) + 'ly ' + m.group(3),
            corrected, flags=re.IGNORECASE,
        )

        # ── Number + singular countable → plural ─────────────────
        corrected = re.sub(
            r'\b(\d+)\s+([a-z]+)\b',
            lambda m: m.group(1) + ' ' + (
                _pluralize_noun(m.group(2))
                if m.group(2).lower() in _COUNTABLE_SINGULARS
                and not m.group(2).lower().endswith('s')
                else m.group(2)
            ),
            corrected, flags=re.IGNORECASE,
        )

        # ── "other" + singular countable → plural ────────────────
        corrected = re.sub(
            r'\b(other)\s+([a-z]+)\b',
            lambda m: m.group(1) + ' ' + (
                _pluralize_noun(m.group(2))
                if m.group(2).lower() in _COUNTABLE_SINGULARS
                and not m.group(2).lower().endswith('s')
                else m.group(2)
            ),
            corrected, flags=re.IGNORECASE,
        )

        # ── Missing copula: "it happy" → "it is happy" ──────────
        _adj_pat = '|'.join(sorted(_COMMON_ADJECTIVES, key=len, reverse=True))
        corrected = re.sub(
            r'\b(it|he|she)\s+(' + _adj_pat + r')\b',
            lambda m: m.group(1) + ' is ' + m.group(2),
            corrected, flags=re.IGNORECASE,
        )

        # ── "will supposed to" → "was supposed to" ────────────────
        # "will supposed" is never valid; the speaker likely meant
        # "was supposed" (past) or "is supposed" (present).
        corrected = re.sub(
            r'\bwill\s+supposed\s+to\b',
            'was supposed to',
            corrected, flags=re.IGNORECASE,
        )

        # ── Missing auxiliary before "either + past participle" ──
        # "you either born with" → "you are either born with"
        corrected = re.sub(
            r'\b(you|we|they)\s+either\s+(born|made|given|chosen|raised|blessed)\b',
            lambda m: m.group(1) + ' are either ' + m.group(2),
            corrected, flags=re.IGNORECASE,
        )
        corrected = re.sub(
            r'\b(he|she|it)\s+either\s+(born|made|given|chosen|raised|blessed)\b',
            lambda m: m.group(1) + ' is either ' + m.group(2),
            corrected, flags=re.IGNORECASE,
        )

        # ── "make/makes + noun + base adj" → past participle ─────
        corrected = re.sub(
            r'\b(makes?)\s+(\w+)\s+(confuse|surprise|bore|interest|excite'
            r'|amaze|annoy|disappoint|embarrass|exhaust|frighten'
            r'|overwhelm|relax|satisfy|shock|worry|tire|scare)\b',
            lambda m: m.group(1) + ' ' + m.group(2) + ' ' + m.group(3) + 'd',
            corrected, flags=re.IGNORECASE,
        )

        # ── Stray "are" after action verb ────────────────────────
        # "walk are very quietly" → "walk very quietly"
        corrected = re.sub(
            r'\b(walk|run|move|go|come|talk|speak|play|work|swim|fly|drive)\s+are\s+',
            lambda m: m.group(1) + ' ',
            corrected, flags=re.IGNORECASE,
        )

        # ── "don't" after singular subject → "doesn't" ───────────
        corrected = re.sub(
            r"\b(it|he|she)\s+don'?t\b",
            lambda m: m.group(1) + " doesn't",
            corrected, flags=re.IGNORECASE,
        )

        # ── "they/we/you <verb> does not" → "do not" ─────────────
        # Handles plural subject + relative clause: "they say does not"
        corrected = re.sub(
            r'\b(they|we|you)\s+(?!he\b|she\b|it\b)(\w+)\s+does\s+not\b',
            lambda m: m.group(1) + ' ' + m.group(2) + ' do not',
            corrected, flags=re.IGNORECASE,
        )

        # ── "me and my/his/her X" → "my/his/her X and I" ────────
        corrected = re.sub(
            r'\b[Mm]e\s+and\s+(my|his|her|our|their)\s+(\w+)',
            lambda m: m.group(1).capitalize() + ' ' + m.group(2) + ' and I'
                      if m.start() == 0 or corrected[m.start()-1] in '.!?\n '
                      else m.group(1) + ' ' + m.group(2) + ' and I',
            corrected,
        )

        # ── "mines" (possessive) → "mine" ────────────────────────
        corrected = re.sub(
            r'\b[Mm]ines\b(?!\s+(?:of|are|were)\b)',
            lambda m: 'Mine' if m.group(0)[0] == 'M' else 'mine',
            corrected,
        )

        # ── "on way" → "on the way" (missing article) ────────────
        corrected = re.sub(
            r'\bon\s+way\s+(home|back|there|here|to|out)\b',
            lambda m: 'on the way ' + m.group(1),
            corrected, flags=re.IGNORECASE,
        )

        # ── "who" for non-human antecedents → "that" ─────────────
        _nonhuman = (r'(?:cat|dog|bird|fish|animal|video|movie|book|car'
                     r'|phone|computer|house|building|table|chair|thing'
                     r'|game|song|food|drink|machine|robot|device)')
        corrected = re.sub(
            r'\b(' + _nonhuman + r')\s+who\b',
            lambda m: m.group(1) + ' that',
            corrected, flags=re.IGNORECASE,
        )

        # ── "very/really + -ly adverb" after sense/taste verbs → adjective ─
        # "tastes very deliciously" → "tastes very delicious"
        def _sense_verb_adv_fix(m):
            adverb = m.group(3)
            if len(adverb) > 4 and adverb.lower().endswith('ly'):
                base = adverb[:-2]
                if base.lower() in _COMMON_ADJECTIVES:
                    return m.group(1) + ' ' + m.group(2) + ' ' + base
            return m.group(0)

        corrected = re.sub(
            r'\b(tastes?|smells?|looks?|feels?|sounds?)\s+'
            r'(very|really|so|quite|extremely|pretty)\s+'
            r'(\w+ly)\b',
            _sense_verb_adv_fix,
            corrected, flags=re.IGNORECASE,
        )

        # ── "more + comparative" (double comparative) ────────────
        # "more better" → "better", "more prettier" → "prettier"
        corrected = re.sub(
            r'\bmore\s+(better|worse|bigger|smaller|faster|slower|older'
            r'|younger|taller|shorter|longer|stronger|weaker|easier'
            r'|harder|prettier|uglier|nicer|louder|quieter|cheaper'
            r'|richer|poorer|thinner|thicker|lighter|heavier|darker'
            r'|brighter|closer|farther|further)\b',
            lambda m: m.group(1),
            corrected, flags=re.IGNORECASE,
        )

        # ── Word-by-word subject-verb agreement ──────────────────
        corrected = self._fix_subject_verb_agreement(corrected)

        return corrected

    def _fix_subject_verb_agreement(self, text: str) -> str:
        """Fix subject-verb agreement by word-by-word analysis.

        Handles:
        - Plural subject + conjugated verb → base form
        - Singular subject + base verb → conjugated form
        - Article + singular noun + base verb → conjugated form
        """
        words = text.split()
        if len(words) < 2:
            return text
        result = list(words)

        def _strip_punct(w):
            stripped = w.rstrip('.,!?;:')
            return stripped, w[len(stripped):]

        i = 0
        while i < len(words):
            w_clean, _ = _strip_punct(words[i])
            w_low = w_clean.lower()

            # Plural subject + verb-s → base form
            if w_low in _PLURAL_SUBJECTS:
                j = i + 1
                while j < len(words) and _strip_punct(words[j])[0].lower() in _INTERPOSING_ADVERBS:
                    j += 1
                if j < len(words):
                    v_clean, v_punct = _strip_punct(words[j])
                    base = _CONJ_TO_BASE.get(v_clean.lower())
                    if base:
                        result[j] = base + v_punct

            # Singular subject + base verb → conjugated form
            elif w_low in _SINGULAR_SUBJECTS:
                if i > 0 and _strip_punct(words[i - 1])[0].lower() in _CAUSATIVE_WORDS:
                    i += 1
                    continue
                j = i + 1
                while j < len(words) and _strip_punct(words[j])[0].lower() in _INTERPOSING_ADVERBS:
                    j += 1
                if j < len(words):
                    v_clean, v_punct = _strip_punct(words[j])
                    conj = _BASE_TO_CONJ.get(v_clean.lower())
                    if conj:
                        result[j] = conj + v_punct

            # Article + singular noun + base verb → conjugated
            elif w_low in ('a', 'an', 'this', 'that', 'each', 'every'):
                if i + 2 < len(words):
                    noun_clean, _ = _strip_punct(words[i + 1])
                    if noun_clean.lower() in _COUNTABLE_SINGULARS:
                        j = i + 2
                        while (j < len(words)
                               and _strip_punct(words[j])[0].lower() in _INTERPOSING_ADVERBS):
                            j += 1
                        if j < len(words):
                            v_clean, v_punct = _strip_punct(words[j])
                            conj = _BASE_TO_CONJ.get(v_clean.lower())
                            if conj:
                                result[j] = conj + v_punct

            i += 1

        return ' '.join(result)

    @staticmethod
    def _split_run_ons(text: str) -> str:
        """Insert sentence boundaries in run-on passages.

        When the speaker omits punctuation (common in ESL speech), the
        transcript is one giant clause.  This heuristic inserts a period
        before clause-starting patterns like 'he/she/I/we/they + verb'
        that appear mid-sentence after a long stretch (>15 words since
        the last period).
        """
        # Only process chunks that lack internal punctuation
        parts = re.split(r'(?<=[.!?])\s+', text)
        result_parts = []
        for part in parts:
            words = part.split()
            if len(words) <= 20:
                result_parts.append(part)
                continue

            # Walk through tokens, inserting periods before clause starters
            out_tokens = []
            since_break = 0
            i = 0
            while i < len(words):
                w = words[i].lower().rstrip('.,;:!?')
                next_w = words[i + 1].lower().rstrip('.,;:!?') if i + 1 < len(words) else ''
                # Check for clause-starting pronoun + verb pattern
                is_clause_start = (
                    w in ('he', 'she', 'i', 'we', 'they', 'it')
                    and next_w
                    and (next_w in _BASE_TO_CONJ or next_w in _CONJ_TO_BASE
                         or next_w in ("don't", "doesn't", "didn't", "wasn't",
                                       "weren't", "won't", "can't", "couldn't",
                                       "wouldn't", "shouldn't", "said", "told",
                                       "asked", "think", "thought", "know",
                                       "knew", "want", "wanted", "lend",
                                       "lent"))
                )
                if is_clause_start and since_break >= 12 and i > 0:
                    # Skip split if a determiner is nearby — likely a
                    # relative clause: "the sentences they say ..."
                    recent = [words[k].lower().rstrip('.,;:!?')
                              for k in range(max(0, i - 3), i)]
                    if any(r in ('the', 'a', 'an', 'this', 'that',
                                 'these', 'those')
                           for r in recent):
                        out_tokens.append(words[i])
                        since_break += 1
                        i += 1
                        continue
                    # End previous clause with period, capitalize new one
                    if out_tokens and not out_tokens[-1].endswith(('.', '!', '?')):
                        out_tokens[-1] = out_tokens[-1].rstrip(',;:') + '.'
                    out_tokens.append(words[i][0].upper() + words[i][1:] if len(words[i]) > 1 else words[i].upper())
                    since_break = 1
                else:
                    out_tokens.append(words[i])
                    since_break += 1
                i += 1
            result_parts.append(' '.join(out_tokens))

        return ' '.join(result_parts)

    def _apply_t5_corrections(self, text: str) -> str:
        """Apply T5 neural grammar correction sentence-by-sentence.

        Uses vennify/t5-base-grammar-correction as the main grammar
        correction engine after regex pre-processing.  Processes each sentence
        individually to avoid truncation (T5-base max = 512 tokens).
        """
        if self._t5_model is None or self._t5_tokenizer is None:
            return text

        # Split run-on passages into smaller sentences first
        text = self._split_run_ons(text)

        # Split into sentences (preserve delimiters)
        sentence_parts = re.split(r'(?<=[.!?])\s+', text)
        if not sentence_parts:
            return text

        corrected_parts = []
        for sent in sentence_parts:
            stripped = sent.strip()
            if not stripped or len(stripped.split()) < 3:
                # Too short to correct meaningfully
                corrected_parts.append(sent)
                continue
            try:
                inputs = self._t5_tokenizer(
                    "grammar: " + stripped,
                    return_tensors="pt",
                    max_length=256,
                    truncation=True,
                )
                with torch.no_grad():
                    outputs = self._t5_model.generate(
                        **inputs,
                        max_length=256,
                        num_beams=5,
                        early_stopping=True,
                    )
                result = self._t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
                if result and result.strip():
                    res = result.strip()
                    # Hallucination guard 1: reject if T5 splits one
                    # sentence into multiple (adds sentence-ending punct).
                    orig_ends = len(re.findall(r'[.!?]', stripped))
                    res_ends = len(re.findall(r'[.!?]', res))
                    if res_ends > orig_ends:
                        corrected_parts.append(sent)
                        continue
                    # Hallucination guard 2: reject if word-level
                    # similarity drops too low (semantic drift).
                    orig_words = stripped.lower().split()
                    res_words = res.lower().split()
                    sim = difflib.SequenceMatcher(None, orig_words, res_words).ratio()
                    if sim >= 0.5:
                        corrected_parts.append(res)
                    else:
                        corrected_parts.append(sent)
                else:
                    corrected_parts.append(sent)
            except Exception:
                corrected_parts.append(sent)

        return ' '.join(corrected_parts)

    def _validate_semantics(self, text: str) -> List[str]:
        """
        Validate sentence semantics using heuristic checks.
        
        Catches nonsensical but syntactically valid sentences like:
        "The Dog Bark Cloud Every Morning"
        
        Args:
            text: Text to validate
            
        Returns:
            List of semantic issue descriptions
        """
        issues = []
        words = text.split()
        
        if len(words) < 2:
            return issues
        
        # Check for obvious subject-verb agreement issues
        # Pattern: Singular noun + plural verb or vice versa
        for i in range(len(words) - 1):
            curr_word = words[i].lower()
            next_word = words[i + 1].lower()
            
            # Check for article + noun + base verb (should be verb+s)
            if i > 0 and words[i - 1].lower() in ['the', 'a', 'an']:
                # "The dog bark" should be "The dog barks"
                if next_word in ['bark', 'run', 'walk', 'jump', 'swim', 'fly', 'speak', 'talk']:
                    issues.append(f"Subject-verb disagreement: '{curr_word} {next_word}' should likely be '{curr_word} {next_word}s'")
        
        # Check for nonsensical noun-noun sequences that don't make sense
        # Pattern: [Common noun] [Verb used as noun in wrong context]
        nonsensical_patterns = [
            ('dog', 'bark'),  # "dog bark" as consecutive nouns is odd
            ('cloud', 'bark'),  # "cloud bark" makes no sense
        ]
        
        for i in range(len(words) - 1):
            word_pair = (words[i].lower(), words[i + 1].lower())
            # Check if this pair appears in suspicious contexts
            if word_pair[0] in ['bark', 'jump', 'run'] and word_pair[1] in ['cloud', 'sky', 'water']:
                issues.append(f"Unusual word combination: '{word_pair[0]} {word_pair[1]}' may not make semantic sense")
        
        # Check for verb + noun where adverb expected
        # Pattern: [verb] [noun] where [verb] [adverb] expected
        adverb_context_verbs = ['bark', 'run', 'speak', 'walk', 'jump', 'fly', 'swim']
        unlikely_adverb_nouns = ['cloud', 'dog', 'cat', 'tree', 'house', 'table']
        
        for i in range(len(words) - 1):
            if words[i].lower() in adverb_context_verbs:
                if words[i + 1].lower() in unlikely_adverb_nouns:
                    issues.append(f"Possible word order issue: '{words[i]} {words[i + 1]}' - expected adverb after verb")
        
        # NOTE: Removed hardcoded capitalization check
        # Problem: Flagged proper nouns as errors (e.g., "I went to Paris")
        # Solution: Use NER in production or remove this check entirely
        # For now, we rely on LanguageTool's context-aware grammar checking
        
        return issues
    
    def _apply_semantic_corrections(self, text: str, semantic_issues: List[str]) -> str:
        """
        Apply corrections for detected semantic issues.
        
        Args:
            text: Original text
            semantic_issues: List of semantic issue descriptions
            
        Returns:
            Corrected text
        """
        corrected = text
        
        # Fix subject-verb agreement issues
        for issue in semantic_issues:
            if "Subject-verb disagreement" in issue:
                # Extract the problematic words and correction from issue description
                # e.g., "'dog bark' should likely be 'dog barks'"
                import re
                match = re.search(r"'([^']+)'\s+should likely be\s+'([^']+)'", issue)
                if match:
                    wrong_phrase = match.group(1)
                    correct_phrase = match.group(2)
                    # Replace in text (case-insensitive but preserve original case of first letter)
                    corrected = re.sub(
                        r'\b' + re.escape(wrong_phrase) + r'\b',
                        correct_phrase,
                        corrected,
                        flags=re.IGNORECASE
                    )
        
        return corrected
    
    def _deduplicate_suggestions(self, suggestions: List[str]) -> List[str]:
        """Remove duplicate suggestions (case-insensitive)."""
        unique = []
        seen = set()
        
        for suggestion in suggestions:
            normalized = suggestion.lower().strip()
            if normalized not in seen:
                unique.append(suggestion)
                seen.add(normalized)
        
        return unique
    
    def get_semantic_issues(self, text: str) -> List[str]:
        """
        Get semantic issues from text validation.
        
        Args:
            text: Text to validate
            
        Returns:
            List of semantic issue descriptions
        """
        return self._validate_semantics(text)
    
    def extract_grammar_issues(self, original: str, corrected: str) -> List[str]:
        """
        Extract words with grammar issues by comparing original and corrected text.

        Filters out:
        - **ASR artifacts**: hyphenated non-words like ``early-year-dence``
          that Whisper misheard; correcting these is meaningless.
        - **Compound-word merges**: ``Cyber security`` → ``Cybersecurity``
          is a trivial spacing fix, not a grammar error worth highlighting
          an entire sentence for.
        
        Args:
            original: Original transcript
            corrected: Grammar-corrected transcript
            
        Returns:
            List of words that were corrected (meaningful grammar issues only)
        """
        orig_words = original.lower().split()
        corr_words = corrected.lower().split()
        
        if orig_words == corr_words:
            return []
        
        issues = []
        matcher = difflib.SequenceMatcher(a=orig_words, b=corr_words)
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag not in ('replace', 'delete'):
                continue

            orig_segment = orig_words[i1:i2]
            corr_segment = corr_words[j1:j2]

            # Skip compound-word merges (e.g. "cyber security" → "cybersecurity")
            if self._is_compound_merge(orig_segment, corr_segment):
                continue

            for idx in range(i1, i2):
                if idx >= len(orig_words):
                    continue
                word = orig_words[idx]
                # Skip ASR artifacts (hyphenated non-words)
                if self._is_asr_artifact(word):
                    continue
                issues.append(word)
        
        return issues

    # ------------------------------------------------------------------
    # Helpers for filtering grammar noise
    # ------------------------------------------------------------------

    @staticmethod
    def _is_compound_merge(orig_segment: List[str], corr_segment: List[str]) -> bool:
        """Return True when the correction simply merges adjacent words.

        Example: ``["cyber", "security"]`` → ``["cybersecurity"]``
        """
        if len(orig_segment) < 2 or len(corr_segment) != 1:
            return False
        joined = "".join(w.lower().strip(".,!?;:") for w in orig_segment)
        return corr_segment[0].lower().strip(".,!?;:") == joined

    @staticmethod
    def _is_asr_artifact(word: str) -> bool:
        """Return True when *word* looks like an ASR transcription glitch.

        Whisper sometimes produces hyphenated gibberish such as
        ``early-year-dence`` when it can't confidently decode a phrase.
        These should not be flagged as grammar errors.
        """
        # Two or more hyphens → almost certainly ASR noise
        if word.count("-") >= 2:
            return True
        return False

    @staticmethod
    def _post_t5_cleanup(text: str) -> str:
        """Apply targeted fixes for common errors that T5 tends to miss.

        Runs AFTER T5 so we can clean up anything it left behind.
        """
        corrected = text

        # "my mine" → "mine" (redundant possessive, often from T5/paraphrase)
        corrected = re.sub(
            r'\bmy\s+mine\b', 'mine', corrected, flags=re.IGNORECASE,
        )

        # "mines" (possessive pronoun) → "mine"
        # Only skip when it's clearly a plural noun ("mines of coal",
        # "mines are", "mines were").  "mines was dead" is possessive.
        corrected = re.sub(
            r'\b[Mm]ines\b(?!\s+(?:of|are|were)\b)',
            lambda m: 'Mine' if m.group(0)[0] == 'M' else 'mine',
            corrected,
        )

        # "on way home/back" → "on the way home/back"
        corrected = re.sub(
            r'\bon\s+way\s+(home|back|there|here|to|out)\b',
            lambda m: 'on the way ' + m.group(1),
            corrected, flags=re.IGNORECASE,
        )

        # "me and my/his/her X" → "my/his/her X and I"
        corrected = re.sub(
            r'(?:^|(?<=[.!?]\s))[Mm]e\s+and\s+(my|his|her|our|their)\s+(\w+)',
            lambda m: m.group(1).capitalize() + ' ' + m.group(2) + ' and I',
            corrected,
        )

        # "video/cat/dog who" → "video/cat/dog that"
        _nonhuman = (r'(?:cat|dog|bird|fish|animal|video|movie|book|car'
                     r'|phone|computer|house|building|table|chair|thing'
                     r'|game|song|food|drink|machine|robot|device)')
        corrected = re.sub(
            r'\b(' + _nonhuman + r')\s+who\b',
            lambda m: m.group(1) + ' that',
            corrected, flags=re.IGNORECASE,
        )

        # "more better/worse/prettier" (double comparative)
        corrected = re.sub(
            r'\bmore\s+(better|worse|bigger|smaller|faster|slower|older'
            r'|younger|taller|shorter|longer|stronger|weaker|easier'
            r'|harder|prettier|uglier|nicer|louder|quieter|cheaper'
            r'|richer|poorer|thinner|thicker|lighter|heavier|darker'
            r'|brighter|closer|farther|further)\b',
            lambda m: m.group(1),
            corrected, flags=re.IGNORECASE,
        )

        # ── Past-narrative tense correction ─────────────────────
        # When the text opens with a clear past-time marker ("Last weekend",
        # "Last Saturday", "Yesterday"), the entire passage is a past
        # narrative.  Fix specific present-tense verbs that T5 and earlier
        # regex layers missed.  Only applies to irregular verbs where the
        # base/present form is unambiguously wrong in past context.
        _past_openers = re.match(
            r'(?i)^(?:last\s+(?:week|weekend|month|year|night|time'
            r'|monday|tuesday|wednesday|thursday|friday|saturday|sunday)'
            r'|yesterday|the\s+other\s+day)',
            corrected,
        )
        if _past_openers:
            # "I leave" / "he leaves" → "I left" / "he left"
            corrected = re.sub(
                r'\b(I|he|she|we|they|it)\s+leave(?:s)?\b',
                lambda m: m.group(1) + ' left',
                corrected, flags=re.IGNORECASE,
            )
            # "he lends" / "he lend" / "I lend" → "... lent"
            corrected = re.sub(
                r'\b(I|he|she|we|they|it)\s+lend(?:s)?\b',
                lambda m: m.group(1) + ' lent',
                corrected, flags=re.IGNORECASE,
            )
            # "I spend" / "he spends" → "... spent"
            corrected = re.sub(
                r'\b(I|he|she|we|they|it)\s+spend(?:s)?\b',
                lambda m: m.group(1) + ' spent',
                corrected, flags=re.IGNORECASE,
            )
            # "I send" / "he sends" → "... sent"
            corrected = re.sub(
                r'\b(I|he|she|we|they|it)\s+send(?:s)?\b',
                lambda m: m.group(1) + ' sent',
                corrected, flags=re.IGNORECASE,
            )

        # "tastes/smells/... very deliciously" → "very delicious"
        # Also handle adverbs like "beautifully", "wonderfully" etc.
        def _fix_sense_verb_adverb(m):
            adverb = m.group(3)
            if len(adverb) > 4 and adverb.lower().endswith('ly'):
                # Try stripping -ly to get adjective
                base = adverb[:-2]  # "deliciously" → "delicious"
                if base.lower() in _COMMON_ADJECTIVES:
                    return m.group(1) + ' ' + m.group(2) + ' ' + base
            return m.group(0)

        corrected = re.sub(
            r'\b(tastes?|smells?|looks?|feels?|sounds?)\s+'
            r'(very|really|so|quite|extremely|pretty)\s+'
            r'(\w+ly)\b',
            _fix_sense_verb_adverb,
            corrected, flags=re.IGNORECASE,
        )

        return corrected

    @staticmethod
    def _revert_asr_artifacts(original: str, corrected: str) -> str:
        """Restore ASR artifact tokens that the grammar pipeline changed.

        Works by aligning original and corrected word-by-word; wherever the
        original token is an ASR artifact (2+ hyphens) and the corrected
        version differs, we put the original token back.
        """
        orig_words = original.split()
        corr_words = corrected.split()

        orig_lower = [w.lower() for w in orig_words]
        corr_lower = [w.lower() for w in corr_words]

        if orig_lower == corr_lower:
            return corrected

        matcher = difflib.SequenceMatcher(a=orig_lower, b=corr_lower)
        # Build a mutable list from the corrected tokens
        result = list(corr_words)

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != "replace":
                continue
            # Only 1-to-1 replacements are safe to revert
            if (i2 - i1) == 1 and (j2 - j1) == 1:
                orig_tok = orig_words[i1]
                if orig_tok.count("-") >= 2:
                    result[j1] = orig_tok  # put original back

        return " ".join(result)
