"""Lightweight phonetics helpers.

This project uses ARPABET phones in a few places (CMU dict, MFA english_us_arpa).
For user-facing display we convert to:
  - A compact IPA string   (e.g. /bəˈnænə/)
  - A human-readable guide  (e.g. buh·NA·nuh)
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# ARPABET → IPA mapping (US English-ish). Stress digits handled separately.
# ---------------------------------------------------------------------------
_ARPABET_TO_IPA = {
	# Vowels
	"AA": "ɑ",
	"AE": "æ",
	"AH": "ə",
	"AO": "ɔ",
	"AW": "aʊ",
	"AY": "aɪ",
	"EH": "ɛ",
	"ER": "ɜr",
	"EY": "eɪ",
	"IH": "ɪ",
	"IY": "i",
	"OW": "oʊ",
	"OY": "ɔɪ",
	"UH": "ʊ",
	"UW": "u",
	# Consonants
	"B": "b",
	"CH": "tʃ",
	"D": "d",
	"DH": "ð",
	"F": "f",
	"G": "ɡ",
	"HH": "h",
	"JH": "dʒ",
	"K": "k",
	"L": "l",
	"M": "m",
	"N": "n",
	"NG": "ŋ",
	"P": "p",
	"R": "r",
	"S": "s",
	"SH": "ʃ",
	"T": "t",
	"TH": "θ",
	"V": "v",
	"W": "w",
	"Y": "j",
	"Z": "z",
	"ZH": "ʒ",
}

# ---------------------------------------------------------------------------
# ARPABET → human-readable English approximation
# ---------------------------------------------------------------------------
_ARPABET_TO_READABLE = {
	# Vowels – short, intuitive spellings a native reader can sound out
	"AA": "ah",    # fAther
	"AE": "a",     # cAt
	"AH": "uh",    # bUt / About (schwa)
	"AO": "aw",    # thOUGHt
	"AW": "ow",    # hOW
	"AY": "y",     # mY, trY  – reads naturally in context (e.g. MY)
	"EH": "e",     # bEd
	"ER": "er",    # bIRd
	"EY": "ay",    # sAY, dAY
	"IH": "i",     # bIt
	"IY": "ee",    # sEE
	"OW": "oh",    # gO
	"OY": "oy",    # bOY
	"UH": "uu",    # bOOk
	"UW": "oo",    # fOOd
	# Consonants
	"B":  "b",
	"CH": "ch",
	"D":  "d",
	"DH": "th",
	"F":  "f",
	"G":  "g",
	"HH": "h",
	"JH": "j",
	"K":  "k",
	"L":  "l",
	"M":  "m",
	"N":  "n",
	"NG": "ng",
	"P":  "p",
	"R":  "r",
	"S":  "s",
	"SH": "sh",
	"T":  "t",
	"TH": "th",
	"V":  "v",
	"W":  "w",
	"Y":  "y",
	"Z":  "z",
	"ZH": "zh",
}

# Two-consonant clusters that can legally begin an English syllable.
# Used by the syllabifier to decide where to place inter-vocalic consonants.
_VALID_ONSETS: set[Tuple[str, ...]] = {
	("B", "L"), ("B", "R"),
	("D", "R"), ("D", "W"),
	("F", "L"), ("F", "R"),
	("G", "L"), ("G", "R"), ("G", "W"),
	("HH", "Y"),
	("K", "L"), ("K", "R"), ("K", "W"),
	("P", "L"), ("P", "R"),
	("S", "K"), ("S", "L"), ("S", "M"), ("S", "N"),
	("S", "P"), ("S", "T"), ("S", "W"),
	("SH", "R"),
	("T", "R"), ("T", "W"),
	("TH", "R"), ("TH", "W"),
}


# ── helper: parse tokens ──────────────────────────────────────────────────

def _parse_tokens(
	phones: Union[str, Iterable[str], None],
) -> Optional[List[str]]:
	"""Normalise *phones* into a plain list of ARPABET token strings."""
	if not phones:
		return None
	if isinstance(phones, str):
		tokens = [t for t in phones.split() if t.strip()]
	else:
		tokens = [t for t in phones if isinstance(t, str) and t.strip()]
	return tokens or None


# ── IPA conversion ─────────────────────────────────────────────────────────

def arpabet_to_ipa(
	phones: Union[str, Iterable[str], None],
	*,
	wrap_slashes: bool = True,
) -> Optional[str]:
	"""Convert ARPABET phones into a compact IPA string.

	Args:
		phones: Either a space-delimited ARPABET string (e.g. "B AH0 N AE1 N AH0")
			or an iterable of ARPABET tokens (e.g. ["B", "AH0", ...]).
		wrap_slashes: If True, wrap result in /.../.

	Returns:
		IPA string or None if conversion yields nothing.
	"""

	tokens = _parse_tokens(phones)
	if not tokens:
		return None

	ipa_chars: list[str] = []

	for token in tokens:
		base = re.sub(r"[012]", "", token)
		stress = re.search(r"[012]", token)

		ipa = _ARPABET_TO_IPA.get(base)
		if not ipa:
			continue

		# Stress marker goes before stressed vowel
		if stress and stress.group() == "1":
			ipa_chars.append("ˈ")
		elif stress and stress.group() == "2":
			ipa_chars.append("ˌ")

		ipa_chars.append(ipa)

	if not ipa_chars:
		return None

	ipa_string = "".join(ipa_chars)
	return f"/{ipa_string}/" if wrap_slashes else ipa_string


# ── Readable pronunciation ────────────────────────────────────────────────

def arpabet_to_readable(
	phones: Union[str, Iterable[str], None],
) -> Optional[str]:
	"""Convert ARPABET phones into a human-readable pronunciation guide.

	Syllables are separated by a middle-dot (·).
	Stressed syllables are UPPER-CASED (primary *and* secondary stress).

	Examples::

		"AO1 F AH0 N"              → "AW·fuhn"       (often)
		"W EH1 N Z D EY0"          → "WENZ·day"      (Wednesday)
		"N IY1 SH"                 → "NEESH"          (niche)
		"D AH0 B R IY1"            → "duh·BREE"       (debris)
		"HH ER0 M AY1 AH0 N IY0"  → "her·MY·uh·nee"  (Hermione)

	Args:
		phones: Space-delimited ARPABET string or iterable of ARPABET tokens.

	Returns:
		Readable pronunciation string, or *None* if conversion fails.
	"""
	tokens = _parse_tokens(phones)
	if not tokens:
		return None

	# 1. Parse each token into a structured dict
	parsed: list[dict] = []
	for token in tokens:
		base = re.sub(r"[012]", "", token)
		stress_m = re.search(r"[012]", token)
		stress = int(stress_m.group()) if stress_m else None
		is_vowel = stress is not None          # only vowels carry stress in ARPABET
		readable = _ARPABET_TO_READABLE.get(base)
		if readable is None:
			continue
		parsed.append({
			"base": base,
			"stress": stress,
			"is_vowel": is_vowel,
			"readable": readable,
		})

	if not parsed:
		return None

	# 2. Find vowel positions
	vowel_indices = [i for i, p in enumerate(parsed) if p["is_vowel"]]

	if not vowel_indices:
		# No vowels at all – just concatenate everything
		return "".join(p["readable"] for p in parsed)

	# 3. Syllabify (onset-maximisation with valid-onset check)
	boundaries = [0]

	for idx in range(len(vowel_indices) - 1):
		vi_curr = vowel_indices[idx]
		vi_next = vowel_indices[idx + 1]

		# Consonant indices between the two vowels
		cons = list(range(vi_curr + 1, vi_next))
		n = len(cons)

		if n == 0:
			boundary = vi_next
		elif n == 1:
			boundary = cons[0]                       # single C → onset of next
		elif n == 2:
			c1 = parsed[cons[0]]["base"]
			c2 = parsed[cons[1]]["base"]
			if (c1, c2) in _VALID_ONSETS:
				boundary = cons[0]                   # both → onset of next
			else:
				boundary = cons[1]                   # split: C|C
		else:
			# 3+ consonants: try last-two as onset, else just last one
			c_pen = parsed[cons[-2]]["base"]
			c_last = parsed[cons[-1]]["base"]
			if (c_pen, c_last) in _VALID_ONSETS:
				boundary = cons[-2]
			else:
				boundary = cons[-1]

		boundaries.append(boundary)

	boundaries.append(len(parsed))

	# 4. Build syllable strings, applying stress casing
	parts: list[str] = []
	for i in range(len(boundaries) - 1):
		syl_phones = parsed[boundaries[i] : boundaries[i + 1]]
		# Determine stress from the vowel in this syllable
		stress = 0
		for p in syl_phones:
			if p["is_vowel"] and p["stress"] is not None:
				stress = p["stress"]
				break

		text = "".join(p["readable"] for p in syl_phones)
		if stress in (1, 2):                     # primary or secondary → UPPERCASE
			text = text.upper()
		parts.append(text)

	return "·".join(parts)