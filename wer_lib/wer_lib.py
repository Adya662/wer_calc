import time
import os
import json
from pathlib import Path
import difflib
import re
import string
from typing import List, Tuple, Dict
import unicodedata
from rapidfuzz.fuzz import ratio
import phonetics
from metaphone import doublemetaphone
from concurrent.futures import ThreadPoolExecutor
import sys
import shutil
import threading
from functools import lru_cache

# Optional dependency: g2p_en for grapheme-to-phoneme
try:
    from g2p_en import G2p  # type: ignore
except Exception:  # pragma: no cover - optional
    G2p = None  # type: ignore


# Resolve all paths relative to this file's directory (wer_lib)
BASE_DIR = Path(__file__).resolve().parent

# Directory containing all call subfolders
CALLS_DIR = BASE_DIR / "calls"
# Root-level output for global WER
GLOBAL_OUTPUT_DIR = BASE_DIR / "output"
GLOBAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Log file to capture terminal output
OUTPUT_LOG_FILE = BASE_DIR / "output_lib.txt"

class TeeStream:
    """
    Simple thread-safe tee for capturing terminal output to a file while still
    printing to the original stdout/stderr.
    """
    def __init__(self, *streams):
        self.streams = streams
        self._lock = threading.Lock()

    def write(self, data: str) -> None:
        with self._lock:
            for stream in self.streams:
                stream.write(data)
                stream.flush()

    def flush(self) -> None:
        with self._lock:
            for stream in self.streams:
                stream.flush()

# ------------------ Phonetic utilities (cached, deterministic) ------------------

@lru_cache(maxsize=4096)
def _jw_sim(a: str, b: str) -> float:
    """
    Simple Jaro-Winkler similarity (0..1). Deterministic and dependency-free.
    Good enough for short phonetic codes.
    """
    return _jaro_winkler(a, b)

def _jaro_winkler(s1: str, s2: str, p: float = 0.1) -> float:
    # Jaro
    s1, s2 = s1 or "", s2 or ""
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0
    match_distance = max(len1, len2)//2 - 1
    s1_matches = [False]*len1
    s2_matches = [False]*len2
    matches = 0
    transpositions = 0
    # matches
    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if s2_matches[j]:
                continue
            if s1[i] != s2[j]:
                continue
            s1_matches[i] = s2_matches[j] = True
            matches += 1
            break
    if matches == 0:
        return 0.0
    # transpositions
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    transpositions //= 2
    jaro = (matches/len1 + matches/len2 + (matches - transpositions)/matches)/3.0
    # winkler
    prefix = 0
    for i in range(min(4, len1, len2)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break
    return jaro + prefix * p * (1 - jaro)

def _levenshtein(a: List[str], b: List[str]) -> int:
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n+1))
    curr = [0]*(n+1)
    for i in range(1, m+1):
        curr[0] = i
        ai = a[i-1]
        for j in range(1, n+1):
            cost = 0 if ai == b[j-1] else 1
            curr[j] = min(prev[j] + 1, curr[j-1] + 1, prev[j-1] + cost)
        prev, curr = curr, prev
    return prev[n]

@lru_cache(maxsize=4096)
def metaphone_codes(token: str) -> Tuple[str, str]:
    """
    Returns (primary, alternate) Double Metaphone codes; empty strings if not available.
    """
    t = (token or "").strip().lower()
    if not t:
        return ("", "")
    try:
        p, a = doublemetaphone(t)
        return (p or "", a or "")
    except Exception:
        # fallback: coarse code (vowel removal + basic collapsing)
        import re as _re
        s = _re.sub(r"[^a-z0-9]", "", t)
        s = s.replace("ph", "f").replace("gh", "g").replace("kn", "n").replace("wr", "r")
        s = _re.sub(r"[aeiou]+", "", s)
        return (s[:8], "")

def metaphone_similar(a: str, b: str, jw_threshold: float = 0.92) -> bool:
    pa, aa = metaphone_codes(a)
    pb, ab = metaphone_codes(b)
    if not (pa or aa or pb or ab):
        return False
    # exact code match
    if pa and (pa == pb or pa == ab):
        return True
    if aa and (aa == pb or aa == ab):
        return True
    # soft code similarity
    codes_a = [c for c in (pa, aa) if c]
    codes_b = [c for c in (pb, ab) if c]
    return any(_jw_sim(x, y) >= jw_threshold for x in codes_a for y in codes_b)

_g2p = G2p() if 'G2p' in globals() and G2p is not None else None  # type: ignore

@lru_cache(maxsize=4096)
def g2p_arpabet(token: str) -> List[str]:
    """
    Convert token to ARPABET phoneme list using g2p_en when available.
    Returns [] if unavailable or token empty.
    """
    t = (token or "").strip()
    if not t or _g2p is None:
        return []
    seq = _g2p(t)  # type: ignore
    phones = [p for p in seq if p and p[0].isalpha() and p[0].isupper()]
    return [p.rstrip("012") for p in phones]

def phoneme_similarity(a: str, b: str) -> float:
    """
    0..1 similarity using normalized edit distance on ARPABET phones.
    Falls back to 0 if G2P unavailable or one side empty.
    """
    pa = g2p_arpabet(a)
    pb = g2p_arpabet(b)
    if not pa or not pb:
        return 0.0
    dist = _levenshtein(pa, pb)
    denom = max(len(pa), len(pb))
    return 1.0 - (dist / denom) if denom else 0.0

def sounds_alike(a: str, b: str, meta_jw: float = 0.92, g2p_thresh: float = 0.80) -> bool:
    """
    Decide if two tokens sound alike.
    1) Double Metaphone (codes equal or Jaro-Winkler >= meta_jw) ⇒ True
    2) Else G2P ARPABET similarity >= g2p_thresh ⇒ True
    Else False.
    All results are cached.
    """
    if not a or not b:
        return False
    if a == b:
        return True
    if metaphone_similar(a, b, jw_threshold=meta_jw):
        return True
    if phoneme_similarity(a, b) >= g2p_thresh:
        return True
    return False

def _tokens_equal(a: str, b: str) -> bool:
    # 1) strict equality first
    if a == b:
        return True

    # 2) BRAND::<brand> special-case: compare brands phonetically too
    if a.startswith("BRAND::") and b.startswith("BRAND::"):
        u = a.split("::", 1)[1]
        v = b.split("::", 1)[1]
        from_either_fuzzy = _levenshtein([u], [v]) <= 1  # cheap typo tolerance
        return from_either_fuzzy or sounds_alike(u, v)

    # 3) Otherwise, fall back to phonetic equality on the bucket term itself.
    au = a.split("::", 1)
    bu = b.split("::", 1)
    atok = au[1] if len(au) == 2 else a
    btok = bu[1] if len(bu) == 2 else b

    # short-circuit: if tokens are trivially close by typos
    if _jw_sim(atok, btok) >= 0.97:
        return True

    # phonetic check
    return sounds_alike(atok, btok)

# Number mappings for cardinal numbers only
WORD_TO_DIGIT = {
    'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
    'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
    'eleven': '11', 'twelve': '12', 'thirteen': '13', 'fourteen': '14', 'fifteen': '15',
    'sixteen': '16', 'seventeen': '17', 'eighteen': '18', 'nineteen': '19', 'twenty': '20',
    'thirty': '30', 'forty': '40', 'fifty': '50', 'sixty': '60', 'seventy': '70',
    'eighty': '80', 'ninety': '90', 'hundred': '100', 'thousand': '1000', 'million': '1000000'
}

DIGIT_TO_WORD = {v: k for k, v in WORD_TO_DIGIT.items()}

# Ordinal mappings for better normalization
ORDINAL_MAPPINGS = {
    'first': '1st', 'second': '2nd', 'third': '3rd', 'fourth': '4th', 'fifth': '5th',
    'sixth': '6th', 'seventh': '7th', 'eighth': '8th', 'ninth': '9th', 'tenth': '10th',
    'eleventh': '11th', 'twelfth': '12th', 'thirteenth': '13th', 'fourteenth': '14th',
    'fifteenth': '15th', 'sixteenth': '16th', 'seventeenth': '17th', 'eighteenth': '18th',
    'nineteenth': '19th', 'twentieth': '20th', 'thirtieth': '30th'
}

# Time pattern normalizations
TIME_PATTERNS = {
    'five five': '5:05',
    'five oh five': '5:05',
    'twelve thirty': '12:30',
    'two fifteen': '2:15',
    'three forty five': '3:45'
}

# Common variant spellings normalization (can be extended)
VARIANT_SPELLINGS = {
    'apki': 'aapki',
    'kese': 'kaise',
    'rey': 'ray',
    'u': 'you',
    'ur': 'your',
    'thru': 'through',
    'nite': 'night',
    'lite': 'light',
    'tho': 'though',
    'gonna': 'going to',
    'wanna': 'want to',
    'gotta': 'got to',
    'dunno': 'do not know',
    'kinda': 'kind of',
    'sorta': 'sort of',
    'coulda': 'could have',
    'shoulda': 'should have',
    'woulda': 'would have'
}

def are_phonetically_similar(word1: str, word2: str, threshold: float = 0.8) -> bool:
    """
    Check if two words are phonetically similar. Uses cached, deterministic pipeline:
    - Double Metaphone with Jaro-Winkler soft check
    - Fallback: ARPABET G2P phoneme similarity
    - Final fallback: very high string similarity
    """
    if not word1 or not word2:
        return False

    word1_clean = word1.lower().strip()
    word2_clean = word2.lower().strip()

    # Treat exact text equality as similar
    if word1_clean == word2_clean:
        return True

    # Phonetic pipeline
    if sounds_alike(word1_clean, word2_clean):
        return True

    # Fallback: high string similarity for very close matches
    if ratio(word1_clean, word2_clean) >= 90:  # ratio returns 0-100
        return True

    return False

def normalize_time_expressions(text: str) -> str:
    """Normalize time expressions like 'five five' to '5:05'"""
    for time_phrase, time_format in TIME_PATTERNS.items():
        text = re.sub(r'\b' + re.escape(time_phrase) + r'\b', time_format, text, flags=re.IGNORECASE)
    return text

def normalize_ordinals(text: str) -> str:
    """Normalize ordinal words to their numeric form"""
    words = text.split()
    normalized_words = []
    
    for word in words:
        if word.lower() in ORDINAL_MAPPINGS:
            normalized_words.append(ORDINAL_MAPPINGS[word.lower()])
        else:
            normalized_words.append(word)
    
    return ' '.join(normalized_words)


def transliterate_to_roman(text: str) -> str:
    """
    Transliterate non-Roman scripts to Roman/ASCII.
    This is a basic implementation - can be enhanced with specific transliteration libraries.
    """
    # Normalize unicode characters
    normalized = unicodedata.normalize('NFKD', text)
    
    # Basic transliteration for common scripts
    transliterated = ""
    for char in normalized:
        if ord(char) < 128:  # ASCII characters
            transliterated += char
        else:
            # Basic transliteration mapping (can be extended)
            # For now, just replace with ASCII equivalent or remove
            ascii_equiv = unicodedata.normalize('NFKD', char).encode('ascii', 'ignore').decode('ascii')
            if ascii_equiv:
                transliterated += ascii_equiv
            else:
                # For scripts like Devanagari, you'd need a proper transliteration library
                # For now, we'll skip non-ASCII chars that don't have simple equivalents
                continue
    
    return transliterated

def normalize_numbers(text: str) -> str:
    """
    Normalize cardinal numbers (convert between digit and word forms).
    Avoid ordinals (1st, 2nd, etc.)
    """
    words = text.split()
    normalized_words = []
    
    for word in words:
        # Skip ordinals (contains 'st', 'nd', 'rd', 'th' after digits)
        if re.match(r'^\d+(st|nd|rd|th)$', word.lower()):
            normalized_words.append(word)
            continue
            
        # Convert word to digit
        if word.lower() in WORD_TO_DIGIT:
            normalized_words.append(WORD_TO_DIGIT[word.lower()])
        # Convert digit to word (for single digits and common numbers)
        elif word in DIGIT_TO_WORD:
            normalized_words.append(DIGIT_TO_WORD[word])
        else:
            normalized_words.append(word)
    
    return ' '.join(normalized_words)

def normalize_variants(text: str) -> str:
    """Normalize variant spellings to canonical forms"""
    words = text.split()
    normalized_words = []
    
    for word in words:
        if word.lower() in VARIANT_SPELLINGS:
            normalized_words.append(VARIANT_SPELLINGS[word.lower()])
        else:
            normalized_words.append(word)
    
    return ' '.join(normalized_words)

def canonicalize_and_normalize(text: str) -> str:
    """
    Comprehensive canonicalization and normalization pipeline
    """
    if not text or not text.strip():
        return ""
    
    # Step 1: Transliterate non-Roman scripts to Roman/ASCII
    text = transliterate_to_roman(text)
    
    # Step 2: Normalize time expressions
    text = normalize_time_expressions(text)
    
    # Step 3: Normalize ordinals
    text = normalize_ordinals(text)
    
    # Step 4: Normalize variant spellings
    text = normalize_variants(text)
    
    # Step 5: Convert to lowercase
    text = text.lower()
    
    # Step 6: Remove all punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Step 7: Normalize numbers (cardinal only)
    text = normalize_numbers(text)
    
    # Step 8: Collapse repeated whitespace
    text = ' '.join(text.split())
    
    return text.strip()

def find_word_differences(ref: str, hyp: str, ref_orig: str, hyp_orig: str) -> List[Tuple[str, str]]:
    """Find word-level differences between normalized texts, but return original words"""
    ref_words = ref.split()
    hyp_words = hyp.split()
    ref_orig_words = ref_orig.split()
    hyp_orig_words = hyp_orig.split()
    
    # Use difflib to find differences in normalized text
    differ = difflib.SequenceMatcher(None, ref_words, hyp_words)
    differences = []
    
    for tag, i1, i2, j1, j2 in differ.get_opcodes():
        if tag != 'equal':
            # Use original words for display
            ref_part = ' '.join(ref_orig_words[i1:i2]) if i1 < i2 and i1 < len(ref_orig_words) else '[missing]'
            hyp_part = ' '.join(hyp_orig_words[j1:j2]) if j1 < j2 and j1 < len(hyp_orig_words) else '[missing]'
            differences.append((ref_part, hyp_part))
    
    return differences[:10]  # Return first 10 differences for analysis

def compute_normalized_wer(ref: str, hyp: str, fuzzy_threshold: int = 85, use_phonetic: bool = True) -> Tuple[int, int, List[Tuple[str, str]]]:
    """
    Compute WER with comprehensive normalization and phonetic matching
    """
    # Keep original for display
    ref_orig = ref
    hyp_orig = hyp
    
    # Normalize both texts
    ref_norm = canonicalize_and_normalize(ref)
    hyp_norm = canonicalize_and_normalize(hyp)
    
    r_words = ref_norm.split()
    h_words = hyp_norm.split()
    m, n = len(r_words), len(h_words)
    
    if m == 0 and n == 0:
        return 0, 0, []
    if m == 0:
        return n, n, find_word_differences(ref_norm, hyp_norm, ref_orig, hyp_orig)
    if n == 0:
        return m, m, find_word_differences(ref_norm, hyp_norm, ref_orig, hyp_orig)
    
    # DP table for WER computation
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i  # deletions
    for j in range(n + 1):
        dp[0][j] = j  # insertions
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Check for matches
            cost = 1  # Default: substitution needed

            rw = r_words[i-1]
            hw = h_words[j-1]
            # Exact or phonetic equality → treat as equal (cost 0)
            if rw == hw:
                cost = 0
            elif use_phonetic and are_phonetically_similar(rw, hw):
                cost = 0
            elif ratio(rw, hw) >= fuzzy_threshold:
                cost = 0
            
            dp[i][j] = min(
                dp[i-1][j] + 1,      # deletion
                dp[i][j-1] + 1,      # insertion
                dp[i-1][j-1] + cost  # substitution
            )
    
    # Find differences for analysis
    differences = find_word_differences(ref_norm, hyp_norm, ref_orig, hyp_orig)
    
    return dp[m][n], m, differences

def process_call_dir(call_dir: Path) -> Tuple[List[str], List[str]]:
    refs = []
    hyps = []
    ref_file = call_dir / "ref_transcript.json"
    gt_file = call_dir / "gt_transcript.json"
    output_dir = call_dir / "output"
    output_dir.mkdir(exist_ok=True)

    # Empty out the output folder for this call
    try:
        for child in list(output_dir.iterdir()):
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
    except Exception as e:
        print(f"Skipping cleanup for {call_dir.name}: {e}")

    try:
        ref_data = json.loads(ref_file.read_text(encoding="utf-8"))
        gt_data = json.loads(gt_file.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Skipping {call_dir.name}: {e}")
        return refs, hyps

    refs = [t["content"].strip() for t in ref_data
            if t.get("speaker") == "assistant" and t.get("content", "").strip()]
    hyps = [t["content"].strip() for t in gt_data
            if t.get("speaker") == "Agent" and t.get("content", "").strip()]

    mismatches = []
    for ref, hyp in zip(refs, hyps):
        # IMPORTANT: Use GT (hyp) as reference and REF (ref) as hypothesis
        # Also compute explicit substitutions/insertions/deletions for terminal output
        errors, total, diffs = compute_normalized_wer(hyp, ref)
        # Simple token comparison on normalized text to extract S/I/D
        ref_norm = canonicalize_and_normalize(hyp)
        hyp_norm = canonicalize_and_normalize(ref)
        ref_tokens = ref_norm.split()
        hyp_tokens = hyp_norm.split()
        # Build edit ops using difflib-based alignment
        sm = difflib.SequenceMatcher(None, ref_tokens, hyp_tokens)
        substitution_pairs = []
        deletion_words = []
        insertion_words = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == 'replace':
                # Pair words roughly position-wise within the span
                span_len = max(i2 - i1, j2 - j1)
                for k in range(span_len):
                    r = ref_tokens[i1 + k] if i1 + k < i2 else ''
                    h = hyp_tokens[j1 + k] if j1 + k < j2 else ''
                    if r and h:
                        substitution_pairs.append({"reference": r, "hypothesis": h})
                    elif r and not h:
                        deletion_words.append(r)
                    elif h and not r:
                        insertion_words.append(h)
            elif tag == 'delete':
                deletion_words.extend(ref_tokens[i1:i2])
            elif tag == 'insert':
                insertion_words.extend(hyp_tokens[j1:j2])
        if diffs:
            mismatches.append({
                # IMPORTANT: reference is GT, hypothesis is REF
                "reference_sentence": hyp,
                "hypothesis_sentence": ref,
                "word_differences": diffs
            })
        # Print explicit differences for each pair
        print("  Substitutions:")
        for idx, p in enumerate(substitution_pairs, 1):
            print(f"    {idx}. '{p['reference']}' -> '{p['hypothesis']}'")
        print("  Deletions:")
        for idx, w in enumerate(deletion_words, 1):
            print(f"    {idx}. '{w}' — missing in hypothesis")
        print("  Insertions:")
        for idx, w in enumerate(insertion_words, 1):
            print(f"    {idx}. '{w}' — extra in hypothesis")

    with open(output_dir / "mismatches.json", "w", encoding="utf-8") as f:
        json.dump({"call_id": call_dir.name, "mismatches": mismatches}, f, indent=2)

    return refs, hyps

def main():
    # Capture terminal output to file while still printing to console
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with open(OUTPUT_LOG_FILE, "w", encoding="utf-8") as log_file:
        tee = TeeStream(original_stdout, log_file)
        sys.stdout = tee
        sys.stderr = tee
        try:
            start_time = time.time()
            global_refs = []
            global_hyps = []

            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_call_dir, call_dir) for call_dir in CALLS_DIR.iterdir() if call_dir.is_dir()]
                for f in futures:
                    refs, hyps = f.result()
                    global_refs.extend(refs)
                    global_hyps.extend(hyps)
            print(f"Checkpoint: processed {len(global_refs)} utterances so far, elapsed time: {time.time() - start_time:.2f} sec")

            # Calculate global WER by concatenating all utterances and computing WER
            # IMPORTANT: Use concatenated GT as reference and REF as hypothesis
            concatenated_ref = " ".join(global_hyps)
            concatenated_hyp = " ".join(global_refs)
            print(f"Token counts - Ref: {len(concatenated_ref.split())}, Hyp: {len(concatenated_hyp.split())}")
            start_wer_time = time.time()
            errors, total, _ = compute_normalized_wer(concatenated_ref, concatenated_hyp)
            print(f"Global WER computed in {time.time() - start_wer_time:.2f} sec")
            percentage = round((errors / total * 100) if total > 0 else 0, 2)

            with open(GLOBAL_OUTPUT_DIR / "global_wer.json", "w", encoding="utf-8") as f:
                json.dump({
                    "total_errors": errors,
                    "total_words": total,
                    "wer_percentage": percentage
                }, f, indent=2)

            print(f"Processed {len(global_refs)} assistant utterances across {CALLS_DIR}.")
            print(f"Global WER: {errors}/{total} = {percentage}%")
            elapsed_time = time.time() - start_time
            print(f"Total processing time: {elapsed_time:.2f} seconds")
        finally:
            # Restore original stdout/stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr

if __name__ == "__main__":
    main()