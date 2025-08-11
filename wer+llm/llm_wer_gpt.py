import json
import os
from typing import List, Dict, Any, Tuple
import openai
import logging
from dotenv import load_dotenv
import sys
from pathlib import Path
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Resolve log file path and tee for terminal mirroring
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_LOG_FILE = BASE_DIR / "output_gpt.txt"

class TeeStream:
    def __init__(self, *streams):
        self.streams = streams
        self._lock = threading.Lock()
    def write(self, data: str) -> None:
        with self._lock:
            for s in self.streams:
                s.write(data)
                s.flush()
    def flush(self) -> None:
        with self._lock:
            for s in self.streams:
                s.flush()

# ------------------ Phonetic utilities (cached, deterministic) ------------------
from functools import lru_cache

# Optional deps (graceful fallback)
try:
    from metaphone import doublemetaphone  # type: ignore
except Exception:  # pragma: no cover
    doublemetaphone = None  # type: ignore

try:
    from g2p_en import G2p  # type: ignore
except Exception:  # pragma: no cover
    G2p = None  # type: ignore

@lru_cache(maxsize=4096)
def _jw_sim(a: str, b: str) -> float:
    return _jaro_winkler(a, b)

def _jaro_winkler(s1: str, s2: str, p: float = 0.1) -> float:
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
    prefix = 0
    for i in range(min(4, len1, len2)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break
    return jaro + prefix * p * (1 - jaro)

def _levenshtein_list(a, b) -> int:
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
def metaphone_codes(token: str):
    t = (token or "").strip().lower()
    if not t:
        return ("", "")
    if doublemetaphone is None:
        import re as _re
        s = _re.sub(r'[^a-z0-9]', '', t)
        s = s.replace('ph', 'f').replace('gh', 'g').replace('kn', 'n').replace('wr', 'r')
        s = _re.sub(r'[aeiou]+', '', s)
        return (s[:8], "")
    p, a = doublemetaphone(t)
    return (p or "", a or "")

def metaphone_similar(a: str, b: str, jw_threshold: float = 0.92) -> bool:
    pa, aa = metaphone_codes(a)
    pb, ab = metaphone_codes(b)
    if not (pa or aa or pb or ab):
        return False
    if pa and (pa == pb or pa == ab):
        return True
    if aa and (aa == pb or aa == ab):
        return True
    codes_a = [c for c in (pa, aa) if c]
    codes_b = [c for c in (pb, ab) if c]
    return any(_jw_sim(x, y) >= jw_threshold for x in codes_a for y in codes_b)

_g2p = G2p() if 'G2p' in globals() and G2p is not None else None  # type: ignore

@lru_cache(maxsize=4096)
def g2p_arpabet(token: str):
    t = (token or "").strip()
    if not t or _g2p is None:
        return []
    seq = _g2p(t)  # type: ignore
    phones = [p for p in seq if p and p[0].isalpha() and p[0].isupper()]
    return [p.rstrip("012") for p in phones]

def phoneme_similarity(a: str, b: str) -> float:
    pa = g2p_arpabet(a)
    pb = g2p_arpabet(b)
    if not pa or not pb:
        return 0.0
    dist = _levenshtein_list(pa, pb)
    denom = max(len(pa), len(pb))
    return 1.0 - (dist / denom) if denom else 0.0

def sounds_alike(a: str, b: str, meta_jw: float = 0.92, g2p_thresh: float = 0.80) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    if metaphone_similar(a, b, jw_threshold=meta_jw):
        return True
    if phoneme_similarity(a, b) >= g2p_thresh:
        return True
    return False

def tokens_equal_phonetic(a: str, b: str) -> bool:
    a = (a or '').strip()
    b = (b or '').strip()
    if not a and not b:
        return True
    if a == b:
        return True
    return sounds_alike(a.lower(), b.lower())

def compute_phonetic_aware_alignment(ref_tokens, hyp_tokens):
    """DP alignment with phonetic equality treated as exact match (cost 0).
    Returns dict with wer, insertions, deletions, substitutions and ops list.
    """
    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0]*(n+1) for _ in range(m+1)]
    back = [[None]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        dp[i][0] = i
        back[i][0] = 'D'
    for j in range(1, n+1):
        dp[0][j] = j
        back[0][j] = 'I'
    for i in range(1, m+1):
        ri = ref_tokens[i-1]
        for j in range(1, n+1):
            hj = hyp_tokens[j-1]
            sub_cost = 0 if tokens_equal_phonetic(ri, hj) else 1
            del_cost = dp[i-1][j] + 1
            ins_cost = dp[i][j-1] + 1
            rep_cost = dp[i-1][j-1] + sub_cost
            best = min(del_cost, ins_cost, rep_cost)
            dp[i][j] = best
            if best == rep_cost:
                back[i][j] = 'M' if sub_cost == 0 else 'R'
            elif best == del_cost:
                back[i][j] = 'D'
            else:
                back[i][j] = 'I'
    i, j = m, n
    ops = []
    insertions = deletions = substitutions = 0
    while i > 0 or j > 0:
        action = back[i][j]
        if action == 'M':
            ops.append(('match', i-1, j-1))
            i -= 1; j -= 1
        elif action == 'R':
            substitutions += 1
            ops.append(('replace', i-1, j-1))
            i -= 1; j -= 1
        elif action == 'D':
            deletions += 1
            ops.append(('delete', i-1, j))
            i -= 1
        elif action == 'I':
            insertions += 1
            ops.append(('insert', i, j-1))
            j -= 1
        else:
            break
    ops.reverse()
    errors = substitutions + deletions + insertions
    wer = (errors / m) if m > 0 else 0.0
    return {
        'wer': wer,
        'insertions': insertions,
        'deletions': deletions,
        'substitutions': substitutions,
        'ops': ops,
    }

class LLMTranscriptProcessor:
    def __init__(self, openai_api_key: str, model: str = "gpt-4"):
        """
        Initialize the transcript processor with OpenAI API key
        
        Args:
            openai_api_key: OpenAI API key for LLM normalization and WER calculation
            model: OpenAI model to use
        """
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.model = model
        self.normalization_prompt = self._get_normalization_prompt()
        self.wer_prompt = self._get_wer_prompt()

    def _extract_json_object(self, text: str) -> str:
        """Try to extract the first top-level JSON object from an arbitrary string."""
        if not text:
            return ""
        # Strip code fences if present
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        # Fast path: looks like a pure JSON object
        if cleaned.strip().startswith('{') and cleaned.strip().endswith('}'):
            return cleaned.strip()

        # Scan for the first balanced {...}
        brace = 0
        start = -1
        for i, ch in enumerate(cleaned):
            if ch == '{':
                if brace == 0:
                    start = i
                brace += 1
            elif ch == '}':
                if brace > 0:
                    brace -= 1
                    if brace == 0 and start != -1:
                        return cleaned[start:i+1]
        return cleaned

    def _relax_to_json(self, text: str) -> str:
        """Apply tolerant transformations to coerce near-JSON into JSON (best-effort)."""
        import re
        s = text
        # Remove trailing commas before } or ]
        s = re.sub(r",\s*([}\]])", r"\1", s)
        # Convert single-quoted keys to double-quoted
        s = re.sub(r"([,{]\s*)'([^'\n]+)'\s*:", r'\1"\2":', s)
        # Convert single-quoted string values to double-quoted
        s = re.sub(r":\s*'([^'\n]*)'", r': "\1"', s)
        return s

    def _parse_llm_json_result(self, result_text: str, reference: str) -> Dict[str, Any]:
        """Robustly parse the LLM's JSON; return {} on failure."""
        import json
        # 1) Try as-is
        try:
            return json.loads(result_text)
        except Exception:
            pass

        # 2) Extract a plausible JSON object
        extracted = self._extract_json_object(result_text)
        try:
            return json.loads(extracted)
        except Exception:
            pass

        # 3) Relax to JSON and try again
        relaxed = self._relax_to_json(extracted)
        try:
            return json.loads(relaxed)
        except Exception:
            return {}

    def _finalize_metrics(self, result: Dict[str, Any], reference: str) -> Dict[str, Any]:
        """Fill missing fields and compute counts/ratios when absent."""
        if not isinstance(result, dict):
            result = {}
        substitutions_list = result.get("substitution_pairs") or []
        deletions_list = result.get("deletion_words") or []
        insertions_list = result.get("insertion_words") or []

        # Ensure numeric counts exist
        result["substitutions"] = result.get("substitutions") or (len(substitutions_list) if isinstance(substitutions_list, list) else 0)
        result["deletions"] = result.get("deletions") or (len(deletions_list) if isinstance(deletions_list, list) else 0)
        result["insertions"] = result.get("insertions") or (len(insertions_list) if isinstance(insertions_list, list) else 0)

        # Ensure denominator (N)
        if not isinstance(result.get("total_reference_words"), int):
            result["total_reference_words"] = len(reference.split())

        # Ensure WER value
        if not isinstance(result.get("conversational_wer"), (int, float)):
            denom = max(result["total_reference_words"], 1)
            result["conversational_wer"] = (result["substitutions"] + result["deletions"] + result["insertions"]) / denom

        return result
    
    def _get_normalization_prompt(self) -> str:
        """Get the normalization prompt for the LLM"""
        return """**Canonicalize and Normalisation**:
   - Process ONLY the content from bot/assistant speaker turns provided in the input
   - Transliterate all non-Roman scripts to standardized Roman (ASCII) script (e.g., 'मैं' → 'main').
   - Normalize variant spellings and pronunciations to canonical word forms (e.g., 'apki', 'kese', 'rey' → 'aapki', 'kaise', 'ray').
   - Convert all text to lowercase.
   - Remove all punctuation characters.
   - Normalize number expressions carefully: convert digit ↔ word forms for cardinals only ('one' ↔ '1'), but not for ordinals.
   - Collapse all repeated whitespace into a single space.
   - Join all bot utterances with single spaces to create one continuous canonical transcript.

Please normalize the following text and return only the normalized version without any additional explanation:
"""

    def _get_wer_prompt(self) -> str:
        """Get the WER calculation prompt for the LLM"""
        return """You are an expert in conversational quality assessment. Calculate the Word Error Rate (WER) between two normalized transcripts, but ONLY count errors that would affect conversational quality and understanding.

**Instructions for Conversational Quality WER:**
1. DO NOT count as errors:
   - Words that sound phonetically identical (e.g., "rey" vs "ray", "mintra" vs "myntra")
   - Minor spelling variations that don't change meaning
   - Filler words or hesitations that don't affect understanding
   - Different but semantically equivalent expressions

2. DO count as errors:
   - Words that change the meaning of the conversation
   - Missing or added content words that affect understanding
   - Incorrect information that would confuse the user

3. Calculate WER using the formula: WER = (S + D + I) / N
   Where:
   - S = Substitutions (words that change meaning)
   - D = Deletions (missing important words)
   - I = Insertions (extra words that change meaning)
   - N = Total number of words in reference

 4. Provide your analysis in this exact JSON format (ensure valid JSON):
{
    "substitutions": <number>,
    "deletions": <number>,
    "insertions": <number>,
    "total_reference_words": <number>,
    "conversational_wer": <decimal>,
    "explanation": "<brief explanation of what errors were counted and why>",
    "substitution_pairs": [{"reference": "<ref_word_or_phrase>", "hypothesis": "<hyp_word_or_phrase>"}],
    "deletion_words": ["<word_or_phrase_missing_in_hypothesis>", ...],
    "insertion_words": ["<word_or_phrase_extra_in_hypothesis>", ...]
}

Reference (Ground Truth): {reference}
Hypothesis (Prediction): {hypothesis}

Analyze these transcripts and return only the JSON response:"""

    def extract_utterances_by_speaker(self, transcript_data: List[Dict[str, Any]], speaker_label: str) -> str:
        """
        Extract and concatenate utterances from specific speaker
        
        Args:
            transcript_data: List of transcript entries with timestamp, speaker, content
            speaker_label: Speaker to extract ("assistant" or "Agent")
            
        Returns:
            Concatenated utterances from the specified speaker
        """
        utterances = []
        
        for entry in transcript_data:
            if entry.get("speaker") == speaker_label:
                content = entry.get("content", "").strip()
                if content:  # Only add non-empty content
                    utterances.append(content)
        
        return ' '.join(utterances)

    def normalize_with_llm(self, text: str) -> str:
        """
        Normalize text using LLM according to the specified canonicalization rules
        
        Args:
            text: Raw text to normalize
            
        Returns:
            Normalized text
        """
        if not text.strip():
            return ""
            
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a text normalization expert. Follow the exact instructions provided for canonicalization and normalization."},
                    {"role": "user", "content": f"{self.normalization_prompt}\n\nText to normalize: {text}"}
                ],
                max_tokens=2000,
                temperature=0.1
            )
            
            normalized_text = response.choices[0].message.content.strip()
            logger.info(f"Successfully normalized text of length {len(text)} to {len(normalized_text)}")
            return normalized_text
            
        except Exception as e:
            logger.error(f"Error normalizing text with LLM: {e}")
            return text.lower().strip()

    def calculate_conversational_wer_with_llm(self, reference: str, hypothesis: str) -> Dict[str, Any]:
        """
        Calculate conversational quality WER using LLM
        
        Args:
            reference: Reference transcript (ground truth)
            hypothesis: Hypothesis transcript (predicted)
            
        Returns:
            Dictionary with WER metrics and analysis
        """
        if not reference.strip() or not hypothesis.strip():
            return {
                "substitutions": 0,
                "deletions": 0,
                "insertions": 0,
                "total_reference_words": len(reference.split()) if reference.strip() else 0,
                "conversational_wer": 0.0,
                "explanation": "One or both transcripts are empty"
            }
        
        try:
            # Avoid Python format brace conflicts by manual placeholder replacement
            prompt = self.wer_prompt.replace("{reference}", reference).replace("{hypothesis}", hypothesis)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in conversational quality assessment and WER calculation. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            # Force the model to return pure JSON by adding a system guard
            result_text = response.choices[0].message.content.strip()

            # Robust parsing
            result = self._parse_llm_json_result(result_text, reference)
            if not result:
                # One retry with stricter instruction
                retry_prompt = prompt + "\n\nReturn strictly minified JSON per the schema. Do not include any commentary or markdown fences."
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "Return only valid JSON for the requested schema. No extra text."},
                        {"role": "user", "content": retry_prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.0
                )
                result_text = response.choices[0].message.content.strip()
                result = self._parse_llm_json_result(result_text, reference)

            if result:
                result = self._finalize_metrics(result, reference)
                logger.info(f"Successfully calculated conversational WER: {result.get('conversational_wer', 0):.4f}")
                return result
            else:
                raise ValueError("Could not parse JSON response")
                    
        except Exception as e:
            logger.error(f"Error calculating WER with LLM: {e}")
            # Fallback calculation
            ref_words = len(reference.split())
            hyp_words = len(hypothesis.split())
            fallback_wer = abs(ref_words - hyp_words) / max(ref_words, 1)
            
            return {
                "substitutions": 0,
                "deletions": max(0, ref_words - hyp_words),
                "insertions": max(0, hyp_words - ref_words),
                "total_reference_words": ref_words,
                "conversational_wer": fallback_wer,
                "explanation": f"Fallback calculation due to LLM error: {str(e)}"
            }

    def process_call_transcripts(self, calls_dir: str) -> Dict[str, Dict[str, Any]]:
        """
        Process all call transcripts in the specified directory
        
        Args:
            calls_dir: Directory containing call folders with ref_transcript and gt_transcript files
            
        Returns:
            Dictionary with results for each call
        """
        results = {}
        
        if not os.path.exists(calls_dir):
            logger.error(f"Directory {calls_dir} does not exist")
            return results
        
        # Iterate through all call directories
        for call_dir in os.listdir(calls_dir):
            call_path = os.path.join(calls_dir, call_dir)
            
            if not os.path.isdir(call_path):
                continue
            
            logger.info(f"Processing call: {call_dir}")
            
            ref_path = os.path.join(call_path, 'ref_transcript.json')
            gt_path = os.path.join(call_path, 'gt_transcript.json')
            
            if not (os.path.exists(ref_path) and os.path.exists(gt_path)):
                logger.warning(f"Missing transcript files for call {call_dir}")
                continue
            
            try:
                # Load transcripts
                with open(ref_path, 'r', encoding='utf-8') as f:
                    ref_data = json.load(f)
                
                with open(gt_path, 'r', encoding='utf-8') as f:
                    gt_data = json.load(f)
                
                # Extract assistant/agent utterances
                ref_utterances = self.extract_utterances_by_speaker(ref_data, "assistant")
                gt_utterances = self.extract_utterances_by_speaker(gt_data, "Agent")
                
                if not ref_utterances.strip() or not gt_utterances.strip():
                    logger.warning(f"No assistant/agent utterances found for call {call_dir}")
                    continue
                
                # Normalize using LLM
                logger.info(f"Normalizing reference transcript for call {call_dir}")
                normalized_ref = self.normalize_with_llm(ref_utterances)
                
                logger.info(f"Normalizing ground truth transcript for call {call_dir}")
                normalized_gt = self.normalize_with_llm(gt_utterances)
                
                # Calculate conversational WER using LLM
                logger.info(f"Calculating conversational WER for call {call_dir}")
                # IMPORTANT: Use GT as the reference and REF as hypothesis
                wer_metrics = self.calculate_conversational_wer_with_llm(normalized_gt, normalized_ref)
                
                results[call_dir] = {
                    'raw_ref_utterances': ref_utterances,
                    'raw_gt_utterances': gt_utterances,
                    'normalized_ref': normalized_ref,
                    'normalized_gt': normalized_gt,
                    'wer_metrics': wer_metrics
                }
                
                logger.info(f"Call {call_dir} - Conversational WER: {wer_metrics.get('conversational_wer', 0):.4f}")
                
            except Exception as e:
                logger.error(f"Error processing call {call_dir}: {e}")
                continue
        
        return results

    def calculate_global_metrics(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate global conversational WER metrics across all calls
        
        Args:
            results: Results from process_call_transcripts
            
        Returns:
            Global metrics dictionary
        """
        if not results:
            return {}
        
        # Collect all individual WER scores
        conversational_wers = []
        total_substitutions = 0
        total_deletions = 0
        total_insertions = 0
        total_reference_words = 0
        
        for call_id, call_data in results.items():
            wer_metrics = call_data.get('wer_metrics', {})
            if 'conversational_wer' in wer_metrics:
                conversational_wers.append(wer_metrics['conversational_wer'])
                total_substitutions += wer_metrics.get('substitutions', 0)
                total_deletions += wer_metrics.get('deletions', 0)
                total_insertions += wer_metrics.get('insertions', 0)
                total_reference_words += wer_metrics.get('total_reference_words', 0)
        
        # Calculate aggregate metrics
        avg_conversational_wer = sum(conversational_wers) / len(conversational_wers) if conversational_wers else 0
        
        # Calculate global WER using total counts
        global_conversational_wer = (total_substitutions + total_deletions + total_insertions) / max(total_reference_words, 1)
        
        # Also calculate WER on concatenated transcripts for comparison
        all_ref = []
        all_gt = []
        
        for call_id, call_data in results.items():
            if 'normalized_ref' in call_data and 'normalized_gt' in call_data:
                all_ref.append(call_data['normalized_ref'])
                all_gt.append(call_data['normalized_gt'])
        
        # Calculate WER on concatenated transcripts
        global_ref = ' '.join(all_ref)
        global_gt = ' '.join(all_gt)
        
        # IMPORTANT: Use concatenated GT as reference and REF as hypothesis
        concatenated_wer_metrics = self.calculate_conversational_wer_with_llm(global_gt, global_ref)
        
        return {
            'total_calls': len(results),
            'average_conversational_wer': avg_conversational_wer,
            'global_conversational_wer_aggregate': global_conversational_wer,
            'global_conversational_wer_concatenated': concatenated_wer_metrics.get('conversational_wer', 0),
            'total_substitutions': total_substitutions,
            'total_deletions': total_deletions,
            'total_insertions': total_insertions,
            'total_reference_words': total_reference_words,
            'concatenated_analysis': concatenated_wer_metrics
        }

# Usage example
def main():
    # Load environment variables
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Tee stdout/stderr to file while preserving console output
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    OUTPUT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_LOG_FILE, "w", encoding="utf-8") as log_file:
        tee = TeeStream(original_stdout, log_file)
        sys.stdout = tee
        sys.stderr = tee
        # Also add a file handler so logging module writes to the same file
        file_handler = logging.FileHandler(OUTPUT_LOG_FILE, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        try:
            # Initialize processor
            processor = LLMTranscriptProcessor(
                openai_api_key=openai_api_key,
                model="gpt-4o"
            )

            # Process all calls (resolve relative to this file's folder)
            calls_directory = str(BASE_DIR / "calls")
            results = processor.process_call_transcripts(calls_directory)

            # Calculate global metrics
            global_metrics = processor.calculate_global_metrics(results)

            # Print results
            print("\n" + "="*60)
            print("GLOBAL CONVERSATIONAL QUALITY WER METRICS")
            print("="*60)

            for metric, value in global_metrics.items():
                if isinstance(value, float):
                    print(f"{metric}: {value:.4f}")
                elif isinstance(value, dict):
                    print(f"{metric}:")
                    for k, v in value.items():
                        if isinstance(v, float):
                            print(f"  {k}: {v:.4f}")
                        else:
                            print(f"  {k}: {v}")
                else:
                    print(f"{metric}: {value}")

            # Print per-call results
            print("\n" + "="*60)
            print("PER-CALL CONVERSATIONAL WER RESULTS")
            print("="*60)

            for call_id, call_data in results.items():
                wer_metrics = call_data.get('wer_metrics', {})
                print(f"\nCall: {call_id}")
                print(f"  Conversational WER: {wer_metrics.get('conversational_wer', 0):.4f}")
                print(f"  Substitutions: {wer_metrics.get('substitutions', 0)}")
                print(f"  Deletions: {wer_metrics.get('deletions', 0)}")
                print(f"  Insertions: {wer_metrics.get('insertions', 0)}")
                print(f"  Reference Words: {wer_metrics.get('total_reference_words', 0)}")
                if wer_metrics.get('explanation'):
                    print(f"  Analysis: {wer_metrics['explanation']}")

            # Save results to file
            output_file = "global_wer_gpt_llm.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'global_metrics': global_metrics,
                    'per_call_results': results
                }, f, indent=2, ensure_ascii=False)

            print(f"\nResults saved to {output_file}")
        finally:
            logger.removeHandler(file_handler)
            sys.stdout = original_stdout
            sys.stderr = original_stderr

if __name__ == "__main__":
    main()