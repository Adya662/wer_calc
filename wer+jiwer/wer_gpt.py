#!/usr/bin/env python3
import os
import sys
import json
import time
import shutil
import subprocess
import glob
from pathlib import Path
from dotenv import load_dotenv
import openai
from openai import OpenAI
import Levenshtein
from jiwer import process_words, wer as jiwer_wer
import jellyfish
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import threading
from functools import lru_cache

# Resolve paths relative to this file (for saving global outputs in wer+jiwer root)
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_LOG_FILE = BASE_DIR / "output_gpt.txt"

class TeeStream:
    """
    Thread-safe tee for mirroring stdout/stderr to a log file while keeping console output.
    """
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
    """Simple Jaro-Winkler similarity (0..1)."""
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
    """DP alignment with phonetic equality as exact match (cost 0).
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

def get_openai_client():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in .env")
    return OpenAI(api_key=api_key)

# Global variables for tracking across all calls
all_entity_types = set()
wer_values = []
all_canonical_ref_texts = []  # For global WER calculation
all_canonical_hyp_texts = []  # For global WER calculation

# ------------------------------------------------------------------------------
# Bot Utterance Filtering Function
# ------------------------------------------------------------------------------
def filter_bot_utterances(transcript_text: str, bot_speaker_label: str):
    """
    Extract only bot utterances from transcript JSON for precise WER calculation.
    
    Args:
        transcript_text: Raw JSON transcript text
        bot_speaker_label: Either "assistant" (for ref) or "Agent" (for gt)
    
    Returns:
        Filtered transcript JSON containing only bot utterances
    """
    try:
        transcript_data = json.loads(transcript_text)
        
        # Filter for only bot utterances
        bot_utterances = []
        for utterance in transcript_data:
            if isinstance(utterance, dict) and utterance.get("speaker") == bot_speaker_label:
                bot_utterances.append(utterance)
        
        print(f"🎯 Filtered {len(bot_utterances)} {bot_speaker_label} utterances from {len(transcript_data)} total utterances")
        
        # Return filtered transcript as JSON string
        return json.dumps(bot_utterances, ensure_ascii=False, indent=2)
        
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing transcript JSON: {e}")
        # If JSON parsing fails, try to extract bot utterances from raw text
        return extract_bot_utterances_from_text(transcript_text, bot_speaker_label)
    except Exception as e:
        print(f"❌ Error filtering bot utterances: {e}")
        raise

def extract_bot_utterances_from_text(text: str, bot_speaker_label: str):
    """
    Fallback method to extract bot utterances from malformed JSON or text format.
    """
    import re
    
    # Try to find JSON-like structures with the bot speaker
    pattern = rf'"speaker":\s*"{re.escape(bot_speaker_label)}"[^}}]*"content":\s*"([^"]*)"'
    matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
    
    if matches:
        # Create a simple JSON structure with just the content
        bot_utterances = [{"speaker": bot_speaker_label, "content": match} for match in matches]
        print(f"🎯 Extracted {len(bot_utterances)} {bot_speaker_label} utterances using regex fallback")
        return json.dumps(bot_utterances, ensure_ascii=False, indent=2)
    else:
        print(f"⚠️  No {bot_speaker_label} utterances found in text")
        return json.dumps([], ensure_ascii=False, indent=2)

# ------------------------------------------------------------------------------
# Core WER Processing Function (from wer1.py)
# ------------------------------------------------------------------------------
def process_transcripts(folder: str, ref_name: str, hyp_name: str):
    """Process individual transcript pair and return canonical texts for global WER"""
    client = get_openai_client()
    
    folder = Path(folder)
    ref_path = folder / ref_name
    hyp_path = folder / hyp_name
    
    # Read and filter transcripts for only bot utterances
    ref_text = filter_bot_utterances(ref_path.read_text(encoding="utf-8"), "assistant")
    hyp_text = filter_bot_utterances(hyp_path.read_text(encoding="utf-8"), "Agent")
    
    # LLM prompt for canonicalization
    prompt = (
        "Act as a world-class multilingual transcript cleaner and named entity recognition (NER) specialist, optimized for voice assistant systems.\n"
        "Given the following instructions, canonicalize the provided text only for speaker labels - 'assistant' or 'Agent' through transliteration into a standardized Roman script form, then identify and extract named entities comprehensively.\n\n"
        "## Task Overview\n"
        "You have two main tasks:\n"
        "1. **Canonicalize and Normalisation**:\n"
        "   - Transliterate all non-Roman scripts to standardized Roman (ASCII) script (e.g., 'मैं' → 'main').\n"
        "   - Normalize variant spellings and pronunciations to canonical word forms (e.g., 'apki', 'kese', 'rey' → 'aapki', 'kaise', 'ray').\n"
        "   - Convert all text to lowercase.\n"
        "   - Remove all punctuation characters.\n"
        "   - Normalize number expressions carefully: convert digit ↔ word forms for cardinals only ('one' ↔ '1'), but not for ordinals.\n"
        "   - Collapse all repeated whitespace into a single space.\n\n"
        "2. **Named Entity Recognition (NER)**:\n"
        "   Identify all named entities in the canonicalized sentence. Extract each entity explicitly according to the entity types provided below.\n\n"
        "## Entity Types to Extract\n"
        "- PERSON: Names of individuals\n"
        "- ORGANIZATION (ORG): Companies, agencies, institutions\n"
        "- LOCATION/GPE: Cities, states, countries, geographic areas\n"
        "- DATE/TIME: Dates, time expressions\n"
        "- MONEY: Monetary values, currencies, financial amounts\n"
        "- PRODUCT: Products, services, brands\n"
        "- EVENT: Events or notable occasions\n"
        "- LANGUAGE: Languages\n"
        "- NORP: Nationalities, religious or political groups\n"
        "- FAC: Facilities (buildings, airports, highways)\n"
        "- OTHER: Any other proper nouns not fitting above categories\n\n"
        "## Output Format\n"
        "Provide your answer strictly as a JSON object with two keys:\n"
        "{\n"
        "  \"canonical_transcript\": \"<canonicalized transliterated sentence here>\",\n"
        "  \"entities\": [\n"
        "    {\n"
        "      \"index\": <word_index_starting_from_0>,\n"
        "      \"value\": \"<exact_entity_text>\",\n"
        "      \"type\": \"<entity_type_from_list_above>\"\n"
        "    },\n"
        "    ...\n"
        "  ]\n"
        "}\n\n"
        "Please apply this transformation only to the agent or assistant speaker turns. The input will be a transcript in JSON format.\n\n"
        "Make sure all normalisation steps are followed strictly to give an output field \"canonical_transcript\" in the JSON we are expecting."
    )
    
    out_dir = folder
    
    # Parallel LLM calls for reference and hypothesis
    def call_llm(transcript):
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                return client.chat.completions.create(
                    model="gpt-4",
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": transcript}
                    ],
                    timeout=120
                ).choices[0].message.content
            except openai.APITimeoutError:
                if attempt < max_retries:
                    wait = 2 ** attempt
                    print(f"Warning: API timeout, retrying in {wait}s (attempt {attempt}/{max_retries})")
                    time.sleep(wait)
                else:
                    raise
    
    print("Sending transcripts to LLM in parallel...")
    with ThreadPoolExecutor() as executor:
        future_ref = executor.submit(call_llm, ref_text)
        future_hyp = executor.submit(call_llm, hyp_text)
        resp_ref = future_ref.result()
        resp_hyp = future_hyp.result()
    
    # Save and validate responses
    canon_ref_path = out_dir / "canon_ref_transcript_gpt_lib.json"
    canon_ref_path.write_text(json.dumps(json.loads(resp_ref.strip()), indent=2), encoding="utf-8")
    print(f"Saved canonical reference -> {canon_ref_path}")
    
    canon_hyp_path = out_dir / "canon_gt_transcript_gpt_lib.json"
    canon_hyp_path.write_text(json.dumps(json.loads(resp_hyp.strip()), indent=2), encoding="utf-8")
    print(f"Saved canonical hypothesis -> {canon_hyp_path}")
    
    # Load back and process
    print("Loading canonical transcripts for WER calculation...")
    t_script1 = json.loads(canon_hyp_path.read_text(encoding="utf-8"))
    t_script2 = json.loads(canon_ref_path.read_text(encoding="utf-8"))
    
    # Validate canonical transcript fields
    assert isinstance(t_script1, dict) and "canonical_transcript" in t_script1, "Missing canonical_transcript in hypothesis"
    assert isinstance(t_script2, dict) and "canonical_transcript" in t_script2, "Missing canonical_transcript in reference"
    assert isinstance(t_script1["canonical_transcript"], str), "Hypothesis transcript must be string"
    assert isinstance(t_script2["canonical_transcript"], str), "Reference transcript must be string"
    
    # Clean transcripts
    def clean_transcript(text):
        import re
        text = re.sub(r"\b(user|customer|user:|customer:|user )[:]? ?", "", text)
        return text
    
    t_script1["canonical_transcript"] = clean_transcript(t_script1["canonical_transcript"])
    t_script2["canonical_transcript"] = clean_transcript(t_script2["canonical_transcript"])
    
    # Tokenize and compute WER
    # IMPORTANT: Use GT (t_script1) as reference and REF (t_script2) as hypothesis
    ref_tokens = t_script1["canonical_transcript"].split()
    hyp_tokens = t_script2["canonical_transcript"].split()
    
    t_start = time.time()
    
    # Compute WER using phonetic-aware DP (treat sounds-alike as equal)
    ref_str = " ".join(ref_tokens)
    hyp_str = " ".join(hyp_tokens)
    pa = compute_phonetic_aware_alignment(ref_tokens, hyp_tokens)
    wer_value = pa['wer']
    insertions = pa['insertions']
    deletions = pa['deletions']
    substitutions = pa['substitutions']
    
    # Generate mismatches and collect explicit S/I/D lists
    # Build mismatches from phonetic-aware ops
    ops = pa['ops']
    mismatches = []
    window_size = 5
    substitution_pairs = []
    deletion_words = []
    insertion_words = []
    
    for op, i, j in ops:
        if op == "match":
            continue
        elif op == "delete":
            ref_word = ref_tokens[i]
            hyp_word = ""
            ref_context = " ".join(ref_tokens[max(0, i-window_size):i+window_size+1])
            hyp_context = ""
            deletion_words.append(ref_word)
            mismatches.append({
                "operation": op,
                "index_ref": i,
                "index_hyp": j,
                "ref_word": ref_word,
                "hyp_word": hyp_word,
                "ref_context": ref_context,
                "hyp_context": hyp_context
            })
        elif op == "insert":
            ref_word = ""
            hyp_word = hyp_tokens[j]
            ref_context = ""
            hyp_context = " ".join(hyp_tokens[max(0, j-window_size):j+window_size+1])
            insertion_words.append(hyp_word)
            mismatches.append({
                "operation": op,
                "index_ref": i,
                "index_hyp": j,
                "ref_word": ref_word,
                "hyp_word": hyp_word,
                "ref_context": ref_context,
                "hyp_context": hyp_context
            })
        elif op == "replace":
            ref_word = ref_tokens[i]
            hyp_word = hyp_tokens[j]
            ref_context = " ".join(ref_tokens[max(0, i-window_size):i+window_size+1])
            hyp_context = " ".join(hyp_tokens[max(0, j-window_size):j+window_size+1])
            substitution_pairs.append({"reference": ref_word, "hypothesis": hyp_word})
            mismatches.append({
                "operation": op,
                "index_ref": i,
                "index_hyp": j,
                "ref_word": ref_word,
                "hyp_word": hyp_word,
                "ref_context": ref_context,
                "hyp_context": hyp_context
            })
    
    # Save mismatches
    mismatches_log = {
        "wer": wer_value,
        "insertions": insertions,
        "deletions": deletions,
        "substitutions": substitutions,
        "mismatches": mismatches,
        "substitution_pairs": substitution_pairs,
        "deletion_words": deletion_words,
        "insertion_words": insertion_words
    }
    mismatches_path = out_dir / "wer_mismatches_gpt_lib.json"
    with open(mismatches_path, "w", encoding="utf-8") as mmf:
        json.dump(mismatches_log, mmf, indent=2, ensure_ascii=False)
    print(f"Logged {len(mismatches)} mismatches and measures to {mismatches_path}")
    
    # Process entities
    entities_gt = t_script1["entities"]
    entities_ref = t_script2["entities"]
    
    unique_entities_gt = {(e['value'], e['type']) for e in entities_gt}
    unique_entities_ref = {(e['value'], e['type']) for e in entities_ref}
    
    concerned_entities_gt = [e for e in entities_gt if e['type'] in ('PERSON', 'ORG', 'PRODUCT')]
    concerned_entities_ref = [e for e in entities_ref if e['type'] in ('PERSON', 'ORG', 'PRODUCT')]
    
    unique_concerned_ner_gt = {(e['value'], e['type']) for e in concerned_entities_gt}
    unique_concerned_ner_ref = {(e['value'], e['type']) for e in concerned_entities_ref}
    
    t_end = time.time()
    time_taken = t_end - t_start
    
    # Create results
    results = {
        "wer": wer_value,
        "wer_value": wer_value,
        "entities_gt": entities_gt,
        "entities_ref": entities_ref,
        "unique_entities_gt": list(unique_entities_gt),
        "unique_entities_ref": list(unique_entities_ref),
        "concerned_entities_gt": concerned_entities_gt,
        "concerned_entities_ref": concerned_entities_ref,
        "unique_concerned_ner_gt": list(unique_concerned_ner_gt),
        "unique_concerned_ner_ref": list(unique_concerned_ner_ref),
        "time_taken": time_taken,
        "ref_tokens": ref_tokens,
        "hyp_tokens": hyp_tokens,
        "mismatches_count": len(mismatches)
    }
    
    # Save results
    results_path = out_dir / "wer+eer_gpt_lib.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved comprehensive results -> {results_path}")
    
    # Print summary
    print(f"\nResults summary:")
    print(f"  WER: {wer_value:.4f}")
    print(f"  Total GT NERs: {len(entities_gt)}")
    print(f"  Total Ref NERs: {len(entities_ref)}")
    print(f"  Unique GT NERs: {len(unique_entities_gt)}")
    print(f"  Unique Ref NERs: {len(unique_entities_ref)}")
    print(f"  Total Concerned GT NERs (PERSON/ORG): {len(concerned_entities_gt)}")
    print(f"  Total Concerned Ref NERs (PERSON/ORG): {len(concerned_entities_ref)}")
    print(f"  Unique Concerned GT NERs: {len(unique_concerned_ner_gt)}")
    print(f"  Unique Concerned Ref NERs: {len(unique_concerned_ner_ref)}")
    print(f"  Time taken: {time_taken:.6f} seconds\n")
    # Print explicit differences per call in terminal
    print("  Substitutions:")
    for idx, p in enumerate(substitution_pairs, 1):
        print(f"    {idx}. '{p['reference']}' -> '{p['hypothesis']}'")
    print("  Deletions:")
    for idx, w in enumerate(deletion_words, 1):
        print(f"    {idx}. '{w}' — missing in hypothesis")
    print("  Insertions:")
    for idx, w in enumerate(insertion_words, 1):
        print(f"    {idx}. '{w}' — extra in hypothesis")
    
    # Return canonical texts for global WER calculation
    return ref_str, hyp_str, results

# ------------------------------------------------------------------------------
# CSV Logging Function
# ------------------------------------------------------------------------------
def log_call_metrics(*args, **kwargs):
    # CSV logging removed by request; keep stub to avoid breaking callers.
    return None

# ------------------------------------------------------------------------------
# Process Single Call (from run_wer1.py)
# ------------------------------------------------------------------------------
def process_single_call(call_dir: str):
    """Process a single call directory"""
    global wer_values, all_entity_types, all_canonical_ref_texts, all_canonical_hyp_texts
    
    start = time.time()
    cd = Path(call_dir)
    print(f"🚀 [{cd.name}] started at {time.strftime('%X')}")
    
    out = cd / "output"
    out.mkdir(exist_ok=True)
    
    # Copy transcripts
    for name in ("ref_transcript.json", "gt_transcript.json"):
        if (cd / name).exists():
            shutil.copy(cd / name, out / name)
    
    # Process transcripts and get canonical texts
    try:
        ref_canonical, hyp_canonical, results = process_transcripts(
            str(out), "ref_transcript.json", "gt_transcript.json"
        )
        
        # Add to global collections for overall WER calculation
        all_canonical_ref_texts.append(ref_canonical)
        all_canonical_hyp_texts.append(hyp_canonical)
        
        # Move and rename result file
        src = out / "wer1_eval.json"
        dst = out / "wer+eer.json"
        if src.exists():
            shutil.move(str(src), str(dst))
        
        # Extract metrics
        wer_value = results.get("wer", results.get("wer_value", 0.0))
        wer_values.append(wer_value)
        
        entities_gt = results.get("entities_gt", results.get("entities", []))
        entities_ref = results.get("entities_ref", entities_gt)
        
        unique_entities_gt = {(e['value'], e['type']) for e in entities_gt}
        unique_entities_ref = {(e['value'], e['type']) for e in entities_ref}
        
        concerned_entities_gt = [e for e in entities_gt if e['type'] in ('PERSON', 'ORG')]
        concerned_entities_ref = [e for e in entities_ref if e['type'] in ('PERSON', 'ORG')]
        
        unique_concerned_ner_gt = {(e['value'], e['type']) for e in concerned_entities_gt}
        unique_concerned_ner_ref = {(e['value'], e['type']) for e in concerned_entities_ref}
        
        # Log metrics to CSV
        log_call_metrics(out, wer_value, entities_gt, entities_ref,
                        unique_entities_gt, unique_entities_ref,
                        concerned_entities_gt, concerned_entities_ref,
                        unique_concerned_ner_gt, unique_concerned_ner_ref)
        
        # Update global entity types
        all_entity_types.update(e["type"] for e in entities_gt)
        all_entity_types.update(e["type"] for e in entities_ref)
        
        elapsed = time.time() - start
        print(f"🏁 [{cd.name}] completed at {time.strftime('%X')}, took {elapsed:.2f}s")
        
    except Exception as e:
        print(f"❌ [{cd.name}] failed: {e}")
        raise

# ------------------------------------------------------------------------------
# Summary Generation (from generate_summary.py)
# ------------------------------------------------------------------------------
def generate_summary(calls_root: Path):
    """Generate summary CSV from all processed calls"""
    SUMMARY_COLUMNS = [
        "Call",
        "WER",
        "Total GT NERs",
        "Total Ref NERs",
        "Unique GT NERs",
        "Unique Ref NERs",
        "Total Concerned GT NERs (PERSON/ORG)",
        "Total Concerned Ref NERs (PERSON/ORG)",
        "Unique Concerned GT NERs",
        "Unique Concerned Ref NERs",
    ]
    
    rows = []
    for call_dir in sorted(calls_root.iterdir()):
        if not call_dir.is_dir():
            continue
        
        out_dir = call_dir / "output"
        csv_files = sorted(glob.glob(str(out_dir / "*.csv")))
        
        if not csv_files:
            continue
        
        last_rows = []
        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path)
                if df.shape[0] == 0:
                    continue
                last_rows.append(df.iloc[-1])
            except Exception as e:
                print(f"Warning: Could not read {csv_path}: {e}")
                continue
        
        if not last_rows:
            continue
        
        # Use the last row from the first CSV file
        full = last_rows[0] if len(last_rows) == 1 else pd.concat([row.to_dict() for row in last_rows], axis=1).iloc[:, 0]
        
        summary = {
            "Call": call_dir.name,
            "WER": full.get("WER", pd.NA),
            "Total GT NERs": full.get("Total GT NERs", pd.NA),
            "Total Ref NERs": full.get("Total Ref NERs", pd.NA),
            "Unique GT NERs": full.get("Unique GT NERs", pd.NA),
            "Unique Ref NERs": full.get("Unique Ref NERs", pd.NA),
            "Total Concerned GT NERs (PERSON/ORG)": full.get("Total Concerned GT NERs (PERSON/ORG)", pd.NA),
            "Total Concerned Ref NERs (PERSON/ORG)": full.get("Total Concerned Ref NERs (PERSON/ORG)", pd.NA),
            "Unique Concerned GT NERs": full.get("Unique Concerned GT NERs", pd.NA),
            "Unique Concerned Ref NERs": full.get("Unique Concerned Ref NERs", pd.NA),
        }
        rows.append(summary)
    
    summary_df = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    return summary_df

# ------------------------------------------------------------------------------
# Global WER Calculation
# ------------------------------------------------------------------------------
def calculate_global_wer():
    """Calculate single global WER score from all canonical transcripts"""
    global all_canonical_ref_texts, all_canonical_hyp_texts
    
    if not all_canonical_ref_texts or not all_canonical_hyp_texts:
        print("⚠️  No canonical texts available for global WER calculation")
        return None
    
    # Join all bot utterances from entire canonicalized dataset with spaces
    global_ref_text = " ".join(all_canonical_ref_texts)
    global_hyp_text = " ".join(all_canonical_hyp_texts)
    
    print(f"🌍 Calculating Global WER...")
    print(f"   Total reference text length: {len(global_ref_text)} characters")
    print(f"   Total hypothesis text length: {len(global_hyp_text)} characters")
    
    # Calculate global WER
    try:
        global_wer_output = process_words(global_ref_text, global_hyp_text)
        global_wer = global_wer_output.wer
        
        print(f"🎯 Global WER (entire system performance): {global_wer:.4f}")
        
        # Save global WER details
        global_wer_details = {
            "global_wer": global_wer,
            "global_insertions": global_wer_output.insertions,
            "global_deletions": global_wer_output.deletions,
            "global_substitutions": global_wer_output.substitutions,
            "total_calls_processed": len(all_canonical_ref_texts),
            "total_ref_characters": len(global_ref_text),
            "total_hyp_characters": len(global_hyp_text),
            "individual_call_wers": wer_values
        }
        
        return global_wer, global_wer_details
        
    except Exception as e:
        print(f"❌ Error calculating global WER: {e}")
        return None, None

# ------------------------------------------------------------------------------
# Main Function
# ------------------------------------------------------------------------------
def main():
    """Main execution function"""
    global wer_values, all_entity_types, all_canonical_ref_texts, all_canonical_hyp_texts
    
    total_start = time.time()
    # Tee terminal output to file
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    OUTPUT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_LOG_FILE, "w", encoding="utf-8") as log_file:
        tee = TeeStream(original_stdout, log_file)
        sys.stdout = tee
        sys.stderr = tee
        try:
            if len(sys.argv) != 2:
                print("Usage: python integrated_wer_system.py <calls_directory>")
                sys.exit(1)
            
            calls_root = Path(sys.argv[1])
            
            if not calls_root.exists() or not calls_root.is_dir():
                print(f"❌ Invalid calls directory: {calls_root}")
                sys.exit(1)
            
            print(f"🚀 Starting WER processing for all calls in: {calls_root}")
            print("=" * 80)
            
            # Process all call directories
            processed_calls = 0
            failed_calls = 0
            
            for call_dir in sorted(calls_root.iterdir()):
                if not call_dir.is_dir():
                    continue
                
                # Check if required files exist
                if not ((call_dir / "ref_transcript.json").exists() and 
                        (call_dir / "gt_transcript.json").exists()):
                    print(f"⚠️  Skipping {call_dir.name}: Missing required transcript files")
                    continue
                
                try:
                    process_single_call(str(call_dir))
                    processed_calls += 1
                except Exception as e:
                    print(f"❌ [{call_dir.name}] failed: {e}")
                    failed_calls += 1
            
            print("=" * 80)
            print(f"📈 Processing Summary:")
            print(f"   Processed calls: {processed_calls}")
            print(f"   Failed calls: {failed_calls}")
            
            if all_entity_types:
                print("\n🏷️  All entity types encountered across all calls:")
                for entity_type in sorted(all_entity_types):
                    print(f"   - {entity_type}")
            
            # Calculate average WER across individual calls
            if wer_values:
                avg_wer = sum(wer_values) / len(wer_values)
                print(f"\n📊 Average WER across {len(wer_values)} calls: {avg_wer:.4f}")
            
            # Calculate Global WER (NEW FEATURE)
            global_wer, global_wer_details = calculate_global_wer()
            
            # Generate summary report
            print(f"\n📋 Generating summary report...")
            try:
                summary_df = generate_summary(calls_root)
                summary_path = BASE_DIR / "global_wer_summary_gpt_lib.csv"
                summary_df.to_csv(summary_path, index=False)
                print(f"✅ Summary report saved to: {summary_path}")
            except Exception as e:
                print(f"⚠️  Could not generate summary report: {e}")
            
            # Save global WER details
            if global_wer_details:
                global_wer_path = BASE_DIR / "global_wer_report_gpt_lib.json"
                with open(global_wer_path, "w", encoding="utf-8") as f:
                    json.dump(global_wer_details, f, indent=2, ensure_ascii=False)
                print(f"🌍 Global WER report saved to: {global_wer_path}")
            
            total_time = time.time() - total_start
            print(f"\n⏱️  Total processing time: {total_time:.2f} seconds")
            print("🎉 Processing complete!")
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

if __name__ == "__main__":
    main()