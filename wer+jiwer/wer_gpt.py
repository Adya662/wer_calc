#!/usr/bin/env python3
import os
import sys
import json
import time
import shutil
import subprocess
import glob
import csv
from pathlib import Path
from dotenv import load_dotenv
import openai
from openai import OpenAI
import Levenshtein
from jiwer import process_words, wer as jiwer_wer
import jellyfish
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

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
    
    print("⚙️  Sending transcripts to LLM in parallel...")
    with ThreadPoolExecutor() as executor:
        future_ref = executor.submit(call_llm, ref_text)
        future_hyp = executor.submit(call_llm, hyp_text)
        resp_ref = future_ref.result()
        resp_hyp = future_hyp.result()
    
    # Save and validate responses
    canon_ref_path = out_dir / "canon_ref_transcript.json"
    canon_ref_path.write_text(json.dumps(json.loads(resp_ref.strip()), indent=2), encoding="utf-8")
    print(f"✅ Saved canonical reference → {canon_ref_path}")
    
    canon_hyp_path = out_dir / "canon_gt_transcript.json"
    canon_hyp_path.write_text(json.dumps(json.loads(resp_hyp.strip()), indent=2), encoding="utf-8")
    print(f"✅ Saved canonical hypothesis → {canon_hyp_path}")
    
    # Load back and process
    print("🔄 Loading canonical transcripts for WER calculation...")
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
    ref_tokens = t_script2["canonical_transcript"].split()
    hyp_tokens = t_script1["canonical_transcript"].split()
    
    t_start = time.time()
    
    # Compute WER using jiwer
    ref_str = " ".join(ref_tokens)
    hyp_str = " ".join(hyp_tokens)
    
    output = process_words(ref_str, hyp_str)
    wer_value = output.wer
    insertions = output.insertions
    deletions = output.deletions
    substitutions = output.substitutions
    
    # Generate mismatches
    ops = Levenshtein.editops(ref_tokens, hyp_tokens)
    mismatches = []
    window_size = 5
    
    for op, i, j in ops:
        if op == "delete":
            ref_word = ref_tokens[i]
            hyp_word = ""
            ref_context = " ".join(ref_tokens[max(0, i-window_size):i+window_size+1])
            hyp_context = ""
        elif op == "insert":
            ref_word = ""
            hyp_word = hyp_tokens[j]
            ref_context = ""
            hyp_context = " ".join(hyp_tokens[max(0, j-window_size):j+window_size+1])
        elif op == "replace":
            ref_word = ref_tokens[i]
            hyp_word = hyp_tokens[j]
            if jellyfish.metaphone(ref_word) == jellyfish.metaphone(hyp_word):
                continue
            ref_context = " ".join(ref_tokens[max(0, i-window_size):i+window_size+1])
            hyp_context = " ".join(hyp_tokens[max(0, j-window_size):j+window_size+1])
        
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
        "mismatches": mismatches
    }
    mismatches_path = out_dir / "wer_mismatches.json"
    with open(mismatches_path, "w", encoding="utf-8") as mmf:
        json.dump(mismatches_log, mmf, indent=2, ensure_ascii=False)
    print(f"📝 Logged {len(mismatches)} mismatches and measures to {mismatches_path}")
    
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
    results_path = out_dir / "wer1_eval.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved comprehensive results → {results_path}")
    
    # Print summary
    print(f"\n📊 Results summary:")
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
    
    # Return canonical texts for global WER calculation
    return ref_str, hyp_str, results

# ------------------------------------------------------------------------------
# CSV Logging Function
# ------------------------------------------------------------------------------
def log_call_metrics(call_output_dir, wer_value, entities_gt, entities_ref,
                     unique_entities_gt, unique_entities_ref,
                     concerned_entities_gt, concerned_entities_ref,
                     unique_concerned_ner_gt, unique_concerned_ner_ref):
    csv_path = Path(call_output_dir) / "wer_stats.csv"
    headers = [
        "WER",
        "Total GT NERs",
        "Total Ref NERs",
        "Unique GT NERs",
        "Unique Ref NERs",
        "Total Concerned GT NERs (PERSON/ORG)",
        "Total Concerned Ref NERs (PERSON/ORG)",
        "Unique Concerned GT NERs",
        "Unique Concerned Ref NERs"
    ]
    
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(headers)
        writer.writerow([
            f"{wer_value:.4f}",
            len(entities_gt),
            len(entities_ref),
            len(unique_entities_gt),
            len(unique_entities_ref),
            len(concerned_entities_gt),
            len(concerned_entities_ref),
            len(unique_concerned_ner_gt),
            len(unique_concerned_ner_ref)
        ])

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
        summary_path = calls_root / "wer_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"✅ Summary report saved to: {summary_path}")
    except Exception as e:
        print(f"⚠️  Could not generate summary report: {e}")
    
    # Save global WER details
    if global_wer_details:
        global_wer_path = calls_root / "global_wer_report.json"
        with open(global_wer_path, "w", encoding="utf-8") as f:
            json.dump(global_wer_details, f, indent=2, ensure_ascii=False)
        print(f"🌍 Global WER report saved to: {global_wer_path}")
    
    total_time = time.time() - total_start
    print(f"\n⏱️  Total processing time: {total_time:.2f} seconds")
    print("🎉 Processing complete!")

if __name__ == "__main__":
    main()