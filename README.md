## WER evaluation framework

This repository contains three complementary approaches to compute Word Error Rate (WER) for assistant/bot utterances in phone-call transcripts. Across all methods, the evaluation is aligned so that the reference is always the ground-truth agent transcript and the hypothesis is the assistant reference transcript:

- Reference = `gt_transcript.json` (speaker `Agent`)
- Hypothesis = `ref_transcript.json` (speaker `assistant`)

The tooling extracts only bot/agent turns, applies normalization, computes per‑call differences, prints and saves substitutions/insertions/deletions, and produces global metrics over all calls.

### Repository and dataset layout

Each method lives in its own folder and follows the same call data layout:

- `calls/` directory with one subfolder per call, e.g. `calls/<call_id>/`
  - `annotated.json` – full call annotation (not always consumed)
  - `gt_transcript.json` – ground truth transcript (speaker `Agent`)
  - `ref_transcript.json` – reference transcript to evaluate (speaker `assistant`)
  - `output/` – per‑call outputs produced by the script
- One or two Python entry points in the method folder
- Global output artifacts for the method (aggregates over all calls)
- A text log capturing terminal output with S/I/D lists and global WER for that method, saved next to the method scripts (see per‑method outputs below)

### Methods at a glance

- `wer_lib/wer_lib.py` – Pure library approach. Custom, deterministic normalization + phonetic/fuzzy matching + dynamic‑programming WER.
- `wer+jiwer/wer_gpt.py` and `wer+jiwer/wer_gemini.py` – LLM canonicalization (OpenAI or Gemini) followed by jiwer-based WER and detailed diffs.
- `wer+llm/llm_wer_gpt.py` and `wer+llm/llm_wer_gemini.py` – LLM performs both normalization and the WER judgment (“conversational WER”) and returns structured S/I/D details.

## Normalization pipeline (shared intent)

All approaches aim to compare semantically comparable strings by reducing superficial differences before WER. The pure‑library method implements this directly; the LLM methods prompt the LLM to do the same. The steps include:

- Transliteration of non‑Roman scripts to Roman/ASCII
- Time expression normalization (e.g., “five five” → “5:05”)
- Ordinal normalization (e.g., “eighteenth” ↔ “18th”)
- Variant spelling normalization (e.g., “apki” → “aapki”, “rey” → “ray”)
- Case folding to lowercase
- Punctuation removal
- Cardinal number normalization (word ↔ digit, but not ordinals)
- Whitespace collapsing

Additionally, scoring ignores near‑duplicates via phonetic and fuzzy matching (depending on method; see details below).

## Detailed method descriptions

### 1) Pure library WER: `wer_lib/wer_lib.py`

Core logic:

1. Load `calls/` and, for each call:
   - Extract only bot/agent utterances: `assistant` from `ref_transcript.json`, `Agent` from `gt_transcript.json`.
   - Normalize both strings using the pipeline above (`canonicalize_and_normalize`).
2. Compute WER using a dynamic‑programming edit distance over normalized tokens with phonetic equality:
   - Fast, deterministic, cached phonetics pipeline:
     - Double Metaphone (primary+alt) with a tiny Jaro‑Winkler guard on codes
     - If inconclusive, optional G2P (ARPABET) with phoneme‑level edit‑distance similarity
   - Tokens that “sound alike” are treated as exact matches (cost 0)
   - Otherwise, standard DP costs: deletion = 1, insertion = 1, substitution = 1.
3. Collect first N word‑level differences for analysis and print explicit lists of:
   - Substitution pairs: `reference -> hypothesis`
   - Deletion words: “missing in hypothesis”
   - Insertion words: “extra in hypothesis”
4. Save per‑call mismatches to `wer_lib/calls/<call_id>/output/mismatches.json`.
5. Compute global WER by concatenating all GT utterances as the reference and all REF utterances as the hypothesis:
   - Stored in `wer_lib/output/global_wer.json`.

How to run:

```bash
cd wer/wer_lib
python3 wer_lib.py
```

Outputs:

- Per call: `wer_lib/calls/<call_id>/output/mismatches.json`
- Global: `wer_lib/output/global_wer.json`
- Terminal log: `wer_lib/output_lib.txt` (contains S/I/D checkpoints and global WER)

### 2) LLM canonicalization + phonetic‑aware DP WER: `wer+jiwer/wer_gpt.py`, `wer+jiwer/wer_gemini.py`

Core logic:

1. Filter transcripts to only bot/agent utterances.
2. Send filtered JSON to the LLM to obtain a canonical form plus named entities:
   - OpenAI (`wer_gpt.py`) or Gemini (`wer_gemini.py`) returns a JSON with `canonical_transcript` and `entities`.
   - The canonical transcript is aggressively cleaned of any lingering user artifacts.
3. Tokenize canonical transcripts with GT as reference and REF as hypothesis.
4. Compute WER with a phonetic‑aware dynamic program (same pipeline as in library method):
   - Double Metaphone (+ Jaro‑Winkler on codes); fallback to G2P ARPABET similarity
   - Sounds‑alike tokens are exact equals (cost 0)
5. Generate detailed differences from the DP alignment ops and print explicit S/I/D lists
   - Accumulate and print explicit substitutions, insertions, deletions.
6. Persist artifacts per call in `wer+jiwer/calls/<call_id>/output/`:
   - OpenAI path (`wer_gpt.py`):
     - `canon_ref_transcript_gpt_lib.json`
     - `canon_gt_transcript_gpt_lib.json`
     - `wer_mismatches_gpt_lib.json`
     - `wer+eer_gpt_lib.json`
   - Gemini path (`wer_gemini.py`):
     - `canon_ref_transcript_gemini_lib.json`
     - `canon_gt_transcript_gemini_lib.json`
     - `wer_mismatches_gemini_lib.json`
     - `wer+eer_gemini_lib.json`
7. Compute Global WER on concatenated canonical texts (GT reference vs REF hypothesis):
   - OpenAI path: `wer+jiwer/global_wer_summary_gpt_lib.csv`, `wer+jiwer/global_wer_report_gpt_lib.json`
   - Gemini path: `wer+jiwer/wer_summary_gemini_lib.csv`, `wer+jiwer/global_wer_report_gemini_lib.json`

How to run:

```bash
# OpenAI path
python3 wer/wer+jiwer/wer_gpt.py wer/wer+jiwer/calls

# Gemini path
python3 wer/wer+jiwer/wer_gemini.py wer/wer+jiwer/calls
```

Environment variables:

- `OPENAI_API_KEY` for `wer_gpt.py`
- `GEMINI_API_KEY` for `wer_gemini.py`

Outputs:

- Per call: `wer+jiwer/calls/<call_id>/output/*` as described above
- Global (saved in `wer+jiwer/`):
  - OpenAI: `global_wer_summary_gpt_lib.csv`, `global_wer_report_gpt_lib.json`
  - Gemini: `wer_summary_gemini_lib.csv`, `global_wer_report_gemini_lib.json`
- Terminal logs:
  - OpenAI: `wer+jiwer/output_gpt.txt`
  - Gemini: `wer+jiwer/output_gemini.txt`

Notes:
- Transcripts are read directly from each call folder; the scripts no longer copy them into `output/`.

### 3) LLM‑only conversational WER: `wer+llm/llm_wer_gpt.py`, `wer+llm/llm_wer_gemini.py`

Core logic:

1. Extract bot/agent utterances and have the LLM normalize both sides using the same normalization rules listed above.
2. Ask the LLM to compute “conversational WER” with explicit guidance:
   - Ignore phonetically identical words, minor spelling variants, and fillers.
   - Count only substitutions/insertions/deletions that affect conversational understanding.
   - Return strict JSON with: counts, `substitution_pairs`, `deletion_words`, `insertion_words`, and a short explanation.
3. Print the returned S/I/D lists per call and persist aggregate results:
   - Global metrics include average WER, aggregate WER, and concatenated‑text WER (GT reference vs REF hypothesis).
   - Saved files:
     - OpenAI: `wer+llm/global_wer_gpt_llm.json`
     - Gemini: `wer+llm/global_wer_gemini_llm.json`

How to run:

```bash
cd wer/wer+llm

# OpenAI model
python3 llm_wer_gpt.py

# Gemini model
python3 llm_wer_gemini.py
```

Environment variables:

- `OPENAI_API_KEY` for `llm_wer_gpt.py`
- `GEMINI_API_KEY` for `llm_wer_gemini.py`

Outputs:

- Per call: printed S/I/D, with per‑call data included in the final JSON result map
- Global: `wer+llm/global_wer_gpt_llm.json`, `wer+llm/global_wer_gemini_llm.json`
- Terminal logs:
  - OpenAI: `wer+llm/output_gpt.txt`
  - Gemini: `wer+llm/output_gemini.txt`

## Consistent S/I/D semantics and denominator

Across all methods:

- WER is computed with GT as the reference and REF as the hypothesis.
- Denominator N = total number of reference words in the GT canonical transcript(s).
- The scripts print, per call:
  - Substitutions: list of `reference -> hypothesis` pairs
  - Deletions: tokens present in GT but missing in REF
  - Insertions: tokens present in REF but not in GT

## Global outputs and logs

For each method you will find:

- Per‑call artifacts in `calls/<call_id>/output/`
- Global summary artifacts in the method folder (see per‑method lists above)
- A text file next to the method scripts capturing the terminal output including per‑call S/I/D and the global WER:
  - `wer_lib/output_lib.txt`
  - `wer+jiwer/output_gpt.txt`
  - `wer+jiwer/output_gemini.txt`
  - `wer+llm/output_gpt.txt`
  - `wer+llm/output_gemini.txt`

## Dependencies

- Python 3.10+
- Common libraries: `python-dotenv`, `pandas`, `rapidfuzz`, `phonetics`, `metaphone`, `Levenshtein`, `jellyfish`, `jiwer`
- LLM clients as needed: `openai` or `google-generativeai`

Set environment variables in `.env`:

```env
OPENAI_API_KEY=...
GEMINI_API_KEY=...
```

## Run quick commands

```bash
# Pure library
cd wer/wer_lib && python3 wer_lib.py

# LLM canonicalization + jiwer (OpenAI)
python3 wer/wer+jiwer/wer_gpt.py wer/wer+jiwer/calls

# LLM canonicalization + jiwer (Gemini)
python3 wer/wer+jiwer/wer_gemini.py wer/wer+jiwer/calls

# LLM‑only conversational WER (OpenAI)
cd wer/wer+llm && python3 llm_wer_gpt.py

# LLM‑only conversational WER (Gemini)
cd wer/wer+llm && python3 llm_wer_gemini.py
```

## Global WER comparison table

Fill in after running the methods on the same dataset.

| Script | Method | Global WER | Notes |
|---|---|---:|---|
| `wer_lib/wer_lib.py` | Pure library (custom normalization + DP WER with phonetic/fuzzy) | 23.06% | DP treats sounds‑alike as equal (cost 0); 421 utterances processed |
| `wer+jiwer/wer_gpt.py` | LLM canonicalization (OpenAI) + jiwer WER | 11.73% | Global WER on concatenated bot text; phonetic‑aware alignment |
| `wer+jiwer/wer_gemini.py` | LLM canonicalization (Gemini) + jiwer WER | 24.51% | Global WER on concatenated bot text; phonetic‑aware alignment |
| `wer+llm/llm_wer_gpt.py` | LLM‑only conversational WER (OpenAI) | 10.85% | Global conversational WER (concatenated); JSON parsing hardened |
| `wer+llm/llm_wer_gemini.py` | LLM‑only conversational WER (Gemini) | 11.43% | Global conversational WER (concatenated); JSON response enforced |

