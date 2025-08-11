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
- A text log capturing terminal output with S/I/D lists and global WER for that method, saved at the repo root: `output_{file_name}.txt`

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
2. Compute WER using a dynamic‑programming edit distance over normalized tokens:
   - Substitutions are free if tokens are highly similar by fuzzy ratio or phonetically similar by Soundex/Double‑Metaphone.
   - Otherwise, standard DP costs: deletion = 1, insertion = 1, substitution = 1.
3. Collect first N word‑level differences for analysis and print explicit lists of:
   - Substitution pairs: `reference -> hypothesis`
   - Deletion words: “missing in hypothesis”
   - Insertion words: “extra in hypothesis”
4. Save per‑call mismatches to `calls/<call_id>/output/mismatches.json`.
5. Compute global WER by concatenating all GT utterances as the reference and all REF utterances as the hypothesis:
   - Stored in `wer_lib/global_wer.json`.

How to run:

```bash
cd wer/wer_lib
python3 wer_lib.py
```

Outputs:

- Per call: `calls/<call_id>/output/mismatches.json`
- Global: `wer_lib/global_wer.json`
- Terminal log at repo root: `output_wer_lib.py.txt` (contains S/I/D and global WER)

### 2) LLM canonicalization + jiwer WER: `wer+jiwer/wer_gpt.py`, `wer+jiwer/wer_gemini.py`

Core logic:

1. Filter transcripts to only bot/agent utterances.
2. Send filtered JSON to the LLM to obtain a canonical form plus named entities:
   - OpenAI (`wer_gpt.py`) or Gemini (`wer_gemini.py`) returns a JSON with `canonical_transcript` and `entities`.
   - The canonical transcript is aggressively cleaned of any lingering user artifacts.
3. Tokenize canonical transcripts with GT as reference and REF as hypothesis.
4. Compute WER with `jiwer.process_words` on the canonicalized strings.
5. Generate detailed differences using `Levenshtein.editops`:
   - Skip “replace” ops that are phonetically identical (Metaphone) to reduce false substitutions.
   - Accumulate and print explicit substitutions, insertions, deletions.
6. Persist artifacts per call in `calls/<call_id>/output/`:
   - `canon_gt_transcript.json` and `canon_ref_transcript.json`
   - `wer_mismatches.json` (with `substitution_pairs`, `deletion_words`, `insertion_words`)
   - `wer+eer.json` (aggregated per‑call stats)
7. Compute Global WER on concatenated canonical texts (GT reference vs REF hypothesis):
   - Stored at `wer+jiwer/calls/global_wer_report.json`.

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

- Per call: `calls/<call_id>/output/*` as described above
- Global: `wer+jiwer/calls/global_wer_report.json`
- Terminal logs at repo root: `output_wer_gpt.py.txt`, `output_wer_gemini.py.txt`

### 3) LLM‑only conversational WER: `wer+llm/llm_wer_gpt.py`, `wer+llm/llm_wer_gemini.py`

Core logic:

1. Extract bot/agent utterances and have the LLM normalize both sides using the same normalization rules listed above.
2. Ask the LLM to compute “conversational WER” with explicit guidance:
   - Ignore phonetically identical words, minor spelling variants, and fillers.
   - Count only substitutions/insertions/deletions that affect conversational understanding.
   - Return strict JSON with: counts, `substitution_pairs`, `deletion_words`, `insertion_words`, and a short explanation.
3. Print the returned S/I/D lists per call and persist aggregate results:
   - Global metrics include average WER, aggregate WER, and concatenated‑text WER (GT reference vs REF hypothesis).
   - The full report is saved as `wer+llm/conversational_wer_analysis.json`.

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
- Global: `wer+llm/conversational_wer_analysis.json`
- Terminal logs at repo root: `output_llm_wer_gpt.py.txt`, `output_llm_wer_gemini.py.txt`

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
- Global summary artifacts in the method folder (e.g., `global_wer.json`, `global_wer_report.json`, or `conversational_wer_analysis.json`)
- A text file at the repository root capturing the method’s terminal output including per‑call S/I/D and the global WER:
  - `output_wer_lib.py.txt`
  - `output_wer_gpt.py.txt`
  - `output_wer_gemini.py.txt`
  - `output_llm_wer_gpt.py.txt`
  - `output_llm_wer_gemini.py.txt`

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
| `wer_lib/wer_lib.py` | Pure library (custom normalization + DP WER with phonetic/fuzzy) |  |  |
| `wer+jiwer/wer_gpt.py` | LLM canonicalization (OpenAI) + jiwer WER |  |  |
| `wer+jiwer/wer_gemini.py` | LLM canonicalization (Gemini) + jiwer WER |  |  |
| `wer+llm/llm_wer_gpt.py` | LLM‑only conversational WER (OpenAI) |  |  |
| `wer+llm/llm_wer_gemini.py` | LLM‑only conversational WER (Gemini) |  |  |

