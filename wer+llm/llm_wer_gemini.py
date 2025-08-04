import json
import os
from typing import List, Dict, Any, Tuple
import google.generativeai as genai
import logging
from dotenv import load_dotenv
import re
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMTranscriptProcessor:
    def __init__(self, gemini_api_key: str, model: str = "gemini-1.5-pro"):
        """
        Initialize the transcript processor with Gemini API key
        
        Args:
            gemini_api_key: Google Gemini API key for LLM normalization and WER calculation
            model: Gemini model to use
        """
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel(model)
        self.normalization_prompt = self._get_normalization_prompt()
        self.wer_prompt = self._get_wer_prompt()
    
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

4. Provide your analysis in this exact JSON format:
{
    "substitutions": <number>,
    "deletions": <number>,
    "insertions": <number>,
    "total_reference_words": <number>,
    "conversational_wer": <decimal>,
    "explanation": "<brief explanation of what errors were counted and why>"
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
            prompt = f"{self.normalization_prompt}\n\nText to normalize: {text}"
            
            # Generate response with Gemini
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=2000,
                    temperature=0.1
                )
            )
            
            normalized_text = response.text.strip()
            logger.info(f"Successfully normalized text of length {len(text)} to {len(normalized_text)}")
            return normalized_text
            
        except Exception as e:
            logger.error(f"Error normalizing text with LLM: {e}")
            # Add a small delay before retrying or falling back
            time.sleep(1)
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
            prompt = self.wer_prompt.format(reference=reference, hypothesis=hypothesis)
            
            # Generate response with Gemini
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=1000,
                    temperature=0.1
                )
            )
            
            result_text = response.text.strip()
            
            # Try to parse JSON response
            try:
                result = json.loads(result_text)
                logger.info(f"Successfully calculated conversational WER: {result.get('conversational_wer', 0):.4f}")
                return result
            except json.JSONDecodeError:
                # Extract JSON from response if it's wrapped in other text
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    return result
                else:
                    raise ValueError("Could not parse JSON response")
                    
        except Exception as e:
            logger.error(f"Error calculating WER with LLM: {e}")
            # Add a small delay before falling back
            time.sleep(1)
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
                
                # Add a small delay between API calls to avoid rate limiting
                time.sleep(0.5)
                
                logger.info(f"Normalizing ground truth transcript for call {call_dir}")
                normalized_gt = self.normalize_with_llm(gt_utterances)
                
                # Add a small delay between API calls
                time.sleep(0.5)
                
                # Calculate conversational WER using LLM
                logger.info(f"Calculating conversational WER for call {call_dir}")
                wer_metrics = self.calculate_conversational_wer_with_llm(normalized_ref, normalized_gt)
                
                results[call_dir] = {
                    'raw_ref_utterances': ref_utterances,
                    'raw_gt_utterances': gt_utterances,
                    'normalized_ref': normalized_ref,
                    'normalized_gt': normalized_gt,
                    'wer_metrics': wer_metrics
                }
                
                logger.info(f"Call {call_dir} - Conversational WER: {wer_metrics.get('conversational_wer', 0):.4f}")
                
                # Add delay between processing calls
                time.sleep(1)
                
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
        
        concatenated_wer_metrics = self.calculate_conversational_wer_with_llm(global_ref, global_gt)
        
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
    gemini_api_key = os.getenv("GEMINI_API_KEY")  # Changed from OPENAI_API_KEY
    
    if not gemini_api_key:
        logger.error("GEMINI_API_KEY not found in environment variables")
        return

    # Initialize processor
    processor = LLMTranscriptProcessor(
        gemini_api_key=gemini_api_key,
        model="gemini-1.5-pro"  # Changed from gpt-4o
    )
    
    # Process all calls
    calls_directory = "calls"  # Replace with your actual calls directory path
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
    print(f"\n{'='*60}")
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
    output_file = "conversational_wer_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'global_metrics': global_metrics,
            'per_call_results': results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()