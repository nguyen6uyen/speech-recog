from typing import List, Dict, Any, Optional

class PostProcessor:
    """
    Step 3: Post-Processing & Word Library mapping.
    Cleans ML output, handles Homophenes (Pain Words), and prepares LLM tokens.
    """
    
    def __init__(self, confidence_threshold: float = 0.4):
        self.threshold = confidence_threshold
        
        # --- PAIN WORDS (Homophenes) ---
        # Words that look the same on the lips. 
        # If the model is torn between these, we flag them for LLM/Context.
        self.homophenes = {
            "HELLO": ["HELP", "HOLLOW"],
            "HELP": ["HELLO"],
            "NO": ["KNOW", "GO"],
            "STOP": ["STEP", "TOP"]
        }
        
        # --- WORD LIBRARY / SPECIAL COMMANDS ---
        # Map specific results to actions or formatted strings
        self.library_map = {
            "HELLO": "Hello!",
            "YES": "Yes.",
            "NO": "No.",
            "STOP": "STOP",
            "HELP": "Help!"
        }

    def handle_pain_words(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Adjusts scores if candidate words are known homophenes.
        If the top 2 candidates are a 'Pain Pair' and their scores are close,
        the LLM should handle the tie-break.
        """
        if len(candidates) < 2:
            return candidates
            
        top_word = candidates[0]['text']
        second_word = candidates[1]['text']
        score_diff = candidates[0]['confidence'] - candidates[1]['confidence']
        
        # If the top word has a 'pain pair' that is also in the top results
        if top_word in self.homophenes and second_word in self.homophenes[top_word]:
            if score_diff < 0.2: # They are uncertain
                # Keep both high, let Step 4 (LLM) decide
                print(f"⚠️ Ambiguity detected between homophenes: {top_word} and {second_word}")
                
        return candidates

    def process_prediction(self, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Filters by confidence and maps to library.
        """
        # Filter out junk
        candidates = [c for c in candidates if c['confidence'] >= self.threshold]
        
        if not candidates:
            return None
            
        # Refine homophenes
        candidates = self.handle_pain_words(candidates)
        
        top_cand = candidates[0]
        
        # Map to final output format from library
        final_text = self.library_map.get(top_cand['text'], top_cand['text'])
        
        return {
            "text": final_text,
            "confidence": top_cand['confidence'],
            "alternatives": [c['text'] for c in candidates[1:3]]
        }

    def prepare_llm_prompt(self, processed_results: List[Dict[str, Any]]) -> str:
        """
        Converts a sequence of processed words into a token sequence for the LLM.
        """
        sequence = [res['text'] for res in processed_results]
        raw_sequence = " ".join(sequence)
        
        prompt = f"The following sequence was lip-read and might contain errors: {raw_sequence}. " \
                 f"Please correct the grammar and punctuation while keeping the meaning."
        
        return prompt

# --- TEST ---
if __name__ == "__main__":
    pp = PostProcessor()
    
    # Mock output from Step 2 (The Predictor)
    # Scenario: User said 'HELLO' but it looked a lot like 'HELP'
    mock_ml_output = [
        {"text": "HELLO", "confidence": 0.45},
        {"text": "HELP", "confidence": 0.42},
        {"text": "STOP", "confidence": 0.05}
    ]
    
    result = pp.process_prediction(mock_ml_output)
    
    if result:
        print(f"--- Step 3: Post-Processed Result ---")
        print(f"Output: {result['text']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Possible Alternatives: {result['alternatives']}")
        
        # Simulate building a sequence
        history = [result, {"text": "YES", "confidence": 0.9}]
        print(f"\nFinal LLM Prompt: {pp.prepare_llm_prompt(history)}")
    else:
        print("Confidence too low, ignoring.")
