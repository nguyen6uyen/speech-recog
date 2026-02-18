from typing import List, Dict, Any, Optional
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json

class SentenceGenerator:
    """
    Step 4: LLM Layer - Sentence Generation.
    Uses LangChain and a local LLM (Ollama) to convert word tokens into coherent sentences.
    """

    def __init__(self, model_name: str = "qwen2.5:7b", temperature: float = 0.2):
        """
        Initializes the LangChain agent with a local Ollama model.
        Defaulting to 0.5b for speed and high availability.
        """
        self.llm = ChatOllama(model=model_name, temperature=temperature)
        self.output_parser = StrOutputParser()
        
        # System instructions for the LLM
        self.system_prompt = (
            "You are an assistant that helps refine the output of a silent speech (lip-reading) interface. "
            "The input you receive is a sequence of words that were captured by a VSR (Visual Speech Recognition) model. "
            "These words might be slightly incorrect because many words look similar on the lips (homophenes). "
            "\n\nRules:\n"
            "1. Correct grammar and punctuation.\n"
            "2. If a word looks wrong in context, replace it with a more likely word that looks similar on the lips.\n"
            "3. Keep the meaning as close as possible to the intended speech.\n"
            "4. Only return the corrected sentence, nothing else.\n"
            "5. If the input is empty or nonsensical, return an empty string."
        )

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", "Lip-read sequence: {tokens}")
        ])

        self.chain = self.prompt_template | self.llm | self.output_parser

    def generate_sentence(self, tokens: List[str], confidence_scores: List[float]) -> Dict[str, Any]:
        """
        Processes a sequence of tokens and generates a refined sentence.
        
        :param tokens: List of word tokens mapped from Step 3.
        :param confidence_scores: Average confidence scores from Step 2/3.
        :return: A dictionary with the main sentence and optional alternatives.
        """
        if not tokens:
            return {"sentence": "", "status": "empty"}

        raw_sequence = " ".join(tokens)
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        # Determine if we should provide alternatives based on confidence
        # If confidence is low, we ask the LLM to provide its top 3 best guesses
        if avg_confidence < 0.6:
            # Update prompt temporarily for n-best results
            n_best_prompt = (
                f"{self.system_prompt}\n"
                "Because the confidence of the input is low, provide the TOP 3 most likely "
                "intended sentences as a JSON list of strings."
            )
            temp_template = ChatPromptTemplate.from_messages([
                ("system", n_best_prompt),
                ("user", f"Lip-read sequence: {raw_sequence}")
            ])
            temp_chain = temp_template | self.llm | self.output_parser
            
            try:
                response = temp_chain.invoke({})
                # Clean up response in case LLM adds markdown or chatter
                cleaned_response = response.strip()
                if "```json" in cleaned_response:
                    cleaned_response = cleaned_response.split("```json")[1].split("```")[0].strip()
                elif "```" in cleaned_response:
                    cleaned_response = cleaned_response.split("```")[1].split("```")[0].strip()
                
                candidates = json.loads(cleaned_response)
                return {
                    "sentence": candidates[0] if candidates else "",
                    "alternatives": candidates[1:3] if len(candidates) > 1 else [],
                    "status": "low_confidence_alternatives"
                }
            except Exception as e:
                print(f"Error generating alternatives: {e}")
                # Fallback to single sentence
                pass

        # Normal single-sentence generation
        sentence = self.chain.invoke({"tokens": raw_sequence})
        return {
            "sentence": sentence.strip(),
            "alternatives": [],
            "status": "success"
        }

# --- TEST ---
if __name__ == "__main__":
    # Note: This requires Ollama to be running with the specified model
    import sys
    
    generator = SentenceGenerator()
    
    print("--- Step 4: LLM Layer Test ---")
    
    # Test 1: Clear sequence
    print("\nTest 1: Normal confidence")
    test_tokens = ["HELLO", "STOP"]
    test_scores = [0.9, 0.8]
    # In a real run, this call might fail if Ollama is not running, so we wrap it
    try:
        result = generator.generate_sentence(test_tokens, test_scores)
        print(f"Input: {test_tokens}")
        print(f"Output: {result['sentence']}")
    except Exception as e:
        print(f"Skipping live test: {e} (Is Ollama running?)")

    # Test 2: Low confidence sequence
    print("\nTest 2: Low confidence (Alternatives)")
    test_tokens_low = ["HELP", "NO"]
    test_scores_low = [0.4, 0.3]
    try:
        result_low = generator.generate_sentence(test_tokens_low, test_scores_low)
        print(f"Input: {test_tokens_low}")
        if "alternatives" in result_low and result_low["alternatives"]:
            print(f"Primary: {result_low['sentence']}")
            print(f"Alts: {result_low['alternatives']}")
        else:
            print(f"Output: {result_low['sentence']}")
    except Exception as e:
        print(f"Skipping live test: {e}")
