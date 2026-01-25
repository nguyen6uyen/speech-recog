from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from step2_lip_to_word import LipToWordPredictor
from step3_post_processing import PostProcessor
from step4_llm_layer import SentenceGenerator
import uvicorn

app = FastAPI()

# Initialize our modules
predictor = LipToWordPredictor(model_path="mvp_lip_model.pth")
post_processor = PostProcessor(confidence_threshold=0.3)
sentence_generator = SentenceGenerator()

@app.get("/")
async def root():
    return {
        "app": "Chaplin Silent Speech API",
        "status": "online",
        "endpoint": "/predict [POST]",
        "instructions": "Send a POST request with lip landmark sequences to /predict"
    }

class LandmarkFrame(BaseModel):
    x: float
    y: float

class LipSequence(BaseModel):
    # The browser sends a list of frames, each containing 42 lip landmarks
    # Format: sequence[frame_index][landmark_index]
    data: List[List[List[float]]] 

@app.post("/predict")
async def predict_lips(sequence: LipSequence):
    try:
        # 1. Convert incoming JSON data back to numpy for Step 2
        np_sequence = [np.array(frame) for frame in sequence.data]
        
        # 2. Run Step 2 (ML Prediction)
        candidates = predictor.predict(np_sequence)
        
        # 3. Run Step 3 (Post-Processing)
        result = post_processor.process_prediction(candidates)
        
        if not result:
            return {"tokens": [], "confidence": 0, "status": "low_confidence"}
            
        # 4. Optional: Run Step 4 (LLM Refinement) if it's a short phrase
        # We can also call this separately via /refine for a list of words
        refined = sentence_generator.generate_sentence([result['text']], [result['confidence']])
        
        return {
            "prediction": result['text'],
            "refined_sentence": refined['sentence'],
            "alternatives": refined['alternatives'],
            "confidence": result['confidence'],
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class SentenceInput(BaseModel):
    tokens: List[str]
    confidences: List[float]

@app.post("/refine")
async def refine_sentence(input_data: SentenceInput):
    """
    Step 4: Refine a whole sequence of detected words into a coherent sentence.
    """
    try:
        result = sentence_generator.generate_sentence(input_data.tokens, input_data.confidences)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 Chaplin API Server Starting...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
