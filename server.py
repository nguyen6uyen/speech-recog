from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import os
import torch
import shutil
import uuid
from pipelines.pipeline import InferencePipeline
from step2_lip_to_word import LipToWordPredictor
from step3_post_processing import PostProcessor
from step4_llm_layer import SentenceGenerator
import uvicorn
from elevenlabs.client import ElevenLabs
from fastapi.responses import Response

app = FastAPI()

# --- ELEVENLABS CONFIG ---
class TextPayload(BaseModel):
    text: str

@app.post("/speak")
async def speak(payload: TextPayload):
    """
    Step 5: Text-to-Speech (TTS) via ElevenLabs.
    Returns MPEG audio bytes.
    """
    text = payload.text
    if not text:
        raise HTTPException(status_code=400, detail="Text is empty")

    api_key = os.environ.get("ELEVEN_LABS_API_KEY")
    if not api_key:
        print("⚠️ ELEVEN_LABS_API_KEY not found. Text-to-Speech will allow fallback to browser.")
        raise HTTPException(status_code=503, detail="ElevenLabs API key missing")

    try:
        client = ElevenLabs(api_key=api_key)
        # Using 'eleven_turbo_v2' for lowest latency conversational AI
        audio_generator = client.generate(
            text=text,
            voice="Rachel", 
            model="eleven_turbo_v2_5" 
        )
        audio_bytes = b"".join(audio_generator)
        return Response(content=audio_bytes, media_type="audio/mpeg")
    except Exception as e:
        print(f"ElevenLabs Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- INITIALIZE VSR MODEL ---
# --- INITIALIZE VSR MODEL ---
# --- INITIALIZE VSR MODEL ---
CONFIG_PATH = "./configs/LRS3_V_WER19.1.ini"

# Reverting to CPU due to MPS tensor mismatch issues in ESPNet
DEVICE = "cpu"
print("⚠️ Using CPU for VSR Model (MPS/GPU disabled for stability).")

print(f"⏳ Loading Imperial College VSR Model (Expert Brain) on {DEVICE}...")
vsr_pro_model = InferencePipeline(
    config_filename=CONFIG_PATH,
    detector="mediapipe",
    face_track=True,
    device=DEVICE
)

print(f"⏳ Loading Imperial College VSR Model (Expert Brain) on {DEVICE}...")
vsr_pro_model = InferencePipeline(
    config_filename=CONFIG_PATH,
    detector="mediapipe",
    face_track=True,
    device=DEVICE
)
print("✅ Expert Brain Loaded Successfully!")

# Initialize our secondary modules
try:
    predictor = LipToWordPredictor(model_path="mvp_lip_model.pth")
except Exception as e:
    print(f"⚠️ Custom MVP model not loaded: {e}")
    predictor = None

post_processor = PostProcessor(confidence_threshold=0.3)
sentence_generator = SentenceGenerator()

# Ensure temp directory exists
TEMP_DIR = "temp_clips"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

@app.get("/")
async def root():
    # Serve the front-end directly from the server, disable caching for dev
    response = FileResponse("index.html")
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# --- ENDPOINT 1: PRO MODEL (VIDEO BASED) ---
@app.post("/predict_video")
async def predict_video(file: UploadFile = File(...), fast_mode: bool = False):
    """
    Step 2 PRO: Uses the Imperial College Visual Transformer model.
    Accepts a video file, crops lips automatically, and returns full sentences.
    """
    file_id = str(uuid.uuid4())
    temp_path = os.path.join(TEMP_DIR, f"{file_id}_{file.filename}")
    
    try:
        # 1. Save uploaded video to temp file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        print(f"📁 Video received. Running LRS3 Model inference on {temp_path}...")
        
        # 2. Run Pro Model Inference
        raw_output = vsr_pro_model(temp_path)
        print(f"🤖 LRS3 Raw Output: '{raw_output}'")
        
        if not raw_output or raw_output.strip() == "":
            print("⚠️ Model returned empty result. Check if mouth is visible.")
            return {"prediction": "", "refined_sentence": "", "status": "success"}

        # 3. Polish with Step 4 LLM Layer (Skip if Fast Mode)
        refined_sentence = raw_output
        
        if not fast_mode: 
            try:
                # We fix the input: use a single string token
                refined = sentence_generator.generate_sentence([raw_output], [1.0])
                refined_sentence = refined['sentence']
                print(f"✨ AI Refined: '{refined_sentence}'")
            except Exception as llm_err:
                print(f"LLM Error: {llm_err}")
        else:
            print("⚡ Fast Mode: Skipping LLM refinement.")

        return {
            "prediction": raw_output,
            "refined_sentence": refined_sentence,
            "status": "success",
            "model": "Imperial_College_VSR"
        }
        
    except Exception as e:
        print(f"Prediction Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"VSR Model failed: {str(e)}")
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

# --- ENDPOINT 2: CUSTOM MODEL (LANDMARK BASED) ---
class LipSequence(BaseModel):
    data: List[List[List[float]]] 

@app.post("/predict_landmarks")
async def predict_landmarks(sequence: LipSequence):
    """
    Step 2 CUSTOM: Uses our smaller GRU model trained on custom landmarks.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Custom ML model not loaded.")
    try:
        np_sequence = [np.array(frame) for frame in sequence.data]
        candidates = predictor.predict(np_sequence)
        result = post_processor.process_prediction(candidates)
        
        if not result:
            return {"prediction": "", "status": "low_confidence"}
            
        refined = sentence_generator.generate_sentence([result['text']], [result['confidence']])
        
        return {
            "prediction": result['text'],
            "refined_sentence": refined['sentence'],
            "confidence": result['confidence'],
            "status": "success",
            "model": "Custom_GRU"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
