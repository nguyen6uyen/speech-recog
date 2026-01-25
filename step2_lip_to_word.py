import torch
import torch.nn as nn
import numpy as np
import os
from typing import List, Dict, Any, Optional

# --- CONSISTENT CONFIGURATION ---
LIP_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78,
    185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191
]
PHRASES = ["HELLO", "YES", "NO", "STOP", "HELP"]

# Import model architecture (ensure we use the same one as training)
try:
    from train_mvp_model import LipReadingGRU
except ImportError:
    class LipReadingGRU(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_classes):
            super(LipReadingGRU, self).__init__()
            self.gru = nn.GRU(input_dim, hidden_dim, 2, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(hidden_dim * 2, num_classes)
        def forward(self, x):
            _, h_n = self.gru(x)
            out = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
            return self.fc(out)

class LipToWordPredictor:
    def __init__(self, model_path: str = "mvp_lip_model.pth", device: str = "cpu"):
        self.device = torch.device(device)
        self.phrases = PHRASES
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str):
        if not os.path.exists(model_path):
            print(f"Warning: Model {model_path} not found. Using placeholder results.")
            return None
        
        # Dim = 42 points * 2 (x,y)
        model = LipReadingGRU(input_dim=84, hidden_dim=64, num_classes=len(self.phrases))
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device).eval()
        return model

    def preprocess(self, landmarks_sequence: List[np.ndarray]) -> torch.Tensor:
        """Process Raw MediaPipe landmarks into Model features"""
        processed = []
        for lm in landmarks_sequence:
            if lm is None: continue
            
            # Extract lip points
            lip_points = lm[LIP_INDICES] if len(lm) > 100 else lm
            
            # 1. Center X and Y separately
            center = lip_points.mean(axis=0)
            norm_points = lip_points - center
            
            # 2. Scale Normalize
            scale = np.linalg.norm(norm_points.max(axis=0) - norm_points.min(axis=0))
            if scale > 0:
                norm_points /= scale
            
            processed.append(norm_points.flatten())
        
        if not processed: return torch.empty(0)
        return torch.tensor(np.array(processed), dtype=torch.float32).unsqueeze(0).to(self.device)

    def predict(self, landmarks_sequence: List[np.ndarray]) -> List[Dict[str, Any]]:
        features = self.preprocess(landmarks_sequence)
        
        if self.model is None or features.numel() == 0:
            return [{"text": "HELLO", "confidence": 0.5}, {"text": "WAITING FOR DATA", "confidence": 0.1}]

        with torch.no_grad():
            logits = self.model(features)
            probs = torch.softmax(logits, dim=1)[0]
            
            # Create list of candidates
            results = []
            for i, p in enumerate(self.phrases):
                results.append({"text": p, "confidence": float(probs[i])})
            
            # Sort by confidence
            results.sort(key=lambda x: x['confidence'], reverse=True)
            return results

if __name__ == "__main__":
    predictor = LipToWordPredictor()
    # Test with mock data
    mock_seq = [np.random.rand(42, 2) for _ in range(20)]
    print(predictor.predict(mock_seq))
