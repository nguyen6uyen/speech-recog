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

# Architecture must match train_mvp_model.py
class LipReadingGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(LipReadingGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, 2, batch_first=True, bidirectional=True, dropout=0.5)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
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
            return None
        # input_dim is now 85 (1 ratio + 84 relative points)
        model = LipReadingGRU(input_dim=85, hidden_dim=64, num_classes=len(self.phrases))
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device).eval()
        return model

    def preprocess(self, landmarks_sequence: List[np.ndarray]) -> torch.Tensor:
        processed = []
        for lm in landmarks_sequence:
            if lm is None: continue
            
            # Extract lip points
            lip_points = lm[LIP_INDICES] if len(lm) > 100 else lm
            
            # Feature extraction (matching new training logic)
            width = np.linalg.norm(lip_points[0] - lip_points[10]) + 1e-6
            height = np.linalg.norm(lip_points[37] - lip_points[16])
            ratio = height / width
            
            # Relative centered points
            rel = (lip_points - lip_points.mean(axis=0)).flatten()
            
            # [Ratio, RelPoints...]
            processed.append(np.concatenate([[ratio], rel]))
        
        if not processed: return torch.empty(0)
        return torch.tensor(np.array(processed), dtype=torch.float32).unsqueeze(0).to(self.device)

    def predict(self, landmarks_sequence: List[np.ndarray]) -> List[Dict[str, Any]]:
        features = self.preprocess(landmarks_sequence)
        
        if self.model is None or features.numel() == 0:
            return [{"text": "ERR", "confidence": 0.0}]

        with torch.no_grad():
            logits = self.model(features)
            probs = torch.softmax(logits, dim=1)[0]
            
            results = []
            for i, p in enumerate(self.phrases):
                results.append({"text": p, "confidence": float(probs[i])})
            
            results.sort(key=lambda x: x['confidence'], reverse=True)
            return results
