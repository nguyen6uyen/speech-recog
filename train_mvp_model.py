import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import glob
from torch.utils.data import Dataset, DataLoader

# --- MODEL ARCHITECTURE (IMPROVED FOR SMALL DATA) ---
class LipReadingGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2):
        super(LipReadingGRU, self).__init__()
        # 1. Added Dropout to the GRU itself
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, 
                          batch_first=True, bidirectional=True, dropout=0.5)
        
        # 2. Added a more complex head with its own Dropout layer
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        _, h_n = self.gru(x)
        out = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        out = self.fc(out)
        return out

# --- DATASET LOADER WITH RATIO FEATURES ---
class MVPDataset(Dataset):
    def __init__(self, data_dir, phrase_to_idx, augment=True):
        self.samples = []
        self.labels = []
        
        file_list = glob.glob(os.path.join(data_dir, "*.npy"))
        for f in file_list:
            phrase = os.path.basename(f).split('_')[0]
            if phrase in phrase_to_idx:
                raw_data = np.load(f) # Shape: [Frames, 42, 2]
                
                # Create 15 variants of EVERY sample (more intensive augmentation)
                num_variants = 15 if augment else 1
                for _ in range(num_variants):
                    data = raw_data.copy()
                    if augment:
                        # Add coordinate noise
                        data += np.random.normal(0, 0.003, data.shape)
                        # Random scaling (simulating different distances)
                        data *= np.random.uniform(0.9, 1.1)
                    
                    processed_seq = []
                    for frame in data:
                        # --- FEATURE ENGINEERING: GEOMETRIC RATIOS ---
                        # These features are much more robust than raw X/Y
                        
                        # Lip Width (left corner to right corner)
                        width = np.linalg.norm(frame[0] - frame[10]) + 1e-6
                        # Inner Lip Height
                        height = np.linalg.norm(frame[37] - frame[16])
                        # Aspect Ratio
                        ratio = height / width
                        
                        # Relative coordinates (centered)
                        rel = (frame - frame.mean(axis=0)).flatten()
                        
                        # Combine: [Ratio, Raw landmarks...]
                        processed_seq.append(np.concatenate([[ratio], rel]))
                    
                    self.samples.append(torch.tensor(np.array(processed_seq), dtype=torch.float32))
                    self.labels.append(phrase_to_idx[phrase])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

def pad_collate(batch):
    (xx, yy) = zip(*batch)
    xx_pad = nn.utils.rnn.pad_sequence(xx, batch_first=True, padding_value=0)
    return xx_pad, torch.tensor(yy)

# --- TRAINING SCRIPT ---
def train():
    PHRASES = ["HELLO", "YES", "NO", "STOP", "HELP"]
    phrase_to_idx = {p: i for i, p in enumerate(PHRASES)}
    
    if not os.path.exists("mvp_data") or len(os.listdir("mvp_data")) == 0:
        print("Error: No data found in 'mvp_data/'")
        return

    dataset = MVPDataset("mvp_data", phrase_to_idx, augment=True)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=pad_collate)

    device = torch.device("cpu")
    # input_dim = 1 ratio + 84 relative coords = 85
    model = LipReadingGRU(input_dim=85, hidden_dim=64, num_classes=len(PHRASES)).to(device)
    
    criterion = nn.CrossEntropyLoss()
    # Weight decay (L2 Reg) is essential to combat overfitting
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    print(f"Starting robust training on {len(dataset)} samples...")
    for epoch in range(80):
        model.train()
        total_loss = 0
        for x, y in loader:
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/80], Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "mvp_lip_model.pth")
    print("Robust model saved as 'mvp_lip_model.pth'.")

if __name__ == "__main__":
    train()
