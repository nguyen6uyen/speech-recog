import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import glob
from torch.utils.data import Dataset, DataLoader

# --- MODEL ARCHITECTURE ---
class LipReadingGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2):
        super(LipReadingGRU, self).__init__()
        # input_dim will be len(LIP_INDICES) * 2
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, 
                          batch_first=True, bidirectional=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        # x shape: [Batch, Seq_Len, Features]
        _, h_n = self.gru(x)
        # Concatenate the final hidden state from both directions
        out = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        out = self.fc(out)
        return out

# --- DATASET LOADER ---
class MVPDataset(Dataset):
    def __init__(self, data_dir, phrase_to_idx):
        self.samples = []
        self.labels = []
        
        file_list = glob.glob(os.path.join(data_dir, "*.npy"))
        for f in file_list:
            phrase = os.path.basename(f).split('_')[0]
            if phrase in phrase_to_idx:
                # data shape: [Seq, 42, 2]
                data = np.load(f)
                
                processed_seq = []
                for frame in data:
                    # Center X and Y separately
                    center = frame.mean(axis=0)
                    norm_frame = frame - center
                    
                    # Scale to normalize for distance from camera
                    # (Distance between corner of mouth points)
                    scale = np.linalg.norm(norm_frame.max(axis=0) - norm_frame.min(axis=0))
                    if scale > 0:
                        norm_frame /= scale
                    
                    processed_seq.append(norm_frame.flatten())
                
                self.samples.append(torch.tensor(np.array(processed_seq), dtype=torch.float32))
                self.labels.append(phrase_to_idx[phrase])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

def pad_collate(batch):
    (xx, yy) = zip(*batch)
    # Pad sequences to the same length for the batch
    xx_pad = nn.utils.rnn.pad_sequence(xx, batch_first=True, padding_value=0)
    return xx_pad, torch.tensor(yy)

# --- TRAINING SCRIPT ---
def train():
    PHRASES = ["HELLO", "YES", "NO", "STOP", "HELP"]
    phrase_to_idx = {p: i for i, p in enumerate(PHRASES)}
    
    if not os.path.exists("mvp_data") or len(os.listdir("mvp_data")) == 0:
        print("Error: No data found in 'mvp_data/'. Please run collect_mvp_data.py first.")
        return

    dataset = MVPDataset("mvp_data", phrase_to_idx)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=pad_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # input_dim = 42 lip points * 2 (x,y) = 84
    model = LipReadingGRU(input_dim=84, hidden_dim=64, num_classes=len(PHRASES)).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Starting training...")
    for epoch in range(50):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/50], Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "mvp_lip_model.pth")
    print("Training complete. Model saved as 'mvp_lip_model.pth'.")

if __name__ == "__main__":
    train()
