
from huggingface_hub import hf_hub_download
import os

def download_file(repo_id, filename, local_dir):
    print(f"Downloading {filename} from {repo_id} to {local_dir}...")
    os.makedirs(local_dir, exist_ok=True)
    hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir, local_dir_use_symlinks=False)

if __name__ == "__main__":
    print("Starting model downloads...")
    
    # LM
    lm_dir = "benchmarks/LRS3/language_models/lm_en_subword"
    download_file("Amanvir/lm_en_subword", "model.json", lm_dir)
    download_file("Amanvir/lm_en_subword", "model.pth", lm_dir)

    # VSR
    vsr_dir = "benchmarks/LRS3/models/LRS3_V_WER19.1"
    download_file("Amanvir/LRS3_V_WER19.1", "model.json", vsr_dir)
    download_file("Amanvir/LRS3_V_WER19.1", "model.pth", vsr_dir)
    
    print("All models downloaded successfully.")
