
# Deployment Guide: Hugging Face Spaces

This guide explains how to deploy the Speech Recognition application to Hugging Face Spaces so it can be accessed publicly via a web browser.

## 1. Create a Space
1.  Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **Create new Space**.
2.  **Space name**: `speech-recog` (or your choice).
3.  **License**: `mit` (recommended).
4.  **SDK**: `Docker`.
5.  **Hardware**: `CPU Basic (Free)` is sufficient for the `0.5b` model.

## 2. Upload Files
You can upload files via the web interface or using Git.

### Option A: Web Interface
Upload the following files to your Space:
- `Dockerfile`
- `start.sh`
- `requirements.txt`
- `setup.sh`
- `server.py`
- `main.py`
- `step2_lip_to_word.py`
- `step3_post_processing.py`
- `step4_llm_layer.py`
- `chaplin.py`
- `export_onnx.py`
- `collect_mvp_data.py`
- `index.html`
- `configs/` folder
- `pipelines/` folder
- `hydra_configs/` folder

### Option B: Git (Recommended)
1.  Clone your Space:
    ```sh
    git clone https://huggingface.co/spaces/YOUR_USERNAME/speech-recog
    ```
2.  Copy all project files into the cloned directory.
3.  Push changes:
    ```sh
    git add .
    git commit -m "Initial commit"
    git push
    ```

## 3. Configuration (Optional)
By default, the app uses `qwen2.5:0.5b`. To use a larger model (requires GPU Space):
1.  Go to **Settings** in your Space.
2.  New **Variable**: `OLLAMA_MODEL` value: `qwen2.5:7b`.

## 4. Access
Once built, your app will be available at:
`https://huggingface.co/spaces/DoubleNguyen/SpeechRecognition`

## 5. (Important) Custom Model Upload
The file `mvp_lip_model.pth` (Custom VSR Model) could not be pushed via Git due to binary file restrictions.
If you want to use the "Predict Landmarks" feature (the custom model):
1.  Go to the **Files** tab in your Hugging Face Space.
2.  Click **Add file** -> **Upload files**.
3.  Upload `mvp_lip_model.pth` from your local machine.
4.  Commit changes. The Space will rebuild and load the model.
