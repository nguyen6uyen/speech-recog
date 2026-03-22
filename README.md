# Speech Recognition

A web-based interface that converts lip movements into speech to assist individuals with verbal disabilities

![Guide](https://github.com/nguyen6uyen/speech-recog/blob/main/guide.gif)

- Hold the middle button until you finish your speech, then wait to transform it to script
- The Fast Mode toggle on if you do not want AI automatic sentence improvment
- Auto Speak on if you want to hear the transcript out loud

Web based through Hugging Face: [here](https://huggingface.co/spaces/DoubleNguyen/SpeechRecognition)

Relies on a [model](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages?tab=readme-ov-file#autoavsr-models) trained on the [Lip Reading Sentences 3](https://mmai.io/datasets/lip_reading/) dataset as part of the [Auto-AVSR project](https://github.com/mpc001/auto_avsr).

Adapt and take inspiration from [Chaplin](https://github.com/amanvirparhar/chaplin).

## Quick Start

### 1. Project Setup

First, clone the repository and navigate into the directory:

```sh
git clone git@github.com:nguyen6uyen/speech-recog.git
cd speech-recog
```

### 2. Download Models

Run the setup script to download the required VSR models (approx. 1GB):

```sh
chmod +x setup.sh
./setup.sh
```

### 3. API Keys Configuration
This project uses **Google Gemini** for text refinement and **ElevenLabs** for realistic speech synthesis.

You need to set the following environment variables before running the app:

```sh
export GOOGLE_API_KEY="your_google_api_key"
export ELEVEN_LABS_API_KEY="your_elevenlabs_api_key"
```

> **Note**: You can get a free Google API key [here](https://aistudio.google.com/app/apikey) and an ElevenLabs key [here](https://elevenlabs.io).

### 4. Install Python Dependencies

We use [`uv`](https://github.com/astral-sh/uv) for fast Python package management.

1.  **Install uv** (if you haven't already):
    ```sh
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
2.  **Install Dependencies**:
    The project dependencies will be automatically installed when you run the application using `uv run`.

## Usage

### Web Interface (Recommended)

The web interface is the most reliable way to run Chaplin, especially on macOS where terminal camera permissions can be tricky.

1.  **Start the Server**:
    ```sh
    uv run --with-requirements requirements.txt --python 3.12 server.py
    ```

2.  **Open in Browser**:
    Go to **[http://localhost:8000](http://localhost:8000)** in your web browser.

3.  **Start Lip Reading**:
    -   Allow camera access when prompted.
    -   Click and hold the **"Hold to Record"** button (or press Spacebar).
    -   Mouth a sentence silently.
    -   Release the button to see the transcription.

### Desktop Application (Alternative)

If you prefer a standalone window:

```sh
uv run --with-requirements requirements.txt --python 3.12 main.py config_filename=./configs/LRS3_V_WER19.1.ini detector=mediapipe
```
*Note: You may need to grant your terminal permission to access the camera in your system settings.*

## Troubleshooting

-   **Camera not working?**: Try the Web Interface method (`server.py`). Browsers handle camera permissions much reliably than terminal applications.
-   **Model not found?**: Ensure you ran `./setup.sh` and it completed successfully. Check `benchmarks/LRS3/models/` for the `.pth` file.
-   **LLM/TTS Error?**: Ensure `GOOGLE_API_KEY` and `ELEVEN_LABS_API_KEY` are set in your environment. Use `printenv` to verify.
