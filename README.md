# Speech Recognition

A web-based interface that converts lip movements into speech to assist individuals with verbal disabilities

## Quick Start

### 1. Project Setup

First, clone the repository and navigate into the directory:

```sh
git clone https://github.com/amanvirparhar/chaplin
cd chaplin
```

### 2. Download Models

Run the setup script to download the required VSR models (approx. 1GB):

```sh
chmod +x setup.sh
./setup.sh
```

### 3. Install & Configure Ollama

This project uses [Ollama](https://ollama.com/) for the LLM layer to correct the lip-reading output.

1.  **Install Ollama**: Download from [ollama.com](https://ollama.com/).
2.  **Pull the Model**: Run the following command in your terminal to download the required model (`qwen2.5:7b`):

    ```sh
    ollama pull qwen2.5:7b
    ```
3.  **Start Ollama**: Ensure Ollama is running in the background (usually `ollama serve`).

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
-   **Ollama connection error?**: Make sure Ollama is running (`ollama serve`).
