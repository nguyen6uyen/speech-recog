#!/bin/bash

# Start the application
# Use specific host and port for Hugging Face Spaces (Spaces run on port 7860)
uvicorn server:app --host 0.0.0.0 --port 7860
