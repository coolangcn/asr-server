# GEMINI.md - Project Overview

This document provides a comprehensive overview of the ASR (Automatic Speech Recognition) Server project.

## Project Overview

This project is a Python-based web server that provides advanced speech recognition services. It combines Automatic Speech Recognition (ASR) with Speaker Verification (SV) to not only transcribe audio but also identify who is speaking from a database of registered users.

**Core Technologies:**

*   **Web Framework:** Flask
*   **ASR Engine:** `funasr` (using the `iic/SenseVoiceSmall` model)
*   **Speaker Verification:** `modelscope` (using `eres2net_large` and `rdino_ecapa` models for robust, cross-validated speaker ID)
*   **Machine Learning:** PyTorch
*   **Audio Processing:** FFmpeg

**Key Components:**

*   `asr_server.py`: The main Flask application that exposes the `/transcribe` API endpoint.
*   `register_speaker.py`: A command-line utility to enroll new speakers by creating voice embeddings.
*   `speaker_db_multi.json`: A JSON database that stores the voice embeddings for all registered speakers.
*   `requirements_stable.txt`: A list of all required Python packages.
*   `start_asr_server.bat` / `start_register.bat`: Convenience scripts for running the main services on Windows.

## Building and Running

### Prerequisites

1.  **Python:** A Python environment is required. The project uses a virtual environment located at `D:\AI\asr_env`.
2.  **FFmpeg:** FFmpeg must be installed and accessible in the system's PATH. The server will fail to start if it's not found.
3.  **Python Dependencies:** Install all required packages using the following command:
    ```bash
    pip install -r requirements_stable.txt
    ```

### 1. Registering a Speaker

Before the server can identify speakers, they must be registered.

1.  Place a clear audio sample (3-10 seconds is ideal) of the speaker in the project directory (e.g., `speaker_audio.wav`).
2.  Run the `register_speaker.py` script with the speaker's name and the path to their audio file.

**Command:**

```bash
# Activate the virtual environment first
# Example:
python register_speaker.py --name "JohnDoe" --audio "path/to/speaker_audio.wav"
```

The `start_register.bat` file provides a hardcoded example for registering the speaker "mama" with `mama.wav`.

### 2. Running the ASR Server

Once speakers are registered, you can start the main server.

**Command:**

Use the provided batch file to start the server. It handles activating the virtual environment and running the Python script.

```bash
start_asr_server.bat
```

Alternatively, you can run it manually:

```bash
# Activate the virtual environment first
python asr_server.py
```

The server will start on `http://0.0.0.0:5000`.

### 3. Using the API

Send a `POST` request to the `/transcribe` endpoint with an audio file.

*   **Endpoint:** `http://localhost:5000/transcribe`
*   **Method:** `POST`
*   **Body:** `multipart/form-data` with a key `audio_file` containing the audio to be transcribed.

**Example using cURL:**

```bash
curl -X POST -F "audio_file=@/path/to/your/audio.wav" http://localhost:5000/transcribe
```

The server will return a JSON object containing the full transcription and a breakdown of segments with speaker, emotion, and timing information.

## Development Conventions

*   **Configuration:** All major settings (model names, thresholds, server host/port) are centralized in the `Config` class within `asr_server.py`.
*   **Virtual Environment:** All development and execution should be done within the designated Python virtual environment (`D:\AI\asr_env`).
*   **Speaker Database:** The `speaker_db_multi.json` file is the single source of truth for speaker identity. It is managed by the `register_speaker.py` script.
*   **Cross-Validation:** The system uses two different speaker verification models to improve identification accuracy. A speaker is only confirmed if both models agree.
