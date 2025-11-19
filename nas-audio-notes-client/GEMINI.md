# Project Overview

This project is an AI-powered audio transcription and archiving system. It automatically processes audio files from a designated folder, transcribes them using an external ASR (Automatic Speech Recognition) service, and provides a web-based interface to view and analyze the transcriptions.

The system consists of two main Python services:

1.  `transcribe.py`: A background service that continuously monitors a source directory for new audio files (`.m4a`, `.mp3`, etc.). When a new file appears, it converts it to a compatible WAV format, sends it to an ASR API for transcription, and saves the resulting text and metadata (including speaker diarization) into an SQLite database (`transcripts.db`) and as a `.txt` file. Processed audio files are moved to a `processed` subdirectory.
2.  `web_viewer.py`: A Flask-based web application that provides a user-friendly interface to the transcribed data. It reads from the same SQLite database and offers several views:
    *   **Dashboard**: Shows system status (ASR server online/offline, pending files) and a list of the latest transcriptions.
    *   **Timeline**: Displays transcriptions in a chronological, chat-like format, separating entries by speaker.
    *   **Statistics**: Provides analytics on speaker activity, such as the number of spoken segments and average duration.

The entire application is designed to be deployed on a NAS (Network Attached Storage) device or a similar always-on server.

## Technologies Used

*   **Backend**: Python 3
*   **Web Framework**: Flask
*   **Database**: SQLite
*   **External Tools**: `ffmpeg` for audio conversion.
*   **Deployment**: The project includes shell (`.sh`) and PowerShell (`.ps1`) scripts for easy deployment on Linux and Windows.

# Building and Running

The project does not have a formal build process or package manager like `pip`. Dependencies are expected to be installed manually. The core dependencies mentioned are `flask` and `requests`.

## Running the Services

The two main services, the transcriber and the web viewer, must be run as separate processes. The scripts accept a `--source-path` argument to specify the main directory for audio files, which will also be used for the database and transcript files.

**On Linux (using the deployment script):**

The `deploy_nas.sh` script is the recommended way to run the services on a Linux-based system (like a NAS). It handles stopping any existing processes and starting the new ones in the background.

1.  **Make the script executable:**
    ```bash
    chmod +x deploy_nas.sh
    ```

2.  **Run the deployment script:**
    ```bash
    # To use the default path (/volume2/download/records/Sony-2)
    ./deploy_nas.sh

    # To override the source path
    export SOURCE_PATH_OVERRIDE=/path/to/your/audio
    ./deploy_nas.sh
    ```

**On Windows (Manual Execution):**

The `README.md` mentions a `one_click_deploy.ps1` script, but it is not present in the file list. You can run the services manually from the command line.

1.  **Start the transcription service:**
    ```powershell
    python transcribe.py --source-path D:\path\to\your\audio
    ```

2.  **Start the web viewer service in a separate terminal:**
    ```powershell
    # The web interface will be available at http://localhost:5009
    python web_viewer.py --source-path D:\path\to\your\audio
    ```

## Viewing the Output

*   **Web Interface**: Access `http://<server_ip>:5009` in a web browser.
*   **Log Files**: `transcribe.log` and `web_viewer.log` are created to log the output of the respective services.
*   **Database**: The raw data is stored in `transcripts.db` in the source directory.
*   **Text Transcripts**: Formatted text files are saved in the `transcripts` subdirectory.

# Development Conventions

*   **Configuration**: Configuration is handled via hardcoded constants at the top of each Python file. Key settings include the ASR server URL, file paths, and the web server port.
*   **Command-line Arguments**: Both scripts accept a `--source-path` argument to allow for some flexibility in the location of the audio files.
*   **Error Handling**: The scripts have basic error handling, primarily through `try...except` blocks, with errors logged to the console or log files.
*   **Frontend**: The web interface is self-contained within `web_viewer.py`. All HTML, CSS, and JavaScript are embedded in a single string, making it a single-file distribution.
*   **API**: The web viewer exposes two simple JSON API endpoints: `/api/status` and `/api/data`.
