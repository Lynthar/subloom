"""HTTP server exposing subloom as a job-based API.

`create_app` is intentionally not re-exported here — importing it eagerly
would chain into the transcribe pipeline (ffmpeg, faster-whisper, ollama)
just to look at request schemas. Import directly when you need it:

    from subloom.server.app import create_app
"""
