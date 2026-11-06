# subloom

[![license](https://img.shields.io/github/license/Lynthar/subloom)](LICENSE)

Local-first CLI and HTTP pipeline that transcribes video with faster-whisper and translates subtitles with a local LLM (Ollama or any OpenAI-compatible server) or with DeepL / Google

English | [简体中文](README.zh-CN.md)

Give it a video file and it transcribes the subtitles, then translates them into
the language you specify. What I set out to build is a scriptable pipeline for
producing subtitles in bulk, and unless you point translation at another machine
or a cloud API, none of your media or subtitle content leaves this one. The CLI and the HTTP
service run the same code; the service just puts a job queue in front of it.

```mermaid
flowchart LR
    V[video.mp4] --> F[ffmpeg<br>audio] --> W[faster-whisper<br>transcription] --> X[WhisperX<br>word alignment, optional] --> T[timing<br>VAD, padding, splitting] --> O[LLM<br>translation] --> S[video.zh.srt]
```

## Install

Not published on PyPI; install from source. Prerequisites: Python 3.11+, `ffmpeg`
on PATH, and a translation service: a running Ollama, any OpenAI-compatible server
(LM Studio, llama.cpp, vLLM, a cloud API), or a DeepL / Google Cloud Translation key.

```bash
git clone https://github.com/Lynthar/subloom.git
cd subloom
pip install -e .
```

Two extras worth having:

```bash
pip install -e '.[whisperx]'   # word-level forced alignment
pip install -e '.[serve]'      # the HTTP service
```

For mostly Chinese, Japanese or Korean audio, `pip install -e '.[qwen3-asr]'` adds
Qwen3-ASR as an alternative to Whisper; switch with
`subloom config set asr.backend qwen3_asr`. Details are in the user guide.

An NVIDIA GPU is not required, but it makes a big difference.

## Usage

Run this first: it checks ffmpeg and your GPU, lets you choose the speech
recognition and translation backends, and downloads the models.

```bash
subloom quickstart
```

Then:

```bash
subloom process video.mp4 -t zh
subloom process video.mp4 -t zh -t ja --bilingual
subloom batch ./videos/ -t zh --recursive -w 2
subloom transcribe video.mp4              # transcription only
subloom translate video.en.srt -t zh      # translate an existing SRT
subloom serve --host 127.0.0.1 --port 8765
```

Output files go in the same directory as the source: `{name}.{lang}.srt`, or
`{name}.{source}-{target}.srt` for bilingual. Translation uses ten built-in
prompt templates. `batch` processes a directory, optionally recursively, with a
worker count you set.

Before translating, subtitles cut mid-sentence are joined back up, and an LLM backend reads
the whole transcript once so each character's name comes out the same throughout; `--names`
gives the official spellings. `--drop-interjections` leaves out lines that are only "uh" or
"ah". The user guide has the details.

`serve` binds to `127.0.0.1` and expects a bearer token — generate one with
`openssl rand -hex 32` and pass it as `SUBLOOM_TOKEN`. The token travels
over plain HTTP, so put TLS in front if you move that binding off loopback. Also
treat it as equivalent to filesystem access: anyone holding it can ask the
service to read any path the process can read. `--no-auth` exists for
single-user local runs and only warns when used off loopback.

## Configuration

`~/.config/subloom/config.yaml`, created on first run. The keys you're
likely to change:

| Key | Default |
|---|---|
| `whisper.model` | `large-v3` |
| `whisper.device` / `compute_type` | `cuda` / `float16` |
| `whisper.use_whisperx` | `true` |
| `translation.backend` | `ollama` (or `openai`, `deepl`, `google`) |
| `ollama.model` | `qwen2.5:14b` |
| `ollama.host` | `http://localhost:11434` |
| `timestamp.mode` | `minimal` |
| `max_workers` | `2` |

`config/default.yaml` in the repository is a commented reference copy — it isn't
read at runtime.

## Limitations

- **SRT output only** — no WebVTT, ASS or TTML.
- **Translation needs a service you configure**: Ollama, an OpenAI-compatible server,
  DeepL or Google. Without it you can still `transcribe`, but that's as far as it
  goes. Only the configured backend is used: no fallback to another, no bundled
  model, and nothing goes to a cloud service unless you set one up.
- **Whisper's 99 languages refer to transcription.** Translation has proper
  language names for fourteen; anything else gets passed to the model as a bare
  language code.
- **It picks the audio track by channel count**, so a file with a 2.0 main track
  and a 5.1 commentary track can end up transcribing the commentary. There's no
  `--audio-track` flag yet.
- **The HTTP service keeps its job queue in memory.** Restarting loses the queue,
  and there's no per-job cancel or real progress percentage.

## Documentation

- [User guide](docs/user-guide.md) — installation per platform, every option,
  privacy and network boundaries, troubleshooting. Written in Chinese.

## License

GNU Affero General Public License v3.0 only — see [LICENSE](LICENSE).
Copyright (c) 2026 Lynthar.

This project depends on [pysrt](https://pypi.org/project/pysrt/), which is
GPLv3; the AGPLv3 of the combined work is what you receive it under.
