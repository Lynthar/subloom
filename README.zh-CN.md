# subloom

[![license](https://img.shields.io/github/license/Lynthar/subloom)](LICENSE)

本地优先的视频字幕流水线：faster-whisper 转写 + 本地大模型翻译（Ollama 或任意 OpenAI 兼容服务），也可用 DeepL / Google，CLI 与 HTTP 服务同源

[English](README.md) | 简体中文

给它一个视频文件，它转写出字幕，再翻译成你指定的语言。我计划的是一条能脚本化的自动
批量字幕生产流水线，并且除非你把翻译指向另一台机器或云端 API，否则你的音视频和字幕
内容都不会离开本地。CLI 和 HTTP 服务跑的是同一份代码，服务端只是在前面加了一个任务
队列。

```mermaid
flowchart LR
    V[video.mp4] --> F[ffmpeg<br>抽音] --> W[faster-whisper<br>转写] --> X[WhisperX<br>逐词对齐，可选] --> T[时轴后处理<br>VAD、留白、拆句] --> O[大模型<br>翻译] --> S[video.zh.srt]
```

## 安装

**没有发布到 PyPI**，需要从源码安装。前置条件：Python 3.11+、PATH 里有 `ffmpeg`、以及
一个翻译服务：正在运行的 Ollama，任意 OpenAI 兼容服务（LM Studio、llama.cpp、vLLM、
云端 API），或者 DeepL / Google Cloud Translation 的 key。

```bash
git clone https://github.com/Lynthar/subloom.git
cd subloom
pip install -e .
```

两个值得装的可选组件：

```bash
pip install -e '.[whisperx]'   # 逐词强制对齐
pip install -e '.[serve]'      # HTTP 服务
```

片源以中文、日语、韩语为主时，`pip install -e '.[qwen3-asr]'` 可以装上 Qwen3-ASR 代替 Whisper，
再用 `subloom config set asr.backend qwen3_asr` 切换。细节见用户手册。

NVIDIA 显卡不是必需的，但有没有它差别很大。

## 用法

先跑这一条，它会检查 ffmpeg 和显卡，让你选语音识别与翻译后端，并把模型下载下来：

```bash
subloom quickstart
```

然后：

```bash
subloom process video.mp4 -t zh
subloom process video.mp4 -t zh -t ja --bilingual
subloom batch ./videos/ -t zh --recursive -w 2
subloom transcribe video.mp4              # 只转写
subloom translate video.en.srt -t zh      # 翻译现成的 SRT
subloom serve --host 127.0.0.1 --port 8765
```

输出文件放在源文件的同一个目录下：`{文件名}.{语言}.srt`，双语是
`{文件名}.{源}-{目标}.srt`。翻译内置十套提示词模板。`batch` 处理一个目录，可以递归，
并发数自己指定。

翻译前会把说到一半被切开的字幕并回整句；大模型后端还会先通读一遍全文，让同一个人名从头到尾
译法一致，`--names` 可以给出正式写法。`--drop-interjections` 滤掉只有「嗯」「啊」的字幕行。
细节见用户手册。

`serve` 默认绑定 `127.0.0.1`，需要 bearer token——用 `openssl rand -hex 32` 生成一个，
通过 `SUBLOOM_TOKEN` 传入。token 走的是明文 HTTP，所以一旦把绑定地址挪出环回，
前面就得加 TLS。另外要把它当成文件系统访问权限来对待：拿着 token 的人可以让服务读取
它权限内的任意路径。`--no-auth` 是给单人本机用的，在非环回地址上用它只会警告、不会拒绝。

## 配置

`~/.config/subloom/config.yaml`，首次运行时创建。常改的几个键：

| 键 | 默认 |
|---|---|
| `whisper.model` | `large-v3` |
| `whisper.device` / `compute_type` | `cuda` / `float16` |
| `whisper.use_whisperx` | `true` |
| `translation.backend` | `ollama`（或 `openai`、`deepl`、`google`） |
| `ollama.model` | `qwen2.5:14b` |
| `ollama.host` | `http://localhost:11434` |
| `timestamp.mode` | `minimal` |
| `max_workers` | `2` |

仓库里的 `config/default.yaml` 是一份带注释的参考副本，**运行时不读它**。

## 能力边界

- **只输出 SRT**，不支持 WebVTT、ASS 和 TTML。
- **翻译需要一个你配置好的服务**：Ollama、OpenAI 兼容服务、DeepL 或 Google。没有的话仍然可以
  `transcribe`，但也只能做到这一步。只用配置里选中的那一个后端，不会换用别的，不自带模型，
  没配云端就不会往云端发。
- **Whisper 的 99 种语言指的是转写能力。** 翻译这边只有十四种有正式语言名，其余的会把
  语言码原样交给模型。
- **音轨是按声道数选的**，所以一个 2.0 主音轨加 5.1 导演解说的片源，可能转写的是解说。
  目前还没有 `--audio-track` 这个旗标。
- **HTTP 服务的任务队列在内存里。** 重启会丢掉队列，也没有单任务取消和真实的进度百分比。

## 文档

- [用户指南](docs/user-guide.md) —— 分平台安装、每个选项、隐私与网络边界、故障排查。

## 许可证

GNU Affero 通用公共许可证 v3.0 only —— 见 [LICENSE](LICENSE)。Copyright (c) 2026 Lynthar。

本项目依赖 [pysrt](https://pypi.org/project/pysrt/)（GPLv3），整体以 AGPLv3 分发。
