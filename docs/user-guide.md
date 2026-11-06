# subloom 使用指南

本地化视频字幕生成和翻译工具的完整使用指南。

---

## 目录

1. [安装与配置](#一安装与配置)
   - [系统要求](#系统要求)
   - [各平台安装](#各平台安装)
   - [GPU 加速设置](#gpu-加速设置)
   - [HuggingFace 镜像](#huggingface-镜像中国用户)
   - [硬件与模型推荐](#硬件与模型推荐)
   - [翻译后端](#翻译后端)
   - [更新方法](#更新方法)
2. [基本使用](#二基本使用)
   - [快速开始](#快速开始)
   - [路径格式说明](#路径格式说明)
   - [常见用法示例](#常见用法示例)
   - [配置文件](#配置文件)
3. [命令索引](#三命令索引)
   - [process - 完整处理](#process---完整处理)
   - [batch - 批量处理](#batch---批量处理)
   - [transcribe - 仅转录](#transcribe---仅转录)
   - [translate - 仅翻译](#translate---仅翻译)
   - [config - 配置管理](#config---配置管理)
   - [quickstart - 初始化向导](#quickstart---初始化向导)
4. [HTTP 服务模式](#四http-服务模式)
   - [安装与启动](#安装与启动)
   - [接口说明](#接口说明)
   - [作为守护进程运行](#作为守护进程运行)
   - [服务模式注意事项](#服务模式注意事项)
5. [隐私与网络边界](#五隐私与网络边界)
   - [四条边界](#四条边界)
   - [完全离线部署](#完全离线部署)
6. [故障排查](#六故障排查)
7. [附录](#七附录)
   - [支持的语言](#支持的语言)
   - [时间戳后处理](#时间戳后处理)
   - [语气词过滤](#语气词过滤)
   - [碎句合并](#碎句合并)
   - [人名统一](#人名统一)
   - [翻译提示词配置](#翻译提示词配置)
   - [降级与退出码](#降级与退出码)

---

# 一、安装与配置

## 系统要求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| **操作系统** | Windows 10 / macOS 10.15 / Ubuntu 20.04 | Windows 11 / macOS 14 / Ubuntu 22.04 |
| **Python** | 3.11+ | 3.12+ |
| **内存** | 8 GB | 16 GB+ |
| **显卡** | 无（CPU 可运行，较慢） | NVIDIA GPU 6GB+ VRAM |
| **存储** | 10 GB（模型缓存） | SSD 推荐 |

**必需依赖**：
- Python 3.11+
- ffmpeg
- Ollama

---

## 各平台安装

### Windows

```powershell
# 1. 安装 Python（从 https://python.org/downloads 下载）
#    重要：勾选 "Add Python to PATH"

# 2. 安装 ffmpeg（任选一种）
choco install ffmpeg          # Chocolatey
winget install ffmpeg         # winget
# 或手动下载：https://ffmpeg.org/download.html

# 3. 安装 Ollama（从 https://ollama.ai/download 下载）

# 4. 启动 Ollama 服务（保持终端运行）
ollama serve

# 5. 安装 subloom（新终端）
git clone https://github.com/Lynthar/subloom.git
cd subloom
pip install -e .

# 6. 运行设置向导
subloom quickstart
```

### macOS

```bash
# 使用 Homebrew
brew install python@3.11 ffmpeg ollama

# 启动 Ollama
brew services start ollama  # 后台服务
# 或 ollama serve           # 前台运行

# 安装 subloom
git clone https://github.com/Lynthar/subloom.git
cd subloom
pip install -e .
subloom quickstart
```

### Linux (Ubuntu/Debian)

```bash
# 安装依赖
sudo apt update
sudo apt install python3 python3-pip python3-venv ffmpeg

# 安装 Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve  # 保持运行

# 安装 subloom（推荐使用虚拟环境）
git clone https://github.com/Lynthar/subloom.git
cd subloom
python3 -m venv venv
source venv/bin/activate
pip install -e .
subloom quickstart
```

**其他发行版**：
- **Fedora**: `sudo dnf install python3 python3-pip ffmpeg`
- **Arch**: `sudo pacman -S python python-pip ffmpeg`

### 可选安装项（extras）

`pip install -e .` 是基础安装，根据使用场景可以追加：

| 命令 | 提供的功能 |
|------|----------|
| `pip install -e '.[whisperx]'` | **强烈推荐**。WhisperX 用 wav2vec2 强制对齐，词级时间戳精度比基础 faster-whisper 高很多，直接影响字幕同步感 |
| `pip install -e '.[qwen3-asr]'` | 可选的 Qwen3-ASR 语音识别后端，中日韩识别更强，见[下文](#qwen3-asr语音识别可选后端) |
| `pip install -e '.[serve]'` | 启用 HTTP 服务模式（`subloom serve`），加 fastapi + uvicorn |
| `pip install -e '.[dev]'` | 开发依赖：pytest / ruff / mypy |

可以叠加：`pip install -e '.[whisperx,serve]'`。

开发与 CI 走 [uv](https://docs.astral.sh/uv/)：`uv sync --locked --extra dev --extra serve` 按仓里的 `uv.lock` 装出与 CI 完全相同的一套版本，之后用 `uv run pytest` / `uv run ruff check` 跑门禁。`pip install -e '.[dev]'` 仍然可用，只是版本不锁。

---

## GPU 加速设置

GPU 加速可大幅提升处理速度（5-10 倍）。

### 检查 GPU 状态

```bash
# 检查 PyTorch 是否识别 GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 查看 GPU 信息
nvidia-smi
```

### 按显卡型号安装 PyTorch

| GPU 系列 | 架构 | 安装命令 |
|----------|------|----------|
| RTX 50xx (5090, 5080...) | Blackwell | `pip install torch --index-url https://download.pytorch.org/whl/cu128` |
| RTX 40xx (4090, 4080...) | Ada | `pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| RTX 30xx (3090, 3080...) | Ampere | `pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| RTX 20xx / GTX 16xx | Turing | `pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| 无 GPU | - | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |

> **注意**：RTX 50 系列必须使用 CUDA 12.8 (`cu128`)，否则会报 "sm_120 is not compatible" 错误。

### 完整安装流程

```bash
# 1. 卸载现有 PyTorch
pip uninstall torch torchvision torchaudio -y

# 2. 安装对应版本（以 RTX 40 系列为例）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. 验证
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"
```

---

## HuggingFace 镜像（中国用户）

Whisper 模型从 HuggingFace 下载，国内用户可配置镜像加速。

### 临时设置

```bash
# Linux / macOS
export HF_ENDPOINT=https://hf-mirror.com
subloom process video.mp4 -t zh

# Windows PowerShell
$env:HF_ENDPOINT = "https://hf-mirror.com"

# Windows CMD
set HF_ENDPOINT=https://hf-mirror.com
```

### 永久设置

```bash
# macOS (zsh)
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.zshrc && source ~/.zshrc

# Linux (bash)
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc && source ~/.bashrc

# Windows (需要管理员权限)
[Environment]::SetEnvironmentVariable("HF_ENDPOINT", "https://hf-mirror.com", "User")
```

### 命令行指定

```bash
subloom process video.mp4 -t zh --hf-mirror https://hf-mirror.com
```

**可用镜像**：`https://hf-mirror.com`、`https://huggingface.sukaka.top`

---

## 硬件与模型推荐

### Whisper 模型（语音识别）

| 显存 | 推荐模型 | 速度 | 准确度 |
|------|---------|------|--------|
| < 4 GB | `small` | 快 | 较好 |
| 4-6 GB | `medium` | 中等 | 高 |
| 6 GB+ | `large-v3` | 较慢 | 最高 |

```bash
# 设置默认模型
subloom config set whisper.model large-v3

# 自动选择
subloom transcribe video.mp4 --auto-model
```

### Qwen3-ASR（语音识别，可选后端）

默认后端是 Whisper。片源以中文、日语、韩语为主时可以换成 Qwen3-ASR：
它负责识别，配套的 Qwen3-ForcedAligner 负责词级时间轴，两者都在本地运行。

```bash
pip install -e '.[qwen3-asr]'
subloom config set asr.backend qwen3_asr
```

只想某一次换后端时，`process`、`batch`、`transcribe` 都可以加 `--asr-backend qwen3_asr`，不改配置。

| 配置项 | 默认 | 说明 |
|------|------|------|
| `qwen3_asr.model` | `Qwen/Qwen3-ASR-1.7B` | 约 4.7 GB；显存紧张换 `Qwen/Qwen3-ASR-0.6B`（约 1.9 GB） |
| `qwen3_asr.aligner_model` | `Qwen/Qwen3-ForcedAligner-0.6B` | 约 1.8 GB，提供词级时间轴 |
| `qwen3_asr.device` | `cuda` | `cuda` 或 `cpu`；没有 CUDA 时自动退到 CPU 并提示 |
| `qwen3_asr.batch_size` | `4` | 音频按约 15 秒一块切开，每批推理这么多块；显存不够就调小，显存富余调大能提速 |

需要知道的几点：

- **识别覆盖 30 种语言，时间轴对齐只覆盖 11 种**：中、英、粤、法、德、意、日、韩、葡、俄、西。
  其余语言能识别，但字幕时间可能不准，结果里会列为降级（见[降级与退出码](#降级与退出码)）。
- **建议装 FlashAttention 2**（`flash-attn`，需要 CUDA）。没有它时，超过约 8 秒的片段识别准确度会下降；
  程序照常运行，但结果里会列为降级。flash-attn 在 Windows 上没有官方安装包，多数 Windows 用户都会遇到这一条。
- **不报告语种识别的置信度**，处理结果里只显示语言码。
- **Whisper 专属的参数不起作用**：`--whisperx`、`--vad-filter`、`--vad-mode`、`--speech-pad`、
  `--min-silence`、`--auto-model`、`--batch-size`、`--whisper-model`。给了会提示「已忽略」。`transcribe --model` 对两个后端都有效。
- 显存：1.7B 组合在默认 `batch_size` 下实测峰值约 8 GB；0.6B 组合按权重估约 3.7 GB，另加推理开销，未实测。

### Ollama 模型（翻译）

| 显存 | 推荐模型 | 翻译速度 | 翻译质量 |
|------|---------|---------|---------|
| < 6 GB | `qwen2.5:7b` | 快 | 一般 |
| 6-12 GB | `qwen2.5:14b` | 中等 | 较好 |
| 16 GB+ | `qwen2.5:32b` | 较慢 | 最佳 |

```bash
# 设置默认模型
subloom config set ollama.model qwen2.5:14b
```

---

## 翻译后端

默认用 Ollama 翻译。`translation.backend` 选后端，后端自己的连接设置写在同名的一节里；
重试次数与超时放在 `translation` 节，各后端共用，换后端不用重配。

后端分两类，翻译方式不同：

| 后端 | `translation.backend` | 类别 | 适用 |
|------|----------------------|------|------|
| Ollama 原生接口 | `ollama`（默认） | 大模型 | 本机或局域网里的 Ollama |
| OpenAI 兼容接口 | `openai` | 大模型 | LM Studio、llama.cpp（`llama-server`）、vLLM、Ollama 的 `/v1`，或云端 API |
| DeepL | `deepl` | 机器翻译 | DeepL API（Free 或 Pro） |
| Google | `google` | 机器翻译 | Google Cloud Translation Basic（v2） |

- **大模型**照提示词翻：吃 `translation` 节里的 `temperature`、`max_batch_size` 和提示词模板，
  按模型大小收窄批量，漏句会逐条重试。
- **机器翻译**一批字幕作为一组文本发出，逐条对应返回：提示词模板、`temperature`、`max_batch_size`
  对它们不起作用（`--prompt-template` 会提示「已忽略」），也不做逐条重试——同样的原文，
  引擎每次给的结果一样。

**只会用你选中的那一个。** 它连不上、没有配置的模型或 API key 被拒时，`process` 在转录之前就报错退出
（退出码 1），不会改用别的后端，更不会转去云端。没设 API key 时，所有命令在动手之前就会报错。

### OpenAI 兼容接口

先填地址和模型，再切后端——`translation.backend` 设成 `openai` 时这两项必须已经有值，否则配置不予保存：

```bash
subloom config set openai.base_url http://localhost:1234/v1
subloom config set openai.model qwen2.5-14b-instruct
subloom config set translation.backend openai
subloom config check --verbose
```

- `base_url` 要带 `/v1`。常见的地址：LM Studio `http://localhost:1234/v1`，llama.cpp
  `http://localhost:8080/v1`，vLLM `http://localhost:8000/v1`，Ollama `http://localhost:11434/v1`。
  它没有默认值，不填就会报错。
- `model` 填服务端在 `<base_url>/models` 里列出的 id，`config check --verbose` 会把它们列出来。
  llama.cpp 列的是启动时的 `--alias`，没设就是模型文件名。
- **API key**：本地服务一般用不着。需要时设环境变量 `SUBLOOM_OPENAI_API_KEY`（优先），
  或者 `config set openai.api_key <key>`，后者会把 key 明文写进配置文件，`config show` 只显示「已设置」。
  通用的 `OPENAI_API_KEY` 不读，免得把给 OpenAI 的 key 发到别的地址。
- 用默认提示词时，会请服务端按 JSON Schema 约束输出，每条字幕都是必填项。服务端不接受就退到
  `json_object`，还不接受就不加约束、靠解析兜底。LM Studio 只认前一种，DeepSeek 只认后一种，两边都能用。
- 批量大小按模型名里的参数量自动收紧：32B 及以上用配置值，13B 到 32B 之间最多 6，再小最多 4。
  名字里看不出参数量的（比如 `gpt-4o-mini`），按云端大模型对待，直接用配置值。
- OpenAI 的推理模型（o 系列、GPT-5）不接受 `temperature: 0`，请求会被 400 拒掉，请换用非推理模型。
- `config pull-model` 只管 Ollama。其他服务的模型，在它们自己那边下载和加载。

### DeepL 与 Google

```bash
export SUBLOOM_DEEPL_API_KEY=<你的 key>     # 或 SUBLOOM_GOOGLE_API_KEY
subloom config set translation.backend deepl # 或 google
subloom config check
```

- key 也可以 `config set deepl.api_key <key>` / `config set google.api_key <key>` 写进配置文件，
  环境变量优先。`config check` 会实际请求一次（DeepL 查额度、Google 查语言列表），确认 key 被接受，不发字幕。
- **DeepL**：key 以 `:fx` 结尾的是 Free 版，自动走 `api-free.deepl.com`，否则走 Pro 的 `api.deepl.com`。
  每批最多 50 条；这一批连同前后各 3 条原文作为 `context` 一起发出，帮引擎理解上下文，DeepL 不对它计费。
  目标语言用带变体的代码：中文 `ZH-HANS` / `ZH-HANT`，英语 `EN-US`，葡萄牙语 `PT-BR`。
- **Google**：Cloud Translation Basic（v2），用 API key 认证，key 放在请求头里、不进 URL。
  每批最多 128 条；v2 没有上下文参数，每条独立翻译。中文目标 `zh` 发成 `zh-CN`。
- **额度用尽或 key 被拒**（DeepL 456 / 403，Google 403 / 400）**整次失败**，那个语言不写半翻的字幕；
  429 限流会退避重试，次数取 `translation.max_retries`。
- 某条返回空译文时保留原文，结果里记一条 `translation_incomplete` 降级（退出码 3）。

后端指向云端时，字幕文本会发给那家服务，见[隐私与网络边界](#五隐私与网络边界)第 2 条。
`process`、`batch` 和 `translate` 都可以临时用 `--translation-backend` 换后端，
`process` 和 `batch` 用 `--translation-model` 换模型（`translate` 用 `-m`；机器翻译后端没有模型可换）。

---

## 更新方法

```bash
# 标准更新
cd subloom
git pull origin main
pip install -e .

# 一键更新（含缓存清理）
cd subloom && git pull && find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null; pip install -e .

# 查看版本
subloom version
```

---

# 二、基本使用

## 快速开始

```bash
# 1. 确保 Ollama 正在运行
ollama serve

# 2. 运行设置向导（首次使用）
subloom quickstart

# 3. 处理视频
subloom process video.mp4 -t zh
```

---

## 路径格式说明

不同操作系统的路径格式有所不同：

### Windows

```powershell
# 使用反斜杠或正斜杠
subloom process "D:\Videos\movie.mp4" -t zh
subloom process "D:/Videos/movie.mp4" -t zh

# 路径包含空格时必须加引号
subloom process "C:\My Videos\movie.mp4" -t zh

# 当前目录
subloom process .\video.mp4 -t zh

# 批量处理
subloom batch "D:\Videos\" -t zh
```

### macOS / Linux

```bash
# 使用正斜杠
subloom process /Users/name/Videos/movie.mp4 -t zh

# 路径包含空格时加引号
subloom process "/Users/name/My Videos/movie.mp4" -t zh

# 使用 ~ 表示主目录
subloom process ~/Videos/movie.mp4 -t zh

# 当前目录
subloom process ./video.mp4 -t zh
```

### 通用建议

- **路径包含空格**：始终使用引号包裹
- **特殊字符**：避免使用 `& | < > ! $ ; ` 等特殊字符
- **中文路径**：支持，但建议使用英文路径避免潜在问题

---

## 常见用法示例

### 生成字幕

```bash
# 基本用法：视频 → 中文字幕
subloom process video.mp4 -t zh

# 多语言字幕
subloom process video.mp4 -t zh -t ja -t ko

# 双语字幕（原文 + 译文）
subloom process video.mp4 -t zh --bilingual

# 指定输出目录
subloom process video.mp4 -t zh -o ./output/

# 指定源语言（跳过自动检测）
subloom process video.mp4 -s en -t zh

# 滤掉只有「ああ」「うん」这类语气词的字幕行
subloom process video.mp4 -t zh --drop-interjections
```

### 批量处理

```bash
# 处理文件夹中所有视频
subloom batch ./videos/ -t zh

# 递归处理子文件夹
subloom batch ./videos/ -t zh --recursive

# 并发处理
subloom batch ./videos/ -t zh --workers 4
```

### 仅转录（不翻译）

```bash
# 生成原语言字幕
subloom transcribe video.mp4

# 指定语言
subloom transcribe video.mp4 --language en
```

### 仅翻译（已有字幕）

```bash
# 翻译现有字幕文件
subloom translate video.en.srt -t zh

# 生成双语字幕
subloom translate video.en.srt -t zh --bilingual
```

---

## 配置文件

### 位置

| 系统 | 路径 |
|------|------|
| Linux/macOS | `~/.config/subloom/config.yaml` |
| Windows | `%APPDATA%\subloom\config.yaml` |

从旧名字 subtitle-forge 升级时：同一位置的 `subtitle-forge` 目录会继续生效，直到你建了 `subloom` 目录（想换过来直接改目录名）。环境变量要跟着改名，`SUBTITLE_FORGE_*` 不再读取，换成 `SUBLOOM_*`；没改的话，用到它们的翻译后端和 `serve` 会在开工前报缺少密钥或 token。

### 常用命令

```bash
# 查看当前配置
subloom config show

# 修改配置
subloom config set whisper.model large-v3
subloom config set ollama.model qwen2.5:32b
subloom config set max_workers 4

# 重置配置
subloom config reset

# 系统检查
subloom config check --verbose
```

### 配置示例

```yaml
asr:
  backend: whisper               # 语音识别后端：whisper 或 qwen3_asr，设置写在同名的一节里

whisper:
  model: large-v3                # Whisper 模型
  device: cuda                   # cuda 或 cpu
  compute_type: float16          # float16, int8_float16, int8

  # VAD（语音活动检测）调参 - 影响转录段落的边界
  speech_pad_ms: 250             # VAD 检测到语音前后的填充（毫秒）
  min_silence_duration_ms: 700   # 切分相邻语句的最小静音时长（毫秒）

  # WhisperX（强烈推荐，需要 [whisperx] 安装 extra）
  use_whisperx: true             # 用 wav2vec2 做强制对齐
  whisperx_align: true

translation:
  backend: ollama                # 翻译后端：ollama、openai、deepl、google，设置写在同名的一节里
  request_timeout: 180.0         # 单次请求超时（秒）
  # 以下三项只对大模型后端（ollama、openai）有效
  temperature: 0.0               # 翻译用 0；之前是 0.3 容易丢段
  max_batch_size: 10             # 上限值；实际会按模型自适应：14B→6, ≤8B→4
  prompt_template_id: null       # 提示词模板 id；null 为内置提示词
  names: []                      # 人名的正式写法，见附录「人名统一」
  find_names: true               # 翻译前通读全片找人名的各种写法

ollama:
  model: qwen2.5:14b             # 翻译模型
  host: http://localhost:11434

output:
  encoding: utf-8
  keep_original: true            # 保留原语言字幕
  bilingual: false               # 默认双语模式
  drop_interjections: false      # 滤掉只有语气词的字幕行，见附录「语气词过滤」
  merge_fragments: true          # 把说到一半被切开的字幕并回整句，见附录「碎句合并」

max_workers: 2                   # 并发数
log_level: INFO
```

完整字段说明详见仓库内的 `config/default.yaml`。

早先版本把 `temperature`、`max_batch_size`、`max_retries`、`request_timeout` 和提示词都写在
`ollama` 节里。这样的配置文件照样能读，读取时会给出提示；用
`subloom config import <配置文件路径>` 可以按新格式写回。旧的自定义提示词正文会存成
用户模板 `custom`，并且仍是选中状态。

---

# 三、命令索引

## process - 完整处理

**功能**：视频 → 音频提取 → 语音识别 → 翻译 → 字幕文件

```bash
subloom process <video> -t <target_lang> [options]
```

| 参数 | 简写 | 说明 |
|------|------|------|
| `<video>` | - | 视频文件路径（必需） |
| `--target-lang` | `-t` | 目标语言，可多次指定（必需） |
| `--source-lang` | `-s` | 源语言（默认自动检测） |
| `--output-dir` | `-o` | 输出目录（默认视频所在目录） |
| `--asr-backend` | - | 语音识别后端：whisper, qwen3_asr（默认取配置） |
| `--whisper-model` | - | 指定 Whisper 模型 |
| `--translation-backend` | - | 翻译后端：ollama, openai, deepl, google（默认取配置） |
| `--translation-model` | - | 指定翻译模型（旧名 `--ollama-model` 仍可用） |
| `--keep-original / --no-keep-original` | - | 是否同时保存原语言字幕（默认保留）|
| `--bilingual` | - | 生成双语字幕（原文上译文下） |
| `--drop-interjections / --keep-interjections` | - | 滤掉只有语气词的字幕行（默认取配置 `output.drop_interjections`，见[语气词过滤](#语气词过滤)） |
| `--merge-fragments / --no-merge-fragments` | - | 把说到一半被切开的字幕并回整句（默认开，取配置 `output.merge_fragments`，见[碎句合并](#碎句合并)） |
| `--names` | - | 人名的正式写法，逗号或顿号分隔，如 `光希,琴音`（默认取配置 `translation.names`，见[人名统一](#人名统一)） |
| `--find-names / --no-find-names` | - | 翻译前通读全片原文找人名和它的各种写法（默认开，取配置 `translation.find_names`） |
| `--whisperx / --no-whisperx` | - | 使用/不使用 WhisperX |
| `--post-process / --no-post-process` | - | 启用/禁用时间戳后处理 |
| `--timestamp-mode` | - | 后处理模式：off, minimal, full |
| `--split-sentences / --no-split-sentences` | - | 按句子拆分多句字幕 |
| `--vad-mode` | - | VAD 预设：default, aggressive, relaxed, precise |
| `--speech-pad` | - | 语音填充时间（毫秒），覆盖配置 |
| `--min-silence` | - | 最小静音时长（毫秒），覆盖配置 |
| `--prompt-template` | `-p` | 使用指定提示词模板 |
| `--hf-mirror` | - | HuggingFace 镜像 URL |
| `--save-failed-log` | - | 仅保存翻译失败的 JSON 详情 |
| `--save-debug-log` | - | 保存完整调试日志 + 失败 JSON（推荐排查时用） |

**示例**：

```bash
# 基本用法
subloom process movie.mp4 -t zh

# 高质量处理
subloom process movie.mp4 -t zh --whisper-model large-v3 --translation-model qwen2.5:32b

# 使用动漫提示词模板
subloom process anime.mp4 -t zh -p anime

# 保存调试日志
subloom process movie.mp4 -t zh --save-debug-log
```

---

## batch - 批量处理

**功能**：批量处理多个视频文件

```bash
subloom batch <directory> -t <target_lang> [options]
```

| 参数 | 简写 | 说明 |
|------|------|------|
| `<directory>` | - | 视频目录路径（必需） |
| `--target-lang` | `-t` | 目标语言（必需） |
| `--recursive` | `-r` | 递归处理子目录 |
| `--workers` | `-w` | 并发数（默认取配置 `max_workers`，即 2；上限 4）；大于 1 时翻译期间识别模型不卸载，见[翻译速度慢](#翻译速度慢) |
| `--file-list` | - | 从文件列表读取 |
| `--output-dir` | `-o` | 输出目录 |
| `--asr-backend` | - | 语音识别后端：whisper, qwen3_asr |
| `--whisper-model` | - | Whisper 模型 |
| `--translation-backend` | - | 翻译后端：ollama, openai, deepl, google |
| `--translation-model` | - | 翻译模型（旧名 `--ollama-model` 仍可用） |
| `--keep-original / --no-keep-original` | - | 保留/不保留原语言字幕 |
| `--bilingual / --no-bilingual` | - | 生成双语字幕 |
| `--drop-interjections / --keep-interjections` | - | 滤掉只有语气词的字幕行（默认取配置 `output.drop_interjections`，见[语气词过滤](#语气词过滤)） |
| `--merge-fragments / --no-merge-fragments` | - | 把说到一半被切开的字幕并回整句（默认开，取配置 `output.merge_fragments`，见[碎句合并](#碎句合并)） |
| `--names` | - | 人名的正式写法，逗号或顿号分隔，如 `光希,琴音`（默认取配置 `translation.names`，见[人名统一](#人名统一)） |
| `--find-names / --no-find-names` | - | 翻译前通读全片原文找人名和它的各种写法（默认开，取配置 `translation.find_names`） |
| `--timestamp-mode` | - | 后处理模式 |
| `--split-sentences / --no-split-sentences` | - | 按句子拆分多句字幕 |
| `--save-debug-log` | - | 保存调试日志（每个视频一个 `{文件名}_debug/`，此时逐个处理） |

**示例**：

```bash
# 处理目录
subloom batch ./videos/ -t zh

# 递归处理 + 4 线程
subloom batch ./videos/ -t zh -r --workers 4

# 从列表文件处理
subloom batch --file-list files.txt -t zh

# 双语字幕
subloom batch ./videos/ -t zh --bilingual
```

---

## transcribe - 仅转录

**功能**：语音识别，生成原语言字幕（不翻译）

```bash
subloom transcribe <video> [options]
```

| 参数 | 简写 | 说明 |
|------|------|------|
| `<video>` | - | 视频文件路径（必需） |
| `--output` | `-o` | 输出文件路径 |
| `--language` | `-l` | 源语言（默认自动检测） |
| `--asr-backend` | - | 语音识别后端：whisper, qwen3_asr |
| `--model` | `-m` | 识别模型（对当前后端生效） |
| `--vad-filter / --no-vad-filter` | - | 开/关 VAD 过滤（默认取配置 `whisper.vad_filter`；WhisperX 下不起作用，何时关见[整段台词没识别出来](#整段台词没识别出来)） |
| `--auto-model` | - | 根据 GPU 自动选择模型 |
| `--whisperx / --no-whisperx` | - | 使用/不使用 WhisperX |
| `--timestamp-mode` | - | 后处理模式 |
| `--split-sentences / --no-split-sentences` | - | 按句子拆分多句字幕 |
| `--drop-interjections / --keep-interjections` | - | 滤掉只有语气词的字幕行（默认取配置 `output.drop_interjections`，见[语气词过滤](#语气词过滤)） |
| `--merge-fragments / --no-merge-fragments` | - | 把说到一半被切开的字幕并回整句（默认开，取配置 `output.merge_fragments`，见[碎句合并](#碎句合并)） |
| `--save-debug-log` | - | 保存调试日志 |

**示例**：

```bash
# 基本转录
subloom transcribe video.mp4

# 指定语言和模型
subloom transcribe video.mp4 -l en -m large-v3

# 自动选择最佳模型
subloom transcribe video.mp4 --auto-model
```

---

## translate - 仅翻译

**功能**：翻译现有字幕文件

```bash
subloom translate <subtitle> -t <target_lang> [options]
```

| 参数 | 简写 | 说明 |
|------|------|------|
| `<subtitle>` | - | 字幕文件路径（必需） |
| `--target-lang` | `-t` | 目标语言（必需） |
| `--source-lang` | `-s` | 源语言 |
| `--output` | `-o` | 输出文件路径 |
| `--translation-backend` | - | 翻译后端：ollama, openai, deepl, google |
| `--model` | `-m` | 翻译模型 |
| `--bilingual / --no-bilingual` | - | 生成双语字幕（默认取配置 `output.bilingual`） |
| `--drop-interjections / --keep-interjections` | - | 滤掉只有语气词的字幕行（默认取配置 `output.drop_interjections`，见[语气词过滤](#语气词过滤)） |
| `--merge-fragments / --no-merge-fragments` | - | 把说到一半被切开的字幕并回整句（默认开，取配置 `output.merge_fragments`，见[碎句合并](#碎句合并)） |
| `--names` | - | 人名的正式写法，逗号或顿号分隔，如 `光希,琴音`（默认取配置 `translation.names`，见[人名统一](#人名统一)） |
| `--find-names / --no-find-names` | - | 翻译前通读全片原文找人名和它的各种写法（默认开，取配置 `translation.find_names`） |

**示例**：

```bash
# 翻译字幕
subloom translate video.en.srt -t zh

# 双语字幕
subloom translate video.en.srt -t zh --bilingual
```

---

## config - 配置管理

**功能**：查看和修改配置

| 子命令 | 说明 |
|--------|------|
| `show` | 显示当前配置 |
| `set <key> <value>` | 设置配置项 |
| `reset` | 重置为默认配置 |
| `check` | 检查系统环境 |
| `pull-model` | 下载翻译模型（仅 Ollama） |
| `show-prompt` | 显示当前翻译提示词 |
| `list-prompts` | 列出可用提示词模板 |
| `use-prompt <id>` | 使用指定模板 |
| `export-prompt` | 导出提示词 |
| `set-prompt` | 设置自定义提示词（存为模板 `custom` 并选中） |
| `reset-prompt` | 重置为默认提示词 |

**示例**：

```bash
# 查看配置
subloom config show

# 修改模型
subloom config set whisper.model large-v3
subloom config set ollama.model qwen2.5:32b

# 系统检查
subloom config check --verbose

# 下载模型
subloom config pull-model

# 使用动漫模板
subloom config use-prompt anime
```

---

## quickstart - 初始化向导

**功能**：首次使用的设置向导

```bash
subloom quickstart
```

向导分六步，每一步的选择当场写进配置文件（`--config` 指定的那份，默认位置见[配置文件](#配置文件)）：

1. 检查 ffmpeg。
2. 检测显卡。没有 NVIDIA 显卡时，把所选识别后端的 `device` 设成 `cpu`（Whisper 同时设 `compute_type: int8`）。
3. 选语音识别后端：`whisper` 或 `qwen3_asr`。选 Whisper 时按显存推荐模型（没有显卡推荐 `small`），确认后才改。
   缺可选依赖时提示安装命令：选 Whisper 而没装 `[whisperx]`（且 `whisper.use_whisperx` 开着），
   或选 Qwen3-ASR 而没装 `[qwen3-asr]`。
4. 检查识别模型是否已下载，没有就问要不要现在下。
5. 选翻译后端：
   - `ollama`：按显存推荐翻译模型，确认后才改。
   - `openai`：填服务地址，向导从服务端取模型列表让你按编号选；取不到就手输模型 id。
   - `deepl` / `google`：API key 只从环境变量读，向导不往配置文件里写 key；没设时告诉你怎么设。
6. 检查翻译服务：连得上、模型在、key 被接受。Ollama 缺模型时问要不要现在拉取。

没解决的问题在最后汇总列出，修好后可以用 `subloom config check` 复查。

---

# 四、HTTP 服务模式

把 subloom 跑成一个 HTTP 服务，供媒体服务器（Jellyfin 等）或其他机器上的自动化脚本提交任务。
不需要这类集成就不用看这一章——命令行已经够用。

## 安装与启动

```bash
# 需要 serve 额外依赖（fastapi + uvicorn）
pip install -e '.[serve]'

# 生成一个长随机 token，客户端要用它做 Bearer 鉴权
export SUBLOOM_TOKEN="$(openssl rand -hex 32)"

# 默认只监听本机回环；要让其他机器访问必须显式给 --host
subloom serve --host 0.0.0.0 --port 8765
```

启动后访问 `http://<主机>:8765/docs` 是自动生成的 OpenAPI 交互文档。

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `127.0.0.1` | 绑定地址。默认仅本机可访问；`0.0.0.0` 暴露到网络时 Bearer token 走明文 HTTP，跨出可信局域网请在前面加 TLS 反代 |
| `--port` | `8765` | 绑定端口 |
| `--workers` | `1` | 并发任务数，上限 4。**GPU 是瓶颈，没实测过就别调** |
| `--no-auth` | 关 | 关闭鉴权。仅在绑定 `127.0.0.1` 的单用户机器上安全 |
| `--log-level` | `info` | uvicorn 日志级别 |

不加 `--no-auth` 而又没设 `$SUBLOOM_TOKEN` 时，服务会拒绝启动；
加了 `--no-auth` 又绑非回环地址时，会打印警告——**同网段任何人都能提交任务**。

## 接口说明

| 方法 | 路径 | 鉴权 | 用途 |
|------|------|------|------|
| `POST` | `/jobs` | Bearer | 提交字幕生成任务 |
| `GET` | `/jobs/{id}` | Bearer | 查询任务状态 / 获取输出路径 |
| `GET` | `/health` | 无 | 存活检查 + 队列统计；worker 全部死亡时返回 503（此时接受的任务永远不会执行） |

`POST /jobs` 请求体：

```json
{
  "video_path": "/服务端能看到的/绝对路径.mp4",
  "target_languages": ["zh"],
  "source_language": null,
  "bilingual": false,
  "keep_original": true
}
```

`bilingual` 与 `keep_original` 可以省略：省略时取服务端配置文件里的 `output.bilingual` / `output.keep_original`，
与命令行的 `process` 一致。

提交时就会做校验，**不合规立刻返回错误，不会等到几分钟后 worker 跑起来才失败**：

| 情况 | 响应 |
|------|------|
| 路径不是绝对路径 / 不存在 / 不是文件 | `400` |
| 扩展名不在支持列表内（如误传 `.md`） | `400` |
| 语言代码含路径分隔符等非法字符 | `422` |
| `target_languages` 超过 10 个 | `422` |
| 待处理队列已满（100 个 pending） | `503` |

输出文件写在**源视频同目录**，命名规则与命令行一致：`<视频名>.<语言>.srt`，
双语为 `<视频名>.<源语言>-<目标语言>.srt`。服务模式不支持指定输出目录。

任务完成后，`GET /jobs/{id}` 的 `degradations` 列出这次运行的降级，每条是
`{"code": ..., "message": ...}`，没有降级时为空列表；有降级的任务状态仍是 `completed`。
code 的含义见[降级与退出码](#降级与退出码)。

## 作为守护进程运行

**macOS（launchd）**——存为 `~/Library/LaunchAgents/com.subloom.serve.plist`：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.subloom.serve</string>
  <key>ProgramArguments</key>
  <array>
    <string>/usr/local/bin/subloom</string>
    <string>serve</string>
    <string>--host</string><string>0.0.0.0</string>
    <string>--port</string><string>8765</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>SUBLOOM_TOKEN</key><string>你的长随机-TOKEN</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>/tmp/subloom.log</string>
  <key>StandardErrorPath</key><string>/tmp/subloom.err</string>
</dict>
</plist>
```

然后 `launchctl load ~/Library/LaunchAgents/com.subloom.serve.plist`。

**Linux（systemd）**——存为 `/etc/systemd/system/subloom.service`：

```ini
[Unit]
Description=subloom HTTP server
After=network.target ollama.service

[Service]
Type=simple
User=你的用户名
Environment=SUBLOOM_TOKEN=你的长随机-TOKEN
ExecStart=/usr/local/bin/subloom serve --host 0.0.0.0 --port 8765
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

然后 `systemctl enable --now subloom`。

## 服务模式注意事项

- **模型必须提前缓存好**。服务端不会跑交互式下载。先在这台机器上执行一次
  `subloom transcribe <任意视频>` 把 Whisper 权重下下来，再启动服务；
  否则第一个任务会带着明确提示失败。
- **Ollama 必须在跑**。翻译要调它，上线前用 `subloom config check --verbose` 确认。
- **默认单 worker**。并发跑 Whisper 只会让显存互相踩踏，这也是默认值是 1 的原因。
- **第一个任务慢（约 10–30 秒）**。Whisper 模型是首次请求时才加载的，之后常驻显存，后续任务立刻开始。
- **任务只存在内存里**。重启服务会丢掉任务记录，正在跑的任务被标记为 `failed`（免得永远卡在 `processing`）。
  需要重试就从客户端重新提交。这是有意为之——单用户场景上持久化不划算。
- **停止服务杀不掉正在跑的任务**。Python 线程无法强制终止，正在转录的任务会一直占着 GPU
  跑到自己结束或进程退出，期间**仍可能写出字幕文件**——哪怕它的状态已经是 `failed`。
- **路径是严格的**。`video_path` 必须是绝对路径，而且要是**这台服务器**能看到的路径。
  客户端和服务端挂载点不同时（例如 `/media/videos` 对 `/Volumes/nas-videos`），换算要在客户端做。

---

# 五、隐私与网络边界

默认配置下推理全部在本机完成：**转录和翻译不会把音频或字幕发到任何云服务**。
只有你把翻译后端配成云端服务，字幕文本才会发出去（第 2 条）。
但"本地优先"不等于"永不联网"，下面四条边界值得先看清楚，尤其是要处理敏感内容的时候。

## 四条边界

### 1. 首次获取模型需要联网

| 模型 | 来源 | 默认缓存位置 |
|------|------|--------------|
| Whisper / Qwen3-ASR 语音识别 | HuggingFace（`huggingface.co`） | `~/.cache/huggingface/hub` |
| Ollama 翻译模型 | Ollama 官方仓库 | `~/.ollama/models` |

模型下载完成后，日常使用不再需要这两个来源。Whisper 与 Qwen3-ASR 的模型缓存齐了就直接从本地加载，
不再向 HuggingFace 检查新版本，所以上游更新了也不会自动换上；想换新版本，删掉缓存里对应的模型目录，
下次运行会重新下载。**但要注意**：WhisperX 的对齐模型（日语、中文等用 HuggingFace 上的模型）每次加载
仍会发一次 HTTP 请求检查新版本。要彻底切断，见下面的[完全离线部署](#完全离线部署)。

中国大陆用户可用镜像替代官方源，见 [HuggingFace 镜像](#huggingface-镜像中国用户)。
缓存位置可以用 `whisper.download_root` 配置项或 `HF_HOME` 环境变量改。

### 2. 翻译可以发往别的机器

`ollama.host` 默认是 `http://localhost:11434`，此时翻译请求不出本机。
但翻译后端的地址接受任意值：

```bash
# 一旦改成远程地址，字幕原文和译文都会发送到那台机器
subloom config set ollama.host http://192.168.1.100:11434

# OpenAI 兼容后端：openai.base_url 指向哪里，字幕就发到哪里，云端服务也一样
subloom config set openai.base_url https://api.openai.com/v1

# DeepL、Google 是云端机器翻译：选了它们，字幕就发给这两家
subloom config set translation.backend deepl
```

把翻译放到局域网里的另一台强机上、或者交给云端 API，都是合理用法，**但要清楚字幕内容离开了本机**。
请求只发给 `translation.backend` 选中的那一个后端，没有配置云端就绝不会外发；发出去的只有
字幕文本和提示词，不含音频和视频。处理敏感内容前，用 `subloom config show` 确认
`translation.backend` 和对应那一节的地址。

### 3. 调试日志会记录台词原文

以下两个开关的产物**包含字幕内容**，默认都是关闭的：

| 开关 | 产物 | 里面有什么 |
|------|------|------------|
| `--save-debug-log` | `<视频名>_debug/run.log` | DEBUG 级日志，含模型返回的译文片段（每次最多 500 字符） |
| `--save-debug-log` | `<视频名>_debug/translation_failures.json` | 每条失败字幕的**原文全文**，以及模型响应的前 200 字符 |
| `--save-failed-log` | `<视频名>_translation_failures.json` | 同上，但直接放在输出目录里，不建 `_debug/` 子目录 |

产物都落在输出目录（默认是视频所在目录）。`transcribe` 命令不翻译，
所以它的 `--save-debug-log` 只产出 `run.log`，没有失败报告。

这些文件是排查问题最有用的东西，但**发给别人或贴到 issue 之前，先自己看一遍内容**。
不加这两个开关时，日志里不会出现台词。

### 4. 输出位置与文件权限

- 命令行默认把字幕写在**源视频同目录**，`-o/--output-dir` 可以改。
- 服务模式**恒定写在源视频同目录**，不支持指定别处。
- 文件权限继承操作系统的默认设置（umask）。视频放在多用户共享目录里时，
  生成的字幕对同一批人同样可见。

### 附：关闭 HuggingFace 生态遥测

HuggingFace 的库默认会收集一些使用数据。要全局关掉：

```bash
export HF_HUB_DISABLE_TELEMETRY=1
# 或使用通用开关（对 HF 生态等同）
export DO_NOT_TRACK=1
```

## 完全离线部署

目标：让 subloom 在一台**完全不联网**的机器上跑起来。

**第一步，在有网的机器上把模型下全**

```bash
# Whisper：跑一次转录即可触发下载（或直接用向导）
subloom quickstart

# Ollama：拉取配置里的翻译模型
ollama pull qwen2.5:14b
```

**第二步，把两个缓存目录拷到离线机器的相同位置**

```
~/.cache/huggingface/hub     →  离线机器的 ~/.cache/huggingface/hub
~/.ollama/models             →  离线机器的 ~/.ollama/models
```

如果配置了 `whisper.download_root`，拷这个目录而不是默认的 HF 缓存。

**第三步，在离线机器上禁止一切外呼**

```bash
# 只用本地缓存；缓存里没有的文件直接报错，而不是尝试下载
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
```

写进 `~/.zshrc` / `~/.bashrc` 可以永久生效。设了 `HF_HUB_OFFLINE=1` 之后，
连"检查新版本"那次请求也会被跳过，顺带让模型加载更快一点。

**第四步，验证**

```bash
subloom config check --verbose
```

语音识别和翻译两项都显示 ready，就说明离线环境是完整的。
此时整条流水线（ffmpeg 抽音频 → 转录 → 翻译 → 写 SRT）不需要任何外部网络。

> **注意**：翻译后端必须是本机服务（`ollama.host` 或 `openai.base_url` 指向本机；DeepL、Google 无法离线），否则第 2 条边界依然成立——
> 离线机器上指向局域网另一台机器，字幕一样会离开这台电脑。

---

# 六、故障排查

## 无法连接 Ollama

**症状**：`Cannot connect to Ollama` / `Connection refused`

**解决方案**：

```bash
# 1. 确认 Ollama 正在运行
ollama serve

# 2. 检查服务状态
curl http://localhost:11434

# 3. 检查端口占用
lsof -i :11434          # Linux/macOS
netstat -ano | findstr 11434  # Windows
```

---

## GPU 未识别

**症状**：`CUDA not available` / `Using CPU`

**解决方案**：

```bash
# 1. 系统诊断
subloom config check --verbose

# 2. 检查 PyTorch
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 3. 检查驱动
nvidia-smi

# 4. 重新安装正确版本 PyTorch（见 GPU 加速设置）
```

---

## RTX 50 系列 sm_120 错误

**症状**：`sm_120 is not compatible with the current PyTorch installation`

**解决方案**：

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

---

## 下载速度慢

**症状**：Whisper 模型下载缓慢

**解决方案**：配置 HuggingFace 镜像（见 [HuggingFace 镜像](#huggingface-镜像中国用户)）

---

## 模型下载中断

**症状**：下载过程中断

**解决方案**：直接重新运行命令，支持断点续传

```bash
subloom process video.mp4 -t zh  # Whisper 模型会自动续传
subloom config pull-model         # Ollama 模型
```

---

## 处理速度慢

### 转录速度慢

**参考速度（GPU）**：1 小时视频约 2-5 分钟

**解决方案**：

```bash
# 确认 GPU 使用
nvidia-smi -l 1  # 处理时监控

# 使用较小模型
subloom config set whisper.model medium
```

### 翻译速度慢

**参考速度**：GPU 约 10-20 条/秒，CPU 约 1-3 条/秒

**解决方案**：

```bash
# 使用较小模型
subloom config set ollama.model qwen2.5:7b
```

**本地翻译模型和识别模型挤在同一张显卡上**：显存一满，翻译会慢好几倍。
`process` 和单 worker 的 `batch` 翻译前会先卸载识别模型；`batch --workers` 大于 1 时不卸载，
因为下一个视频马上要用它转写。显存不宽裕就用 `--workers 1`。

---

## 整段台词没识别出来

字幕成段空白，声音里却有人在说话，多半是 VAD（语音活动检测）把那段判成了非语音：
低声、气声、喘息，或者压着背景音乐的对白，都可能被整段删掉，根本到不了识别那一步。

**换 Qwen3-ASR**：片源是中文、日语、韩语时首选。它不靠 VAD 删音频，
速度也比下面关 VAD 的做法快（见 [Qwen3-ASR](#qwen3-asr语音识别可选后端)）。

**关掉 VAD**：`transcribe` 加 `--no-whisperx --no-vad-filter`，或者
`config set whisper.vad_filter false`。这个开关只对 faster-whisper 生效，WhisperX 有自己的 VAD。

- 关掉后整段音频都要解码，音乐、喘息也在内。程序会让 Whisper 不再以自己的上文为条件，
  并跳过静音附近的幻觉，免得一句编造的话连着出现几分钟。
- 代价是慢：2 小时的片子在 RTX 5090 上要一个多小时。加 `--batch-size 16`（或
  `whisper.batch_size`）走批量解码只要几分钟，但质量差一截：长段和重复更多，
  静音附近的幻觉也不再跳过。
- VAD 没误删台词的片源，关掉它不会更好，漏掉的句子反而略多。

---

## 字幕时间轴感觉不对

字幕和声音的同步问题往往是 `lead_in_ms` / `linger_ms` 没调好。
**先用 `--save-debug-log` 跑一次**，对照实际效果调下面的参数。

### 字幕比声音晚出现（"还没看到字幕，话已经讲了一半"）

```bash
subloom config set timestamp.lead_in_ms 150   # 默认 80，调大
```

### 字幕在声音停的瞬间消失（"字还没看完就没了"）

```bash
subloom config set timestamp.linger_ms 500    # 默认 300，调大
```

### 字幕一直显示，对话结束后还在屏幕上

通常是某些段在转录时 `segment.end` 被错误延伸到下一段开始或文件末尾。
首先确认你的 `mode` 是 `minimal`（默认），它会自动修这种情况：

```bash
subloom config set timestamp.mode minimal
```

如果仍然超长，可能是没装 `[whisperx]`——基础 faster-whisper 的对齐精度会更弱。
推荐安装：

```bash
pip install -e '.[whisperx]'
```

### 短字幕快速闪过

通常发生在密集对话场景。如果字幕之间间隔过近，`lead_in_ms + linger_ms + min_gap`
的总和会超过实际词间距，linger 会被自动截短以避免重叠。这是设计取舍——宁可短一些，
不让两段字幕同屏。可以略微减小 lead_in/linger 试试：

```bash
subloom config set timestamp.lead_in_ms 50
subloom config set timestamp.linger_ms 200
```

---

## 翻译有些段没翻译

字幕里出现少数几句仍然是原文（未翻译）。完成面板会以 `translation_incomplete` 报出每个目标语言缺了几段，
命令以退出码 3 结束（见[降级与退出码](#降级与退出码)）。

### 第一步：看失败原因

跑一次带 `--save-debug-log`，打开 `<video>_debug/translation_failures.json`：

```bash
subloom process video.mp4 -t zh --save-debug-log
```

里面的 `failures[*].reason` 字段会分类：

| reason | 含义 | 处理 |
|--------|------|------|
| `模型拒绝翻译` | LLM 内置安全限制（如成人内容） | 换更松的模型，或用 `-p adult` 模板 |
| `内容被过滤` | 同上 | 同上 |
| `LLM返回空响应` | 模型输出空 | 换模型或减小 batch_size |
| `LLM响应过短` | 输出太短被认为失败 | 通常无害，可以忽略 |
| `响应中未找到段落索引` | 解析失败 | 一般会被自动重试修复 |
| `可能是语气词被跳过` | 模型跳过短叹词 | 可忽略 |

### 第二步：手动减小 batch_size（极少需要）

正常情况 batch_size 已按模型大小自动调整（14B → 6, ≤8B → 4）。
如果某个特殊模型还是丢段：

```bash
subloom config set translation.max_batch_size 3
```

### 第三步：换更大模型

```bash
subloom config set ollama.model qwen2.5:32b   # 需要 16GB+ VRAM
```

---

## 保存调试日志

遇到问题时，保存完整日志便于排查：

```bash
subloom process video.mp4 -t zh --save-debug-log
```

生成文件：

```
output/
├── video.zh.srt
└── video_debug/
    ├── run.log                    # 完整运行日志
    └── translation_failures.json  # 翻译失败详情
```

---

## 完整系统诊断

```bash
subloom config check --verbose
```

显示：Python 版本、PyTorch/CUDA 版本、GPU 信息、Ollama 状态、已安装模型

---

# 七、附录

## 支持的语言

subloom 支持 Whisper 识别的 99+ 种语言。

### 常用语言代码

| 代码 | 语言 | 代码 | 语言 |
|------|------|------|------|
| `en` | English | `zh` | 简体中文 |
| `ja` | 日本語 | `ko` | 한국어 |
| `es` | Español | `fr` | Français |
| `de` | Deutsch | `ru` | Русский |
| `pt` | Português | `it` | Italiano |
| `ar` | العربية | `vi` | Tiếng Việt |
| `th` | ไทย | `zh-TW` | 繁體中文 |

---

## 时间戳后处理

### 处理模式

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `off` | 完全禁用 | WhisperX 对齐效果好 |
| `minimal` | 仅修复重叠和最小时长（**默认**） | 大多数情况 |
| `full` | 完整处理：分割、延长等 | 需要分割多句话 |

### 结构检查

`minimal` 和 `full` 模式下，每个写出的字幕文件都会过一遍结构检查：时间轴不重叠、每条时长为正、
不超出音频结尾，译文还要和原文逐条对应（条数与编号一致）。检查不通过时文件**照常写出**，
但命令以退出码 1 结束并列出有问题的条目；服务模式下任务记为 `failed`，`outputs` 里仍列出已写出的文件。
`off` 和 `--no-post-process` 本来就是要原始时间轴，不做这项检查。

两条字幕起点挨得太近、各自保住最短时长就必然重叠时，后处理会把它们合并成一条，不让它们同屏重叠。

### 命令行用法

```bash
# 默认 (minimal)
subloom process video.mp4 -t zh

# 指定模式
subloom process video.mp4 -t zh --timestamp-mode off
subloom process video.mp4 -t zh --timestamp-mode full

# 完全禁用后处理
subloom process video.mp4 -t zh --no-post-process
```

### 句子级分割

当视频中多句对话被合并到一个字幕段时，可以使用 `--split-sentences` 选项按句子拆分：

```bash
# 启用句子分割（推荐用于电影、电视剧、访谈等对话密集的视频）
subloom process video.mp4 -t zh --split-sentences

# 也可在 transcribe 命令中使用
subloom transcribe video.mp4 --split-sentences
```

**工作原理**：
- 利用 WhisperX 提供的词级时间戳
- 在句末标点（。！？等）处拆分
- 根据词时间戳精确计算每个句子的开始和结束时间
- 独立于 `--timestamp-mode` 选项，可与任何模式组合使用

**适用场景**：
- 对话密集的电影、电视剧
- 访谈节目
- 需要逐句显示字幕的场景

### 配置选项

```yaml
timestamp:
  enabled: true
  mode: "minimal"              # off, minimal, full
  min_duration: 1.0            # 最小显示时长（秒）；< 1s 太短读不过来
  max_duration: 8.0            # 最大显示时长（秒）
  min_gap: 0.05                # 相邻字幕的最小间隙（秒）
  max_gap_warning: 10.0        # 间隙超过该值会在日志警告"可能漏录"（秒）
  chars_per_second: 15.0       # 西文阅读速度
  cjk_chars_per_second: 10.0   # 中日韩阅读速度
  split_threshold: 30          # full 模式下的最小拆分字符数
  split_sentences: true        # 句子级分割（默认启用）

  # 字幕显示时间补偿（影响"字幕和声音对齐感"，毫秒）
  # 这两个参数对所有 mode 都生效（包括 mode=off）
  lead_in_ms: 80               # 字幕比第一个字发音提前多少出现
  linger_ms: 300               # 字幕比最后一个字发音延后多少消失
```

#### lead-in / linger 调优指南

声学模型给的"词开始/结束时间"只是发音边界，**不等于字幕该出现/消失的时间**。
为了让观众看得舒服，行业惯例（BBC / Netflix）是字幕略早出现、声音停后再保留一会儿。

- **声音出来了字幕还没出现** → 增大 `lead_in_ms`（试 120 / 150 / 180）
- **字幕在声音停的瞬间就消失了** → 增大 `linger_ms`（试 400 / 500 / 600）
- **字幕拖得太长侵占下一段** → 减小 `linger_ms`（如 150-200）；
  注意：相邻段重叠时，前段的 linger 会被自动截到下一段开始前，不会真的重叠

调整命令：

```bash
subloom config set timestamp.lead_in_ms 120
subloom config set timestamp.linger_ms 500
```

### CJK 语言优化

系统自动检测中文、日文、韩文，使用针对性优化：
- 更慢的阅读速度（10 字/秒 vs 15 字/秒）
- 更低的分割阈值（15 字符 vs 30 字符）
- 字数按 Netflix 口径计：全角字符计 1，半角字符（拉丁字母、数字、半角假名）计 0.5，中日文里夹的英文不再按字数吃满阅读时间

---

## 语气词过滤

日语片源里常有整条只有「あ」「うん」「はぁ」的字幕行，人工字幕一般不打。打开过滤后，这些行在翻译前就去掉：
原语言字幕和译文里都没有，也不占翻译时间。默认关闭。

```bash
subloom process video.mp4 -t zh --drop-interjections
subloom config set output.drop_interjections true    # 设为默认
```

- 只滤**整条都是语气词**的行：「あ、うん、」「うーん」「嗯嗯」「Uh, um.」会去掉，「ああ、そうだね」这种夹着正经话的保留。
  有词义的应答和呼唤不算语气词，「はい」「ね」「ほら」「おい」「いや」都保留。
- 一张表认日语、中文、英语、韩语的常见语气词，不分源语言。只由あ、う、え、お、ん、わ和は行假名拼成的词
  （比如写成假名的「うえ」）也会被当成语气词；识别模型一般把这类词写成汉字。
- 过滤在拆句之后：「ああ。そうだね。」先拆成两条，再滤掉「ああ。」。
- `process`、`batch`、`transcribe`、`translate` 都有 `--drop-interjections / --keep-interjections`；
  `translate` 滤的是读进来的字幕，原文件不动。HTTP 服务按配置文件里的 `output.drop_interjections` 走。

---

## 碎句合并

识别模型在说话人停顿半秒左右就切一刀，一句话常被拆成两三条，比如「見せ」「てみろ。」。分开翻译时模型只看到半句，会译成「给」「我看看。」，甚至把「聞こえ」「ねえ」译成「听」「喂」。碎句合并在翻译前把这样的碎段并回整句，默认开启。满足下面三条才并：

- 前一条没有以句末标点（。！？.!?）结尾；
- 两条在屏幕上的间隔不超过 1 秒；
- 并完不超过 7 秒、两行（中日韩 32 个全角字，其它语言 84 个字符），即 Netflix 字幕规范的上限。

原语言字幕和译文都按合并后的分条输出。代价是一条字幕会横跨一段不超过 1 秒的停顿。想保持识别模型原来的分条：

```bash
subloom process video.mp4 -t zh --no-merge-fragments
subloom config set output.merge_fragments false    # 设为默认
```

- 开了[语气词过滤](#语气词过滤)时先滤再并，单独的「うん、」不会粘到下一句开头。
- `process`、`batch`、`transcribe`、`translate` 都有这对开关；`translate` 并的是读进来的字幕，原文件不动。HTTP 服务按配置文件里的 `output.merge_fragments` 走。

---

## 人名统一

识别模型会把同一个名字写成好几种：一部片里的女主「みつき」，日文识别结果里有三月、みつき、ミツキ、美月几种写法，逐批翻译时就成了三月、美月、光希。所以翻译前会先把全片原文通读一遍，找出片中叫到的人名和它们在识别结果里的各种写法，再把这张名单放进每一批翻译的提示词，同一个人从头到尾译成同一个名字。默认开启。

正式写法（比如「光希」）光听声音推不出来。知道的话用 `--names` 给出：

```bash
subloom process video.mp4 -t zh --names 光希,琴音
subloom translate video.ja.srt -t zh --names 光希，琴音
```

- 名字之间用逗号或顿号分隔，中英文标点都行。写法以你给的为准，通读负责把识别结果里的各种写法对到这些名字上。
- 没给 `--names` 时，名字怎么译由翻译模型自己定：全片统一，但不一定是官方写法，甚至可能选到别人的名字（实测有部片子把女主统一成了一个男性名）。名单会打在日志里（`Character names:`），不对就用 `--names` 重跑。模型给的名字还是假名时（目标语言不是日语），这个人不进名单，照普通翻译走。
- 通读是每个目标语言一次长请求，2 小时的片子在 qwen3.8:27b 上约 10–50 秒。关掉用 `--no-find-names` 或 `config set translation.find_names false`，这时只有 `--names` 给的名字进提示词。
- 通读要求模型的上下文装得下全片原文（2 小时的片子去掉语气词后约 5 千到 2 万字，qwen3.8:27b 上约 4 千到 1.3 万 token），还要给思考留几千 token。Ollama 按显存定默认上下文（[低于 24 GiB 是 4k，24–48 GiB 是 32k](https://docs.ollama.com/context-length)），24 GiB 以下默认就装不下。原文超出上下文时，Ollama 只留片尾、大约半个上下文的原文，名单会缺人，记 `names_partial`（退出码 3）；装得下但思考空间不够时拿不到回答，记 `names_not_found`。解决办法是调大上下文（桌面版设置里的滑块，或环境变量 `OLLAMA_CONTEXT_LENGTH`），或者用 `--names` 加 `--no-find-names`。
- 通读偶尔拿不到可用的回答，连试 3 次都没有时按没有名单继续翻译，结果里记一条 `names_not_found` 降级（退出码 3）。
- 只对大模型翻译后端（ollama、openai）有效；DeepL、Google 不接受名单，配了 `--names` 会在开工前报错。
- `batch` 的 `--names` 对这一批的所有视频生效，适合同一部剧的几集。HTTP 服务按配置文件里的 `translation.names` / `translation.find_names` 走。

---

## 翻译提示词配置

### 内置模板

| 模板 ID | 名称 | 适用场景 |
|---------|------|----------|
| `movie-general` | 通用电影 | 大多数电影/电视剧 |
| `movie-scifi` | 科幻电影 | 科幻、太空题材 |
| `movie-fantasy` | 奇幻电影 | 魔法、奇幻题材 |
| `documentary` | 纪录片 | 纪录片、科教片 |
| `anime` | 动漫 | 日本动漫 |
| `adult` | 成人内容 | 成人影视 |
| `technical` | 技术教程 | 编程、软件教程 |

### 使用模板

```bash
# 查看可用模板
subloom config list-prompts

# 设置默认模板
subloom config use-prompt anime

# 临时使用模板
subloom process anime.mp4 -t zh -p anime
```

### 自定义模板

```bash
# 1. 导出现有模板
subloom config export-prompt -o my_prompt.txt

# 2. 编辑文件

# 3. 应用自定义提示词（存为用户模板 custom 并选中，再执行一次会覆盖它）
subloom config set-prompt -f my_prompt.txt

# 4. 重置为默认
subloom config reset-prompt
```

### 可用占位符

| 占位符 | 说明 |
|--------|------|
| `{source_lang}` | 源语言名称（必需） |
| `{target_lang}` | 目标语言名称（必需） |
| `{segments}` | 要翻译的字幕（必需） |
| `{context_before}` | 前文上下文 |
| `{context_after}` | 后文上下文 |

---

## 降级与退出码

处理跑完了、但没按配置跑足，叫降级：GPU 不可用退到 CPU、强制对齐失败退回段级时间轴、
有几段没翻出来，都算。降级不会让任务失败，字幕照常写出，但每一处都会在结果里说出来：

- `process`、`transcribe`、`translate` 的完成面板变黄，标题为 `Complete (degraded)`，下方逐条列出；
- `batch` 汇总表里该视频的状态是 `Degraded`，表下列明细，所有视频共有的那几条只列一次；
- 服务模式下见 `GET /jobs/{id}` 的 `degradations` 字段。

日志里的 WARNING 照旧保留。`--quiet` 会隐藏完成面板，但退出码不变。

| code | 含义 | 怎么办 |
|------|------|--------|
| `whisperx_missing` | 配置要用 WhisperX 但没装，时间轴来自普通 faster-whisper | 装 `[whisperx]` 额外依赖，或用 `--no-whisperx` 明确不用 |
| `alignment_failed` | 强制对齐失败，时间轴退回段级 | message 里带着原因；常见是该语言没有对齐模型，或模型缓存损坏 |
| `alignment_unsupported_language` | Qwen3-ForcedAligner 不覆盖识别出的语言 | 改用 Whisper 后端 |
| `cpu_fallback` | 配置要用 CUDA 但不可用，模型跑在 CPU 上 | 见[GPU 未识别](#gpu-未识别) |
| `flash_attention_missing` | Qwen3-ASR 没有 FlashAttention 2，长片段准确度下降 | 见 [Qwen3-ASR](#qwen3-asr语音识别可选后端) 一节 |
| `auto_model_fallback` | `--auto-model` 读不到空闲显存，退到 `small` | 用 `--model` 指定模型 |
| `translation_incomplete` | 有段落模型始终没给出译文，保留了原文 | 见[翻译有些段没翻译](#翻译有些段没翻译) |
| `names_not_found` | 翻译前通读人名，连试 3 次都没拿到可用的回答，按没有名单翻译 | 人名可能前后不一；重跑一次，或用 `--names` 给出名字，见[人名统一](#人名统一) |
| `names_partial` | 全片原文超出模型的上下文，通读只读到一部分，名单可能缺人 | 调大上下文（Ollama 桌面版设置的滑块或 `OLLAMA_CONTEXT_LENGTH`），或用 `--names` 给出名字，见[人名统一](#人名统一) |

code 是稳定标识，供脚本按类判断；message 是给人看的，措辞可能改。

| 退出码 | 含义 |
|--------|------|
| `0` | 成功，没有降级 |
| `1` | 失败，包括结构检查不通过；`batch` 里任一视频失败也是 1 |
| `2` | 命令行参数有误；`serve` 没有设置 token |
| `3` | 成功但有降级；`batch` 是没有视频失败、至少一个降级 |
