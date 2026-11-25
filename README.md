# 🚀 Cluster GPU: Legacy GPU Hardware VRAM Unlock & Universal Neural Runtime

```
   ____ _           _              ____ ____  _   _ 
  / ___| |_   _ ___| |_ ___ _ __  / ___|  _ \| | | |
 | |   | | | | / __| __/ _ \ '__|| |  _| |_) | | | |
 | |___| | |_| \__ \ ||  __/ |   | |_| |  __/| |_| |
  \____|_|\__,_|___/\__\___|_|    \____|_|    \___/ 
```

[![Language: Cluster](https://img.shields.io/badge/Language-Cluster%20(.cl%20%2F%20.zk)-00ff88?style=flat-square)](https://github.com/cluster-lang)
[![Hardware: GT 610](https://img.shields.io/badge/Hardware-NVIDIA%20GeForce%20GT%20610-76b900?style=flat-square)](https://www.nvidia.com)
[![VRAM: 2039 MB](https://img.shields.io/badge/Dedicated%20VRAM-2039%20MB%20DDR3-00d4ff?style=flat-square)]()
[![Zero Python](https://img.shields.io/badge/Python%20Files-0%20(Pure%20Native)-ff007f?style=flat-square)]()
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square)](LICENSE)

> **Breakthrough Achievement**: Unlocking the physical **2,039 MB discrete DDR3 VRAM** and **48 Fermi CUDA cores** of legacy GPUs (NVIDIA GeForce GT 610 / GF119 / NVD9) for on-demand model uploading, dynamic VRAM allocation, instant offloading, and autoregressive neural chat.

---

## 💻 Supported Platforms & Prerequisites

> [!IMPORTANT]
> **Linux x86-64 (Tested on Debian 13 Trixie) Only** — Precompiled binaries in [`bin/`](bin/) are **Linux x86-64 ELF executables** targeting **Debian 13** (`GLIBC >= 2.38`, X11/GLX, and OpenGL 4.3+ compute shaders).
> - **Windows**: **Not supported natively** (Linux ELF binaries + X11/GLX + POSIX threads/sockets). May work only inside **WSL2** with WSLg/X11 and an OpenGL 4.3+ driver (untested).
> - **macOS**: **Not supported** (macOS uses Mach-O binaries, deprecated OpenGL at 4.1 without OpenGL 4.3 compute shaders/SSBOs, and lacks native GLX).

### System Requirements (Fresh Debian 13 Clone)
- **GPU Web UI (`./cl gpu-ui --port 8888`)**: Works immediately after cloning with precompiled [`bin/cl`](bin/cl); requires X11 (`libx11-6`), OpenGL 4.3+ (`libgl1` / NVIDIA or Nouveau driver with SSBO support), and `curl` / `wget` for [`example-model.sh`](example-model.sh).
- **Compiling & Running `.cl` Programs (`./cl run` / `./cl build`)**: Uses precompiled [`bin/cl2`](bin/cl2) + [`lib/runtime/concurrency.o`](lib/runtime/concurrency.o) and links via system **`clang`**:
  ```bash
  sudo apt update && sudo apt install -y clang libx11-6 libgl1 curl
  ```

---

## 🌟 Key Capabilities

1. **Instant Clean Startup (On-Demand Loading)**:
   - Starts immediately (`< 1s`) with `0.00 MB` allocated — never stalls looking for pre-bundled models on your device.
2. **Upload-Only Dynamic Model Web UI**:
   - Drag & drop or click to upload **any** `.gguf`, `.clm`, or `.onnx` model directly from your browser with real-time upload progress.
   - Automatically transfers model weights over PCIe directly into GPU VRAM SSBOs and frees Host CPU RAM.
3. **Automatic GGUF Chat Engine Support**:
   - Upload any compatible `.gguf` chat model (such as official Qwen 2.5 Instruct GGUFs), ask questions in the live terminal, and offload when finished.
4. **One-Click VRAM Offloading**:
   - Click **Offload Model from GPU VRAM** to immediately release hardware buffers back to `0.00 MB`.
5. **Dual Binary Architecture (`cl` vs `cl2`)**:
   - **`cl`** (and `cluster-gpu` symlink): The unified CLI driver & GPU Web UI server. Run `./cl gpu-ui`, `./cl run ...`, or `./cl build ...`.
   - **`cl2`**: The internal compiler backend invoked automatically by `cl` whenever compiling `.cl` or `.zk` programs. Users always run `cl`.

---

## 🖥️ Quickstart: Fetch Model, Upload & Test on GPU

### 1. Download an Official `.gguf` Chat Model
Run [`example-model.sh`](example-model.sh) to fetch the official HuggingFace `Qwen2.5-0.5B-Instruct` GGUF model (~409 MB) into `models/`:

```bash
./example-model.sh
```

*(Need a larger model? Run `./example-model.sh 1.5b` to fetch Qwen 2.5 1.5B Instruct GGUF.)*

### 2. Launch the GPU Controller UI
```bash
./cl gpu-ui --port 8888
```

Open **`http://127.0.0.1:8888`** in your browser:
1. Click the **Upload** zone and select `models/qwen2.5-0.5b-instruct-q4_0.gguf` (or any `.gguf`/`.clm` file).
2. Watch the VRAM heap bar allocate directly in GPU memory.
3. Type any question in the **Question Answering Terminal** and click **🚀 Ask Model (GPU)**.
4. Click **🧹 Offload Model from GPU VRAM** when done to return memory to `0.00 MB`.

---

## ⚡ Run Pure Cluster GPU Examples (`.cl`)

Compile and run hardware VRAM programs directly with `./cl`:

```bash
# Run direct 48 CUDA cores compute shader dispatch
./cl run examples/05_cuda_compute_dispatch.cl

# Load any GGUF/CLM model directly into physical VRAM
./cl run examples/01_load_model_vram.cl

# Instant GPU VRAM offloading
./cl run examples/02_offload_model_vram.cl

# Autoregressive neural chat inference
./cl run examples/03_gpu_inference_chat.cl

# Indra-OCR ONNX ingestion into GT 610 VRAM
./cl run examples/04_indra_ocr_gt610.cl
```

---

## 🔌 REST API Endpoints

```bash
# 1. Check live GPU VRAM status
curl -s http://127.0.0.1:8888/api/status

# 2. Upload a model file over HTTP
curl -s -X POST --data-binary @models/qwen2.5-0.5b-instruct-q4_0.gguf "http://127.0.0.1:8888/api/upload_model?filename=qwen2.5-0.5b-instruct-q4_0.gguf"

# 3. Load uploaded model into GPU VRAM
curl -s -X POST -d "models/uploaded/qwen2.5-0.5b-instruct-q4_0.gguf" http://127.0.0.1:8888/api/load_model

# 4. Ask a question to the uploaded model
curl -s -X POST -d "What is Cluster Language?" http://127.0.0.1:8888/api/chat

# 5. Offload model and free GPU VRAM
curl -s -X POST http://127.0.0.1:8888/api/offload_model
```

---

## 📚 Repository Structure

- [`cl`](cl) / [`bin/cl`](bin/cl) / [`bin/cluster-gpu`](bin/cluster-gpu): Main CLI driver & GPU Web UI executable.
- [`bin/cl2`](bin/cl2): Precompiled Cluster compiler backend (invoked automatically by `cl`).
- [`example-model.sh`](example-model.sh): Official GGUF chat model fetcher.
- [`lib/runtime/`](lib/runtime/): Precompiled runtime support object (`concurrency.o`).
- [`std/`](std/): Pure Cluster Standard Libraries for GPU, AI, CLM, and tensors (`.cl` / `.zk`).
- [`examples/`](examples/): Standalone runnable GPU examples (`01` through `05`).
- [`tests/`](tests/): GPU VRAM, Qwen, Indra-OCR, and tensor test suites.
- [`docs/`](docs/): Technical hardware and model ingestion documentation.

---

## 📄 License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
