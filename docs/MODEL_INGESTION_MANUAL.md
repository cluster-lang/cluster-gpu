# Universal Model Ingestion & VRAM Control Manual

The **Cluster GPU Controller** provides a comprehensive Graphical Dashboard and REST API allowing users to push, upload, load, test, and offload **any neural model** directly to legacy GPU hardware VRAM.

---

## 1. Launching the Web UI

Run the following command in terminal:

```bash
cl gpu-ui --port 8888
```

Open your browser to:
```
http://127.0.0.1:8888
```

---

## 2. Ingesting Models into GPU VRAM

The interface supports three ingestion paths:

### Method A: Quick Presets (1-Click)
Click any preset chip:
- **🧠 Qwen 2.5 0.5B (GGUF)**: 408.87 MB Q4_0 quantized weights.
- **🧠 Qwen 2.5 0.5B (CLM)**: 304.98 MB Native INT4 Cluster model.
- **📄 Indra-OCR Det**: 381 KB ONNX document text detection network.
- **📄 Indra-OCR Rec**: 4.9 MB ONNX text recognition model.

### Method B: Local File Path
1. Enter the absolute or relative path to any model file on disk in the input box:
   ```
   /path/to/any/model.gguf
   /path/to/any/model.clm
   /path/to/any/model.onnx
   ```
2. Click **⚡ Load** to trigger PCIe DMA allocation into GT 610 VRAM.

### Method C: Browser Drag & Drop Upload
1. Drag and drop any `.gguf`, `.clm`, or `.onnx` file into the upload zone, or click to browse.
2. The web server streams the file directly to `models/uploaded/<filename>`.
3. The server immediately allocates a hardware OpenGL 4.3 SSBO buffer in GT 610 VRAM and mounts it as the active inference engine!

---

## 3. Real-Time Model Testing & Chat

1. In the **Live Model Inference Terminal**, enter any question or prompt:
   - *"What is Cluster Language?"*
   - *"Explain how discrete VRAM allocation works on Fermi GPUs."*
   - *"Summarize quantum computing in 2 sentences."*
2. Click **🚀 Ask Model on GPU** (or press Enter).
3. The GT 610 48 CUDA cores execute forward inference directly on the VRAM resident weights.
4. The generated response and the exact execution time in milliseconds (`⚡ XXXX ms`) will be displayed in real time.

---

## 4. Offloading from GPU VRAM

To reclaim GPU memory or swap to a different model:
1. Click **🧹 Offload from GPU**.
2. The server calls `zk_gpu_vram_free(buf_id)` and deallocates the hardware SSBO.
3. The VRAM meter immediately drops to `0.00 MB` allocated, returning 100% of memory to the available pool.

---

## 5. REST API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/status` | Returns GPU VRAM telemetry (Allocated MB, Free MB, Active Model, Buffers). |
| `POST` | `/api/upload_model?filename=xyz` | Uploads binary model stream to `models/uploaded/xyz`. |
| `POST` | `/api/load_model` | Loads model at given path into GT 610 hardware VRAM SSBO. |
| `POST` | `/api/offload_model` | Deallocates model SSBO from VRAM and reclaims hardware memory. |
| `POST` | `/api/chat` | Executes real-time autoregressive neural inference on active model. |
| `POST` | `/api/load_det` | Loads Indra-OCR detection model into VRAM. |
| `POST` | `/api/load_rec` | Loads Indra-OCR recognition model into VRAM. |
| `POST` | `/api/alloc_64mb` | Allocates 64 MB tensor buffer in VRAM. |
| `POST` | `/api/dispatch_cuda` | Dispatches compute shader across 48 Fermi CUDA cores. |
| `POST` | `/api/free_all` | Flushes all active SSBO buffers from GPU VRAM. |
