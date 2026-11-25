# Universal Model Upload & VRAM Control Manual

The **Cluster GPU Controller** (`./cl gpu-ui`) starts instantaneously with an empty VRAM pool so you can upload and manage **any model** according to your needs.

---

## 1. Download an Official Chat Model (Optional)

If you do not already have a `.gguf` model on your device, run the included helper script:

```bash
./example-model.sh
```

This fetches the official HuggingFace **`Qwen/Qwen2.5-0.5B-Instruct-GGUF`** (`models/qwen2.5-0.5b-instruct-q4_0.gguf`, ~409 MB), which fits comfortably in 2 GB legacy VRAM and answers user prompts cleanly.

*(For a larger model, pass `1.5b`: `./example-model.sh 1.5b`)*

---

## 2. Launch the Web UI

```bash
./cl gpu-ui --port 8888
```

Open **`http://127.0.0.1:8888`** in your browser. Startup is instant because no model is pre-loaded on boot.

---

## 3. Upload, Chat & Offload

1. **Upload Model**: Click or drag & drop any `.gguf`, `.clm`, `.onnx`, or weight file into the Upload box.
   - Real-time upload progress tracks transfer from browser to `models/uploaded/`.
   - The controller allocates a hardware OpenGL 4.3 SSBO directly in GPU VRAM and frees host CPU RAM.
   - For `.gguf` chat models, the runtime automatically prepares the CLM inference engine so you can chat immediately.
2. **Test / Ask Questions**: Type any prompt in the **Model Test & Question Answering Terminal** and click **🚀 Ask Model (GPU)**.
3. **Offload Model**: Click **🧹 Offload Model from GPU VRAM** whenever you want to free GPU memory (`0.00 MB` allocated).

---

## 4. REST API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/status` | Returns GPU VRAM telemetry (Allocated MB, Free MB, Active Model, SSBO ID). |
| `POST` | `/api/upload_model?filename=xyz` | Uploads binary model stream to `models/uploaded/xyz`. |
| `POST` | `/api/load_model` | Loads model at given path into GPU hardware VRAM SSBO (auto-packs GGUF for chat). |
| `POST` | `/api/chat` | Runs autoregressive neural inference on the currently loaded GPU model. |
| `POST` | `/api/offload_model` | Deallocates model SSBO from VRAM and reclaims hardware memory. |
