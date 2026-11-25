// Example 4: Indra-OCR Document Text Recognition on GT 610 VRAM
// Loads detection (det.onnx) and recognition (rec.onnx) models directly into hardware VRAM.

import "std/tensor.cl"
import "std/ai/gpu.cl"
import "std/math.zk"

fn main():
    println("=======================================================")
    println(" Cluster GPU: Indra-OCR Hardware Ingestion on GT 610")
    println("=======================================================")

    ok := gpu_vram_init()
    if ok != 1:
        println("[ERROR] Failed to initialize GPU OpenGL 4.3 context")
        return

    det_path := "/home/alamgir-zk/.cache/huggingface/hub/models--datalab-to--indra-ocr/snapshots/bd02167fdf2005a19962cc15188f8496a2c87d3b/det.onnx"
    rec_path := "/home/alamgir-zk/.cache/huggingface/hub/models--datalab-to--indra-ocr/snapshots/bd02167fdf2005a19962cc15188f8496a2c87d3b/rec.onnx"

    det_buf := gpu_vram_load_file(det_path)
    if det_buf.buf_id > 0:
        println("[OK] Indra-OCR Detector loaded into VRAM: SSBO #" + to_text(det_buf.buf_id) + " (" + to_text(det_buf.num_elements) + " bytes)")
        gpu_vram_release(det_buf)

    rec_buf := gpu_vram_load_file(rec_path)
    if rec_buf.buf_id > 0:
        println("[OK] Indra-OCR Recognizer loaded into VRAM: SSBO #" + to_text(rec_buf.buf_id) + " (" + to_text(rec_buf.num_elements) + " bytes)")
        gpu_vram_release(rec_buf)

    println("OCR ingestion check completed.")
