"""
Diarizacion con pyannote.audio 4.0 / speaker-diarization-community-1.
Corre DENTRO de venv-dia-community1 (aislado del venv principal).

Hace UN solo intento en el dispositivo/batch indicado y devuelve un codigo de
salida; el orquestador (diarizers.py) decide los reintentos en procesos frescos
(una OOM de CUDA corrompe el contexto, asi que no se reintenta en el mismo proceso):
    GPU normal -> GPU batch reducido -> CPU.

Recibe un WAV de 16 kHz mono y lo pasa como waveform en memoria (evita torchcodec).

Ademas de <out_json>, en exito escribe un sidecar "<out_json sin extension>.stats.json"
con {"peak_vram_gb": float|null, "elapsed_sec": float} para que el orquestador
(diarizers.py) pueda armar el benchmark (--bench-report) sin tocar el contrato
existente de diarize() -> DataFrame.

Uso:
    python _diar_pyannote4.py <wav16k> <out_json> <hf_token> <device> <batch> [min] [max]
    device: cuda | cpu     batch: entero o "" (default del modelo)

Codigos de salida: 0 OK | 2 gated/terminos | 3 sin memoria (probar siguiente) | 1 otro
"""

import json
import sys
import time
from pathlib import Path

BATCH_ATTRS = ("embedding_batch_size", "segmentation_batch_size")


def main() -> int:
    wav_path = sys.argv[1]
    out_json = sys.argv[2]
    hf_token = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else None
    device = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] else "cuda"
    batch = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] else None
    min_s = int(sys.argv[6]) if len(sys.argv) > 6 and sys.argv[6] else None
    max_s = int(sys.argv[7]) if len(sys.argv) > 7 and sys.argv[7] else None

    import soundfile as sf
    import torch
    from pyannote.audio import Pipeline

    if device == "cuda" and not torch.cuda.is_available():
        print("   CUDA no disponible; usar cpu", file=sys.stderr)
        return 3

    if device == "cuda":
        # Ayuda a cuDNN 9 a encontrar un algoritmo para las convoluciones del modelo
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    wav, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    waveform = torch.from_numpy(wav).unsqueeze(0)
    audio_input = {"waveform": waveform, "sample_rate": sr}

    try:
        pipe = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1", token=hf_token,
        )
    except Exception as e:  # noqa: BLE001
        s = str(e).lower()
        if any(k in s for k in ("gated", "403", "401", "awaiting", "restricted")):
            print("ERROR: community-1 gated; acepta los terminos en\n"
                  "https://huggingface.co/pyannote/speaker-diarization-community-1",
                  file=sys.stderr)
            return 2
        raise

    if batch is not None:
        for a in BATCH_ATTRS:
            if hasattr(pipe, a):
                try:
                    setattr(pipe, a, batch)
                except Exception:  # noqa: BLE001
                    pass

    kwargs = {}
    if min_s:
        kwargs["min_speakers"] = min_s
    if max_s:
        kwargs["max_speakers"] = max_s

    tag = device + (f" (batch {batch})" if batch else "")
    print(f"   community-1: intentando en {tag}...", file=sys.stderr)
    try:
        pipe.to(torch.device(device))
        result = pipe(audio_input, **kwargs)
    except Exception as e:  # noqa: BLE001
        msg = str(e).lower()
        # Cualquier fallo de GPU (sin memoria, cuDNN sin engine, etc.) -> cae al
        # siguiente dispositivo. Solo se propaga si pasa en CPU o no es de GPU.
        gpu_fail = any(k in msg for k in (
            "out of memory", "memory", "alloc", "cuda", "cudnn",
            "engine", "cublas", "device-side", "kernel image",
        ))
        if device == "cuda" and gpu_fail:
            print(f"   fallo en {tag}: {str(e)[:80]}", file=sys.stderr)
            return 3  # el orquestador probara el siguiente
        raise

    elapsed = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None

    ann = getattr(result, "speaker_diarization", result)
    segs = [
        {"start": float(turn.start), "end": float(turn.end), "speaker": str(spk)}
        for turn, _, spk in ann.itertracks(yield_label=True)
    ]
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(segs, f)
    stats_path = Path(out_json).with_suffix(".stats.json")
    stats_path.write_text(json.dumps({"peak_vram_gb": peak_gb, "elapsed_sec": elapsed}),
                          encoding="utf-8")
    print(f"OK en {tag}: {len(segs)} segmentos, {elapsed:.1f}s"
          + (f", VRAM pico {peak_gb:.2f} GB" if peak_gb is not None else ""),
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
