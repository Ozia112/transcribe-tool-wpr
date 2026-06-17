"""
Diarizacion con NVIDIA NeMo Sortformer STREAMING
(nvidia/diar_streaming_sortformer_4spk-v2).
Corre DENTRO de venv-dia-nemo (Linux/WSL2), aislado del venv principal.

Mismo contrato que _diar_pyannote4.py: un solo intento, codigo de salida,
el orquestador (diarizers.py) decide reintentos en procesos frescos.

Por que la variante streaming y no la offline (v1): medido en RTX 3050 4 GB,
v1 hace OOM ya con audio de 15 min (procesa todo el clip de una pasada).
v2 streaming procesa en ventanas acotadas y diarizo 15 min en 1.24 GB / 8 s
y 30 min en 1.98 GB / 17 s. Limite real encontrado: el extractor de
features (mel-spectrograma) SI procesa el audio completo de una sola vez
(no esta chunked), asi que clips muy largos (60 min, 3 h) igual hacen OOM
en esa etapa aunque el modelo de diarizacion en si sea liviano. Por eso
se valida la duracion ANTES de cargar el modelo.

Limites conocidos (RTX 3050 4 GB):
  - Maximo 4 oradores (checkpoint publico de NVIDIA).
  - Audio mas largo que MAX_SAFE_SEC (margen sobre el limite real ~30-60 min)
    se rechaza con exit 5 en vez de arriesgar un OOM a mitad de proceso.

Ademas de <out_json>, en exito escribe un sidecar "<out_json sin extension>.stats.json"
con {"peak_vram_gb": float, "elapsed_sec": float} para que el orquestador
(diarizers.py) pueda armar el benchmark (--bench-report) sin tocar el contrato
existente de diarize() -> DataFrame.

Uso:
    python _diar_nemo.py <wav16k> <out_json> [min_speakers] [max_speakers]

Codigos de salida:
  0 OK | 1 otro | 2 gated/terminos | 3 sin memoria | 4 oradores>4 | 5 audio muy largo
"""

import json
import sys
import time
from pathlib import Path

MODEL_NAME = "nvidia/diar_streaming_sortformer_4spk-v2"
MAX_SPEAKERS = 4
MAX_SAFE_SEC = 1800  # 30 min: el ultimo punto medido que SI entra en 4 GB


def main() -> int:
    wav_path = sys.argv[1]
    out_json = sys.argv[2]
    min_s = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else None
    max_s = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else None

    if (min_s and min_s > MAX_SPEAKERS) or (max_s and max_s > MAX_SPEAKERS):
        print(f"ERROR: Sortformer ({MODEL_NAME}) soporta como maximo "
              f"{MAX_SPEAKERS} oradores; se pidieron min={min_s} max={max_s}",
              file=sys.stderr)
        return 4

    import soundfile as sf
    info = sf.info(wav_path)
    if info.duration > MAX_SAFE_SEC:
        print(f"ERROR: audio de {info.duration / 60:.1f} min supera el limite "
              f"seguro de {MAX_SAFE_SEC / 60:.0f} min para una sola pasada "
              f"(el extractor de features de NeMo no esta chunked; hace OOM "
              f"en 4 GB con audio largo aunque el modelo si lo este). "
              f"Hay que partir el audio en bloques antes de llamar a este script.",
              file=sys.stderr)
        return 5

    import torch

    if not torch.cuda.is_available():
        print("ERROR: CUDA no disponible en este venv/WSL2", file=sys.stderr)
        return 1

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    try:
        from nemo.collections.asr.models import SortformerEncLabelModel
    except Exception as e:  # noqa: BLE001
        print(f"ERROR importando NeMo: {e}", file=sys.stderr)
        return 1

    try:
        print(f"   cargando {MODEL_NAME}...", file=sys.stderr)
        diar_model = SortformerEncLabelModel.from_pretrained(MODEL_NAME)
        diar_model.eval()
        diar_model.sortformer_modules.chunk_len = 340
        diar_model.sortformer_modules.chunk_right_context = 40
        diar_model.sortformer_modules.fifo_len = 40
        diar_model.sortformer_modules.spkcache_update_period = 300
        diar_model = diar_model.to("cuda")
    except Exception as e:  # noqa: BLE001
        s = str(e).lower()
        if any(k in s for k in ("gated", "403", "401", "awaiting", "restricted")):
            print(f"ERROR: {MODEL_NAME} gated; acepta los terminos en "
                  f"https://huggingface.co/{MODEL_NAME}", file=sys.stderr)
            return 2
        if any(k in s for k in ("memory", "alloc", "cuda", "cudnn")):
            print(f"   sin memoria cargando el modelo: {str(e)[:120]}",
                  file=sys.stderr)
            return 3
        raise

    try:
        print("   diarizando...", file=sys.stderr)
        predicted_segments = diar_model.diarize(audio=[wav_path], batch_size=1)
    except Exception as e:  # noqa: BLE001
        msg = str(e).lower()
        if any(k in msg for k in ("out of memory", "memory", "alloc", "cuda", "cudnn")):
            print(f"   sin memoria diarizando: {str(e)[:120]}", file=sys.stderr)
            return 3
        raise

    elapsed = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"   VRAM pico: {peak_gb:.2f} GB, {elapsed:.1f}s", file=sys.stderr)

    segs = []
    for line in predicted_segments[0]:
        # formato: "begin_seconds end_seconds speaker_N" (p.ej. "speaker_0")
        parts = line.split()
        start, end, spk = float(parts[0]), float(parts[1]), parts[2]
        spk_idx = int(spk.rsplit("_", 1)[-1])
        segs.append({"start": start, "end": end,
                     "speaker": f"SPEAKER_{spk_idx:02d}"})

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(segs, f)
    stats_path = Path(out_json).with_suffix(".stats.json")
    stats_path.write_text(json.dumps({"peak_vram_gb": peak_gb, "elapsed_sec": elapsed}),
                          encoding="utf-8")
    print(f"OK: {len(segs)} segmentos, VRAM pico {peak_gb:.2f} GB", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
