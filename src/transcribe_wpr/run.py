"""
Pipeline de WhisperX con progreso en tiempo real, estado consultable y
gestion de VRAM por etapas (pensado para GPUs de 4 GB como la RTX 3050).

Por que un pipeline propio en vez de la CLI de whisperx:
  - Libera la VRAM del modelo ASR antes de alinear y diarizar (la CLI no lo hace,
    y por eso la diarizacion daba "CUDA out of memory" en 4 GB).
  - Si la diarizacion aun no cabe en GPU, cae automaticamente a CPU.
  - Emite progreso/etapa en vivo y escribe status.json para consultarlo por CLI.

Etapas: cargando -> vad/transcripcion -> alineacion -> diarizacion -> escribiendo.

Uso (normalmente lo llama transcribe.bat):
    python run.py <audio> <output_dir> [num_speakers] [lang]
"""

import gc
import json
import os
import platform
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from transcribe_wpr import paths


def _quiet_known_warnings() -> None:
    """Silencia SOLO mensajes conocidos e inofensivos del stack. Son filtros por
    MENSAJE exacto (no por categoria global), asi que:
      - NO ocultan errores: las excepciones no pasan por el modulo `warnings`.
      - NO ocultan avisos futuros reales: cualquier warning con otro texto sigue
        mostrandose.
    Debe ejecutarse antes de importar whisperx, porque el de pkg_resources se
    emite al importar ctranslate2."""
    import warnings
    for msg in (
        r"pkg_resources is deprecated as an API",                  # ctranslate2 -> pkg_resources
        r"You are using `torch\.load` with `weights_only=False`",  # lightning_fabric / torch.load
        r"Module 'speechbrain\.pretrained' was deprecated",        # speechbrain 1.0 (lo usa pyannote 3.4)
        r"TensorFloat-32 \(TF32\) has been disabled",              # pyannote ReproducibilityWarning (#6)
    ):
        warnings.filterwarnings("ignore", message=msg)


def _silence_pyannote_version_prints() -> None:
    """El aviso 'Model was trained with ... Bad things might happen' de pyannote
    NO se emite por el modulo `warnings` sino con print() (ver
    pyannote/audio/utils/version.py), asi que filterwarnings no lo toca. Lo
    anulamos sustituyendo check_version por un no-op en los DOS modulos que lo
    importan por nombre (core.model y core.pipeline). Es puramente cosmetico: el
    '0.0.1' es un placeholder de los metadatos del checkpoint y el modelo corre
    bien. Mejor esfuerzo: si pyannote cambia de estructura, no rompe nada.

    Debe llamarse despues de importar whisperx (que arrastra pyannote)."""
    import importlib

    def _noop(*_args, **_kwargs):
        return None

    for modname in ("pyannote.audio.core.model", "pyannote.audio.core.pipeline"):
        try:
            mod = importlib.import_module(modname)
            if hasattr(mod, "check_version"):
                mod.check_version = _noop
        except Exception:  # noqa: BLE001 - cosmetico, nunca debe romper el pipeline
            pass


_quiet_known_warnings()

GLOBAL_STATUS = paths.GLOBAL_STATUS
BENCH_FILE = paths.BENCH_FILE

# Cada etapa mapea su 0-100% local a un rango del progreso global.
STAGE_RANGES = {
    "cargando":      (0, 3),
    "transcripcion": (3, 65),
    "alineacion":    (65, 85),
    "diarizacion":   (85, 98),
    "escribiendo":   (98, 100),
    "completado":    (100, 100),
}
STAGE_LABELS = {
    "cargando": "Cargando modelos",
    "transcripcion": "Transcripcion (GPU)",
    "alineacion": "Alineacion de palabras",
    "diarizacion": "Diarizacion de oradores",
    "escribiendo": "Escribiendo archivos",
    "completado": "Completado",
}
PROGRESS_RE = re.compile(r"Progress:\s*([\d.]+)%")

# Adaptativo ante "CUDA out of memory":
#  - batch_size arranca alto (velocidad) y baja de 1 en 1 hasta 1.
#  - beam_size arranca alto (precision) y baja de 2 en 2 hasta 5.
# Estrategia: primero reducir batch_size (gran impacto en VRAM) manteniendo
# beam alto (su costo es minimo); solo como ultimo recurso, bajar beam.
# START_BATCH=4 es el maximo medido que cabe en 4 GB (batch 5 hace OOM).
# Si tienes mas VRAM o quieres auto-sondear, subelo; el fallback baja solo.
START_BATCH = 4
START_BEAM = 10
MIN_BEAM = 5


def beam_sequence(start: int) -> list[int]:
    """Secuencia de beam_size: baja de 2 en 2 hasta el minimo (5).
    Ej.: 10 -> [10, 8, 6, 5]; 7 -> [7, 5]; 5 -> [5]. (de 6 baja a 5, no a 4)."""
    seq = []
    b = start
    while b > MIN_BEAM:
        seq.append(b)
        b -= 2
    seq.append(MIN_BEAM)
    return list(dict.fromkeys(seq))


def batch_sequence(start: int) -> list[int]:
    """Secuencia de batch_size: baja de 1 en 1 hasta 1. Ej.: 5 -> [5,4,3,2,1]."""
    return list(range(start, 0, -1))


def transcribe_attempts(start_batch: int = START_BATCH,
                        start_beam: int = START_BEAM) -> list[tuple[int, int]]:
    """Lista de intentos (batch_size, beam_size) del mejor al mas seguro:
    baja primero el batch (con beam alto) y, como ultimo recurso, baja el beam.
    Ej.: start_batch=4 -> [(4,10),(3,10),(2,10),(1,10),(1,8),(1,6),(1,5)]."""
    seq = [(b, start_beam) for b in batch_sequence(start_batch)]
    seq += [(1, bm) for bm in beam_sequence(start_beam)[1:]]
    # quita duplicados conservando el orden
    return list(dict.fromkeys(seq))


def append_bench_report(diarizer: str, speakers_label: str, diar_batch,
                        audio_duration_sec: float, elapsed_wall_sec: float,
                        success: bool, error: str | None = None) -> None:
    """Agrega una linea a benchmarks.jsonl con GPU/VRAM/tiempo de esta
    diarizacion. Solo se llama si el usuario paso --bench-report true; nunca
    se escribe nada por defecto. Mejor esfuerzo: nunca rompe el pipeline."""
    try:
        import torch
        gpu_name = vram_total_gb = None
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu_name = props.name
            vram_total_gb = round(props.total_memory / 1e9, 2)

        from transcribe_wpr import diarizers as _diarizers
        stats = dict(_diarizers.LAST_DIAR_STATS)

        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "os": platform.platform(),
            "gpu_name": gpu_name,
            "vram_total_gb": vram_total_gb,
            "diarizer": diarizer,
            "num_speakers_requested": speakers_label,
            "diar_batch_requested": diar_batch,
            "audio_duration_sec": round(audio_duration_sec, 1),
            "elapsed_wall_sec": round(elapsed_wall_sec, 1),
            "diar_peak_vram_gb": stats.get("peak_vram_gb"),
            "diar_elapsed_sec": stats.get("elapsed_sec"),
            "diar_device_used": stats.get("device"),
            "diar_batch_used": stats.get("diar_batch_used"),
            "success": success,
            "error": error[:300] if error else None,
        }
        with open(BENCH_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 - el benchmark nunca debe romper el pipeline
        pass


def is_oom(err: Exception) -> bool:
    msg = str(err).lower()
    return any(k in msg for k in ("out of memory", "outofmemory", "cuda error",
                                  "cublas", "cudnn", "alloc"))


def parse_speakers(s: str) -> tuple[int | None, int | None]:
    """Interpreta el argumento de oradores:
      ""    -> (None, None)  deteccion automatica
      "4"   -> (4, 4)        exactamente 4
      "4-8" -> (4, 8)        entre 4 y 8 (min, max)
    """
    s = (s or "").strip()
    if not s:
        return None, None
    if "-" in s:
        lo, hi = (p.strip() for p in s.split("-", 1))
        lo_i, hi_i = int(lo), int(hi)
        if lo_i > hi_i:
            lo_i, hi_i = hi_i, lo_i  # tolera "8-4"
        return lo_i, hi_i
    v = int(s)
    return v, v

# ----- estado global mutable (lo actualizan las etapas) -----
state = {
    "start": time.time(),
    "stage": "cargando",
    "stage_pct": 0.0,
    "obj": None,        # dict que se serializa a status.json
    "last_write": 0.0,
}
_write_lock = threading.Lock()
_stop_heartbeat = threading.Event()


def fmt_elapsed(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def overall_percent(stage: str, stage_pct: float) -> float:
    lo, hi = STAGE_RANGES.get(stage, (0, 0))
    return round(lo + (hi - lo) * (stage_pct / 100.0), 1)


def write_status(force: bool = False) -> None:
    now = time.time()
    obj = state["obj"]
    obj["stage"] = state["stage"]
    obj["stage_label"] = STAGE_LABELS.get(state["stage"], state["stage"])
    obj["stage_percent"] = round(state["stage_pct"], 1)
    obj["overall_percent"] = overall_percent(state["stage"], state["stage_pct"])
    obj["elapsed_seconds"] = int(now - state["start"])
    obj["elapsed"] = fmt_elapsed(now - state["start"])
    obj["updated_at"] = datetime.now().isoformat(timespec="seconds")
    if not force and now - state["last_write"] < 1.0:
        return
    state["last_write"] = now
    payload = json.dumps(obj, ensure_ascii=False, indent=2)
    with _write_lock:
        for target in (obj["_local_path"], str(GLOBAL_STATUS)):
            try:
                tmp = target + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    f.write(payload)
                os.replace(tmp, target)
            except OSError:
                pass


def _heartbeat() -> None:
    """Mantiene el estado 'vivo' (tiempo transcurrido) aunque no haya
    lineas de progreso (carga de modelos, VAD, diarizacion)."""
    while not _stop_heartbeat.wait(3.0):
        write_status(force=True)
        draw_bar()


def cleanup_status_file() -> None:
    """Elimina los archivos de estado al terminar:
    el .transcribe_status.json (carpeta de salida) y el last_status.json (global).

    - En terminal interactiva: pide confirmacion (Enter / S = si).
    - No interactiva (lanzado por un agente o en segundo plano): NO borra ni
      bloquea; conserva los archivos (done=True) para que el agente confirme y
      los elimine cuando acabe.
    """
    targets = [state["obj"]["_local_path"], str(GLOBAL_STATUS)]
    existentes = [t for t in targets if os.path.exists(t)]
    if not existentes:
        return

    def _borrar():
        for t in existentes:
            try:
                os.remove(t)
            except OSError as e:
                print(f"No se pudo eliminar {t}: {e}", file=sys.__stdout__, flush=True)
        print("Archivos de estado eliminados.", file=sys.__stdout__, flush=True)

    interactive = bool(getattr(sys, "stdin", None)) and sys.stdin.isatty()
    if interactive:
        try:
            ans = input(
                "\n¿Eliminar los archivos de estado (status.json)? [S/n]: "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = "s"
        if ans in ("", "s", "si", "sí", "y", "yes"):
            _borrar()
        else:
            print("Archivos de estado conservados.", file=sys.__stdout__, flush=True)
    else:
        print(
            "NOTA: ejecucion no interactiva; los archivos de estado se conservan "
            "(done=True) para que el agente los confirme y elimine.",
            file=sys.__stdout__, flush=True,
        )


def draw_bar() -> None:
    obj = state["obj"]
    pct = overall_percent(state["stage"], state["stage_pct"])
    n = int(pct / 5)
    bar = "#" * n + "-" * (20 - n)
    sys.__stdout__.write(
        f"\r[{bar}] {pct:5.1f}% | {STAGE_LABELS.get(state['stage'], state['stage'])} "
        f"{state['stage_pct']:5.1f}% | {fmt_elapsed(time.time() - state['start'])}   "
    )
    sys.__stdout__.flush()


def set_stage(stage: str) -> None:
    state["stage"] = stage
    state["stage_pct"] = 0.0
    sys.__stdout__.write("\n")
    print(f">> {STAGE_LABELS.get(stage, stage)}", file=sys.__stdout__, flush=True)
    write_status(force=True)
    draw_bar()


class ProgressTee:
    """Reemplaza sys.stdout: reenvia todo a la consola y captura 'Progress: X%'."""

    def __init__(self, real):
        self.real = real
        self.buf = ""

    def write(self, s):
        self.real.write(s)
        self.buf += s
        while True:
            idx = min((self.buf.find(c) for c in "\r\n" if c in self.buf), default=-1)
            if idx < 0:
                break
            line, self.buf = self.buf[:idx], self.buf[idx + 1:]
            m = PROGRESS_RE.search(line)
            if m:
                state["stage_pct"] = float(m.group(1))
                write_status()
                draw_bar()

    def flush(self):
        self.real.flush()


def main(argv=None) -> int:
    import argparse
    from transcribe_wpr import diarizers

    p = argparse.ArgumentParser(prog="transcribe", add_help=True)
    p.add_argument("audio")
    p.add_argument("speakers", nargs="?", default="")
    p.add_argument("lang", nargs="?", default="es")
    p.add_argument("diarizer", nargs="?", default="")
    p.add_argument("--output", required=True, help="carpeta de salida")
    p.add_argument("--batch", type=int, default=None,
                   help=f"batch_size de transcripcion (def {START_BATCH}; sin limite "
                        "duro: si no cabe en VRAM baja solo de 1 en 1 hasta 1)")
    p.add_argument("--beam", type=int, default=None,
                   help=f"beam_size de transcripcion (def {START_BEAM}; sin limite duro: "
                        f"si hace OOM baja de 2 en 2 hasta el piso {MIN_BEAM})")
    p.add_argument("--diar-batch", dest="diar_batch", type=int, default=None,
                   help="batch de diarizacion (independiente; def por modelo 64, "
                        "baja solo 64->32->16->8->4->CPU)")
    p.add_argument("--bench-report", dest="bench_report", default="false",
                   help="true/false: agrega una linea a benchmarks.jsonl con "
                        "GPU/VRAM/tiempo de la diarizacion (def false, no escribe nada)")
    a = p.parse_args(argv)

    audio_path = a.audio
    output_dir = a.output
    num_speakers = a.speakers or ""
    lang = a.lang or "es"
    diarizer = a.diarizer or diarizers.DEFAULT_DIARIZER
    start_batch = a.batch if a.batch else START_BATCH
    start_beam = a.beam if a.beam else START_BEAM
    diar_batch = a.diar_batch  # None -> el backend usa su propio default per-modelo
    bench_report = str(a.bench_report).strip().lower() in ("true", "1", "yes", "si", "sí")
    # Acepta "4" (exacto: min=max=4) o "4-8" (min=4, max=8). Vacio = automatico.
    min_spk, max_spk = parse_speakers(num_speakers)
    speakers_label = (
        f"{min_spk}-{max_spk}" if min_spk != max_spk
        else (str(min_spk) if min_spk is not None else "auto")
    )

    os.makedirs(output_dir, exist_ok=True)
    local_status = str(Path(output_dir) / ".transcribe_status.json")
    started_at = datetime.now().isoformat(timespec="seconds")

    state["obj"] = {
        "_local_path": local_status,
        "audio": audio_path,
        "output_dir": output_dir,
        "num_speakers": speakers_label,
        "language": lang,
        "diarizer": diarizer,
        "batch_size": None,
        "beam_size": None,
        "stage": "cargando",
        "stage_label": STAGE_LABELS["cargando"],
        "stage_percent": 0.0,
        "overall_percent": 0.0,
        "elapsed": "00:00:00",
        "elapsed_seconds": 0,
        "started_at": started_at,
        "updated_at": started_at,
        "done": False,
        "error": None,
        "pid": os.getpid(),
    }

    print("=" * 60)
    print(f"  Audio     : {audio_path}")
    print(f"  Oradores  : {speakers_label}   Idioma: {lang}")
    print(f"  Diarizador: {diarizer}")
    print(f"  Salida    : {output_dir}")
    print("=" * 60, flush=True)
    write_status(force=True)

    # Validar el diarizador elegido antes del trabajo pesado
    diarizers.get_backend(diarizer)  # lanza si el nombre es invalido

    # Imports pesados despues de configurar el estado (tardan unos segundos)
    import torch
    import whisperx
    from whisperx.diarize import assign_word_speakers
    from whisperx.utils import get_writer

    _silence_pyannote_version_prints()  # #5: anula el print de version de pyannote

    sys.stdout = ProgressTee(sys.__stdout__)

    hb = threading.Thread(target=_heartbeat, daemon=True)
    hb.start()

    def free_gpu():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    diar_attempted = False
    t_diar_start = 0.0
    try:
        # ---- 1. Transcripcion (GPU) con batch_size/beam_size adaptativos ----
        set_stage("transcripcion")
        audio = whisperx.load_audio(audio_path)
        attempts = transcribe_attempts(start_batch, start_beam)
        result = None
        for i, (batch, beam) in enumerate(attempts):
            try:
                state["stage_pct"] = 0.0
                state["obj"]["batch_size"] = batch
                state["obj"]["beam_size"] = beam
                print(f">> batch_size={batch} beam_size={beam} "
                      f"(intento {i + 1}/{len(attempts)})",
                      file=sys.__stdout__, flush=True)
                model = whisperx.load_model(
                    "large-v3", device="cuda", compute_type="int8",
                    language=lang, asr_options={"beam_size": beam},
                )
                result = model.transcribe(
                    audio, batch_size=batch, chunk_size=20, language=lang,
                    print_progress=True,
                )
                del model
                free_gpu()
                write_status(force=True)
                break
            except Exception as e:  # noqa: BLE001
                try:
                    del model
                except NameError:
                    pass
                free_gpu()
                if is_oom(e) and i < len(attempts) - 1:
                    nb, nbe = attempts[i + 1]
                    print(f"   OOM (batch {batch}, beam {beam}); "
                          f"reintentando con batch {nb}, beam {nbe}",
                          file=sys.__stdout__, flush=True)
                    continue
                raise
        if result is None:
            raise RuntimeError("No se pudo transcribir ni en la configuracion minima")

        # ---------- 2. Alineacion (GPU) ----------
        set_stage("alineacion")
        align_model, metadata = whisperx.load_align_model(
            language_code=result.get("language", lang), device="cuda",
        )
        result = whisperx.align(
            result["segments"], align_model, metadata, audio, "cuda",
            return_char_alignments=False, print_progress=True,
        )
        del align_model, metadata
        free_gpu()

        # ---------- 3. Diarizacion (backend seleccionable y desacoplado) ----------
        set_stage("diarizacion")
        from transcribe_wpr import diarizers
        backend = diarizers.get_backend(diarizer)
        ok, reason = backend.availability()
        if not ok:
            raise RuntimeError(
                f"Diarizador '{diarizer}' no disponible: {reason}"
            )
        print(f">> Diarizador: {backend.label}", file=sys.__stdout__, flush=True)
        diar_attempted = True
        t_diar_start = time.time()
        diarize_segments = backend.diarize(
            audio, min_spk, max_spk,
            os.environ.get("HF_TOKEN", ""), "cuda", diar_batch=diar_batch,
            log=lambda m: print(m, file=sys.__stdout__, flush=True),
        )
        if bench_report:
            append_bench_report(diarizer, speakers_label, diar_batch,
                                len(audio) / 16000, time.time() - t_diar_start,
                                success=True)
        free_gpu()
        result = assign_word_speakers(diarize_segments, result)

        # ---------- 4. Escritura de todos los formatos ----------
        set_stage("escribiendo")
        result["language"] = lang
        writer = get_writer("all", output_dir)
        writer(result, audio_path, {
            "highlight_words": False, "max_line_count": None, "max_line_width": None,
        })

        # JSON con formato legible (indentado). WhisperX lo escribe compacto.
        for jf in Path(output_dir).glob("*.json"):
            if jf.name.startswith("."):
                continue
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
                jf.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except (OSError, json.JSONDecodeError):
                pass

        # Markdown con formato "Orador n" (in-process: convert_to_md es stdlib pura)
        from transcribe_wpr import convert_to_md
        try:
            convert_to_md.convert_dir(output_dir)
        except Exception as e:  # noqa: BLE001 - no romper si el .md falla
            print(f"AVISO: no se pudo generar el .md: {e}",
                  file=sys.__stdout__, flush=True)

        state["stage"] = "completado"
        state["stage_pct"] = 100.0
        state["obj"]["done"] = True
        write_status(force=True)
        draw_bar()
        sys.__stdout__.write("\n")
        print(f"LISTO en {state['obj']['elapsed']}. Archivos en: {output_dir}",
              file=sys.__stdout__, flush=True)

        cleanup_status_file()
        return 0

    except Exception as e:  # noqa: BLE001
        import traceback
        if bench_report and diar_attempted:
            append_bench_report(diarizer, speakers_label, diar_batch,
                                len(audio) / 16000, time.time() - t_diar_start,
                                success=False, error=f"{type(e).__name__}: {e}")
        state["obj"]["error"] = f"{type(e).__name__}: {e}"
        write_status(force=True)
        sys.__stdout__.write("\n")
        traceback.print_exc(file=sys.__stdout__)
        print(f"ERROR: {state['obj']['error']}", file=sys.__stdout__, flush=True)
        return 1
    finally:
        _stop_heartbeat.set()
        sys.stdout = sys.__stdout__


if __name__ == "__main__":
    sys.exit(main())
