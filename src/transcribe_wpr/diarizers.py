"""
Backends de diarizacion seleccionables y DESACOPLADOS.

Cada backend implementa el mismo contrato y devuelve un DataFrame con columnas
[start, end, speaker], compatible con whisperx.assign_word_speakers. Las
dependencias pesadas de cada backend se importan SOLO cuando se usa ese backend,
para que un backend roto/no instalado no afecte a los demas.

Backends:
  - pyannote-3.1          : in-process, venv principal (pyannote.audio 3.4). Siempre.
  - pyannote-community-1  : venv separado (pyannote.audio 4.0). Requiere su setup.
  - nemo                  : NVIDIA NeMo Sortformer streaming. Solo Linux/WSL2. Cabe en
                            4 GB VRAM (medido: 1.98 GB para 30 min), pero solo audio
                            de hasta ~30 min de una pasada: el extractor de features
                            de NeMo no esta chunked y hace OOM con audio mas largo
                            aunque el modelo de diarizacion si lo este. Para audio
                            largo usar pyannote-3.1 o pyannote-community-1.

Por que venvs separados: community-1 necesita pyannote.audio 4.0, que choca con el
pin de WhisperX (<4.0). NeMo no tiene soporte oficial en Windows nativo.
"""

import json
import os
import platform
import subprocess
import time
from pathlib import Path

from transcribe_wpr import paths

MIN_VRAM_NEMO_GB = 3.0
MAX_AUDIO_SEC_NEMO = 1800  # 30 min: limite medido del extractor de features de NeMo

# Estadisticas (VRAM pico, tiempo) del ULTIMO diarize() exitoso, para que run.py
# las use al armar la linea de --bench-report sin cambiar el contrato de
# diarize() (que sigue devolviendo solo el DataFrame). Cada diarize() la limpia
# al empezar y la rellena al terminar con exito; si falla, queda vacia.
LAST_DIAR_STATS: dict = {}

# Batch de DIARIZACION (independiente del --batch de transcripcion). Controla
# cuantas ventanas/turnos pasan a la vez por los modelos de segmentacion y
# embeddings: mas alto = mas rapido y mas VRAM; mas bajo = menos VRAM y mas lento.
# 32 es el valor por defecto de pyannote. El pipeline baja de 32->16->8->4 ante
# falta de memoria, y solo entonces cae a CPU.
DIAR_BATCH = 32
MIN_DIAR_BATCH = 4


def diar_batch_sequence(start: int) -> list[int]:
    """Secuencia de batch de diarizacion: baja a la mitad hasta el minimo (4).
    Ej.: 32 -> [32, 16, 8, 4]; 8 -> [8, 4]; 2 -> [2]."""
    start = max(int(start), 1)
    seq = []
    b = start
    while b >= MIN_DIAR_BATCH:
        seq.append(b)
        b //= 2
    if not seq:
        seq = [start]
    return list(dict.fromkeys(seq))

# venvs de backends que corren en proceso aparte (en la carpeta de datos)
VENV_COMMUNITY1 = paths.VENV_COMMUNITY1
VENV_NEMO = paths.VENV_NEMO


def gpu_vram_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1e9
    except Exception:  # noqa: BLE001
        pass
    return 0.0


def _is_wsl() -> bool:
    if platform.system() != "Linux":
        return False
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def _wsl_available() -> bool:
    """True si, desde Windows, se puede invocar `wsl.exe` (WSL2 instalado y con
    al menos una distro registrada)."""
    try:
        r = subprocess.run(["wsl.exe", "--", "true"],
                           capture_output=True, timeout=15)
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def _to_wsl_path(win_path) -> str:
    """Convierte una ruta Windows absoluta (C:\\dev\\...) a su equivalente
    bajo WSL2 (/mnt/c/dev/...). El proyecto vive en C:, por eso es visible
    desde WSL2 sin copiar nada."""
    p = Path(win_path).resolve()
    drive = p.drive.rstrip(":").lower()
    rest = p.as_posix()[len(p.drive):]
    return f"/mnt/{drive}{rest}"


def _segments_to_df(segments: list[dict]):
    import pandas as pd
    if not segments:
        return pd.DataFrame(columns=["start", "end", "speaker"])
    return pd.DataFrame(segments)[["start", "end", "speaker"]]


def _read_stats_sidecar(out_json: Path) -> dict:
    """Lee el sidecar "<out_json>.stats.json" que escriben los backends de
    subproceso (_diar_pyannote4.py, _diar_nemo.py) con peak_vram_gb/elapsed_sec."""
    stats_path = out_json.with_suffix(".stats.json")
    if not stats_path.exists():
        return {}
    try:
        return json.loads(stats_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


# ----------------------------------------------------------------------------
class DiarizerBackend:
    name = "base"
    label = "base"
    default_batch = DIAR_BATCH  # batch de diarizacion por defecto (per-modelo)

    def availability(self) -> tuple[bool, str]:
        """(disponible, razon)."""
        return True, ""

    def diarize(self, audio, min_speakers, max_speakers, hf_token, device,
                diar_batch=None, log=print):
        raise NotImplementedError


# ----------------------------------------------------------------------------
class Pyannote31(DiarizerBackend):
    name = "pyannote-3.1"
    label = "pyannote 3.1 (local, por defecto)"
    default_batch = 64  # medido: cabe en 4 GB

    def availability(self):
        return True, "siempre disponible (entorno del paquete)"

    def diarize(self, audio, min_speakers, max_speakers, hf_token, device,
                diar_batch=None, log=print):
        if diar_batch is None:
            diar_batch = self.default_batch
        # audio: array numpy 16 kHz mono (ya decodificado por whisperx)
        from whisperx.diarize import DiarizationPipeline  # import diferido
        from transcribe_wpr import hf_patch
        hf_patch.apply()  # pyannote 3.4 usa use_auth_token (removido en hf_hub 1.x)

        import torch
        import gc

        def free():
            gc.collect()
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001 - tras OOM el contexto puede fallar
                pass

        # GPU bajando el batch -> CPU como ultimo recurso
        if device == "cuda":
            attempts = [("cuda", b) for b in diar_batch_sequence(diar_batch)]
            attempts.append(("cpu", None))
        else:
            attempts = [("cpu", None)]

        last_err = None
        LAST_DIAR_STATS.clear()
        for dev, b in attempts:
            tag = dev + (f" batch {b}" if b else "")
            try:
                log(f">> Diarizacion (pyannote 3.1) en {tag}...")
                if dev == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                dia = DiarizationPipeline(
                    model_name="pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token, device=dev,
                )
                if b is not None:
                    for a in ("embedding_batch_size", "segmentation_batch_size"):
                        if hasattr(dia.model, a):
                            setattr(dia.model, a, b)
                df = dia(audio, min_speakers=min_speakers,
                         max_speakers=max_speakers)
                LAST_DIAR_STATS.update({
                    "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9
                    if dev == "cuda" else None,
                    "elapsed_sec": time.time() - t0,
                    "device": dev,
                    "diar_batch_used": b,
                })
                del dia
                free()
                return df
            except Exception as e:  # noqa: BLE001
                last_err = e
                free()
                msg = str(e).lower()
                if dev == "cuda" and any(k in msg for k in
                                         ("memory", "cuda", "cudnn", "alloc", "engine")):
                    log(f"   sin memoria en {tag}; siguiente fallback")
                    continue
                raise
        raise last_err


# ----------------------------------------------------------------------------
class PyannoteCommunity1(DiarizerBackend):
    name = "pyannote-community-1"
    label = "pyannote community-1 (venv separado, pyannote.audio 4.0)"
    default_batch = 64  # benchmark: cabe en 4 GB (GPU)

    def availability(self):
        py = VENV_COMMUNITY1 / "Scripts" / "python.exe"
        if not py.exists():
            py = VENV_COMMUNITY1 / "bin" / "python"  # por si Linux
        if not py.exists():
            return False, ("no instalado. Ejecuta: transcribe --setup-diarizer "
                           "pyannote-community-1")
        return True, "venv-dia-community1 listo"

    # Orden de intentos: GPU bajando el batch -> CPU (ultimo recurso). Cada intento
    # corre en un PROCESO FRESCO porque una OOM de CUDA corrompe el contexto y no se
    # puede reintentar GPU/CPU de forma fiable en el mismo proceso.
    def diarize(self, audio, min_speakers, max_speakers, hf_token, device,
                diar_batch=None, log=print):
        if diar_batch is None:
            diar_batch = self.default_batch
        import soundfile as sf
        attempts = [("cuda", str(b)) for b in diar_batch_sequence(diar_batch)]
        attempts.append(("cpu", ""))
        py = VENV_COMMUNITY1 / "Scripts" / "python.exe"
        if not py.exists():
            py = VENV_COMMUNITY1 / "bin" / "python"
        tmp_wav = paths.DIAR_TMP_WAV
        out_json = paths.DIAR_TMP_JSON
        for f in (tmp_wav, out_json):
            if f.exists():
                f.unlink()
        # Audio ya decodificado (16 kHz mono) a wav temporal -> evita torchcodec
        sf.write(str(tmp_wav), audio, 16000)

        # El subproceso usa su PROPIO torch (cuDNN 9 incluido, lo halla solo al
        # importar). Le damos un PATH MINIMO y limpio para evitar que el cuDNN 8
        # del venv principal lo contamine ("unable to find an engine ...").
        env = os.environ.copy()
        windir = env.get("SystemRoot", r"C:\Windows")
        env["PATH"] = os.pathsep.join([
            str(py.parent),
            os.path.join(windir, "System32"),
            windir,
            os.path.join(windir, "System32", "Wbem"),
        ])

        last_out = ""
        LAST_DIAR_STATS.clear()
        try:
            for dev, batch in attempts:
                tag = dev + (f" batch {batch}" if batch else "")
                log(f">> Diarizacion (community-1) en {tag}...")
                cmd = [
                    str(py), str(paths.DIAR_PYANNOTE4_SCRIPT),
                    str(tmp_wav), str(out_json), hf_token or "", dev, batch,
                    str(min_speakers or ""), str(max_speakers or ""),
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True,
                                      encoding="utf-8", errors="replace", env=env)
                last_out = proc.stderr or proc.stdout
                if proc.returncode == 0 and out_json.exists():
                    segs = json.loads(out_json.read_text(encoding="utf-8"))
                    stats = _read_stats_sidecar(out_json)
                    if stats:
                        stats.update({"device": dev, "diar_batch_used": batch})
                        LAST_DIAR_STATS.update(stats)
                    return _segments_to_df(segs)
                if proc.returncode == 3:
                    log(f"   sin memoria en {tag}; siguiente fallback")
                    continue  # probar siguiente dispositivo/batch
                # gated (2) u otro error (1): no tiene sentido seguir
                raise RuntimeError(
                    "Fallo la diarizacion community-1:\n" + last_out
                )
            raise RuntimeError(
                "community-1 no pudo diarizar (ni en CPU):\n" + last_out
            )
        finally:
            tmp_wav.unlink(missing_ok=True)
            out_json.unlink(missing_ok=True)
            out_json.with_suffix(".stats.json").unlink(missing_ok=True)


# ----------------------------------------------------------------------------
class Nemo(DiarizerBackend):
    name = "nemo"
    label = "NVIDIA NeMo Sortformer streaming (WSL2, audio <=30 min)"

    def _venv_python(self):
        """Ruta al python de venv-dia-nemo, en la forma que entienda quien lo
        ejecute: WSL2/Linux nativo usa la ruta tal cual; Windows nativo la
        traduce a /mnt/c/... porque la invocacion real pasa por wsl.exe."""
        py = VENV_NEMO / "bin" / "python"
        if platform.system() == "Windows":
            return _to_wsl_path(py)
        return str(py)

    def availability(self):
        system = platform.system()
        if system == "Windows":
            if not _wsl_available():
                return False, ("NeMo requiere WSL2 (no se encontro wsl.exe o "
                               "no hay distros registradas).")
            check = subprocess.run(
                ["wsl.exe", "--", "test", "-x", self._venv_python()],
                capture_output=True, timeout=15,
            )
            if check.returncode != 0:
                return False, ("no instalado en WSL2. Dentro de WSL2: "
                               "transcribe --setup-diarizer nemo")
            return True, "venv-dia-nemo listo en WSL2 (via wsl.exe)"
        if system != "Linux":
            return False, "NeMo requiere Linux o WSL2."
        vram = gpu_vram_gb()
        if vram and vram < MIN_VRAM_NEMO_GB:
            return False, (f"VRAM insuficiente ({vram:.1f} GB); "
                           f"NeMo requiere >= {MIN_VRAM_NEMO_GB:.0f} GB.")
        py = VENV_NEMO / "bin" / "python"
        if not py.exists():
            return False, ("no instalado. En Linux/WSL2: "
                           "transcribe --setup-diarizer nemo")
        return True, "venv-dia-nemo listo" + (" (WSL2)" if _is_wsl() else "")

    def diarize(self, audio, min_speakers, max_speakers, hf_token, device,
                diar_batch=None, log=print):
        LAST_DIAR_STATS.clear()
        duration_sec = len(audio) / 16000
        if duration_sec > MAX_AUDIO_SEC_NEMO:
            raise RuntimeError(
                f"Audio de {duration_sec / 60:.1f} min supera el limite practico de "
                f"NeMo en 4 GB ({MAX_AUDIO_SEC_NEMO / 60:.0f} min; el extractor de "
                f"features de NeMo no esta chunked). Usa pyannote-3.1 o "
                f"pyannote-community-1 para este archivo."
            )
        import soundfile as sf
        tmp_wav = paths.DIAR_TMP_WAV
        out_json = paths.DIAR_TMP_JSON
        for f in (tmp_wav, out_json):
            if f.exists():
                f.unlink()
        sf.write(str(tmp_wav), audio, 16000)

        on_windows = platform.system() == "Windows"
        py = self._venv_python()
        script = _to_wsl_path(paths.DIAR_NEMO_SCRIPT) if on_windows \
            else str(paths.DIAR_NEMO_SCRIPT)
        wav_arg = _to_wsl_path(tmp_wav) if on_windows else str(tmp_wav)
        json_arg = _to_wsl_path(out_json) if on_windows else str(out_json)
        args = [py, script, wav_arg, json_arg,
                str(min_speakers or ""), str(max_speakers or "")]
        cmd = ["wsl.exe", "--"] + args if on_windows else args

        log(">> Diarizacion (NeMo Sortformer streaming) en "
            + ("WSL2 via wsl.exe..." if on_windows else "venv aislado..."))
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
        last_out = proc.stderr or proc.stdout
        if proc.returncode == 4:
            raise RuntimeError(
                f"NeMo Sortformer soporta como maximo 4 oradores "
                f"(min={min_speakers} max={max_speakers} pedidos):\n{last_out}"
            )
        if proc.returncode == 5:
            raise RuntimeError(
                f"Audio demasiado largo para NeMo en una sola pasada; usa "
                f"pyannote-3.1 o pyannote-community-1:\n{last_out}"
            )
        if proc.returncode != 0 or not out_json.exists():
            raise RuntimeError("Fallo la diarizacion NeMo:\n" + last_out)
        segs = json.loads(out_json.read_text(encoding="utf-8"))
        stats = _read_stats_sidecar(out_json)
        if stats:
            stats["device"] = "cuda"
            LAST_DIAR_STATS.update(stats)
        out_json.unlink(missing_ok=True)
        out_json.with_suffix(".stats.json").unlink(missing_ok=True)
        return _segments_to_df(segs)


# ----------------------------------------------------------------------------
_BACKENDS = {b.name: b for b in (Pyannote31(), PyannoteCommunity1(), Nemo())}
DEFAULT_DIARIZER = "pyannote-3.1"


def get_backend(name: str) -> DiarizerBackend:
    if name not in _BACKENDS:
        raise ValueError(
            f"Diarizador desconocido: {name}. Opciones: {', '.join(_BACKENDS)}"
        )
    return _BACKENDS[name]


def list_backends() -> list[tuple[str, str, bool, str]]:
    """[(name, label, disponible, razon), ...]"""
    out = []
    for name, b in _BACKENDS.items():
        ok, reason = b.availability()
        out.append((name, b.label, ok, reason))
    return out


if __name__ == "__main__":
    # `python diarizers.py` -> lista de backends y su disponibilidad
    print("Diarizadores disponibles:\n")
    for name, label, ok, reason in list_backends():
        mark = "[OK]   " if ok else "[NO]   "
        star = " (por defecto)" if name == DEFAULT_DIARIZER else ""
        print(f"  {mark}{name}{star}")
        print(f"         {label}")
        print(f"         {reason}\n")
