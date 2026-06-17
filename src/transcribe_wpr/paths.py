"""Rutas de la herramienta, separando CODIGO de DATOS de runtime.

- CODE_DIR : donde vive el paquete instalado (site-packages). Solo lectura.
             Aqui estan los scripts _diar_*.py que se ejecutan bajo OTROS venvs
             y data/heavy_requirements.txt.
- DATA_DIR : carpeta de datos del usuario (escribible). Aqui van los venvs de
             diarizadores, ffmpeg, el estado global, benchmarks.jsonl, etc.
             En site-packages no se debe escribir (puede ser de solo lectura y
             no debe albergar venvs de varios GB).
"""

from pathlib import Path

import platformdirs

CODE_DIR = Path(__file__).resolve().parent
DATA_FILES_DIR = CODE_DIR / "data"
HEAVY_REQUIREMENTS = DATA_FILES_DIR / "heavy_requirements.txt"

# Scripts que se invocan como subproceso bajo venvs distintos (no se importan)
DIAR_PYANNOTE4_SCRIPT = CODE_DIR / "_diar_pyannote4.py"
DIAR_NEMO_SCRIPT = CODE_DIR / "_diar_nemo.py"
CONVERT_TO_MD_SCRIPT = CODE_DIR / "convert_to_md.py"

DATA_DIR = Path(platformdirs.user_data_dir("transcribe-wpr", appauthor=False))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Entornos de diarizadores que corren aislados
VENV_COMMUNITY1 = DATA_DIR / "venv-dia-community1"
VENV_NEMO = DATA_DIR / "venv-dia-nemo"

# ffmpeg/ffprobe (los baja `transcribe --start` en Windows)
BIN_DIR = DATA_DIR / "bin"

# Estado global y benchmark
GLOBAL_STATUS = DATA_DIR / "last_status.json"
BENCH_FILE = DATA_DIR / "benchmarks.jsonl"

# WAV/JSON temporales que intercambian los diarizadores de subproceso
DIAR_TMP_WAV = DATA_DIR / "_diar_in.wav"
DIAR_TMP_JSON = DATA_DIR / "_diar_out.json"

# Marcadores de setup (los escribe `transcribe --start`)
READY_MARKER = DATA_DIR / ".setup_done"
TOKEN_FILE = DATA_DIR / "hf_token.txt"
