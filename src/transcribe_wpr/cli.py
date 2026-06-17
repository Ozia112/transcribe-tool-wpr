"""
Dispatcher del comando `transcribe` (entry point del paquete).

Reemplaza al antiguo transcribe.bat e install.bat:
  - Los subcomandos (--status, --to-md, --list-diarizers, --setup-diarizer,
    --rediarize) llaman in-process a los modulos del paquete.
  - `transcribe --start` hace el setup pesado que antes hacia install.bat:
    instala el stack pesado en ESTE venv (opcion A: mismo entorno de pipx),
    baja ffmpeg y, opcionalmente, guarda el token de HuggingFace.

Uso (resumen):
  transcribe <audio> [oradores] [idioma] [diarizador] [--batch N] [--beam N]
             [--diar-batch N] [--bench-report true|false]
  transcribe --start
  transcribe --status [--watch | --watch N | --json]
  transcribe --setup-token
  transcribe --to-md [carpeta]
  transcribe --list-diarizers
  transcribe --setup-diarizer <pyannote-community-1|nemo>
  transcribe --rediarize <json> <audio> [oradores] [diarizador]
  transcribe --version
  transcribe --help
"""

import os
import platform
import subprocess
import sys
from pathlib import Path

from transcribe_wpr import __version__, paths

PYTORCH_CU121_INDEX = "https://download.pytorch.org/whl/cu121"


# --------------------------------------------------------------------------
# Token: el HF_TOKEN puede venir del entorno o del archivo de la carpeta de
# datos (lo escribe --setup-token). Lo cargamos al entorno una sola vez.
def _load_token_into_env() -> None:
    if os.environ.get("HF_TOKEN"):
        return
    if paths.TOKEN_FILE.exists():
        tok = paths.TOKEN_FILE.read_text(encoding="utf-8").strip()
        if tok:
            os.environ["HF_TOKEN"] = tok


# --------------------------------------------------------------------------
# Runtime: antes de transcribir hay que poner en el PATH ffmpeg y las DLLs de
# cuDNN/cuBLAS que vienen como paquetes pip dentro de ESTE venv (lo que antes
# hacia transcribe.bat con set PATH=...).
def _setup_runtime_path() -> None:
    parts = []
    if paths.BIN_DIR.is_dir():
        parts.append(str(paths.BIN_DIR))
    import importlib.util
    for sub in ("cudnn", "cublas", "cuda_nvrtc"):
        try:
            spec = importlib.util.find_spec(f"nvidia.{sub}")
        except (ImportError, ValueError):
            spec = None
        if spec and spec.submodule_search_locations:
            binp = Path(list(spec.submodule_search_locations)[0]) / "bin"
            if binp.is_dir():
                parts.append(str(binp))
    if parts:
        os.environ["PATH"] = os.pathsep.join(parts + [os.environ.get("PATH", "")])


def _setup_done() -> bool:
    return paths.READY_MARKER.exists()


def _require_setup() -> None:
    if not _setup_done():
        print("La herramienta aun no esta preparada. Ejecuta primero:\n"
              "  transcribe --start\n"
              "(instala el stack pesado, ffmpeg y, opcionalmente, el token).",
              file=sys.stderr)
        sys.exit(2)


# --------------------------------------------------------------------------
def _download_ffmpeg() -> bool:
    """Descarga ffmpeg.exe/ffprobe.exe a la carpeta de datos (Windows).
    En Linux/WSL2 se asume ffmpeg del sistema (apt)."""
    if platform.system() != "Windows":
        from shutil import which
        if which("ffmpeg"):
            print("   ffmpeg del sistema encontrado.")
            return True
        print("   ffmpeg no encontrado. Instalalo con: sudo apt install ffmpeg")
        return False

    if (paths.BIN_DIR / "ffmpeg.exe").exists() and (paths.BIN_DIR / "ffprobe.exe").exists():
        print("   ffmpeg ya presente.")
        return True

    import io
    import urllib.request
    import zipfile

    url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    print(f"   descargando ffmpeg desde {url} ...")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            data = resp.read()
        paths.BIN_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for name in zf.namelist():
                base = os.path.basename(name)
                if base in ("ffmpeg.exe", "ffprobe.exe"):
                    with zf.open(name) as src, open(paths.BIN_DIR / base, "wb") as dst:
                        dst.write(src.read())
        ok = (paths.BIN_DIR / "ffmpeg.exe").exists()
        print("   ffmpeg instalado." if ok else "   ADVERTENCIA: no se hallaron los .exe en el zip.")
        return ok
    except Exception as e:  # noqa: BLE001
        print(f"   ADVERTENCIA: no se pudo descargar ffmpeg: {e}")
        print("   Descargalo manualmente de https://www.gyan.dev/ffmpeg/builds/ y "
              f"copia ffmpeg.exe/ffprobe.exe a:\n     {paths.BIN_DIR}")
        return False


def _setup_token_interactive(optional: bool) -> None:
    print()
    if optional:
        print("Token de HuggingFace (necesario para los diarizadores pyannote).")
        print("Puedes OMITIRLO ahora (Enter vacio) y ponerlo despues con:")
        print("  transcribe --setup-token")
    print("Obtenlo en https://huggingface.co/settings/tokens (tipo 'read') y acepta")
    print("los terminos de pyannote/speaker-diarization-3.1 y segmentation-3.0.")
    try:
        tok = input("Pega tu token HuggingFace" + (" (o Enter para omitir): "
                    if optional else ": ")).strip()
    except (EOFError, KeyboardInterrupt):
        tok = ""
    if not tok:
        print("Token omitido." if optional else "No ingresaste token.")
        return
    paths.TOKEN_FILE.write_text(tok, encoding="utf-8")
    os.environ["HF_TOKEN"] = tok
    if platform.system() == "Windows":
        subprocess.run(["setx", "HF_TOKEN", tok], capture_output=True)
    print(f"Token guardado en {paths.TOKEN_FILE}")


# --------------------------------------------------------------------------
def cmd_start() -> int:
    print("=" * 60)
    print("  transcribe --start : preparando el entorno")
    print("=" * 60)

    if not (sys.version_info[:2] == (3, 11)):
        print(f"AVISO: estas en Python {sys.version_info.major}.{sys.version_info.minor}; "
              "el stack esta probado en 3.11. Si falla, reinstala con "
              "'pipx install --python python3.11 transcribe-tool-wpr'.")

    print("\n[1/3] Instalando el stack pesado (torch+CUDA, WhisperX, pyannote)...")
    print("      (descarga grande, varios minutos)")
    rc = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(paths.HEAVY_REQUIREMENTS)],
    ).returncode
    if rc != 0:
        print("ERROR: fallo la instalacion del stack pesado.", file=sys.stderr)
        return 1

    print("\n[2/3] ffmpeg...")
    _download_ffmpeg()

    print("\n[3/3] Token de HuggingFace...")
    if os.environ.get("HF_TOKEN") or paths.TOKEN_FILE.exists():
        print("   token ya configurado; se omite.")
    else:
        _setup_token_interactive(optional=True)

    paths.READY_MARKER.write_text(__version__, encoding="utf-8")
    print("\n" + "=" * 60)
    print("  Listo. Prueba:  transcribe --list-diarizers")
    print("  Uso:            transcribe \"audio.wav\" 4-8")
    print("  Diarizadores extra (opcionales):")
    print("    transcribe --setup-diarizer pyannote-community-1")
    print("    transcribe --setup-diarizer nemo   (requiere WSL2)")
    print("=" * 60)
    return 0


def cmd_setup_diarizer(name: str) -> int:
    if name == "pyannote-community-1":
        return _setup_community1()
    if name == "nemo":
        return _setup_nemo_instructions()
    print(f"Backend desconocido: {name}\nOpciones: pyannote-community-1 | nemo")
    return 1


def _setup_community1() -> int:
    venv = paths.VENV_COMMUNITY1
    print(f"Creando venv aislado para community-1 en:\n  {venv}")
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=False)

    # El ejecutable de Python del venv recien creado (Windows vs POSIX).
    py = venv / "Scripts" / "python.exe"
    if not py.exists():
        py = venv / "bin" / "python"
    if not py.exists():
        print(f"ERROR: no se encontro el Python del venv en {py}\n"
              "       ¿Fallo la creacion del venv? Revisa los mensajes anteriores.")
        return 1
    pip = [str(py), "-m", "pip"]

    print("Actualizando pip...")
    subprocess.run(pip + ["install", "--upgrade", "pip"], check=False)
    print("Instalando pyannote.audio 4.0.4...")
    subprocess.run(pip + ["install", "pyannote.audio==4.0.4"], check=False)
    print("Forzando torch CUDA (cu126)...")
    subprocess.run(pip + ["install", "torch==2.11.0+cu126", "torchaudio==2.11.0+cu126",
                          "--index-url", "https://download.pytorch.org/whl/cu126"], check=False)
    print("Instalando soundfile...")
    subprocess.run(pip + ["install", "soundfile"], check=False)
    print("\nVenv listo. Verificando acceso al modelo en HuggingFace...")
    _check_community1_access()
    return 0


def _check_community1_access() -> int:
    """Verifica, via API de HuggingFace, si el token ya tiene acceso al modelo
    'gated' community-1. Devuelve 0 si todo esta listo, 1 si falta algo.

    Nota: aceptar los terminos de un modelo gated SOLO se puede hacer desde el
    navegador (formulario 'Agree'); HuggingFace no expone una API publica para
    aceptarlos por terminal. Lo mejor que se puede automatizar es comprobar si
    ya estan aceptados y, si no, dar el enlace exacto.
    """
    repo_id = "pyannote/speaker-diarization-community-1"
    url = f"https://huggingface.co/{repo_id}"
    token = os.environ.get("HF_TOKEN")

    if not token:
        print("\n[!] No hay token de HuggingFace configurado.")
        print("    1) Crea uno (tipo 'read') en https://huggingface.co/settings/tokens")
        print("    2) Guardalo con:  transcribe --setup-token")
        print(f"    3) Acepta los terminos del modelo en:\n       {url}")
        return 1

    try:
        from huggingface_hub import auth_check
        auth_check(repo_id, token=token)
    except ImportError:
        print(f"\n[!] No se pudo importar huggingface_hub para verificar el acceso.")
        print(f"    Si aun no lo hiciste, acepta los terminos en:\n       {url}")
        return 1
    except Exception as e:  # noqa: BLE001
        name = type(e).__name__
        if name == "GatedRepoError":
            print("\n[!] Tu token es valido, pero FALTA aceptar los terminos del modelo.")
            print("    Debes aceptarlos desde el navegador (no hay API para hacerlo por")
            print("    terminal). Abre el enlace, inicia sesion y pulsa 'Agree':")
            print(f"       {url}")
        elif name == "RepositoryNotFoundError":
            print("\n[!] No se pudo acceder al repositorio. ¿El token es valido y de la")
            print(f"    cuenta correcta? Modelo:\n       {url}")
        else:
            print(f"\n[!] No se pudo verificar el acceso automaticamente ({name}).")
            print(f"    Si aun no lo hiciste, acepta los terminos en:\n       {url}")
        return 1

    print("\n[OK] Tu token ya tiene acceso al modelo: todo esta listo.")
    print("Uso: transcribe \"audio.wav\" 4-8 es pyannote-community-1")
    return 0


def _setup_nemo_instructions() -> int:
    print()
    print("NVIDIA NeMo (Sortformer streaming) NO tiene soporte en Windows nativo.")
    print("Requiere Linux o WSL2. Funciona en 4 GB VRAM, pero solo audio de hasta")
    print("~30 min por pasada. El venv debe crearse DENTRO de WSL2 en:")
    print(f"  {paths.VENV_NEMO}")
    print("\nPasos en WSL2/Ubuntu:")
    print("  1. python3 -m venv --without-pip <ruta-de-arriba>")
    print("     curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py")
    print("     <ruta>/bin/python3 /tmp/get-pip.py")
    print("  2. <ruta>/bin/pip install torch --index-url "
          "https://download.pytorch.org/whl/cu124")
    print("  3. <ruta>/bin/pip install \"nemo_toolkit[asr]\"")
    print("  4. El script _diar_nemo.py ya viene en el paquete; el CLI lo invoca solo.")
    return 0


# --------------------------------------------------------------------------
HELP = __doc__


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    _load_token_into_env()

    cmd = args[0] if args else ""
    rest = args[1:]

    if cmd in ("", "-h", "--help"):
        print(HELP)
        return 0
    if cmd in ("--version", "-V"):
        print(f"transcribe-tool-wpr {__version__}")
        return 0
    if cmd == "--start":
        return cmd_start()
    if cmd == "--status":
        from transcribe_wpr import status
        return status.main(rest)
    if cmd == "--setup-token":
        _setup_token_interactive(optional=False)
        return 0
    if cmd in ("--to-md", "--regenerate"):
        from transcribe_wpr import convert_to_md
        target = rest[0] if rest else os.getcwd()
        convert_to_md.main([target])
        return 0
    if cmd == "--list-diarizers":
        from transcribe_wpr import diarizers
        print("Diarizadores disponibles:\n")
        for name, label, ok, reason in diarizers.list_backends():
            mark = "[OK]   " if ok else "[NO]   "
            star = " (por defecto)" if name == diarizers.DEFAULT_DIARIZER else ""
            print(f"  {mark}{name}{star}")
            print(f"         {label}")
            print(f"         {reason}\n")
        return 0
    if cmd == "--setup-diarizer":
        if not rest:
            print("Falta el nombre: pyannote-community-1 | nemo")
            return 1
        return cmd_setup_diarizer(rest[0])
    if cmd == "--rediarize":
        _require_setup()
        _setup_runtime_path()
        from transcribe_wpr import rediarize
        return rediarize.main(rest + ["--output", os.getcwd()])

    # ---- default: transcribir ----
    if cmd.startswith("-"):
        print(f"Opcion desconocida: {cmd}\nUsa 'transcribe --help'.", file=sys.stderr)
        return 1
    audio = cmd
    if not Path(audio).exists():
        print(f"ERROR: no se encontro el archivo: {audio}\n"
              "Usa 'transcribe --help' para ver el uso.", file=sys.stderr)
        return 1
    _require_setup()
    _setup_runtime_path()
    from transcribe_wpr import run
    return run.main(args + ["--output", os.getcwd()])


if __name__ == "__main__":
    sys.exit(main())
