@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM  transcribe - wrapper unico de la herramienta WhisperX
REM
REM  TRANSCRIBIR (por defecto, ejecutar DESDE la carpeta destino):
REM    transcribe <ruta_audio> [oradores] [idioma] [diarizador]
REM    oradores  : numero exacto (4) o rango min-max (4-8) o vacio (auto)
REM    diarizador: pyannote-3.1 (def) | pyannote-community-1 | nemo
REM    transcribe "C:\audio\junta.wav" 4-8 es pyannote-community-1
REM
REM  SUBCOMANDOS:
REM    transcribe --status [--watch | --watch N | --json]
REM    transcribe --setup-token
REM    transcribe --to-md [carpeta]
REM    transcribe --list-diarizers
REM    transcribe --setup-diarizer <nombre>
REM    transcribe --help
REM ============================================================

set "SCRIPT_DIR=%~dp0"
set "PY=%SCRIPT_DIR%venv\Scripts\python.exe"
set "PY311=%USERPROFILE%\.pyenv\pyenv-win\versions\3.11.9\python.exe"

REM ffmpeg.exe + DLLs de cuDNN 8 y cuBLAS (CTranslate2)
set "PATH=%SCRIPT_DIR%;%SCRIPT_DIR%venv\Lib\site-packages\nvidia\cudnn\bin;%SCRIPT_DIR%venv\Lib\site-packages\nvidia\cublas\bin;%PATH%"

set "CMD=%~1"

if "%CMD%"==""                  goto :help
if /i "%CMD%"=="--help"         goto :help
if /i "%CMD%"=="-h"             goto :help
if /i "%CMD%"=="--status"       goto :status
if /i "%CMD%"=="--setup-token"  goto :setup
if /i "%CMD%"=="--to-md"        goto :tomd
if /i "%CMD%"=="--regenerate"   goto :tomd
if /i "%CMD%"=="--list-diarizers"  goto :listdiar
if /i "%CMD%"=="--setup-diarizer"  goto :setupdiar
if /i "%CMD%"=="--rediarize"       goto :rediarize
goto :transcribe


:transcribe
if not exist "%~1" (
    echo ERROR: No se encontro el archivo: %~1
    echo Usa  transcribe --help  para ver el uso.
    exit /b 1
)
REM Salida = directorio actual. Reenvia TODOS los args del usuario a run.py
REM (posicionales: audio [oradores] [idioma] [diarizador]; flags: --batch --beam).
"%PY%" -u "%SCRIPT_DIR%run.py" --output "%CD%" %*
exit /b %errorlevel%


:status
REM Reenvia el resto de argumentos (--watch [N] | --json) a status.py
"%PY%" "%SCRIPT_DIR%status.py" %2 %3 %4
exit /b %errorlevel%


:tomd
set "MDDIR=%~2"
if "%MDDIR%"=="" set "MDDIR=%CD%"
"%PY%" "%SCRIPT_DIR%convert_to_md.py" "%MDDIR%"
exit /b %errorlevel%


:listdiar
"%PY%" "%SCRIPT_DIR%diarizers.py"
exit /b %errorlevel%


:rediarize
REM transcribe --rediarize <json> <audio> [oradores] [diarizador]
REM Re-diariza una transcripcion existente sin re-transcribir. Salida = carpeta actual.
"%PY%" -u "%SCRIPT_DIR%rediarize.py" %2 %3 %4 %5 --output "%CD%"
exit /b %errorlevel%


:setupdiar
set "WHICH=%~2"
if /i "%WHICH%"=="pyannote-community-1" goto :setup_c1
if /i "%WHICH%"=="nemo"                  goto :setup_nemo
echo Backend desconocido: %WHICH%
echo Opciones: pyannote-community-1 ^| nemo
exit /b 1

:setup_c1
if not exist "%PY311%" (
    echo ERROR: no se encontro Python 3.11 en %PY311%
    exit /b 1
)
set "C1PIP=%SCRIPT_DIR%venv-dia-community1\Scripts\pip.exe"
echo Creando venv aislado para pyannote community-1 (pyannote.audio 4.0)...
"%PY311%" -m venv "%SCRIPT_DIR%venv-dia-community1"
echo Instalando pyannote.audio 4.0.4...
"%C1PIP%" install "pyannote.audio==4.0.4"
echo Forzando torch CUDA (cu126) para usar GPU (pyannote trae torch CPU)...
"%C1PIP%" install "torch==2.11.0+cu126" "torchaudio==2.11.0+cu126" --index-url https://download.pytorch.org/whl/cu126
echo Instalando soundfile (lectura de audio sin torchcodec)...
"%C1PIP%" install soundfile
echo.
echo Listo. Recuerda aceptar los terminos del modelo en:
echo   https://huggingface.co/pyannote/speaker-diarization-community-1
echo Uso: transcribe "audio.wav" 4-8 es pyannote-community-1
exit /b %errorlevel%

:setup_nemo
echo.
echo NVIDIA NeMo (Sortformer streaming) NO tiene soporte en Windows nativo.
echo Requiere Linux o WSL2. Funciona en 4 GB VRAM, pero solo para audio de
echo hasta ~30 min de una pasada (limite del extractor de features de NeMo).
echo.
echo Setup en WSL2/Ubuntu (ejecutar dentro de wsl, en esta misma carpeta
echo montada como /mnt/c/...):
echo   1. python3 -m venv --without-pip venv-dia-nemo
echo      (si "python3 -m venv" normal falla por ensurepip/python3.X-venv
echo      y no quieres pedir sudo, usa --without-pip y luego:)
echo      curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
echo      venv-dia-nemo/bin/python3 /tmp/get-pip.py
echo   2. venv-dia-nemo/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124
echo   3. venv-dia-nemo/bin/pip install "nemo_toolkit[asr]"
echo   4. _diar_nemo.py ya esta incluido en esta carpeta.
exit /b 0


:setup
echo.
echo Para obtener tu token:
echo   1. Ve a https://huggingface.co/settings/tokens
echo   2. Crea un token (tipo "read" es suficiente)
echo   3. Acepta los terminos de estos modelos:
echo      - https://huggingface.co/pyannote/speaker-diarization-3.1
echo      - https://huggingface.co/pyannote/segmentation-3.0
echo      - https://huggingface.co/pyannote/speaker-diarization-community-1
echo.
set /p "TOKEN=Pega tu token HuggingFace aqui: "
if "%TOKEN%"=="" (
    echo ERROR: No ingresaste ningun token.
    exit /b 1
)
setx HF_TOKEN "%TOKEN%"
echo.
echo Token guardado en la variable HF_TOKEN.
echo Cierra y reabre la terminal para que surta efecto.
exit /b 0


:help
echo.
echo  transcribe - transcripcion con diarizacion (WhisperX)
echo.
echo  TRANSCRIBIR (ejecutar DESDE la carpeta destino):
echo    transcribe ^<ruta_audio^> [oradores] [idioma] [diarizador] [--batch N] [--beam N]
echo    oradores  : exacto (4), rango min-max (4-8) o vacio (auto)
echo    diarizador: pyannote-3.1 (def) ^| pyannote-community-1 ^| nemo
echo    --batch N      : batch de TRANSCRIPCION (def 4; baja solo si falta VRAM)
echo    --beam N       : beam de transcripcion (def 10; baja solo si falta VRAM)
echo    --diar-batch N : batch de DIARIZACION (def 64 por modelo; baja solo y luego CPU)
echo    --bench-report true^|false : agrega linea a benchmarks.jsonl con GPU/VRAM/tiempo
echo                       de la diarizacion (def false, no escribe nada)
echo    Ejemplo: transcribe "C:\audio\junta.wav" 4-8 es pyannote-community-1 --batch 4
echo.
echo  SUBCOMANDOS:
echo    transcribe --status [--watch ^| --watch N ^| --json]   progreso
echo    transcribe --setup-token              guarda el token de HuggingFace
echo    transcribe --to-md [carpeta]          regenera el .md desde el .json
echo    transcribe --list-diarizers           diarizadores y su disponibilidad
echo    transcribe --setup-diarizer ^<nombre^>  instala un diarizador extra
echo    transcribe --rediarize ^<json^> ^<audio^> [oradores] [diarizador]
echo                                          re-diariza sin re-transcribir
echo    transcribe --help                     esta ayuda
echo.
exit /b 0
