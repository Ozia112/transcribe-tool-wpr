@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM  install - instalador maestro de la herramienta transcribe
REM
REM  Prepara SOLO el camino por defecto (transcripcion + diarizador
REM  pyannote-3.1):
REM    1. Verifica Python 3.11
REM    2. Crea el venv principal (venv\) si no existe
REM    3. Instala requirements.txt (torch+cu121, whisperx, pyannote 3.1,
REM       faster-whisper, cuDNN 8 y cuBLAS como paquetes pip, etc.)
REM    4. Verifica/descarga ffmpeg.exe y ffprobe.exe
REM    5. (Opcional) guarda el token de HuggingFace -- se puede omitir y
REM       ponerlo despues con  transcribe --setup-token
REM
REM  NO instala (se hace solo en su momento, no aqui):
REM    - Los modelos (se descargan en el primer  transcribe).
REM    - Los diarizadores extra (pyannote-community-1, nemo): se instalan
REM      on-demand con  transcribe --setup-diarizer <nombre>.
REM ============================================================

set "SCRIPT_DIR=%~dp0"
set "VENV=%SCRIPT_DIR%venv"
set "PY=%VENV%\Scripts\python.exe"

echo.
echo ============================================================
echo   Instalador de transcribe (WhisperX + diarizacion)
echo ============================================================
echo.

REM ---------- 1. Localizar Python 3.11 ----------
set "PY311="
set "PY311_PYENV=%USERPROFILE%\.pyenv\pyenv-win\versions\3.11.9\python.exe"
if exist "%PY311_PYENV%" (
    set "PY311=%PY311_PYENV%"
    goto :py_found
)
REM Probar el lanzador  py -3.11
py -3.11 --version >nul 2>&1
if not errorlevel 1 (
    set "PY311=py -3.11"
    goto :py_found
)
REM Ultimo intento: que "python" sea 3.11
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set "PYVER=%%v"
echo %PYVER% | findstr /b "3.11" >nul 2>&1
if not errorlevel 1 (
    set "PY311=python"
    goto :py_found
)

echo ERROR: no se encontro Python 3.11.
echo.
echo La herramienta requiere Python 3.11 (3.12+ no es compatible con el stack).
echo Opciones:
echo   - Instala pyenv-win y luego:  pyenv install 3.11.9
echo   - O instala Python 3.11 desde https://www.python.org/downloads/
echo Reintenta este instalador despues.
exit /b 1

:py_found
echo [1/5] Python 3.11 encontrado: %PY311%

REM ---------- 2. Crear el venv principal ----------
if exist "%PY%" (
    echo [2/5] venv ya existe; se reutiliza.
) else (
    echo [2/5] Creando venv principal en "%VENV%"...
    %PY311% -m venv "%VENV%"
    if errorlevel 1 (
        echo ERROR: no se pudo crear el venv.
        exit /b 1
    )
)

REM ---------- 3. Instalar dependencias ----------
echo [3/5] Actualizando pip e instalando requirements.txt...
echo       (descarga grande: torch + CUDA, puede tardar varios minutos)
"%PY%" -m pip install --upgrade pip >nul
if exist "%SCRIPT_DIR%constraints.txt" (
    "%PY%" -m pip install -r "%SCRIPT_DIR%requirements.txt" -c "%SCRIPT_DIR%constraints.txt"
) else (
    "%PY%" -m pip install -r "%SCRIPT_DIR%requirements.txt"
)
if errorlevel 1 (
    echo ERROR: fallo la instalacion de dependencias.
    exit /b 1
)

REM ---------- 4. ffmpeg / ffprobe ----------
if exist "%SCRIPT_DIR%ffmpeg.exe" if exist "%SCRIPT_DIR%ffprobe.exe" (
    echo [4/5] ffmpeg.exe y ffprobe.exe ya presentes.
    goto :ffmpeg_done
)
echo [4/5] Descargando ffmpeg (build esencial para Windows)...
set "FF_ZIP=%TEMP%\ffmpeg-essentials.zip"
set "FF_TMP=%TEMP%\ffmpeg-extract"
powershell -NoProfile -Command ^
  "$ErrorActionPreference='Stop';" ^
  "Invoke-WebRequest -Uri 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip' -OutFile '%FF_ZIP%';" ^
  "if (Test-Path '%FF_TMP%') { Remove-Item -Recurse -Force '%FF_TMP%' }" ^
  "Expand-Archive -Path '%FF_ZIP%' -DestinationPath '%FF_TMP%' -Force;" ^
  "$bin = Get-ChildItem -Path '%FF_TMP%' -Recurse -Filter 'ffmpeg.exe' | Select-Object -First 1;" ^
  "Copy-Item $bin.FullName '%SCRIPT_DIR%ffmpeg.exe' -Force;" ^
  "Copy-Item (Join-Path $bin.DirectoryName 'ffprobe.exe') '%SCRIPT_DIR%ffprobe.exe' -Force;"
if errorlevel 1 (
    echo ADVERTENCIA: no se pudo descargar ffmpeg automaticamente.
    echo Descargalo manualmente desde https://www.gyan.dev/ffmpeg/builds/
    echo y coloca ffmpeg.exe y ffprobe.exe en esta carpeta:
    echo   %SCRIPT_DIR%
) else (
    echo       ffmpeg.exe y ffprobe.exe instalados.
)
:ffmpeg_done

REM ---------- 5. Token de HuggingFace (opcional) ----------
echo [5/5] Token de HuggingFace (necesario para los diarizadores pyannote).
if defined HF_TOKEN (
    echo       HF_TOKEN ya esta definido en el entorno; se omite.
    goto :token_done
)
echo.
echo       Puedes omitir este paso (Enter vacio) y configurarlo despues con:
echo         transcribe --setup-token
echo       Para obtener uno: https://huggingface.co/settings/tokens (tipo "read")
echo       y acepta los terminos de pyannote/speaker-diarization-3.1 y segmentation-3.0.
echo.
set /p "TOKEN=      Pega tu token HuggingFace (o Enter para omitir): "
if "%TOKEN%"=="" (
    echo       Omitido. Recuerda:  transcribe --setup-token  antes de diarizar.
) else (
    setx HF_TOKEN "%TOKEN%" >nul
    echo       Token guardado en HF_TOKEN. Cierra y reabre la terminal.
)
:token_done

echo.
echo ============================================================
echo   Instalacion completada.
echo.
echo   Siguiente paso: agrega esta carpeta al PATH para usar "transcribe"
echo   desde cualquier lugar, o invoca transcribe.bat con su ruta completa.
echo.
echo   Prueba:  transcribe --list-diarizers
echo   Uso:     transcribe "C:\ruta\audio.wav" 4-8
echo.
echo   Diarizadores extra (opcionales, se instalan aparte):
echo     transcribe --setup-diarizer pyannote-community-1
echo     transcribe --setup-diarizer nemo      (requiere WSL2)
echo ============================================================
exit /b 0
