# transcribe — Transcripción con diarización de oradores

Herramienta de línea de comandos para **convertir audio a texto en español
identificando quién habla**, con salida en Markdown lista para Obsidian. Usa
WhisperX (Whisper `large-v3`) sobre GPU NVIDIA y pyannote para separar las voces.

```text
**Orador 0 (start 00:00:00.451, end 00:00:03.595):** pero para el público infantil...

**Orador 1 (start 00:00:46.339, end 00:00:48.640):** No, yo manejo siempre los forms.
```

Todo se opera con **un único comando, `transcribe`** (en el PATH), con subcomandos.

---

## Tabla de contenido

- [Características](#características)
- [Requisitos y especificaciones mínimas](#requisitos-y-especificaciones-mínimas)
- [Instalación](#instalación)
- [Comandos](#comandos)
- [Tutorial básico de uso](#tutorial-básico-de-uso)
- [Salida generada](#salida-generada)
- [Información técnica](#información-técnica)
- [Diarizadores (backends)](#diarizadores-backends)
- [Benchmark de diarizadores](#benchmark-de-diarizadores)
- [Stack y librerías](#stack-y-librerías)
- [Limitaciones y notas](#limitaciones-y-notas)

---

## Características

- **Transcripción** con Whisper `large-v3` (alta precisión en español).
- **Diarización**: separa e identifica los turnos de cada orador.
- **Diarizador seleccionable** (modular y desacoplado): pyannote 3.1, pyannote
  community-1 o NVIDIA NeMo, según los requisitos de tu equipo.
- **Número de oradores flexible**: exacto (`N`), rango (`min-max`) o automático.
- **Progreso en vivo** en la terminal y consultable desde otra terminal.
- **Gestión automática de VRAM**: ajusta `batch_size`/`beam_size` para no quedarse
  sin memoria, y la diarización cae a CPU si hace falta.
- **Salida múltiple**: Markdown (`Orador n`), JSON, SRT, VTT, TXT, TSV.
- **Sin alucinaciones de repetición** (problema típico del Whisper crudo).

---

## Requisitos y especificaciones mínimas

| Recurso | Mínimo | Recomendado / probado |
| --- | --- | --- |
| GPU | NVIDIA con CUDA, 4 GB VRAM | RTX 3050 Laptop 4 GB |
| CPU | x64 | Ryzen 7 (la diarización puede usar CPU) |
| RAM | 8 GB | 16 GB |
| Disco | ~10 GB (modelos + entorno) | SSD |
| SO | Windows 10/11 x64 | Windows 11 |
| Internet | Solo la 1.ª vez (descarga de modelos) | — |

> La GPU **debe ser NVIDIA**: WhisperX usa CUDA. Las GPU AMD/integradas no sirven
> para el cómputo en Windows. Lo ideal es que el escritorio corra en la iGPU para
> dejar toda la VRAM de la NVIDIA libre.

---

## Instalación

Se instala como herramienta de línea de comandos con **pipx** (recomendado) en
**dos fases**: `pipx install` trae el CLI ligero (rápido, sin torch), y
`transcribe --start` instala el stack pesado (PyTorch+CUDA, WhisperX, pyannote)
en el mismo entorno aislado.

**Requisito previo:** Python 3.11 (3.12+ no es compatible con el stack) y
[pipx](https://pipx.pypa.io/) instalado (`python -m pip install --user pipx` y
`python -m pipx ensurepath`).

1. **Instala el CLI** (ligero):

   ```bat
   pipx install --python python3.11 transcribe-tool-wpr
   ```

   `--python python3.11` asegura que el entorno aislado use 3.11. pipx deja el
   comando `transcribe` en el PATH automáticamente.

2. **Prepara el entorno** (stack pesado + ffmpeg + token):

   ```bat
   transcribe --start
   ```

   `--start`:
   - instala el stack pesado en el venv de pipx (PyTorch + CUDA 12.1, WhisperX,
     pyannote 3.1, faster-whisper, cuDNN 8 y cuBLAS como paquetes pip);
   - descarga `ffmpeg`/`ffprobe` a la carpeta de datos del usuario;
   - **opcionalmente** te pide el token de HuggingFace (omitible; lo pones luego
     con `transcribe --setup-token`).

   > **Lo que NO instala (y no hace falta):** los **modelos** se descargan solos en
   > la primera transcripción; los **diarizadores extra** se instalan on-demand con
   > `transcribe --setup-diarizer pyannote-community-1` o `... nemo`.

**Actualizar:** `pipx upgrade transcribe-tool-wpr`. **Desinstalar:**
`pipx uninstall transcribe-tool-wpr` (la carpeta de datos con venvs/ffmpeg se borra
aparte; ver [Información técnica](#información-técnica)).

> **Alternativa sin PyPI (bleeding-edge):**
> `pipx install --python python3.11 "git+https://github.com/<OWNER>/transcribe-tool-wpr.git"`,
> luego `transcribe --start` igual. Para actualizar, reinstala apuntando al nuevo tag.

### Token de HuggingFace

Necesario para los diarizadores pyannote. Solo se hace una vez:

1. Crea una cuenta gratuita en [huggingface.co](https://huggingface.co) y acepta los términos de:
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   - *(opcional, solo si vas a usar ese diarizador)*
     [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
2. Genera un token de tipo **read** en
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
3. Guárdalo (si lo omitiste durante `transcribe --start`):

   ```bat
   transcribe --setup-token
   ```

   El token se guarda en la carpeta de datos y la herramienta lo lee sola; no hace
   falta reabrir la terminal.

---

## Comandos

```text
transcribe --start                                        # preparar entorno (1.ª vez)
transcribe <ruta_audio> [oradores] [idioma] [diarizador]  # transcribir
transcribe --status [--watch | --watch N | --json]        # ver progreso
transcribe --setup-token                                  # guardar token HF
transcribe --to-md [carpeta]                              # regenerar el .md
transcribe --list-diarizers                               # diarizadores + estado
transcribe --setup-diarizer <nombre>                      # instalar uno extra
transcribe --rediarize <json> <audio> [oradores] [diariz] # re-diarizar sin re-transcribir
transcribe --version                                      # versión instalada
transcribe --help                                         # ayuda
```

**Argumentos de transcripción:**

- `ruta_audio` — archivo de audio (wav, m4a, mp3, etc.).
- `oradores` *(opcional)* — controla la diarización:
  - número exacto `4` → fija `min=max=4`;
  - rango `4-8` → `min=4`, `max=8` (sabes cuántos había, no cuántos hablaron);
  - vacío → automático.
- `idioma` *(opcional)* — por defecto `es`.
- `diarizador` *(opcional)* — `pyannote-3.1` (por defecto), `pyannote-community-1`
  o `nemo`. Ver [Diarizadores](#diarizadores-backends).
- `--batch N` *(opcional)* — `batch_size` de **transcripción** (def 4). Más alto = más rápido.
- `--beam N` *(opcional)* — `beam_size` de **transcripción** (def 10). Más alto = más preciso.
- `--diar-batch N` *(opcional)* — batch de **diarización** (default por modelo, 64),
  independiente del de transcripción. Más alto = más rápido y más VRAM.
  Ver [Diarizadores](#diarizadores-backends). Los tres bajan solos si falta VRAM.
- `--bench-report true|false` *(opcional, por defecto `false`)* — si es `true`, agrega
  una línea a `benchmarks.jsonl` con GPU/VRAM/tiempo de esa diarización. Por defecto no
  escribe nada. Ver [Benchmark de diarizadores](#benchmark-de-diarizadores).

---

## Tutorial básico de uso

1. Abre una terminal **en la carpeta donde quieres los resultados** (la salida se
   guarda en el directorio actual).

   ```bat
   cd "C:\Transcripciones\MiJunta"
   ```

2. Lanza la transcripción (junta de 8 asistentes, ~4 hablan):

   ```bat
   transcribe "C:\Grabaciones\junta.wav" 4-8
   ```

   Verás una barra de progreso en vivo.

3. *(Opcional)* En **otra** terminal, sigue el avance sin tocar la GPU:

   ```bat
   transcribe --status --watch
   ```

   Se cierra solo al terminar.

4. Al finalizar, abre el `.md` y reemplaza `Orador 0`, `Orador 1`… por los nombres
   reales (buscar/reemplazar en Obsidian). Al inicio del `.md` tienes el índice de
   oradores detectados como referencia.

---

## Salida generada

En la carpeta actual, con el nombre del audio:

| Archivo | Contenido |
| --- | --- |
| `.md` | Markdown `**Orador n (start…, end…):** texto` (para Obsidian) |
| `.json` | Datos completos con timestamps por palabra (indentado) |
| `.srt` / `.vtt` | Subtítulos |
| `.txt` | Texto plano |
| `.tsv` | Tabular |

Encabezado del `.md` con el índice de oradores:

```text
## Oradores detectados
- Orador 0 → SPEAKER_00 (reemplaza con nombre real)
- Orador 1 → SPEAKER_01 (reemplaza con nombre real)
- Orador ? → 1 segmento(s) sin asignar por la diarizacion
```

`Orador ?` marca segmentos breves que la diarización no pudo atribuir.

Los archivos de estado (`.transcribe_status.json`, `last_status.json`) se eliminan
al terminar (pide confirmación en terminal interactiva).

---

## Información técnica

### Parámetros del modelo

| Parámetro | Valor | Razón |
| --- | --- | --- |
| model | large-v3 | Máxima precisión en español |
| compute_type | int8 | Cabe en 4 GB de VRAM |
| batch_size | 4→1 (adaptativo) | Velocidad; baja si falta VRAM |
| beam_size | 10→5 (adaptativo) | Precisión; baja como último recurso |
| chunk_size | 20 | Bloques de 20 s (estable con la VAD) |
| language | es | Fuerza español, evita autodetección errónea |

### Adaptación automática de memoria

El pipeline prueba configuraciones `(batch_size, beam_size)` de la más rápida a la
más segura y usa **la primera que quepa** en la VRAM. Baja primero el `batch_size`
(gran impacto en memoria) manteniendo `beam_size` alto (su costo es mínimo):

```text
(4,10) → (3,10) → (2,10) → (1,10) → (1,8) → (1,6) → (1,5)
```

Si hay un fallo de memoria verás el reintento en la terminal. La configuración
usada queda en el estado (`transcribe --status` muestra `Config: batch N beam M`).

Los **valores iniciales** se pueden fijar por CLI con `--batch N` y `--beam N`
(o cambiar los defaults `START_BATCH`, `START_BEAM`, `MIN_BEAM` en `run.py`). Aunque
los subas por encima de lo que cabe, el fallback los baja solo hasta encontrar el
que entra:

```bat
transcribe "audio.wav" 4-8 es pyannote-3.1 --batch 4 --beam 10
```

Además, el pipeline **libera la VRAM entre etapas** (transcripción → alineación →
diarización) y la **diarización cae a CPU** automáticamente si no entra en la GPU.

### Capacidad y rendimiento (medido en RTX 3050 4 GB)

| Métrica | Valor |
| --- | --- |
| VRAM en uso | ~2.6 GB (batch 4, beam 10) |
| Audio de 20 min | ~3–4 min de proceso |
| Audio de 3 h (176 min) | ~26 min de proceso (batch baja a 3 en este tamaño) |
| Idiomas soportados | 99 (Whisper); por defecto `es` |

### Etapas del proceso

1. **Carga de modelos** → 2. **VAD** (detección de voz) → 3. **Transcripción** (GPU)
   → 4. **Alineación** de palabras (GPU) → 5. **Diarización** (GPU/CPU) →
   6. **Escritura** de archivos.

Solo la transcripción y la alineación reportan `%` por etapa en vivo; la VAD, la
diarización y la escritura no tienen sub-progreso (se muestran como "en curso").

---

## Diarizadores (backends)

La diarización (separar **quién** habla) es **modular y desacoplada**: eliges el
backend según tu equipo. Cada uno carga sus dependencias solo si se usa, así que
uno roto o no instalado no afecta a los demás. Lístalos con:

```bat
transcribe --list-diarizers
```

**Comparación rápida** (detalle de cada uno más abajo):

| Backend | SO | VRAM probada | Máx. oradores | Máx. duración probada | Setup |
| --- | --- | --- | --- | --- | --- |
| `pyannote-3.1` *(def)* | Windows / Linux | 4 GB | sin tope | 3 h | ya instalado |
| `pyannote-community-1` | Windows / Linux | 4 GB | sin tope | 30 min | `transcribe --setup-diarizer pyannote-community-1` |
| `nemo` (Sortformer streaming) | Linux/WSL2 (desde Windows vía `wsl.exe`) | 4 GB | **4** | **30 min** | `transcribe --setup-diarizer nemo` en WSL2 |

### `pyannote-3.1` (por defecto)

- **Sistema(s) operativo(s):** Windows 10/11 x64 (probado, entorno del paquete). pyannote.audio
  es multiplataforma; en Linux nativo debería funcionar igual pero no se ha probado aquí.
- **Recursos mínimos:** GPU NVIDIA CUDA con ~2 GB libres (si no hay suficiente, cae solo a CPU).
- **Recursos recomendados:** GPU NVIDIA con 4 GB VRAM.
- **Probado en:** RTX 3050 Laptop 4 GB, Windows 11. VRAM en uso ~2.6 GB con `--diar-batch 64`.
- **Límites de diarización:** número de oradores libre (exacto, rango o automático, sin tope
  duro). Duración probada hasta **3 h (176 min)** sin problemas. Tiende a sobre-segmentar
  interjecciones cortas ("sí", "claro") como un orador extra.
- **Notas:** backend por defecto, sin instalación adicional. `--diar-batch` por defecto 64,
  baja solo 64→32→16→8→4→CPU si falta VRAM. Sin progreso `%` en vivo durante esta etapa.

### `pyannote-community-1`

- **Sistema(s) operativo(s):** Windows 10/11 x64 (probado, venv separado). El código ya
  soporta `bin/python` como ruta alternativa para Linux, pero no se ha probado ahí.
- **Recursos mínimos / recomendados:** igual que `pyannote-3.1` (GPU NVIDIA 4 GB VRAM; cae a
  CPU si falta memoria, reintentando en un proceso fresco por cada paso GPU/CPU).
- **Probado en:** RTX 3050 Laptop 4 GB, Windows 11, audio de **30 min** (rango 3-4 oradores,
  ~4:30 de proceso total). Todavía no se ha medido en audios de 1 h o más con este backend.
- **Límites de diarización:** número de oradores libre, sin tope duro. Separa mejor las
  interjecciones cortas que `pyannote-3.1` (no las fusiona tanto en el orador dominante).
- **Notas:** vive en `venv-dia-community1` (pyannote.audio 4.0 + torch cu126, incompatible
  con el pin de WhisperX `<4.0`/cu121, por eso corre como subproceso aislado). Requiere
  aceptar los términos del modelo en
  [huggingface.co/pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1).

### `nemo` (NVIDIA NeMo Sortformer streaming)

- **Sistema(s) operativo(s):** Linux o WSL2 — NeMo no tiene soporte nativo en Windows.
  Desde una terminal de **Windows**, `transcribe` lo invoca de forma transparente vía
  `wsl.exe` si WSL2 y `venv-dia-nemo` están listos; no hace falta abrir WSL a mano.
- **Recursos mínimos:** GPU NVIDIA con passthrough activo en WSL2 y ~3 GB VRAM libres.
- **Recursos recomendados:** GPU NVIDIA con 4 GB VRAM (la misma que se probó).
- **Probado en:** RTX 3050 Laptop 4 GB vía WSL2 (Ubuntu 24.04). Mediciones reales:

  | Audio | VRAM pico | Tiempo |
  | --- | --- | --- |
  | 15 min | 1.24 GB | 8 s |
  | 30 min | 1.98 GB | 17.5 s |
  | 60 min / 3 h | — | **OOM** |

- **Límites de diarización:** máximo **4 oradores** (límite del checkpoint público de
  NVIDIA, `nvidia/diar_streaming_sortformer_4spk-v2`) y máximo **~30 min de audio por
  pasada** — el propio backend lo valida y rechaza con un error claro si se excede. La
  causa del límite de duración **no es el modelo de diarización** (que sí trabaja en
  ventanas acotadas) sino su extractor de features (mel-spectrograma), que procesa el
  audio completo de una sola pasada y por eso escala su VRAM con la duración total.
- **Notas:** dentro de sus límites, notablemente más rápido que ambos pyannote. Para
  audio de más de 30 min o con más de 4 oradores, usar `pyannote-3.1` o
  `pyannote-community-1`.

### Batch de diarización (`--diar-batch`)

Aplica solo a `pyannote-3.1` y `pyannote-community-1` (independiente del `--batch` de
transcripción; `nemo` no usa este parámetro). Controla cuántas ventanas/turnos pasan a
la vez por los modelos de segmentación y embeddings — es el **principal factor
ajustable de VRAM** de la diarización (no la duración del audio, que va por ventanas).
Subirlo acelera (más VRAM), bajarlo reduce VRAM (más lento). Ante falta de memoria
**baja a la mitad** y, como último recurso, cae a CPU:

```text
GPU 64 → GPU 32 → GPU 16 → GPU 8 → GPU 4 → CPU (último recurso)
```

**Default por modelo** (medido en RTX 3050 4 GB): `pyannote-3.1` = **64**,
`community-1` = **64** (ambos validados en flujo real). Cada diarizador define su
propio default; otros modelos pueden requerir un valor distinto.

Aplica a `pyannote-3.1` (en proceso) y `community-1` (cada intento en proceso fresco,
porque una OOM de CUDA corrompe el contexto). Ejemplo:

```bat
transcribe "audio.wav" 3-4 es pyannote-community-1 --diar-batch 16
```

**Elegibilidad:** los tres son seleccionables desde el CLI, pero solo se ejecutan si tu
equipo cumple los requisitos (lo valida `--list-diarizers`). En esta máquina (Windows,
4 GB VRAM, con WSL2 configurado), los tres son elegibles.

**Uso:**

```bat
transcribe "C:\Grabaciones\junta.wav" 4-8 es pyannote-community-1
```

### Re-diarizar sin re-transcribir

El **diarizador no afecta la transcripción** (son etapas independientes: Whisper
decide *qué* se dijo; el diarizador, *quién*). Para cambiar de diarizador sobre una
transcripción ya hecha —sin volver a transcribir, mucho más rápido— usa el `.json`
generado y el audio original:

```bat
transcribe --rediarize "Junta.json" "Junta.wav" 3-4 pyannote-community-1
```

Reusa el texto exacto del `.json` y solo reasigna los oradores. Ideal para comparar
diarizadores **sobre el mismo texto** (verificado: 0 diferencias de palabras, solo
cambia la asignación de orador).

## Benchmark de diarizadores

Las tablas de VRAM/duración de la sección anterior se midieron a mano en una sola
GPU (RTX 3050 4 GB). Para ir cubriendo otras GPUs/VRAM con el tiempo, `transcribe`
puede registrar cada diarización en un archivo local `benchmarks.jsonl` (formato
[JSON Lines](https://jsonlines.org/), una línea = un JSON independiente):

```bat
transcribe "C:\Grabaciones\junta.wav" 4-8 es nemo --bench-report true
```

Por defecto (`--bench-report false`, o sin pasar la opción) **no se escribe nada**:
es enteramente opt-in. El archivo se crea/crece en la carpeta de la herramienta
(`benchmarks.jsonl`, junto a `run.py`), nunca se sube automáticamente a ningún lado
y no contiene audio ni texto transcrito — solo metadatos de rendimiento. Cada línea:

```json
{
  "timestamp": "2026-06-17T11:02:30",
  "os": "Windows-11-...",
  "gpu_name": "NVIDIA GeForce RTX 3050 Laptop GPU",
  "vram_total_gb": 4.29,
  "diarizer": "nemo",
  "num_speakers_requested": "1-4",
  "diar_batch_requested": null,
  "audio_duration_sec": 905.3,
  "elapsed_wall_sec": 19.8,
  "diar_peak_vram_gb": 1.98,
  "diar_elapsed_sec": 17.5,
  "diar_device_used": "cuda",
  "diar_batch_used": null,
  "success": true,
  "error": null
}
```

Se registra tanto si la diarización **tiene éxito** como si **falla** (`success: false`
con el mensaje de `error`) — los fallos (p. ej. OOM en cierta VRAM/duración) son tan
útiles como los éxitos para mapear los límites reales de cada backend.

Si quieres aportar tus mediciones (otra GPU, otra VRAM), comparte las líneas de tu
`benchmarks.jsonl` (por ejemplo pegándolas en un issue) — son solo metadatos, sin
audio ni transcripciones.

---

## Stack y librerías

| Componente | Versión / nota |
| --- | --- |
| Python | 3.11 (vía pyenv-win; 3.12+ no es compatible con el stack) |
| PyTorch | 2.5.1 + CUDA 12.1 |
| WhisperX | 3.4.5 |
| faster-whisper / CTranslate2 | 1.2.1 / 4.4.0 (motor de inferencia int8) |
| pyannote.audio | 3.4.0 (diarización) |
| speechbrain | 1.0.3 (la 1.1.0 rompe por un import de `k2` en Windows) |
| cuDNN / cuBLAS | cuDNN 8 (`nvidia-cudnn-cu12==8.9.7.29`) — requerido por CTranslate2 |
| ffmpeg | `ffmpeg`/`ffprobe`, los descarga `transcribe --start` (no van en el repo) |

> Las versiones exactas del stack pesado están fijadas en
> `src/transcribe_wpr/data/heavy_requirements.txt` (lo instala `transcribe --start`).
> El diarizador `nemo` usa su propio venv en WSL2 con torch CUDA +
> `nemo_toolkit[asr]` (ver [Diarizadores](#diarizadores-backends)).

### Estructura del paquete

El comando `transcribe` es el **entry point** del paquete (lo pone pipx en el PATH);
el resto son módulos internos en `src/transcribe_wpr/`.

| Archivo | Qué es |
| --- | --- |
| `cli.py` | Dispatcher del comando `transcribe` (incluye `--start`, el setup pesado). |
| `paths.py` | Separa CÓDIGO (paquete) de DATOS de runtime (carpeta de datos del usuario). |
| `run.py` | Pipeline: VRAM por etapas, batch/beam adaptativos, progreso, status, fallback CPU. |
| `status.py` | Lector de progreso (`--status`; solo stdlib, CPU pura). |
| `convert_to_md.py` | Convierte el JSON al Markdown `Orador n` (`--to-md`). |
| `hf_patch.py` | Parche `use_auth_token`→`token` (pyannote + huggingface_hub 1.x). |
| `diarizers.py` | Backends de diarización desacoplados (`pyannote-3.1`, `community-1`, `nemo`). |
| `_diar_pyannote4.py` | Subproceso de `pyannote-community-1` (corre en `venv-dia-community1`). |
| `_diar_nemo.py` | Subproceso de `nemo` (corre en `venv-dia-nemo`, vía WSL2 si el host es Windows). |
| `rediarize.py` | Re-diariza un `.json` existente sin re-transcribir (`--rediarize`). |
| `data/heavy_requirements.txt` | Stack pesado pineado (lo instala `transcribe --start`). |

### Carpeta de datos del usuario

El código vive en el venv de pipx (solo lectura); los **datos de runtime** van en la
carpeta de datos del usuario (`%LOCALAPPDATA%\transcribe-wpr\` en Windows): los venvs
de diarizadores, `ffmpeg`/`ffprobe`, el token (`hf_token.txt`), el estado global
(`last_status.json`) y `benchmarks.jsonl` (este último solo con `--bench-report true`).
Para **desinstalar del todo**: `pipx uninstall transcribe-tool-wpr` y borrar esa carpeta.

---

## Limitaciones y notas

- **Nombres propios**: WhisperX puede transcribir mal acrónimos o nombres poco
  comunes (p. ej. "FILEY"). Revísalos a mano.
- **Interjecciones cortas** ("sí", "claro") a veces se asignan a un orador extra
  (sobre-segmentación). Aparecen como oradores con muy pocos turnos.
- **Solo NVIDIA**: no se puede paralelizar con la iGPU AMD (no hay ROCm en Windows).
- **Diarización sin % en vivo**: pyannote no reporta avance fino; en esa etapa guíate
  por el tiempo transcurrido.
- Para **rehacer solo el Markdown** desde un `.json` existente:
  `transcribe --to-md "C:\carpeta\con\el\json"` (sin argumento usa la carpeta actual).
