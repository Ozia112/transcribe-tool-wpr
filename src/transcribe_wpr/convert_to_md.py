"""
Convierte la salida JSON de WhisperX al formato Markdown:
  **Orador N (start hh:mm:ss.ms, end hh:mm:ss.ms):** texto

Uso:
    python convert_to_md.py <carpeta_o_archivo_json>

Si se pasa una carpeta, procesa el primer .json encontrado.
"""

import json
import sys
import re
from pathlib import Path


def seconds_to_hms(seconds: float) -> str:
    """Convierte segundos a formato hh:mm:ss.ms (3 decimales)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def speaker_label(speaker_id: str, speaker_map: dict) -> str:
    """
    Devuelve etiqueta numerica del orador.
    speaker_map se llena progresivamente para mantener orden de aparicion.
    Los segmentos sin orador asignado por la diarizacion se etiquetan
    como "Orador ?" y no cuentan como un orador adicional.
    """
    if not speaker_id or speaker_id == "SPEAKER_XX":
        return "Orador ?"
    if speaker_id not in speaker_map:
        speaker_map[speaker_id] = len(speaker_map)
    return f"Orador {speaker_map[speaker_id]}"


def merge_consecutive_segments(segments: list) -> list:
    """
    Une segmentos consecutivos del mismo orador en un solo bloque
    si estan separados por menos de 1.5 segundos, para reducir fragmentacion.
    """
    if not segments:
        return segments

    merged = [dict(segments[0])]
    for seg in segments[1:]:
        prev = merged[-1]
        same_speaker = seg.get("speaker") == prev.get("speaker")
        gap = seg["start"] - prev["end"]
        if same_speaker and gap < 1.5:
            prev["end"] = seg["end"]
            prev["text"] = prev["text"].rstrip() + " " + seg["text"].lstrip()
        else:
            merged.append(dict(seg))
    return merged


def convert(json_path: Path, output_path: Path) -> None:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    if not segments:
        print(f"  Sin segmentos en {json_path.name}")
        return

    segments = merge_consecutive_segments(segments)

    speaker_map: dict[str, int] = {}
    lines = []

    unassigned = 0
    for seg in segments:
        speaker_id = seg.get("speaker", "SPEAKER_XX")
        label = speaker_label(speaker_id, speaker_map)
        if label == "Orador ?":
            unassigned += 1
        start = seconds_to_hms(seg["start"])
        end   = seconds_to_hms(seg["end"])
        text  = seg["text"].strip()

        if not text:
            continue

        lines.append(f"**{label} (start {start}, end {end}):** {text}")
        lines.append("")  # linea en blanco entre turnos

    # Encabezado con indice de oradores detectados
    header = [
        "# Transcripcion WhisperX",
        "",
        "## Oradores detectados",
    ]
    for raw_id, idx in sorted(speaker_map.items(), key=lambda x: x[1]):
        header.append(f"- Orador {idx} → _{raw_id}_ (reemplaza con nombre real)")
    if unassigned:
        header.append(f"- Orador ? → {unassigned} segmento(s) sin asignar por la diarizacion")
    header += ["", "---", ""]

    content = "\n".join(header + lines)
    output_path.write_text(content, encoding="utf-8")
    print(f"  Markdown guardado: {output_path}")


def convert_dir(target) -> None:
    """Resuelve <carpeta_o_archivo.json> y genera el .md correspondiente.
    Reutilizable in-process (lo llaman run.py/rediarize.py) y desde main()."""
    target = Path(target)
    if target.is_dir():
        # Excluye archivos de estado/ocultos (p.ej. .transcribe_status.json)
        json_files = [p for p in target.glob("*.json") if not p.name.startswith(".")]
        if not json_files:
            raise FileNotFoundError(f"No se encontraron archivos .json en {target}")
        # El mas reciente (recien generado por WhisperX)
        json_path = max(json_files, key=lambda p: p.stat().st_mtime)
    elif target.suffix == ".json" and target.exists():
        json_path = target
    else:
        raise ValueError(f"Ruta invalida o no es un .json: {target}")

    output_path = json_path.with_suffix(".md")
    print(f"Procesando: {json_path.name}")
    convert(json_path, output_path)


def main(argv=None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        print("Uso: transcribe --to-md <carpeta_o_archivo.json>")
        sys.exit(1)
    try:
        convert_dir(args[0])
    except (FileNotFoundError, ValueError) as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
