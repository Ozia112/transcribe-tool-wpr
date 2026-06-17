"""
Re-diariza una transcripcion ya existente SIN volver a transcribir.

Toma un .json de WhisperX (con segments + words) y el audio original, corre solo
el diarizador elegido y reasigna los oradores. Util para comparar diarizadores
sobre EXACTAMENTE el mismo texto, y mucho mas rapido (no re-transcribe ni alinea).

Uso (lo llama transcribe --rediarize):
    python rediarize.py <json> <audio> [oradores] [diarizador] --output <dir>
"""

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def main() -> int:
    import diarizers
    from run import parse_speakers  # reutiliza el parser de "4" / "4-8"

    p = argparse.ArgumentParser(prog="rediarize.py")
    p.add_argument("json")
    p.add_argument("audio")
    p.add_argument("speakers", nargs="?", default="")
    p.add_argument("diarizer", nargs="?", default="")
    p.add_argument("--output", required=True)
    p.add_argument("--diar-batch", dest="diar_batch", type=int, default=None,
                   help="batch de diarizacion (def 32, baja solo)")
    a = p.parse_args()

    json_path = Path(a.json)
    if not json_path.exists():
        print(f"ERROR: no existe el JSON: {json_path}")
        return 1
    if not Path(a.audio).exists():
        print(f"ERROR: no existe el audio: {a.audio}")
        return 1

    diarizer = a.diarizer or diarizers.DEFAULT_DIARIZER
    backend = diarizers.get_backend(diarizer)
    ok, reason = backend.availability()
    if not ok:
        print(f"ERROR: diarizador '{diarizer}' no disponible: {reason}")
        return 1

    min_spk, max_spk = parse_speakers(a.speakers)
    speakers_label = (
        f"{min_spk}-{max_spk}" if min_spk != max_spk
        else (str(min_spk) if min_spk is not None else "auto")
    )

    print("=" * 60)
    print(f"  RE-DIARIZAR (sin re-transcribir)")
    print(f"  JSON      : {json_path.name}")
    print(f"  Audio     : {Path(a.audio).name}")
    print(f"  Oradores  : {speakers_label}")
    print(f"  Diarizador: {diarizer}")
    print("=" * 60, flush=True)

    import whisperx
    from whisperx.diarize import assign_word_speakers
    from whisperx.utils import get_writer

    result = json.loads(json_path.read_text(encoding="utf-8"))
    lang = result.get("language", "es")
    audio = whisperx.load_audio(a.audio)

    diarize_segments = backend.diarize(
        audio, min_spk, max_spk, os.environ.get("HF_TOKEN", ""), "cuda",
        diar_batch=a.diar_batch, log=lambda m: print(m, flush=True),
    )
    result = assign_word_speakers(diarize_segments, result)
    result["language"] = lang

    out_dir = a.output
    os.makedirs(out_dir, exist_ok=True)
    # El writer nombra los archivos segun el basename que reciba: usamos el del JSON
    writer = get_writer("all", out_dir)
    writer(result, json_path.name, {
        "highlight_words": False, "max_line_count": None, "max_line_width": None,
    })

    # JSON legible (indentado)
    for jf in Path(out_dir).glob("*.json"):
        if jf.name.startswith("."):
            continue
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            jf.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                          encoding="utf-8")
        except (OSError, json.JSONDecodeError):
            pass

    import subprocess
    subprocess.run([sys.executable, str(SCRIPT_DIR / "convert_to_md.py"), out_dir],
                   check=False)
    print(f"\nLISTO. Re-diarizado en: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
