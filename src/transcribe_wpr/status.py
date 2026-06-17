"""
Lector de estado de la transcripcion en curso (o la ultima).

CPU pura: solo usa la libreria estandar. No importa torch ni toca la GPU,
asi que se puede consultar mientras WhisperX usa la GPU sin interferir.

Uso:
    python status.py                 # ultima transcripcion (global)
    python status.py <carpeta>       # estado de una carpeta de salida concreta
    python status.py --watch         # refresca cada 2 s hasta completar
    python status.py --watch 5       # refresca cada 5 s
    python status.py --json          # imprime el status.json crudo
"""

import json
import sys
import time
from pathlib import Path

from transcribe_wpr import paths

GLOBAL_STATUS = paths.GLOBAL_STATUS

# Etapas que emiten sub-progreso en vivo ("Progress: X%"). En el resto
# (carga, VAD, diarizacion, escritura) no hay % por etapa: pyannote y la
# escritura no reportan avance, asi que se muestra "en curso" en vez de 0 %.
LIVE_SUBPROGRESS = {"transcripcion", "alineacion"}


def find_status(arg_path: str | None) -> Path | None:
    if arg_path:
        p = Path(arg_path)
        if p.is_dir():
            cand = p / ".transcribe_status.json"
            return cand if cand.exists() else None
        return p if p.exists() else None
    return GLOBAL_STATUS if GLOBAL_STATUS.exists() else None


def render(data: dict) -> str:
    pct = data.get("overall_percent", 0.0)
    n = int(pct / 5)
    bar = "#" * n + "-" * (20 - n)

    stage = data.get("stage", "")
    if data.get("done"):
        etapa = "100 %"
    elif stage in LIVE_SUBPROGRESS:
        etapa = f"{data.get('stage_percent', 0.0):.1f}%"
    else:
        etapa = "en curso (sin % en vivo)"

    lines = [
        "-" * 56,
        f"  Estado     : {data.get('stage_label', stage or '?')}",
        f"  Global     : [{bar}] {pct:.1f}%",
        f"  Etapa      : {etapa}",
        f"  Transcurrido: {data.get('elapsed', '?')}",
        f"  Oradores   : {data.get('num_speakers', '?')}   Idioma: {data.get('language', '?')}",
        f"  Diarizador : {data.get('diarizer', '-')}",
        f"  Config     : batch {data.get('batch_size') or '-'}   beam {data.get('beam_size') or '-'}",
        f"  Audio      : {Path(data.get('audio', '?')).name}",
        f"  Salida     : {data.get('output_dir', '?')}",
        f"  Actualizado: {data.get('updated_at', '?')}",
    ]
    if data.get("done"):
        lines.append("  >> COMPLETADO")
    if data.get("error"):
        lines.append(f"  >> ERROR: {data['error']}")
    lines.append("-" * 56)
    return "\n".join(lines)


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else list(argv)
    watch = False
    interval = 2.0
    raw_json = False
    path_arg = None

    i = 0
    while i < len(args):
        a = args[i]
        if a == "--watch":
            watch = True
            if i + 1 < len(args) and args[i + 1].replace(".", "").isdigit():
                interval = float(args[i + 1])
                i += 1
        elif a == "--json":
            raw_json = True
        else:
            path_arg = a
        i += 1

    def load():
        sp = find_status(path_arg)
        if not sp:
            return None
        try:
            return json.loads(sp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    if not watch:
        data = load()
        if not data:
            print("No hay estado disponible todavia (¿ya inicio la transcripcion?).")
            return 1
        print(json.dumps(data, ensure_ascii=False, indent=2) if raw_json else render(data))
        return 0

    # --- modo --watch ---
    # Sale solo cuando: ve done/error, o el estado desaparece tras haber existido
    # (run.py borra el status.json al terminar). Ctrl+C para salir antes.
    seen = False
    try:
        while True:
            data = load()
            sys.stdout.write("\033[2J\033[H")  # limpiar pantalla
            if data:
                seen = True
                print(render(data))
                if data.get("done") or data.get("error"):
                    print("\nProceso finalizado.")
                    return 0
            elif seen:
                # El estado existio y ahora no: la transcripcion termino y se limpio.
                print("Proceso finalizado (estado limpiado). Cerrando watch.")
                return 0
            else:
                print("Esperando a que inicie la transcripcion...  (Ctrl+C para salir)")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nWatch interrumpido.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
