"""transcribe-tool-wpr: transcripcion en espanol con diarizacion seleccionable.

El numero de version es una sola fuente de verdad: vive en pyproject.toml y se
lee aqui desde los metadatos del paquete instalado (no se duplica a mano).
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("transcribe-tool-wpr")
except PackageNotFoundError:  # ejecutado desde el repo sin instalar
    __version__ = "0.0.0.dev0"
