"""
Parche de compatibilidad: pyannote.audio 3.4.0 llama a hf_hub_download() con
el argumento `use_auth_token`, que huggingface_hub >=1.0 ya elimino (ahora es
`token`). transformers (transcripcion/alineacion) exige huggingface_hub >=1.5,
asi que no se puede bajar la version. Este parche traduce use_auth_token->token
en tiempo de ejecucion, sin tocar versiones ni romper el resto del stack.

Llamar a apply() una vez antes de instanciar la diarizacion de pyannote.
"""

import functools
import sys


def apply() -> None:
    import huggingface_hub

    orig = huggingface_hub.hf_hub_download
    if getattr(orig, "_ua_patched", False):
        return  # ya parcheado

    @functools.wraps(orig)
    def wrapper(*args, **kwargs):
        if "use_auth_token" in kwargs:
            tok = kwargs.pop("use_auth_token")
            kwargs.setdefault("token", tok)
        return orig(*args, **kwargs)

    wrapper._ua_patched = True

    # 1) Atributo canonico del paquete
    huggingface_hub.hf_hub_download = wrapper

    # 2) Referencias ya enlazadas via "from huggingface_hub import hf_hub_download"
    #    en modulos ya importados (pyannote.audio.core.pipeline, model, etc.)
    for mod in list(sys.modules.values()):
        try:
            if getattr(mod, "hf_hub_download", None) is orig:
                setattr(mod, "hf_hub_download", wrapper)
        except Exception:  # noqa: BLE001 - algunos modulos no permiten getattr
            pass
