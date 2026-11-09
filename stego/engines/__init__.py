"""Импорт модуля движка регистрирует его в реестре. Новый движок — добавить строку."""

import importlib
import os

_ENGINES = {
    "lsb": "LSBVideoEngine",
    "lfvsn": "LFVSNVideoEngine",
    "steganogan": "SteganoGANImageEngine",
}
_enabled = os.environ.get("STEGO_ENGINES", ",".join(_ENGINES)).split(",")
__all__ = []
for _name in _enabled:
    _name = _name.strip()
    if _name not in _ENGINES:
        raise ValueError(f"неизвестный движок в STEGO_ENGINES: {_name!r}")
    _class_name = _ENGINES[_name]
    globals()[_class_name] = getattr(
        importlib.import_module(f"stego.engines.{_name}.engine"), _class_name
    )
    __all__.append(_class_name)
