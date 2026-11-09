import os
import subprocess
import sys

import pytest

from stego import Steganographer, available
from stego.core.exceptions import MethodNotFound
from stego.engines.lsb.engine import LSBVideoEngine


def test_registry_lists_lsb():
    assert "lsb" in available()


def test_by_name_and_by_instance_same_engine():
    assert isinstance(Steganographer("lsb").engine, LSBVideoEngine)
    assert isinstance(Steganographer(LSBVideoEngine()).engine, LSBVideoEngine)


def test_unknown_engine_raises():
    with pytest.raises(MethodNotFound):
        Steganographer("nope")


def test_light_image_only_registers_lsb():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import stego, sys; assert stego.available() == ['lsb']; "
            "assert 'torch' not in sys.modules",
        ],
        env={**os.environ, "STEGO_ENGINES": "lsb"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
