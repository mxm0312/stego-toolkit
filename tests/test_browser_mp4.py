import shutil
import subprocess

import numpy as np
import pytest

from stego import Secret, capacity, extract, pack
from stego.core.exceptions import CapacityExceeded, ContainerError
from stego.media import video

av = pytest.importorskip("av")
pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="нужен ffmpeg")


@pytest.fixture
def compressed_cover(tmp_path):
    path = tmp_path / "cover.mp4"
    with av.open(str(path), "w") as c:
        stream = c.add_stream("libx264", rate=10)
        stream.width, stream.height = 128, 96
        stream.pix_fmt = "yuv420p"
        stream.options = {"g": "10", "sc_threshold": "0"}
        yy, xx = np.mgrid[:96, :128]
        for i in range(40):
            frame = np.stack(((xx + i) % 256, yy * 2, (xx + yy) % 256), axis=-1).astype(np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                c.mux(packet)
        for packet in stream.encode():
            c.mux(packet)
    return path


def _packets(path):
    with av.open(str(path)) as c:
        return [bytes(p) for p in c.demux(c.streams.video[0]) if p.size]


@pytest.mark.parametrize("bits", [1, 2, 3, 4])
def test_browser_mp4_video_in_video_roundtrip(compressed_cover, tmp_path, bits):
    """Секрет — настоящий MP4. Основное видео копируется, скрытые кадры читаются автоматически."""
    secret = Secret.from_file(compressed_cover)
    output = tmp_path / "stego.mp4"
    info = capacity(compressed_cover, browser_mp4=True, bits_per_channel=bits)
    assert info.details["frames"] == 4
    assert info.total_bytes == 128 * 96 * 3 * 4 * bits // 8
    pack(compressed_cover, secret, output, browser_mp4=True, bits_per_channel=bits)
    assert extract(output, bits_per_channel=bits).secret == secret
    assert _packets(output) == _packets(compressed_cover)
    with av.open(str(output)) as c:
        main, hidden = c.streams.video
        assert main.codec_context.format.name == "yuv420p"
        assert hidden.metadata["handler_name"] == "stego-lsb-v1"
        assert not int(hidden.disposition) & 1
        assert 0 < hidden.frames <= info.details["frames"]


def test_only_needed_keyframes_written(compressed_cover, tmp_path):
    output = tmp_path / "small.mp4"
    pack(compressed_cover, "короткий секрет", output, browser_mp4=True)
    with av.open(str(output)) as c:
        assert c.streams.video[1].frames == 1
    assert extract(output).secret.data.decode() == "короткий секрет"


def test_capacity_uses_keyframes(compressed_cover, tmp_path):
    selected = capacity(compressed_cover, browser_mp4=True)
    assert selected.total_bytes < capacity(compressed_cover).total_bytes
    with pytest.raises(CapacityExceeded):
        pack(
            compressed_cover, b"x" * selected.total_bytes, tmp_path / "large.mp4", browser_mp4=True
        )


def test_transcodes_incompatible_cover(tmp_path):
    cover, output = tmp_path / "cover.mkv", tmp_path / "stego.mp4"
    frames = np.random.default_rng(5).integers(0, 256, (5, 15, 17, 3), dtype=np.uint8)
    video.write_lossless(cover, frames, 10)
    pack(cover, b"hidden", output, browser_mp4=True)
    assert extract(output).secret.data == b"hidden"
    with av.open(str(output)) as c:
        assert c.streams.video[0].codec_context.format.name == "yuv420p"


def test_audio_preserved(compressed_cover, tmp_path):
    cover, output = tmp_path / "audio.mp4", tmp_path / "stego.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(compressed_cover),
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=4",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(cover),
        ],
        check=True,
        capture_output=True,
    )
    pack(cover, "со звуком", output, browser_mp4=True)
    with av.open(str(output)) as c:
        assert len(c.streams.audio) == 1
        assert len(c.streams.video) == 2
        assert c.duration / av.time_base >= 3.9
    assert extract(output).secret.data.decode() == "со звуком"


def test_browser_mode_requires_mp4(compressed_cover, tmp_path):
    with pytest.raises(ContainerError, match=".mp4"):
        pack(compressed_cover, b"x", tmp_path / "wrong.mkv", browser_mp4=True)
