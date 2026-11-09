import numpy as np
import pytest

from stego import Secret, capacity, extract, pack
from stego.core import payload
from stego.core.exceptions import CapacityExceeded, ContainerError, PayloadError
from stego.engines.lsb.bitops import embed
from stego.media import video

av = pytest.importorskip("av")


@pytest.fixture
def sample_video(tmp_path):
    """Синтетическое lossless-видео 64×48, 10 кадров."""
    path = tmp_path / "cover.mkv"
    rng = np.random.default_rng(1)
    with av.open(str(path), mode="w") as c:
        stream = c.add_stream("ffv1", rate=10)
        stream.width, stream.height = 64, 48
        stream.pix_fmt = "bgr0"
        for _ in range(10):
            arr = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
            for pkt in stream.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
                c.mux(pkt)
        for pkt in stream.encode():
            c.mux(pkt)
    return path


def test_capacity(sample_video):
    info = capacity(sample_video, bits_per_channel=1)
    assert info.details["frames"] == 10
    assert info.total_bytes == 64 * 48 * 3 * 10 // 8


@pytest.mark.parametrize("ext", [".mkv", ".mp4"])
def test_pack_extract_roundtrip(sample_video, tmp_path, ext):
    secret = Secret.from_text("совершенно секретно", filename="msg.txt")
    output = tmp_path / f"stego{ext}"

    pack(sample_video, secret, output, bits_per_channel=1)
    result = extract(output, bits_per_channel=1)

    assert result.secret.data == secret.data
    assert result.secret.filename == "msg.txt"


@pytest.mark.parametrize("bits", [1, 2, 3, 4])
def test_streaming_matches_legacy_bit_layout(tmp_path, monkeypatch, bits):
    """Проверяет совместимость старых файлов и перенос неполных байтов между кадрами."""
    frames = np.random.default_rng(42).integers(0, 256, (8, 15, 17, 3), dtype=np.uint8)
    cover, old, new = (tmp_path / name for name in ("cover.mkv", "old.mkv", "new.mkv"))
    video.write_lossless(cover, frames, 10)
    secret = Secret(bytes(range(256)) * 2, filename="данные.bin")
    blob = payload.Payload(1, secret.data, secret.filename, secret.media_type).encode()
    expected = embed(frames.reshape(-1), blob, bits).reshape(frames.shape)
    video.write_lossless(old, expected, 10)
    monkeypatch.setattr(video, "read_frames", lambda *a: pytest.fail("video loaded into memory"))
    assert extract(old, bits_per_channel=bits).secret == secret
    pack(cover, secret, new, bits_per_channel=bits)
    assert extract(new, bits_per_channel=bits).secret == secret
    assert all(np.array_equal(a, b) for a, b in zip(video.iter_frames(new), expected, strict=True))


def test_large_secret_rejected_before_decoding(sample_video, tmp_path, monkeypatch):
    monkeypatch.setattr(
        video, "iter_frames", lambda *a, **kw: pytest.fail("decoded oversized input")
    )
    with pytest.raises(CapacityExceeded):
        pack(sample_video, b"x" * 100_000, tmp_path / "out.mkv")


def test_no_payload(sample_video):
    with pytest.raises(PayloadError):
        extract(sample_video)


@pytest.mark.parametrize("bits", [0, 5, -1, 1.5])
def test_invalid_bit_count(sample_video, bits):
    with pytest.raises(ContainerError):
        capacity(sample_video, bits_per_channel=bits)


@pytest.mark.parametrize("method", ["lsb", "lfvsn"])
def test_streaming_never_overwrites_input(sample_video, method):
    before = sample_video.read_bytes()
    with pytest.raises(ContainerError, match="исходного"):
        pack(sample_video, b"secret", sample_video, method=method)
    assert sample_video.read_bytes() == before
