import io
import shutil
from contextlib import closing

import pytest

from stego import Steganographer, available
from stego.core.exceptions import EngineUnavailable
from stego.engines.lfvsn.engine import LFVSNVideoEngine
from stego.engines.lfvsn.weights import resolve_weights

# --- torch-free: регистрация и разрешение путей не должны тянуть тяжёлые зависимости ---


def test_registry_lists_lfvsn():
    assert "lfvsn" in available()


def test_by_name_is_lfvsn_engine():
    assert isinstance(Steganographer("lfvsn").engine, LFVSNVideoEngine)


def test_missing_weights_raises(tmp_path):
    # num_video=99 гарантированно не имеет файла ни в одном из мест поиска
    with pytest.raises(EngineUnavailable):
        resolve_weights(num_video=99, weights_path=tmp_path)


# --- дальше нужен torch (extra lfvsn); без него тесты пропускаются ---
torch = pytest.importorskip("torch")


def test_resolve_device_cpu_and_auto():
    from stego.engines.lfvsn.device import resolve_device

    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in ("cpu", "cuda")


def test_resolve_device_unknown_raises():
    from stego.engines.lfvsn.device import resolve_device

    with pytest.raises(EngineUnavailable):
        resolve_device("tpu")


def test_inn_invertible():
    """Ядро метода — обратимая сеть: forward ∘ reverse на одних и тех же (y, y_h) = identity."""
    from stego.engines.lfvsn.model.inv_arch import InvNN
    from stego.engines.lfvsn.model.subnet import subnet

    net = InvNN(
        channel_in_ho=4,
        channel_in_hi=4,
        subnet_constructor=subnet("DBNet", "xavier"),
        subnet_constructor_v2=subnet("DBNet", "xavier"),
        block_num=[2],
        down_num=1,
        groups=1,
    ).eval()

    x = torch.randn(1, 4, 8, 8)
    x_h = [torch.randn(1, 4, 8, 8)]
    with torch.no_grad():
        y, y_h = net(x, x_h, rev=False)
        x_rec, x_h_rec = net(y, y_h, rev=True)

    assert torch.allclose(x_rec, x, atol=1e-3)
    assert torch.allclose(x_h_rec[0], x_h[0], atol=1e-3)


def test_build_vsn_forward_reverse_shapes():
    """Полный конфиг VSN: DWT-геометрия и каналы сходятся, прямой/обратный ход дают верные формы."""
    from stego.engines.lfvsn.model import DWT, IWT, build_vsn

    dwt, iwt = DWT(), IWT()
    net = build_vsn(num_video=1).eval()
    gop = 3
    host = torch.rand(1, 3 * gop, 16, 16)
    secret = torch.rand(1, 3 * gop, 16, 16)

    with torch.no_grad():
        out, _ = net(x=dwt(host), x_h=[dwt(secret)])
        out = iwt(out)
        assert out.shape[1] == 3 * gop  # контейнер той же размерности, что и хост-группа

        y = torch.rand(1, 3 * gop, 16, 16)
        _, out_x_h, _ = net(x=dwt(y), rev=True)
        assert iwt(out_x_h[0]).shape == host.shape


def _write_video(path, frames, fps=10):
    import av

    _, h, w, _ = frames.shape
    with av.open(str(path), mode="w") as c:
        stream = c.add_stream("ffv1", rate=fps)
        stream.width, stream.height = w, h
        stream.pix_fmt = "bgr0"
        for f in frames:
            for pkt in stream.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                c.mux(pkt)
        for pkt in stream.encode():
            c.mux(pkt)


@pytest.mark.parametrize("secret_length", [2, 6, 8])
def test_engine_pipeline_end_to_end(tmp_path, monkeypatch, secret_length):
    """Полный тракт движка pack→write→read→extract на CPU.

    Веса — случайно инициализированные (реальный конфиг), сохранённые во временный файл:
    проверяем корректность всего трубопровода (декод кадров, группировка, DWT, запись/чтение,
    ресайз секрета, tempfile), а НЕ качество восстановления.
    """
    import numpy as np

    pytest.importorskip("av")

    from stego import extract, pack
    from stego.engines.lfvsn.model import build_vsn
    from stego.media import video

    monkeypatch.setattr(video, "read_frames", lambda *a: pytest.fail("video loaded into memory"))

    weights = tmp_path / "lfvsn_1video.pth"
    torch.save(build_vsn(num_video=1).state_dict(), weights)

    rng = np.random.default_rng(0)
    cover = rng.integers(0, 256, (6, 16, 16, 3), dtype=np.uint8)
    secret = rng.integers(0, 256, (secret_length, 16, 16, 3), dtype=np.uint8)
    cover_path, secret_path = tmp_path / "cover.mkv", tmp_path / "secret.mkv"
    _write_video(cover_path, cover)
    _write_video(secret_path, secret)

    stego = tmp_path / "stego.mkv"
    pack(cover_path, secret_path, stego, method="lfvsn", device="cpu", weights_path=weights)
    assert stego.exists()

    result = extract(stego, method="lfvsn", device="cpu", weights_path=weights)
    assert result.secret.media_type.startswith("video/")
    assert len(result.secret.data) > 0
    recovered = tmp_path / "recovered.mp4"
    recovered.write_bytes(result.secret.data)
    assert video.probe(recovered).frames == 6


@pytest.fixture(scope="module")
def image_test_weights(tmp_path_factory):
    from stego.engines.lfvsn.model import build_vsn

    path = tmp_path_factory.mktemp("lfvsn_image_weights") / "1video.pth"
    torch.save(build_vsn(num_video=1).state_dict(), path)
    return path


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="нужен ffmpeg")
@pytest.mark.parametrize("suffix", [".mp4", ".mkv"])
@pytest.mark.parametrize("cover_length", [2, 20])
def test_photo_embedded_once_preserves_full_video(
    tmp_path, monkeypatch, image_test_weights, suffix, cover_length
):
    """Длина видео не должна превращать одну фотографию в сотни скрытых кадров."""
    import av
    import numpy as np
    from PIL import Image

    from stego import extract, pack
    from stego.engines.lfvsn import runner
    from stego.media import video

    cover = tmp_path / "cover.mkv"
    frames = np.random.default_rng(123).integers(0, 256, (cover_length, 16, 16, 3), np.uint8)
    _write_video(cover, frames)
    photo = tmp_path / "photo.jpg"
    Image.fromarray(frames[0]).save(photo)
    output = tmp_path / f"stego{suffix}"

    processed = []
    original_hide = runner.iter_hide

    def counted_hide(*args, **kwargs):
        with closing(original_hide(*args, **kwargs)) as generated:
            for frame in generated:
                processed.append(1)
                yield frame

    monkeypatch.setattr(runner, "iter_hide", counted_hide)
    monkeypatch.setattr(video, "read_frames", lambda *a: pytest.fail("video loaded into memory"))
    progress = []
    pack(
        cover,
        photo,
        output,
        method="lfvsn",
        device="cpu",
        weights_path=image_test_weights,
        progress=lambda done, total: progress.append((done, total)),
    )
    assert len(processed) == 1
    assert progress == [(0, 1), (1, 1)]
    index = video.stego_stream_index(output, "lfvsn-image")
    assert index == 1
    assert video.probe(output).frames == cover_length
    assert video.probe(output, video_index=index).frames == 1
    with av.open(str(output)) as c:
        assert not int(c.streams.video[index].disposition) & 1
        if suffix == ".mp4":
            assert c.streams.video[0].codec_context.format.name == "yuv420p"

    result = extract(output, method="lfvsn", device="cpu", weights_path=image_test_weights)
    assert result.secret.filename == "recovered_secret.png"
    assert result.secret.media_type == "image/png"
    with Image.open(io.BytesIO(result.secret.data)) as recovered:
        assert recovered.size == (16, 16)
