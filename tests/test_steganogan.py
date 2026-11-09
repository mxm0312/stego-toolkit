import numpy as np
import pytest
from PIL import Image

from stego import Steganographer, available
from stego.core.exceptions import EngineUnavailable
from stego.engines.steganogan.engine import SteganoGANImageEngine
from stego.engines.steganogan.weights import WEIGHTS_DIR, resolve_weights

# --- torch-free: регистрация, поиск весов и ёмкость не должны тянуть тяжёлые зависимости ---


def test_registry_lists_steganogan():
    assert "steganogan" in available()


def test_by_name_is_steganogan_engine():
    engine = Steganographer("steganogan").engine
    assert isinstance(engine, SteganoGANImageEngine)
    assert engine.container_kind == "image"


def test_missing_weights_raises(tmp_path):
    # несуществующая архитектура гарантированно не имеет файла ни в одном из мест поиска
    with pytest.raises(EngineUnavailable):
        resolve_weights(architecture="does-not-exist", weights_path=tmp_path)


def test_capacity_reads_dimensions(tmp_path):
    cover = tmp_path / "cover.png"
    Image.fromarray(np.zeros((128, 96, 3), np.uint8)).save(cover)
    info = Steganographer("steganogan").capacity(cover, architecture="dense")
    assert info.details["width"] == 96 and info.details["height"] == 128
    assert info.details["architecture"] == "dense"
    assert info.usable_bytes >= 0


# --- дальше нужны torch + reedsolo (extra steganogan); без них тесты пропускаются ---
torch = pytest.importorskip("torch")
pytest.importorskip("reedsolo")


def test_vendored_nets_forward_shapes():
    from stego.engines.steganogan.model import build

    enc, dec = build("dense", data_depth=4, hidden_size=8)
    enc.eval()
    dec.eval()
    cover = torch.rand(1, 3, 24, 32) * 2 - 1
    payload = torch.zeros(1, 4, 24, 32).random_(0, 2)
    with torch.no_grad():
        stego = enc(cover, payload)
        logits = dec(stego)
    assert stego.shape == (1, 3, 24, 32)  # контейнер той же геометрии
    assert logits.shape == (1, 4, 24, 32)  # data_depth каналов бит


def _save_checkpoint(path, data_depth=4, hidden_size=8):
    """Собрать .steg-совместимый чекпоинт (оригинал сохраняет pickled-объект SteganoGAN)."""
    from stego.engines.steganogan.model import SteganoGAN, build

    enc, dec = build("dense", data_depth, hidden_size)
    top = SteganoGAN()
    top.encoder, top.decoder, top.critic, top.data_depth = enc, dec, None, data_depth
    top.optimizer = torch.optim.Adam(enc.parameters())  # тренировочный хлам — должен отброситься
    torch.save(top, str(path))


def test_engine_pipeline_end_to_end(tmp_path):
    """Полный тракт pack→write→read→extract на СЛУЧАЙНЫХ весах (плумбинг, не качество).

    Необученный decoder не восстановит сообщение, поэтому extract обязан аккуратно упасть
    PayloadError — это доводит декод-путь до конца: загрузка .steg с отбросом оптимизатора,
    upgrade_legacy, инференс декодера, разбор бит, голосование.
    """
    from stego import extract, pack
    from stego.core.exceptions import PayloadError
    from stego.core.types import Secret

    weights = tmp_path / "dense.steg"
    _save_checkpoint(weights)

    cover = tmp_path / "cover.png"
    # 160×160 при data_depth=4: меньше не берём — тело должно влезть RELIABLE_COPIES раз
    Image.fromarray(np.random.default_rng(0).integers(0, 256, (160, 160, 3), np.uint8)).save(cover)

    stego = tmp_path / "stego.png"
    pack(
        cover,
        Secret.from_text("hi"),
        stego,
        method="steganogan",
        device="cpu",
        weights_path=weights,
    )
    assert stego.exists()

    with pytest.raises(PayloadError):
        extract(stego, method="steganogan", device="cpu", weights_path=weights)


@pytest.mark.parametrize("architecture", ["dense", "basic"])
def test_real_weights_roundtrip(tmp_path, architecture):
    """Если веса скачаны локально — настоящий round-trip на текстурной картинке 384×384."""
    from stego import extract, pack
    from stego.core.types import Secret

    if not (WEIGHTS_DIR / f"{architecture}.steg").exists():
        pytest.skip(f"нет предобученных весов {architecture}.steg")

    n = 384
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:n, 0:n] / (n - 1)
    base = np.stack(
        [
            0.5 + 0.3 * np.sin(6 * xx) * np.cos(5 * yy),
            0.5 + 0.3 * np.sin(9 * yy + 1),
            0.5 + 0.25 * np.cos(7 * xx + 2 * yy),
        ],
        axis=-1,
    ) + rng.normal(0, 0.03, (n, n, 3))
    cover = tmp_path / "cover.png"
    Image.fromarray(np.clip(base * 255, 0, 255).astype(np.uint8)).save(cover)

    secret = Secret.from_text("Секрет! id=42", filename="note.txt")
    stego = tmp_path / "stego.png"
    pack(cover, secret, stego, method="steganogan", architecture=architecture, device="cpu")
    got = extract(stego, method="steganogan", architecture=architecture, device="cpu")
    assert got.secret.data == secret.data and got.secret.filename == "note.txt"
