"""Инференс LF-VSN над кадрами: hide (спрятать) / reveal (восстановить).

Повторяет то, как сеть обучалась и тестировалась (models/LFVSN.py): работа в домене DWT, кадры
нормированы в [0,1]. Сеть обрабатывает окно из `gop` кадров, но supervised только **центральный**
кадр — то есть один кадр секрета прячется в один stego-кадр, а соседние кадры окна дают лишь
временной контекст. Поэтому здесь соответствие **1:1:1** (кадр cover ↔ stego ↔ кадр секрета):
для каждого кадра берётся gop-окно (с отражением на границах), из контейнера сохраняется его центр.
На восстановлении stego-кадр размножается ×gop и идёт через обратный ход; латент секрета
предсказывает `PredictiveModuleMIMO`, поэтому для извлечения достаточно одного stego-видео.

Импортируется лениво (из pack/extract движка), поэтому здесь можно импортировать torch на уровне
модуля — до вызова pack/extract код сюда не заходит.
"""

import numpy as np
import torch

from stego.core.exceptions import ContainerError
from stego.engines.lfvsn.model import DWT, IWT, Quantization, build_vsn
from stego.engines.lfvsn.weights import resolve_weights

GOP = 3  # размер группы кадров, с которым обучалась сеть

_dwt = DWT()
_iwt = IWT()
_quant = Quantization()


def _even_crop(frames: np.ndarray) -> np.ndarray:
    """DWT ополовинивает пространственные размеры — H и W должны быть чётными."""
    _, h, w, _ = frames.shape
    return frames[:, : h - h % 2, : w - w % 2, :]


def _group_tensor(group: np.ndarray, device) -> torch.Tensor:
    """(gop, H, W, 3) uint8 → (1, 3*gop, H, W) float в [0,1]."""
    t = torch.from_numpy(np.ascontiguousarray(group)).to(device=device, dtype=torch.float32) / 255.0
    t = t.permute(0, 3, 1, 2).contiguous()  # (gop, 3, H, W)
    return t.reshape(1, -1, t.shape[-2], t.shape[-1])


def _frames_uint8(t: torch.Tensor, gop: int) -> np.ndarray:
    """(1, 3*gop, H, W) float → (gop, H, W, 3) uint8."""
    h, w = t.shape[-2], t.shape[-1]
    t = t.clamp(0, 1).reshape(gop, 3, h, w).permute(0, 2, 3, 1)
    return (t * 255.0).round().to(torch.uint8).cpu().numpy()


def _load_net(num_video: int, device, weights_path):
    net = build_vsn(num_video=num_video).to(device)
    net.eval()
    ckpt = torch.load(str(resolve_weights(num_video, weights_path)), map_location=device)
    if isinstance(ckpt, dict):
        for key in ("params", "state_dict", "model", "netG"):
            inner = ckpt.get(key)
            if isinstance(inner, dict) and any(torch.is_tensor(v) for v in inner.values()):
                ckpt = inner
                break
    state = {k[7:] if k.startswith("module.") else k: v for k, v in ckpt.items()}
    net.load_state_dict(state, strict=True)  # строго: несовпадение весов — громкая ошибка
    return net


def _gop_windows(n_frames: int, gop: int) -> list[list[int]]:
    """Для каждого кадра — индексы gop-окна вокруг него, с отражением на границах видео."""
    if n_frames < 1:
        raise ContainerError("нет кадров для обработки")
    half = gop // 2
    return [
        [min(max(i + off, 0), n_frames - 1) for off in range(-half, half + 1)]
        for i in range(n_frames)
    ]


def _center(container: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Центральный кадр gop-контейнера (1, 3*gop, H, W) → (1, 3, H, W)."""
    return container[:, : 3 * GOP].reshape(1, GOP, 3, h, w)[:, GOP // 2]


def _report(progress, done: int, total: int) -> None:
    if progress is not None:
        progress(done, total)


def hide(
    cover: np.ndarray,
    secret: np.ndarray,
    *,
    num_video: int = 1,
    device="cpu",
    weights_path=None,
    progress=None,
) -> np.ndarray:
    """Прячет `secret` в `cover`. Обе последовательности (N, H, W, 3) uint8, одной геометрии.

    Возвращает stego-видео той же длины N — по одному кадру на каждый кадр секрета.
    `progress(done, total)` — опциональный колбэк прогресса по кадрам.
    """
    return np.stack(
        list(
            iter_hide(
                cover[: min(len(cover), len(secret))],
                secret[: min(len(cover), len(secret))],
                num_video=num_video,
                device=device,
                weights_path=weights_path,
                progress=progress,
                total=min(len(cover), len(secret)),
            )
        )
    )


def _groups(frames):
    """Скользящее окно из трёх кадров с повтором крайних; память не зависит от длины видео."""
    frames = iter(frames)
    current = next(frames, None)
    if current is None:
        raise ContainerError("нет кадров для обработки")
    previous = current
    for following in frames:
        yield np.stack((previous, current, following))
        previous, current = current, following
    yield np.stack((previous, current, current))


def iter_hide(
    cover,
    secret,
    *,
    num_video=1,
    device="cpu",
    weights_path=None,
    progress=None,
    total=0,
):
    """Потоковый инференс: модель загружается один раз, кадры выдаются по одному."""
    net = _load_net(num_video, device, weights_path)
    _report(progress, 0, total)
    with torch.no_grad():
        for i, (host_group, sec_group) in enumerate(
            zip(_groups(cover), _groups(secret), strict=False)
        ):
            host_group, sec_group = _even_crop(host_group), _even_crop(sec_group)
            if host_group.shape != sec_group.shape:
                raise ContainerError("контейнер и секрет должны совпадать по разрешению (H×W)")
            h, w = host_group.shape[1:3]
            host = _group_tensor(host_group, device)
            sec = _group_tensor(sec_group, device)
            output, _ = net(x=_dwt(host), x_h=[_dwt(sec)])
            center = _quant(_center(_iwt(output), h, w).clamp(0, 1))
            yield _frames_uint8(center, gop=1)[0]
            _report(progress, i + 1, total)


def reveal(
    stego: np.ndarray,
    *,
    num_video: int = 1,
    device="cpu",
    weights_path=None,
    progress=None,
) -> np.ndarray:
    """Восстанавливает секрет из stego-видео (M, H, W, 3) uint8.

    Возвращает (M, H, W, 3) uint8 — по одному кадру секрета на каждый stego-кадр.
    `progress(done, total)` — опциональный колбэк прогресса по кадрам.
    """
    return np.stack(
        list(
            iter_reveal(
                stego,
                num_video=num_video,
                device=device,
                weights_path=weights_path,
                progress=progress,
                total=len(stego),
            )
        )
    )


def iter_reveal(
    stego,
    *,
    num_video=1,
    device="cpu",
    weights_path=None,
    progress=None,
    total=0,
):
    """Извлечение без накопления распакованных кадров."""
    net = _load_net(num_video, device, weights_path)
    _report(progress, 0, total)
    with torch.no_grad():
        for i, frame in enumerate(stego):
            frame = _even_crop(frame[None, ...])[0]
            h, w = frame.shape[:2]
            cf = torch.from_numpy(np.ascontiguousarray(frame)).to(
                device=device, dtype=torch.float32
            )
            cf = _quant(cf.permute(2, 0, 1).unsqueeze(0) / 255.0)  # (1, 3, H, W)
            y = cf.repeat(1, GOP, 1, 1)  # stego-кадр ×gop → (1, 3*gop, H, W)
            _, out_x_h, _ = net(x=_dwt(y), rev=True)
            yield _frames_uint8(_center(_iwt(out_x_h[0]), h, w), gop=1)[0]
            _report(progress, i + 1, total)
