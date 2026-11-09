"""Движок LF-VSN: нейросетевое сокрытие видео/изображения внутри видео (CVPR 2023).

В отличие от LSB это НЕ bit-exact: секрет восстанавливается обратимой сетью приближённо
(высокий PSNR, но не байт-в-байт), поэтому формат payload STG1/crc32 здесь не применяется —
движок работает напрямую с кадрами. Секрет — видеоролик или изображение, а не произвольные байты.

Модуль намеренно НЕ импортирует torch на уровне файла (и в __init__): регистрация в реестре
не должна тянуть тяжёлую зависимость. torch и вендоренная сеть подгружаются лениво внутри
pack/extract — так же, как `import av` внутри функций media/video.py.
"""

import io
import tempfile
from contextlib import closing
from fractions import Fraction
from itertools import islice
from pathlib import Path

import numpy as np

from stego.core.engine import StegoEngine
from stego.core.exceptions import ContainerError
from stego.core.registry import register
from stego.core.types import CapacityInfo, Secret
from stego.media import video

_VIDEO_SUFFIXES = {".mp4", ".m4v", ".mov", ".mkv", ".avi", ".webm"}


@register
class LFVSNVideoEngine(StegoEngine):
    id = 2
    name = "lfvsn"
    display_name = "LF-VSN"

    def capacity(self, container: str | Path, *, num_video: int = 1, **_) -> CapacityInfo:
        info = video.probe(container)
        usable = info.width * info.height * 3 * info.frames * num_video
        return CapacityInfo(
            total_bytes=usable,
            overhead_bytes=0,
            usable_bytes=usable,
            details={
                "width": info.width,
                "height": info.height,
                "frames": info.frames,
                "fps": info.fps,
                "num_video": num_video,
                "note": "нейросеть: восстановление приблизительное (не bit-exact)",
            },
        )

    def pack(
        self,
        container,
        secret: Secret,
        output,
        *,
        num_video: int = 1,
        device: str = "auto",
        weights_path=None,
        progress=None,
        **_,
    ) -> None:
        if Path(container).resolve() == Path(output).resolve():
            raise ContainerError("выходной файл должен отличаться от исходного контейнера")
        from stego.engines.lfvsn.device import resolve_device

        dev = resolve_device(device)  # проверит torch/устройство до тяжёлого импорта runner
        from stego.engines.lfvsn import runner

        info = video.probe(container)
        fps = Fraction(info.fps).limit_denominator(1_000_000)
        with (
            closing(video.iter_frames(container)) as cover,
            closing(self._iter_secret_frames(secret, (info.width, info.height))) as source,
        ):
            stego = runner.iter_hide(
                cover,
                islice(source, info.frames),
                num_video=num_video,
                device=dev,
                weights_path=weights_path,
                progress=progress,
                total=info.frames,
            )
            video.write_lossless(output, stego, fps)

    def extract(
        self,
        container,
        *,
        num_video: int = 1,
        device: str = "auto",
        weights_path=None,
        progress=None,
        **_,
    ) -> Secret:
        from stego.engines.lfvsn.device import resolve_device

        dev = resolve_device(device)  # проверит torch/устройство до тяжёлого импорта runner
        from stego.engines.lfvsn import runner

        info = video.probe(container)
        fps = Fraction(info.fps).limit_denominator(1_000_000)
        with (
            closing(video.iter_frames(container)) as stego,
            tempfile.TemporaryDirectory(prefix="lfvsn_") as tmp,
        ):
            frames = runner.iter_reveal(
                stego,
                num_video=num_video,
                device=dev,
                weights_path=weights_path,
                progress=progress,
                total=info.frames,
            )
            out = Path(tmp) / "recovered_secret.mp4"
            video.write_viewable(out, frames, fps)
            data = out.read_bytes()
        return Secret(data, filename="recovered_secret.mp4", media_type="video/mp4")

    @staticmethod
    def _iter_secret_frames(secret: Secret, size_wh):
        """Читает секрет по кадру; короткий ролик повторно открывается с диска, а не копируется."""
        from PIL import Image

        def resize(frame):
            if (frame.shape[1], frame.shape[0]) == size_wh:
                return frame
            return np.array(Image.fromarray(frame).resize(size_wh, Image.Resampling.BILINEAR))

        suffix = Path(secret.filename).suffix.lower()
        if secret.media_type.startswith("video/") or suffix in _VIDEO_SUFFIXES:
            with tempfile.TemporaryDirectory(prefix="lfvsn_sec_") as tmp:
                path = Path(tmp) / (Path(secret.filename).name or "secret.mkv")
                path.write_bytes(secret.data)
                while True:
                    count = 0
                    with closing(video.iter_frames(path)) as frames:
                        for frame in frames:
                            yield resize(frame)
                            count += 1
                    if not count:
                        raise ContainerError("в секретном видео нет кадров")
        else:
            try:
                with Image.open(io.BytesIO(secret.data)) as image:
                    frame = resize(np.array(image.convert("RGB")))
            except Exception as e:
                raise ContainerError(
                    "LF-VSN прячет видео или изображение; секрет не читается"
                ) from e
            while True:
                yield frame
