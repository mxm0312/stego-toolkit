"""Потоковый LSB: прежний lossless-формат или браузерный MP4 с ключевыми кадрами."""

from contextlib import closing
from fractions import Fraction
from pathlib import Path

import numpy as np

from stego.core import payload
from stego.core.engine import StegoEngine
from stego.core.exceptions import CapacityExceeded, ContainerError, PayloadError
from stego.core.registry import register
from stego.core.types import CapacityInfo, Secret
from stego.media import video


def _validate_bits(bits: int) -> None:
    if not isinstance(bits, int) or not 1 <= bits <= 4:
        raise ContainerError("LSB: число бит на канал должно быть целым от 1 до 4")


def _embed_frames(frames, blob: bytes, bits: int, *, stop_when_done: bool, progress, total):
    offset = 0
    size = len(blob) * 8
    mask = np.uint8((1 << bits) - 1)
    weights = (1 << np.arange(bits - 1, -1, -1)).astype(np.uint8)
    for i, frame in enumerate(frames):
        frame = np.ascontiguousarray(frame)
        flat = frame.reshape(-1)
        count = min(size - offset, flat.size * bits)
        if count > 0:
            start, end = offset // 8, -(-(offset + count) // 8)
            data = np.unpackbits(
                np.frombuffer(blob, dtype=np.uint8, count=end - start, offset=start)
            )
            data = data[offset % 8 : offset % 8 + count]
            data = np.pad(data, (0, (-len(data)) % bits))
            values = (data.reshape(-1, bits) * weights).sum(axis=1, dtype=np.uint8)
            flat[: len(values)] = (flat[: len(values)] & ~mask) | values
            offset += count
        yield frame
        if progress:
            progress(i + 1, total)
        if stop_when_done and offset >= size:
            break
    if offset < size:
        raise CapacityExceeded(need=len(blob), have=offset // 8)


@register
class LSBVideoEngine(StegoEngine):
    id = 1
    name = "lsb"
    display_name = "LSB — младшие биты пикселей"

    def capacity(
        self, container: str | Path, *, bits_per_channel: int = 1, browser_mp4: bool = False, **_
    ) -> CapacityInfo:
        _validate_bits(bits_per_channel)
        info = video.probe(container, keyframes_only=browser_mp4)
        total = (info.samples * bits_per_channel) // 8
        overhead = payload.header_size()
        return CapacityInfo(
            total_bytes=total,
            overhead_bytes=overhead,
            usable_bytes=max(0, total - overhead),
            details={
                "width": info.width,
                "height": info.height,
                "frames": info.frames,
                "fps": info.fps,
                "bits_per_channel": bits_per_channel,
                "keyframes_only": browser_mp4,
            },
        )

    def pack(
        self,
        container,
        secret: Secret,
        output,
        *,
        bits_per_channel: int = 1,
        browser_mp4: bool = False,
        progress=None,
        **_,
    ) -> None:
        _validate_bits(bits_per_channel)
        if Path(container).resolve() == Path(output).resolve():
            raise ContainerError("выходной файл должен отличаться от исходного контейнера")
        blob = payload.Payload(self.id, secret.data, secret.filename, secret.media_type).encode()
        info = video.probe(container, keyframes_only=browser_mp4)
        have = info.samples * bits_per_channel // 8
        if len(blob) > have:
            raise CapacityExceeded(need=len(blob), have=have)

        fps = Fraction(info.fps).limit_denominator(1_000_000)
        if progress:
            progress(0, info.frames)
        with closing(video.iter_frames(container, keyframes_only=browser_mp4)) as source:
            frames = _embed_frames(
                source,
                blob,
                bits_per_channel,
                stop_when_done=browser_mp4,
                progress=progress,
                total=info.frames,
            )
            if browser_mp4:
                video.write_browser_mp4(container, output, frames, fps)
            else:
                video.write_lossless(output, frames, fps)

    def extract(self, container, *, bits_per_channel: int = 1, progress=None, **_) -> Secret:
        _validate_bits(bits_per_channel)
        index = video.stego_stream_index(container, self.name)
        info = video.probe(container, video_index=index)
        have = info.samples * bits_per_channel // 8
        blob = bytearray()
        tail = np.empty(0, dtype=np.uint8)
        target = None
        shifts = np.arange(bits_per_channel - 1, -1, -1, dtype=np.uint8)
        mask = np.uint8((1 << bits_per_channel) - 1)
        with closing(video.iter_frames(container, video_index=index)) as frames:
            for i, frame in enumerate(frames):
                values = frame.reshape(-1) & mask
                bits = ((values[:, None] >> shifts) & 1).astype(np.uint8).reshape(-1)
                bits = np.concatenate((tail, bits))
                whole = len(bits) // 8 * 8
                blob.extend(np.packbits(bits[:whole]).tobytes())
                tail = bits[whole:].copy()
                if progress:
                    progress(i + 1, info.frames)
                if target is None and len(blob) >= payload.HEADER_SIZE:
                    target = payload.total_size(blob[: payload.HEADER_SIZE])
                    if target > have:
                        raise PayloadError("размер скрытых данных превышает ёмкость контейнера")
                if target is not None and len(blob) >= target:
                    p = payload.Payload.decode(bytes(blob[:target]))
                    if p.method_id != self.id:
                        raise PayloadError("скрытые данные записаны другим методом")
                    return Secret(p.data, filename=p.filename, media_type=p.media_type)
        raise PayloadError("видео не содержит полного секрета")
