"""Видео-IO поверх PyAV. Общий слой для движков, работающих с кадрами.

Кадры — ndarray (N, H, W, 3) uint8 в RGB. Запись только lossless — для LSB-подобных
методов сжатие с потерями уничтожает младшие биты. Кодек выбирается по расширению:
  .mp4/.mov → libx264rgb (H.264 4:4:4 RGB, qp=0); .mkv → FFV1.
Оба сохраняют RGB побитово точно.
"""

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from itertools import chain
from pathlib import Path

import numpy as np

from stego.core.exceptions import ContainerError

CHANNELS = 3  # rgb24

# suffix -> (codec, pix_fmt, options); все варианты lossless и bit-exact по RGB
_ENCODERS = {
    ".mp4": ("libx264rgb", "rgb24", {"qp": "0"}),
    ".m4v": ("libx264rgb", "rgb24", {"qp": "0"}),
    ".mov": ("libx264rgb", "rgb24", {"qp": "0"}),
    ".mkv": ("ffv1", "bgr0", {}),  # упакованный RGB(+пад); rgb24<->bgr0 без потерь
}


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    frames: int
    fps: float

    @property
    def samples(self) -> int:
        return self.width * self.height * CHANNELS * self.frames


def probe(
    container: str | Path, *, video_index: int = 0, keyframes_only: bool = False
) -> VideoInfo:
    import av

    with av.open(str(container)) as c:
        if len(c.streams.video) <= video_index:
            raise ContainerError("в контейнере нет нужной видеодорожки")
        v = c.streams.video[video_index]
        w, h = v.codec_context.width, v.codec_context.height
        fps = v.average_rate or Fraction(25, 1)
        frames = v.frames if not keyframes_only else 0
        if not frames:  # matroska и др. не хранят счётчик — считаем пакеты (1 пакет ≈ 1 кадр)
            frames = sum(
                1 for pkt in c.demux(v) if pkt.size and (not keyframes_only or pkt.is_keyframe)
            )
    if not frames:
        raise ContainerError("не удалось определить число кадров")
    return VideoInfo(width=w, height=h, frames=frames, fps=float(fps))


def read_frames(container: str | Path) -> tuple[np.ndarray, Fraction]:
    import av

    frames = []
    with av.open(str(container)) as c:
        v = c.streams.video[0]
        fps = v.average_rate or Fraction(25, 1)
        for frame in c.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise ContainerError("нет видеокадров")
    return np.stack(frames), fps


def iter_frames(container: str | Path, *, video_index: int = 0, keyframes_only: bool = False):
    """Декодирует по одному RGB-кадру, не удерживая видео в памяти."""
    import av

    with av.open(str(container)) as c:
        if len(c.streams.video) <= video_index:
            raise ContainerError("в контейнере нет видеодорожки")
        v = c.streams.video[video_index]
        if keyframes_only:
            v.codec_context.skip_frame = "NONKEY"
        for frame in c.decode(v):
            yield frame.to_ndarray(format="rgb24")


def stego_stream_index(container: str | Path, method: str) -> int:
    """Номер скрытой видеодорожки; старые однодорожечные контейнеры читаем как раньше."""
    import av

    with av.open(str(container)) as c:
        for i, stream in enumerate(c.streams.video):
            metadata = {key.lower(): value for key, value in stream.metadata.items()}
            if metadata.get("handler_name") == f"stego-{method}-v1":
                return i
    return 0


def write_browser_mp4(container: str | Path, output: str | Path, frames, fps: Fraction) -> None:
    """Браузерный MP4 для LSB с точными RGB-кадрами в дополнительной дорожке."""
    if Path(output).suffix.lower() != ".mp4":
        raise ContainerError("браузерный режим LSB требует выходной файл .mp4")
    write_stego_container(container, output, frames, fps, method="lsb")


def write_stego_container(
    container: str | Path, output: str | Path, frames, fps: Fraction, *, method: str
) -> None:
    """Сжатое основное видео и отдельная дорожка с точными кадрами движка.

    MP4/MOV получают обычную H.264/yuv420p-дорожку для просмотра. Для MKV исходная
    видеодорожка копируется. Скрытые RGB-кадры сохраняются без потерь.
    """
    import av

    output = Path(output)
    suffix = output.suffix.lower()
    if suffix not in _ENCODERS:
        raise ContainerError(f"неподдерживаемый выходной формат: {suffix}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise ContainerError("для записи контейнера со скрытой дорожкой требуется ffmpeg")
    with av.open(str(container)) as c:
        codec = c.streams.video[0].codec_context
        compatible = suffix == ".mkv" or (
            codec.name == "h264" and codec.format.name in ("yuv420p", "yuvj420p")
        )
    with tempfile.TemporaryDirectory(prefix=".stego_", dir=output.parent) as tmp:
        hidden = Path(tmp) / "hidden.mp4"
        result = Path(tmp) / f"result{suffix}"
        write_lossless(hidden, frames, fps)
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(container),
            "-i",
            str(hidden),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-map",
            "1:v:0",
            "-c:v:0",
            "copy" if compatible else "libx264",
            "-c:v:1",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-disposition:v:0",
            "default",
            "-disposition:v:1",
            "0",
            "-metadata:s:v:1",
            f"handler_name=stego-{method}-v1",
        ]
        if suffix != ".mkv":
            cmd += ["-movflags", "+faststart"]
        if not compatible:
            cmd += [
                "-pix_fmt:v:0",
                "yuv420p",
                "-crf:v:0",
                "18",
                "-filter:v:0",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            ]
        try:
            subprocess.run([*cmd, str(result)], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            message = e.stderr.decode("utf-8", errors="replace")[-1500:]
            raise ContainerError(f"не удалось собрать видеоконтейнер: {message}") from e
        result.replace(output)


def write_viewable(output: str | Path, frames, fps: Fraction, crf: str = "18") -> None:
    """Обычный H.264 (yuv420p) — открывается в QuickTime/Preview/браузере.

    Для человеко-смотрибельного результата (восстановленный секрет LF-VSN). Не для носителя и не
    для LSB: yuv420p субдискретизирует цвет и убивает младшие биты, из такого не извлечь секрет.
    """
    import av

    frames = iter(frames)
    first = next(frames, None)
    if first is None:
        raise ContainerError("нет видеокадров")
    h, w, _ = first.shape
    with av.open(str(output), mode="w") as c:
        stream = c.add_stream("libx264", rate=fps)
        stream.width, stream.height = w, h
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": crf}
        for f in chain((first,), frames):
            for pkt in stream.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                c.mux(pkt)
        for pkt in stream.encode():
            c.mux(pkt)


def write_lossless(output: str | Path, frames, fps: Fraction) -> None:
    """Пишет массив или итератор кадров, удерживая в памяти только текущий кадр."""
    import av

    suffix = Path(output).suffix.lower()
    if suffix not in _ENCODERS:
        raise ContainerError(
            f"lossless-запись не поддерживает '{suffix}', доступны: {sorted(_ENCODERS)}"
        )
    codec, pix_fmt, options = _ENCODERS[suffix]

    frames = iter(frames)
    first = next(frames, None)
    if first is None:
        raise ContainerError("нет видеокадров")
    h, w, _ = first.shape
    with av.open(str(output), mode="w") as c:
        stream = c.add_stream(codec, rate=fps)
        stream.width, stream.height = w, h
        stream.pix_fmt = pix_fmt
        stream.options = options
        for f in chain((first,), frames):
            for pkt in stream.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                c.mux(pkt)
        for pkt in stream.encode():
            c.mux(pkt)
