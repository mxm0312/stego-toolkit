import argparse
import sys

from stego import capacity, extract, pack
from stego.core.exceptions import StegoError


def _fmt_bytes(n: int) -> str:
    x = float(n)
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if x < 1024 or unit == "ГБ":
            return f"{x:.0f} {unit}" if unit == "Б" else f"{x:.1f} {unit}"
        x /= 1024


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stego", description="Стеганография в видео и картинках")
    parser.add_argument("--method", default="lsb")
    parser.add_argument("--bits-per-channel", type=int, default=1, help="LSB: бит на канал 1..4")
    parser.add_argument(
        "--browser-mp4",
        action="store_true",
        help="LSB: браузерный MP4; секрет в отдельной дорожке ключевых кадров без потерь",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="нейросети (LF-VSN / SteganoGAN): устройство (auto — GPU при наличии, иначе CPU)",
    )
    parser.add_argument("--num-video", type=int, default=1, help="LF-VSN: число секретных роликов")
    parser.add_argument(
        "--architecture",
        choices=["dense", "basic", "residual"],
        default="dense",
        help="SteganoGAN: архитектура сети (веса {architecture}.steg)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capacity", help="показать ёмкость контейнера")
    c.add_argument("container")

    p = sub.add_parser("pack", help="упаковать секрет в видео")
    p.add_argument("container")
    p.add_argument("secret", help="путь к файлу-секрету")
    p.add_argument("-o", "--output", required=True)

    e = sub.add_parser("extract", help="извлечь секрет из видео")
    e.add_argument("container")
    e.add_argument("-o", "--output-dir", required=True)

    args = parser.parse_args(argv)
    # каждый движок берёт из params свои параметры, лишнее игнорирует через **_
    params = {
        "bits_per_channel": args.bits_per_channel,
        "browser_mp4": args.browser_mp4,
        "device": args.device,
        "num_video": args.num_video,
        "architecture": args.architecture,
    }

    try:
        if args.cmd == "capacity":
            info = capacity(args.container, method=args.method, **params)
            print(
                f"Ёмкость: {_fmt_bytes(info.usable_bytes)} "
                f"(сырьё {_fmt_bytes(info.total_bytes)}, заголовок {info.overhead_bytes} Б)"
            )
            print(f"Параметры: {info.details}")
        elif args.cmd == "pack":
            result = pack(args.container, args.secret, args.output, method=args.method, **params)
            print(f"Готово: {args.output}")
            print(
                f"  {_fmt_bytes(result.secret_bytes)} → {_fmt_bytes(result.stego_size_bytes)} "
                f"· заполнено {result.packing_ratio:.1%} · {result.elapsed_s:.1f}с"
                + (f" · {result.device}" if result.device else "")
            )
        elif args.cmd == "extract":
            result = extract(args.container, args.output_dir, method=args.method, **params)
            secret = result.secret
            dev = f" · {result.device}" if result.device else ""
            size_s = _fmt_bytes(secret.size)
            print(f"Извлечено: {secret.filename} ({size_s}) · {result.elapsed_s:.1f}с{dev}")
    except StegoError as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
