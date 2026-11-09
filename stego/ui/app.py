import tempfile
from pathlib import Path

import streamlit as st

from stego import Secret, capacity, extract, pack
from stego.core import registry
from stego.core.exceptions import CapacityExceeded, PayloadError, StegoError

st.set_page_config(page_title="Стеганография", page_icon="🔏", layout="centered")

# Streamlit не предоставляет перевода служебных надписей загрузчика.
# Селекторы соответствуют версии Streamlit, закреплённой в uv.lock.
st.html("""
<style>
[data-testid="stToolbar"], #MainMenu, footer {display: none;}
[data-testid="stFileUploaderDropzoneInstructions"] {font-size: 0;}
[data-testid="stFileUploaderDropzoneInstructions"] > * {display: none;}
[data-testid="stFileUploaderDropzoneInstructions"]::after {
    content: "Перетащите файл сюда"; font-size: 14px;
}
[data-testid="stFileUploaderDropzone"] button {font-size: 0;}
[data-testid="stFileUploaderDropzone"] button > * {display: none;}
[data-testid="stFileUploaderDropzone"] button::after {
    content: "Выбрать файл"; font-size: 14px;
}
</style>
""")

_NEURAL_ENGINES = {"lfvsn", "steganogan"}


def _fmt(n: int) -> str:
    x = float(n)
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if x < 1024 or unit == "ГБ":
            return f"{x:.0f} {unit}" if unit == "Б" else f"{x:.1f} {unit}"
        x /= 1024


def _save_upload(upload, workdir: Path) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / Path(upload.name).name
    path.write_bytes(upload.getbuffer())
    return path


@st.cache_resource
def _gpu_status() -> dict:
    try:
        import torch

        if torch.cuda.is_available():
            return {
                "available": True,
                "name": torch.cuda.get_device_name(0),
                "count": torch.cuda.device_count(),
                "vram_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
            }
        return {"available": False}
    except ImportError:
        return {"available": None}  # torch не установлен


METHOD_HELP = {
    "lsb": (
        "Прячет любой файл, в том числе видео, в младших битах пикселей. "
        "Секрет восстанавливается байт в байт."
    ),
    "lfvsn": ("Прячет видео или изображение в видео нейросетью. Восстановление приблизительное."),
    "steganogan": (
        "Прячет файл или сообщение в изображение нейросетью. Использует коррекцию ошибок "
        "и повторы — надёжнее короткий секрет в крупной картинке."
    ),
}

_KIND = {
    "video": {
        "pack_label": "Видео-контейнер",
        "extract_label": "Видео со скрытым секретом",
        "in_types": ["mp4", "mkv", "mov", "avi", "webm"],
        "out_types": ["mp4", "mkv", "mov"],
        "suffix": ".mkv",
        "mime": "video/x-matroska",
        "noun": "видео",
    },
    "image": {
        "pack_label": "Картинка-контейнер",
        "extract_label": "Изображение со скрытым секретом",
        "in_types": ["png", "bmp", "tif", "tiff", "jpg", "jpeg", "webp"],
        "out_types": ["png", "bmp", "tif", "tiff"],
        "suffix": ".png",
        "mime": "image/png",
        "noun": "картинку",
    },
}

st.title("Стеганография")

engines = {e.name: e for e in registry.engines()}
method = st.selectbox(
    "Метод",
    sorted(engines, key=lambda n: n != "lsb"),
    format_func=lambda n: engines[n].display_name,
)
if method in METHOD_HELP:
    st.caption(METHOD_HELP[method])

# GPU-статус — показываем только для нейросетевых движков
if method in _NEURAL_ENGINES:
    gpu = _gpu_status()
    if gpu["available"] is True:
        vram = f" · {gpu['vram_gb']:.1f} ГБ видеопамяти" if "vram_gb" in gpu else ""
        st.success(f"Видеокарта: {gpu['name']}{vram} ({gpu['count']} уст.)", icon="⚡")
    elif gpu["available"] is False:
        st.info("Видеокарта не обнаружена — вычисления на процессоре", icon="💻")
    else:
        st.warning("Для этого метода нужен тяжёлый образ с нейросетями и PyTorch", icon="⚠️")

kind = engines[method].container_kind
cfg = _KIND[kind]
mode = st.radio("Режим", ["Упаковать секрет", "Извлечь секрет"], horizontal=True)

params: dict = {}
if method == "lsb":
    params["bits_per_channel"] = st.slider(
        "Бит на канал", 1, 4, 1, help="Больше бит — выше ёмкость, но заметнее искажения"
    )
    if mode == "Упаковать секрет":
        video_format = st.selectbox(
            "Формат результата",
            ["MP4 для браузера", "Видео без потерь (прежний формат, MKV)"],
        )
        params["browser_mp4"] = video_format == "MP4 для браузера"
        if params["browser_mp4"]:
            cfg = {**cfg, "suffix": ".mp4", "mime": "video/mp4"}
            st.caption(
                "Основное видео предназначено для просмотра в браузере. Секрет хранится "
                "в дополнительной дорожке ключевых кадров без потерь. "
                "Ёмкость меньше; размер увеличивается на скрытые кадры."
            )
        else:
            st.caption(
                "Видео без потерь сохраняет все кадры и может быть значительно больше исходного."
            )
    else:
        st.caption(
            "Дорожка с секретом определяется автоматически. Число бит должно совпадать с упаковкой."
        )
elif method == "lfvsn":
    params["num_video"] = 1
elif method == "steganogan":
    params["architecture"] = st.selectbox(
        "Архитектура",
        ["dense", "basic", "residual"],
        format_func=lambda name: {
            "dense": "Плотная",
            "basic": "Базовая",
            "residual": "С остаточными связями",
        }[name],
        help="Плотная — ёмче; базовая — устойчивее на небольших изображениях",
    )
    if mode.startswith("Извлечь"):
        st.caption(
            "Архитектура должна совпадать с той, которой паковали, а картинка — быть тем самым "
            "PNG/BMP/TIFF: повторное сжатие или изменение размера стирают секрет."
        )


def _progress_cb(bar, verb: str):
    return lambda done, total: bar.progress(
        (done / total) if total else 1.0, text=f"{verb}… {done}/{total} кадров"
    )


def _show_pack_metrics(result, *, lfvsn_image: bool = False) -> None:
    st.divider()
    st.caption("Метрики упаковки")
    c1, c2, c3 = st.columns(3)
    if method == "lfvsn":
        c1.metric("Исходный контейнер", _fmt(result.container_size_bytes))
    else:
        c1.metric("Заполнение контейнера", f"{result.packing_ratio:.1%}")
    c2.metric("Время", f"{result.elapsed_s:.1f} с")
    c3.metric(
        "Прирост размера",
        f"{result.overhead_ratio:+.1%}",
        help="(размер результата − размер оригинала) / размер оригинала",
    )
    with st.expander("Подробнее"):
        d1, d2 = st.columns(2)
        d1.metric("Секрет", _fmt(result.secret_bytes))
        d2.metric("Файл с секретом", _fmt(result.stego_size_bytes))
        if method != "lfvsn":
            d1.metric("Бит на пиксель", f"{result.bits_per_pixel:.4f}")
            d2.metric("Доля секрета в результате", f"{result.stego_efficiency:.1%}")
        if result.frames > 1 and not lfvsn_image:
            st.metric("Пропускная способность", f"{result.throughput_fps:.1f} кадров/с")
        if result.device:
            device_name = "видеокарта" if result.device == "cuda" else "процессор"
            st.info(f"Устройство вычисления: **{device_name}**")


def _show_extract_metrics(result) -> None:
    st.divider()
    st.caption("Метрики извлечения")
    c1, c2 = st.columns(2)
    c1.metric("Время", f"{result.elapsed_s:.1f} с")
    c2.metric("Файл с секретом", _fmt(result.stego_size_bytes))
    if result.device:
        st.info(
            f"Устройство вычисления: **{'видеокарта' if result.device == 'cuda' else 'процессор'}**"
        )


def _run(workdir: Path) -> None:
    if mode == "Упаковать секрет":
        container_up = st.file_uploader(cfg["pack_label"], type=cfg["in_types"])
        secret_file = st.file_uploader(
            "Секрет (видео или изображение)" if method == "lfvsn" else "Секрет (любой файл)"
        )
        lfvsn_image = (
            method == "lfvsn"
            and secret_file is not None
            and not (
                secret_file.type.startswith("video/")
                or Path(secret_file.name).suffix.lower()
                in {".mp4", ".m4v", ".mov", ".mkv", ".avi", ".webm"}
            )
        )
        output_suffix = ".mp4" if lfvsn_image else cfg["suffix"]
        output_mime = "video/mp4" if lfvsn_image else cfg["mime"]
        if lfvsn_image:
            st.caption(
                "Изображение встраивается один раз. Результат — MP4 с обычным видео "
                "и одним скрытым кадром без потерь; при извлечении получите PNG."
            )
        st.caption(
            "Предельный размер одного загружаемого файла: "
            f"{st.get_option('server.maxUploadSize')} МБ."
        )

        if container_up:
            container = _save_upload(container_up, workdir / "container")
            try:
                info = capacity(container, method=method, **params)
                if kind == "image":
                    st.info(
                        f"Контейнер: **{info.details['width']}×{info.details['height']}**, "
                        f"прячет **≈{_fmt(info.usable_bytes)}** "
                        "(восстановление приблизительное, лучше короткий секрет)"
                    )
                    if secret_file:
                        st.caption(
                            f"Секрет: {_fmt(secret_file.size)}. Оценка приблизительная — "
                            "текст обычно "
                            "сжимается, а чем крупнее картинка, тем надёжнее извлечение."
                        )
                elif method == "lfvsn":
                    st.info(
                        f"Контейнер: **{info.details['width']}×{info.details['height']}**, "
                        f"{info.details['frames']} кадров · секрет прячется нейросетью "
                        f"(видео/изображение), восстановление приблизительное"
                    )
                else:
                    frame_label = "ключевых кадров" if params.get("browser_mp4") else "кадров"
                    st.info(
                        f"Ёмкость контейнера: **{_fmt(info.usable_bytes)}** "
                        f"· {info.details['width']}×{info.details['height']}, "
                        f"{info.details['frames']} {frame_label}"
                    )
                    if secret_file:
                        fits = info.fits(secret_file.size)
                        st.metric(
                            "Секрет",
                            _fmt(secret_file.size),
                            delta="помещается" if fits else "не помещается",
                            delta_color="normal" if fits else "inverse",
                        )
            except StegoError as e:
                st.error(f"Не удалось прочитать контейнер: {e}")

        if container_up and secret_file and st.button("Упаковать", type="primary"):
            container = _save_upload(container_up, workdir / "container")
            secret_path = _save_upload(secret_file, workdir / "secret")
            output = workdir / f"{Path(container_up.name).stem}_stego{output_suffix}"
            try:
                if kind == "image":
                    with st.spinner("Упаковка…"):
                        result = pack(
                            container,
                            Secret.from_file(secret_path),
                            output,
                            method=method,
                            **params,
                        )
                    st.success("Готово.")
                    st.caption(
                        "Носитель — **PNG без потерь**. "
                        "Не пересохраняйте в JPEG: сжатие повредит секрет."
                    )
                else:
                    bar = st.progress(0, text="Упаковка…")
                    result = pack(
                        container,
                        Secret.from_file(secret_path),
                        output,
                        method=method,
                        progress=_progress_cb(bar, "Упаковка"),
                        **params,
                    )
                    bar.empty()
                    st.success("Готово.")
                    if params.get("browser_mp4") or lfvsn_image:
                        st.video(str(output), format="video/mp4")
                        st.caption(
                            "Сохраните исходный MP4: "
                            "пересылка с перекодированием удалит скрытую дорожку."
                        )
                    else:
                        st.caption(
                            "Видео без потерь. Для просмотра используйте VLC; "
                            "секрет требует точных пикселей."
                        )
                st.download_button(
                    "Скачать файл с секретом",
                    output.read_bytes(),
                    file_name=output.name,
                    mime=output_mime,
                )
                _show_pack_metrics(result, lfvsn_image=lfvsn_image)
            except CapacityExceeded as e:
                st.error(f"Секрет не помещается: {e}")
            except StegoError as e:
                st.error(f"Ошибка упаковки: {e}")

    else:
        container_up = st.file_uploader(cfg["extract_label"], type=cfg["out_types"])
        if container_up and st.button("Извлечь", type="primary"):
            container = _save_upload(container_up, workdir / "container")
            try:
                if kind == "image":
                    with st.spinner("Извлечение…"):
                        result = extract(container, method=method, **params)
                else:
                    bar = st.progress(0, text="Извлечение…")
                    result = extract(
                        container, method=method, progress=_progress_cb(bar, "Извлечение"), **params
                    )
                    bar.empty()
                secret = result.secret
                st.success(f"Извлечено: **{secret.filename}** ({_fmt(secret.size)})")
                st.download_button(
                    "Скачать секрет",
                    secret.data,
                    file_name=secret.filename or "secret.bin",
                    mime=secret.media_type,
                )
                _show_extract_metrics(result)
            except PayloadError as e:
                st.error(f"В контейнере нет корректной нагрузки: {e}")
            except StegoError as e:
                st.error(f"Ошибка извлечения: {e}")


with tempfile.TemporaryDirectory(prefix="stego_") as _tmp:
    _run(Path(_tmp))
