# stego

Сокрытие данных внутри видеоконтейнеров. Сменные движки за общим фасадом.

Методы:

- **`lsb`** — произвольные байты (текст / файл / изображение / видео) в младших битах пикселей.
  **Bit-exact**: секрет восстанавливается байт-в-байт (magic `STG1` + crc32). Выход —
  **lossless**, кодек по расширению: `.mp4` → H.264 4:4:4 RGB (qp=0), `.mkv` → FFV1.
  Сжатие с потерями уничтожает младшие биты. Для браузера есть отдельный режим
  **`browser_mp4=True`** (CLI: `--browser-mp4`): основное видео — обычный H.264/yuv420p,
  секрет — в дополнительной RGB-дорожке без потерь, только в копиях ключевых кадров.
  Записывается столько ключевых кадров, сколько нужно для секрета. Ёмкость уменьшается;
  размер увеличивается на скрытую дорожку. Исходная совместимая H.264-дорожка копируется,
  остальные кодеки перекодируются. Звук переносится в AAC.
- **`lfvsn`** — [LF-VSN](https://github.com/MC-E/LF-VSN) (CVPR 2023): обратимая нейросеть
  прячет **видео/изображение внутрь видео**. Восстановление **приблизительное** (высокий
  PSNR, не байт-в-байт), поэтому payload/crc не используется. Требует PyTorch (extra `lfvsn`)
  и предобученных весов; работает на **GPU и CPU** (параметр `device`). См. ниже.
  Изображение встраивается один раз в дополнительный кадр без потерь, основное видео остаётся
  сжатым; рекомендуемый выход — `.mp4`, извлечённый секрет — PNG. Это устраняет повторное
  встраивание одной фотографии во все кадры. Для секретного видео сохраняется прежний режим.
- **`steganogan`** — [SteganoGAN](https://github.com/DAI-Lab/SteganoGAN) (MIT 2019): GAN-сеть
  прячет **произвольные байты внутрь изображения**. Восстановление **приблизительное**
  (не байт-в-байт). Поддерживает архитектуры `dense` / `basic` / `residual` (параметр
  `architecture`). Требует PyTorch и предобученных весов; работает на **GPU и CPU**
  (параметр `device`). Минимальный размер контейнера — ~320×240 пикселей.

## Структура

```
stego/
  core/                    фреймворк
    engine.py              StegoEngine — интерфейс движка (capacity/pack/extract)
    steganographer.py      Steganographer(engine) — фасад
    types.py               Secret, CapacityInfo
    payload.py             формат payload (magic, метаданные, crc32)
    registry.py            реестр движков  ·  exceptions.py  ·  api.py
  media/
    video.py               PyAV: read_frames / write_lossless / probe (переиспользуемо)
  engines/                 сменные движки
    lsb/
      engine.py            LSBVideoEngine
      bitops.py            LSB-упаковка бит
    lfvsn/                 LF-VSN (нейросеть, видео-в-видео)
      engine.py            LFVSNVideoEngine (torch — лениво, движок остаётся лёгким)
      device.py            выбор устройства (auto/cpu/cuda)
      weights.py           поиск файла весов
      runner.py            инференс: hide / reveal
      model/               вендоренная сеть (только torch)
      weights/             сюда кладут скачанные .pth (в git не коммитятся)
    steganogan/            SteganoGAN (нейросеть, изображение-в-изображение)
      engine.py            SteganoGANEngine (torch — лениво)
      device.py            выбор устройства (auto/cpu/cuda)
      weights.py           поиск файла весов
      runner.py            инференс: encode / decode
      coding.py            бинаризация payload
      weights/             скачанные .steg (в git не коммитятся)
  ui/app.py                Streamlit, два режима
  cli.py                   CLI
tests/
```

Слои: `core` — абстракции и фасад, `engines` — реализации, `media` — видео-IO.
Зависимости идут внутрь: `engines` → `core`/`media`, `core` ни от кого не зависит.

## Установка

```bash
make install        # uv sync --all-extras
make test           # прогон тестов
make lint           # ruff check + format
```

## Публичный API

Класс-фасад, инициализируется движком (по имени из реестра или объектом):

```python
from stego import Steganographer
from stego.engines.lsb.engine import LSBVideoEngine

sg = Steganographer("lsb")                 # или Steganographer(LSBVideoEngine())
info = sg.capacity("in.mp4", bits_per_channel=1)     # CapacityInfo
sg.pack("in.mp4", "secret.png", "out.mkv", bits_per_channel=1)
secret = sg.extract("out.mkv", output_dir="./recovered")   # -> Secret

# MP4 для просмотра в браузере; секретом может быть другой MP4.
sg.pack("in.mp4", "secret.mp4", "out.mp4", browser_mp4=True)
secret = sg.extract("out.mp4")  # скрытая дорожка определяется автоматически
```

Либо функции для быстрых вызовов (`method="lsb"` по умолчанию):

```python
from stego import capacity, pack, extract
pack("in.mp4", "secret.png", "out.mkv")
```

`secret` принимает `Secret`, путь к файлу, текст или `bytes`.

## CLI

```bash
uv run stego capacity in.mp4
uv run stego pack in.mp4 secret.png -o out.mkv
uv run stego extract out.mkv -o ./recovered

# Браузерный MP4 (нужен системный ffmpeg); ёмкость только ключевых кадров
uv run stego --browser-mp4 capacity in.mp4
uv run stego --browser-mp4 pack in.mp4 secret.mp4 -o out.mp4
uv run stego extract out.mp4 -o ./recovered

# LF-VSN (нейросеть): секрет — видео/изображение, --device auto|cpu|cuda
uv run stego --method lfvsn --device cpu pack cover.mkv secret.mkv -o stego.mkv
uv run stego --method lfvsn --device cuda extract stego.mkv -o ./recovered

# SteganoGAN (нейросеть): секрет — произвольные байты, контейнер — изображение
uv run stego --method steganogan --device cuda pack cover.png secret.txt -o stego.png
uv run stego --method steganogan --device cuda --architecture dense extract stego.png -o ./recovered
```

## UI

```bash
uv run streamlit run stego/ui/app.py
```

## Docker

Два образа, оба запускают веб-интерфейс без аргументов и CLI с аргументами:

| Образ | Сборка | Методы |
| :--- | :--- | :--- |
| Лёгкий `stego:light` | `Dockerfile` | LSB, без PyTorch и CUDA |
| Тяжёлый `stego:heavy` | `Dockerfile.cuda` | LSB, LF-VSN, SteganoGAN; CPU или NVIDIA GPU |

```bash
docker build -t stego:light .
docker build -f Dockerfile.cuda -t stego:heavy .

# Веб-интерфейс: http://localhost:8501
docker run --rm -p 8501:8501 stego:light

# Видео в видео, браузерный MP4, точное восстановление файла-секрета
docker run --rm -v "$PWD:/data" stego:light stego --browser-mp4 pack /data/in.mp4 /data/secret.mp4 -o /data/out.mp4
docker run --rm -v "$PWD:/data" stego:light stego extract /data/out.mp4 -o /data/recovered

# Тяжёлый образ с внешними весами; для CPU уберите --gpus all
docker run --rm --gpus all -p 8501:8501 \
  -v "$PWD/stego/engines/lfvsn/weights:/app/stego/engines/lfvsn/weights:ro" \
  -v "$PWD/stego/engines/steganogan/weights:/app/stego/engines/steganogan/weights:ro" \
  stego:heavy
```

Причины прежнего большого размера: обычный образ устанавливал extra `lfvsn`, а Linux-колёса
PyTorch из PyPI включают CUDA-зависимости даже при запуске на CPU. Образ также сохранял кэш uv,
а `.dockerignore` исключал `.pth`, но пропускал веса SteganoGAN `.steg`.
Теперь лёгкий образ не устанавливает нейросети, обе сборки отключают кэш установки и исключают
веса; тяжёлый образ не дублирует CUDA отдельным базовым образом. Его размер всё равно зависит
от PyTorch и CUDA-библиотек. Веса монтируются отдельно:

- LF-VSN — [инструкция](stego/engines/lfvsn/weights/README.md).
- SteganoGAN — [инструкция](stego/engines/steganogan/weights/README.md).

### Видео, совместимость и память

В веб-интерфейсе LSB по умолчанию создаёт браузерный MP4. Прежний режим без потерь остаётся
доступным. API и CLI по умолчанию сохраняют прежнее поведение; новый режим включается явно.
Число бит на канал при извлечении должно совпадать с упаковкой. Упаковка и извлечение LSB,
а также LF-VSN обрабатывают кадры последовательно, без накопления распакованного видео в памяти.
Секрет по публичному API всё ещё передаётся целиком в байтах; для LF-VSN также нужна память модели.
Временные файлы интерфейса удаляются после обработки.

Браузерный MP4 содержит две видеодорожки: обычную для просмотра и дополнительную без потерь
для извлечения LSB. Это не делает секрет устойчивым к перекодированию: мессенджеры, редакторы
и видеохостинги могут удалить дополнительную дорожку. Передавать нужно исходный файл.
Сокрытие только в ключевых кадрах основной дорожки с последующим обычным сжатием H.264
не сохраняет младшие биты. Прирост размера в новом режиме зависит от объёма секрета и содержимого
ключевых кадров; нулевой прирост не гарантируется.

Для изображения LF-VSN сохраняет один скрытый RGB-кадр без потерь, а основную дорожку оставляет
сжатой. Совместимая H.264-дорожка копируется; для других кодеков MP4 создаётся H.264/yuv420p.
При извлечении возвращается PNG с приблизительно восстановленным изображением.
Старые контейнеры по-прежнему читаются. Для секретного видео LF-VSN сохраняет прежний режим
без потерь (для просмотра — VLC); извлечённое видео записывается в обычный H.264 MP4.
Тесты LF-VSN со случайными весами проверяют работу тракта, а качество восстановления нужно
оценивать отдельно с предобученными весами.

## Добавление движка

1. `engines/<name>/engine.py` — класс-наследник `StegoEngine` с `id/name/display_name`
   и методами `capacity/pack/extract`, декоратор `@register`.
2. Импортировать его в `engines/__init__.py` (одна строка).

Движок сразу доступен через `Steganographer("<name>")`, `api`, CLI и UI.
Видео-IO берётся готовым из `media.video`. Готовые движки: `lsb`, `lfvsn`, `steganogan`.
