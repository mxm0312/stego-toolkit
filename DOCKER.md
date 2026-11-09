# Запуск Docker: интерфейс и CLI

Команды выполняются из корня проекта в Bash. На компьютере должен быть запущен Docker.

| Образ | Методы | Веса |
| --- | --- | --- |
| `stego:light` | LSB | Не нужны |
| `stego:heavy` | LSB, LF-VSN, SteganoGAN | Нужны для нейросетей |

## Сборка

```bash
docker build -t stego:light .
docker build -f Dockerfile.cuda -t stego:heavy .
```

## Подключение весов

**Веса не входят в образы. Их нужно скачать отдельно и подключать при запуске через `-v`.**
По умолчанию примеры ниже используют такие файлы на компьютере:

```text
stego/engines/lfvsn/weights/LF-VSN_1video_hiding_250k.pth
stego/engines/steganogan/weights/dense.steg
stego/engines/steganogan/weights/basic.steg
```

Для LF-VSN также подходит имя `lfvsn_1video.pth`. Для SteganoGAN нужен файл выбранной
архитектуры: `dense.steg` или `basic.steg`.
Ссылки для скачивания: [LF-VSN](stego/engines/lfvsn/weights/README.md),
[SteganoGAN](stego/engines/steganogan/weights/README.md).

В `-v "путь_на_компьютере:путь_в_контейнере:ro"` укажите свою папку слева.
Переменные `LFVSN_WEIGHTS` и `STEGANOGAN_WEIGHTS` задают путь **внутри контейнера**.
Суффикс `:ro` подключает веса только для чтения. Это одинаково для UI и CLI.

## Веб-интерфейс (UI)

Лёгкий образ — открыть <http://localhost:8501>:

```bash
docker run --rm -p 127.0.0.1:8501:8501 stego:light
```

Тяжёлый образ с NVIDIA GPU и весами — открыть <http://localhost:8502>:

```bash
docker run --rm --gpus all -p 127.0.0.1:8502:8501 \
  -e LFVSN_WEIGHTS=/weights/lfvsn \
  -e STEGANOGAN_WEIGHTS=/weights/steganogan \
  -v "$PWD/stego/engines/lfvsn/weights:/weights/lfvsn:ro" \
  -v "$PWD/stego/engines/steganogan/weights:/weights/steganogan:ro" \
  stego:heavy
```

`--gpus all` передаёт видеокарту контейнеру. Для запуска UI на CPU уберите этот флаг.
После изменения флага запустите новый контейнер; пересобирать образ не нужно.

Выберите метод и режим, загрузите контейнер и секрет, нажмите «Упаковать» и скачайте результат.
Для извлечения выберите тот же метод и загрузите полученный файл.
Число бит LSB и архитектура SteganoGAN должны совпадать с упаковкой.
Остановить интерфейс — `Ctrl+C` в терминале запуска.

## Командная строка (CLI)

Без команды образ запускает UI. Для CLI после имени образа укажите `stego` и команду.
Том `-v "$PWD:/data"` делает файлы текущей папки доступными внутри контейнера в `/data`;
результаты также сохраняются в эту папку.

### LSB: любой файл в видео

Пример с готовым видео из репозитория: проверка ёмкости, упаковка видео в видео и извлечение.

```bash
docker run --rm -v "$PWD:/data" stego:light \
  stego --browser-mp4 capacity /data/examples/secret.mp4

docker run --rm -v "$PWD:/data" stego:light \
  stego --browser-mp4 pack /data/examples/secret.mp4 /data/examples/secret.mp4 -o /data/stego.mp4

docker run --rm -v "$PWD:/data" stego:light \
  stego extract /data/stego.mp4 -o /data/recovered

cmp examples/secret.mp4 recovered/secret.mp4
```

`cmp` ничего не выводит, если файлы совпали. `stego.mp4` предназначен для просмотра в браузере.
Для прежнего формата уберите `--browser-mp4` и укажите выходной файл `.mkv`.

### LF-VSN: видео или изображение в видео

Положите `cover.mp4` и `secret.mp4` в текущую папку:

```bash
docker run --rm -v "$PWD:/data" \
  -e LFVSN_WEIGHTS=/weights/lfvsn \
  -v "$PWD/stego/engines/lfvsn/weights:/weights/lfvsn:ro" \
  stego:heavy stego --method lfvsn --device cpu \
  pack /data/cover.mp4 /data/secret.mp4 -o /data/stego.mkv

docker run --rm -v "$PWD:/data" \
  -e LFVSN_WEIGHTS=/weights/lfvsn \
  -v "$PWD/stego/engines/lfvsn/weights:/weights/lfvsn:ro" \
  stego:heavy stego --method lfvsn --device cpu \
  extract /data/stego.mkv -o /data/recovered
```

Носитель `stego.mkv` смотрите в VLC. Извлечённый секрет — `recovered/recovered_secret.mp4`;
восстановление приблизительное.

Для фотографии замените `/data/secret.mp4` на `/data/photo.jpg`, а выходной файл и файл
для извлечения — на `/data/stego.mp4`. Изображение встраивается один раз в отдельный кадр
без потерь; основное видео остаётся сжатым и проигрывается в браузере. При извлечении
получите `recovered/recovered_secret.png`. В UI формат MP4 для фотографии выбирается автоматически.

### SteganoGAN: файл в изображение

Положите `cover.png` и небольшой `secret.txt` в текущую папку:

```bash
docker run --rm -v "$PWD:/data" \
  -e STEGANOGAN_WEIGHTS=/weights/steganogan \
  -v "$PWD/stego/engines/steganogan/weights:/weights/steganogan:ro" \
  stego:heavy stego --method steganogan --device cpu --architecture dense \
  pack /data/cover.png /data/secret.txt -o /data/stego.png

docker run --rm -v "$PWD:/data" \
  -e STEGANOGAN_WEIGHTS=/weights/steganogan \
  -v "$PWD/stego/engines/steganogan/weights:/weights/steganogan:ro" \
  stego:heavy stego --method steganogan --device cpu --architecture dense \
  extract /data/stego.png -o /data/recovered
```

Для начала используйте изображение от 512×512 и короткий текст. Для базовой архитектуры
замените `dense` на `basic` в обеих командах и подключите `basic.steg`.

Примеры CLI выше используют CPU. Для NVIDIA GPU добавьте `--gpus all` в параметры
`docker run` перед именем тяжёлого образа и замените `--device cpu` на `--device cuda`.
В UI устройство выбирается автоматически.

Сохраняйте файлы с секретом без перекодирования: оно может повредить или удалить скрытые данные.
