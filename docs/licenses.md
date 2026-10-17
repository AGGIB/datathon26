# Лицензии моделей и пакетов

Только лицензии, допускающие коммерческое использование (Apache-2.0 / MIT / BSD). Без Ultralytics/AGPL и некоммерческих весов.

## Модели

| Модель | Лицензия | Ссылка | Зачем |
|---|---|---|---|
| RT-DETRv2-R18 (`PekingU/rtdetr_v2_r18vd`) | Apache-2.0 | https://huggingface.co/PekingU/rtdetr_v2_r18vd | Детекция людей и COCO-объектов на каждом кадре (`ml/detect.py`) |
| OWLv2 base ensemble (`google/owlv2-base-patch16-ensemble`) | Apache-2.0 | https://huggingface.co/google/owlv2-base-patch16-ensemble | Zero-shot: стол, сейф, стеклянная дверь с ручкой, экран ноутбука, предметы на столе — на 6–8 ключевых кадрах видео (`ml/scene_owl.py`). Веса ~620 МБ, только `model.safetensors` |

## Пакеты

Версии — в [`ml/requirements.txt`](../ml/requirements.txt).

| Пакет | Лицензия |
|---|---|
| torch | BSD-3-Clause |
| transformers | Apache-2.0 |
| opencv-python-headless | Apache-2.0 |
| numpy | BSD-3-Clause |
| pillow | MIT-CMU |
| openpyxl | MIT |
| pyyaml | MIT |
