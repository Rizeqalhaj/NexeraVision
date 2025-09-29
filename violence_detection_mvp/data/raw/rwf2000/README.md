---
license: mit
task_categories:
- video-classification
language:
- en
tags:
- surveillance
pretty_name: 'Real World Fight '
author: 'Cheng, M., Cai, K., & Li, M. (2021, January)'
document: 'https://ieeexplore.ieee.org/abstract/document/9412502/'
size_categories:
- 10B<n<100B
---

Para crear un archivo README adecuado para un dataset en Hugging Face, es importante proporcionar información clara y detallada sobre el contenido del dataset, su uso y cualquier información adicional relevante. Aquí tienes un ejemplo de un README para el dataset **Real World Fight (RWF) 2000**:

---

# Real World Fight (RWF) 2000

![Descripción](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection) <!-- Asegúrate de actualizar esta URL con una imagen representativa del dataset si tienes una. -->

## Descripción

**Real World Fight (RWF) 2000** es un conjunto de datos de video diseñado para el reconocimiento de peleas en videos del mundo real. Este dataset contiene videos etiquetados en dos categorías: **"Fight"** y **"Non-Fight"**, y está destinado a facilitar la investigación en la detección automática de violencia en videos de vigilancia y otros contextos de la vida real.

## Detalles del Dataset

- **Tamaño Total:** 12 GB (asegúrate de ajustar según tu archivo real)
- **Número de Videos:** 2000 videos en total
  - **Fight:** 1000 videos
  - **Non-Fight:** 1000 videos
- **Formato de Video:** .avi
- **Duración del Video:** Cada video tiene una duración de aproximadamente 5 a 10 segundos.
- **Resolución:** 640x360 píxeles
- **Etiquetas:** Binarias (0 = Non-Fight, 1 = Fight)

## Estructura del Dataset

El dataset está organizado en dos conjuntos principales de datos: entrenamiento y validación, cada uno con sus subdirectorios correspondientes.

```
RWF-2000/
    ├── train/
    │   ├── Fight/
    │   │   ├── video1.avi
    │   │   ├── video2.avi
    │   │   └── ...
    │   └── NonFight/
    │       ├── video1.avi
    │       ├── video2.avi
    │       └── ...
    └── val/
        ├── Fight/
        │   ├── video1.avi
        │   ├── video2.avi
        │   └── ...
        └── NonFight/
            ├── video1.avi
            ├── video2.avi
            └── ...
```

## Descarga del Dataset

Puedes descargar el dataset directamente desde este repositorio de Hugging Face. Usa el siguiente comando para clonar el dataset a tu máquina local:

```bash
git lfs install
git clone https://huggingface.co/datasets/DanJoshua/RWF-2000
```

## Uso del Dataset

El dataset está destinado a facilitar la investigación y el desarrollo de modelos para la detección de violencia en videos. Se puede utilizar para tareas como:

- Clasificación de videos de pelea vs. no pelea.
- Entrenamiento de modelos de visión por computadora para la detección de actividades violentas.
- Evaluación de algoritmos de detección de eventos violentos en videos.

### Ejemplo de Uso

Aquí hay un ejemplo de cómo cargar y procesar el dataset usando Python:

```python
import os
import cv2

def load_videos(folder_path):
    videos = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.avi'):
            video_path = os.path.join(folder_path, filename)
            cap = cv2.VideoCapture(video_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            videos.append(frames)
            cap.release()
    return videos

# Cargar videos de entrenamiento
fight_videos = load_videos('RWF-2000/train/Fight/')
nonfight_videos = load_videos('RWF-2000/train/NonFight/')
```

## Cita

```
@inproceedings{cheng2021rwf,
  title={RWF-2000: an open large scale video database for violence detection},
  author={Cheng, Ming and Cai, Kunjing and Li, Ming},
  booktitle={2020 25th International Conference on Pattern Recognition (ICPR)},
  pages={4183--4190},
  year={2021},
  organization={IEEE}
}
```
