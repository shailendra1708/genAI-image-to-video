# GenAI Image-to-Video Generator 🎬

This project converts a **single image into an animated video** using a hybrid AI pipeline.

## Pipeline

Image → CLIP → BLIP → FAISS → OpenCV → RIFE → MoviePy → Video

## Technologies Used

* **CLIP** – image understanding
* **BLIP** – caption generation
* **FAISS** – vector similarity search
* **OpenCV** – animation effects
* **RIFE** – frame interpolation
* **MoviePy** – video generation
* **Gradio** – interactive UI

## Features

* Upload any image
* AI understands image content
* Automatically selects animation style
* Generates smooth animation video
* Web interface for easy testing

## How to Run

```bash
git clone https://github.com/shailendra1708/genAI-image-to-video.git
cd genAI-image-to-video
pip install -r requirements.txt
python gradio_app.py
```

Open:

```
http://localhost:7860
```

## Author

Shailendra
