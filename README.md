
# PalmScript — ASL Sign Language to Text

Real-time American Sign Language alphabet recognition from webcam to text, powered by MediaPipe hand landmarks and a custom trained machine learning model.

**Live Demo:** https://huggingface.co/spaces/Suhailnasar/PalmScript

---

## What is PalmScript?

PalmScript converts ASL fingerspelling into text in real time using just a webcam. No special hardware needed — just your hand and a camera.

Sign letters → Watch them appear on screen → Build full words and sentences

---

## How it Works

Instead of training on raw pixels, PalmScript extracts 21 hand landmark points (x, y, z coordinates = 63 numbers) from each frame using MediaPipe and feeds them into a trained machine learning model.


Webcam → MediaPipe (21 hand landmarks) → ML Model → Predicted Letter → Sentence


This approach gives us:
- Lighting independence — landmarks don't care about shadows or background
- Fast inference — 63 numbers instead of thousands of pixels
- High accuracy — 99.38% on test set

---

## Features

- Real-time hand landmark detection via MediaPipe
- Custom trained Neural Network (99.38% accuracy)
- Supports full ASL alphabet — A to Z, space, delete
- Smart sentence builder — pause hand to separate words
- Auto punctuation — long pause adds period
- Backspace and clear buttons
- Dark themed professional UI
- Deployed as a public web app

---

## Tech Stack

| Purpose | Tool |
|---|---|
| Hand Detection | MediaPipe |
| Webcam Capture | OpenCV |
| Model Training | scikit-learn |
| Web Interface | Gradio |
| Deployment | Hugging Face Spaces |

---

## Model Performance

| Model | Accuracy |
|---|---|
| Random Forest | ~96% |
| Neural Network | 99.38% |

Trained on 66,272 hand landmark samples extracted from 87,000 ASL alphabet images.

---

## Project Structure


PalmScript/
├── model/
│   ├── asl_model.pkl        
│   ├── scaler.pkl           
│   └── label_encoder.pkl    
├── app.py                   
├── requirements.txt         
└── packages.txt             


---

## Run Locally

bash
git clone https://github.com/suhailnasar/PalmScript
cd PalmScript
pip install -r requirements.txt
python app.py


---

## How to Use

1. Open the live demo link
2. Allow camera access
3. Show your hand and sign ASL letters
4. Hold each sign steady for it to register
5. Remove hand briefly to separate words
6. Remove hand for 4 seconds to end sentence

---

## Built By

Mohammed Suhail — [GitHub](https://github.com/suhailnasar) · [Hugging Face](https://huggingface.co/Suhailnasar)
```

