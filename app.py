import gradio as gr
import cv2
import mediapipe as mp
import numpy as np
import pickle
import time

# Load model + scaler + encoder
with open('./model/asl_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('./model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('./model/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# MediaPipe setup
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

mp_hands = solutions.hands
mp_draw = solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# State
current_sentence = []
current_word = []
last_letter = None
letter_count = 0
last_hand_time = time.time()
LETTER_THRESHOLD = 8
WORD_PAUSE = 1.5
SENTENCE_PAUSE = 4.0

def predict(frame):
    global last_letter, letter_count, current_sentence
    global current_word, last_hand_time

    if frame is None:
        return None, "Waiting for sign...", ""

    frame = np.array(frame, dtype=np.uint8)
    img_rgb = frame.copy()
    result = hands.process(img_rgb)
    frame_out = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    predicted_letter = None
    confidence = 0
    now = time.time()
    hand_present = bool(result.multi_hand_landmarks)

    if hand_present:
        last_hand_time = now
        landmarks = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame_out, landmarks, mp_hands.HAND_CONNECTIONS)

        row = []
        for lm in landmarks.landmark:
            row.extend([lm.x, lm.y, lm.z])

        features = scaler.transform(np.array(row).reshape(1, -1))
        pred_encoded = model.predict(features)[0]
        predicted_letter = le.inverse_transform([pred_encoded])[0]
        proba = model.predict_proba(features)[0]
        confidence = round(max(proba) * 100, 1)

        if predicted_letter == last_letter:
            letter_count += 1
        else:
            last_letter = predicted_letter
            letter_count = 0

        if letter_count == LETTER_THRESHOLD:
            if predicted_letter == 'space':
                if current_word:
                    current_sentence.append("".join(current_word))
                    current_word = []
                current_sentence.append(" ")
                last_hand_time = now
            elif predicted_letter == 'del':
                if current_word:
                    current_word.pop()
                elif current_sentence:
                    while current_sentence and current_sentence[-1] == " ":
                        current_sentence.pop()
                    while current_sentence and current_sentence[-1] != " ":
                        current_sentence.pop()
            elif predicted_letter != 'nothing':
                current_word.append(predicted_letter.upper())
            letter_count = 0

        progress = int((letter_count / LETTER_THRESHOLD) * 20)
        bar = f"[{'█' * progress}{'░' * (20 - progress)}] {letter_count}/{LETTER_THRESHOLD}"

        cv2.putText(frame_out, f"{predicted_letter.upper()} {confidence}%",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(frame_out, bar,
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    else:
        absent_time = now - last_hand_time

        if absent_time >= WORD_PAUSE and current_word:
            current_sentence.append("".join(current_word))
            current_word = []
            current_sentence.append(" ")
            last_hand_time = now

        if absent_time >= SENTENCE_PAUSE and current_sentence and not current_word:
            sentence_str = "".join(current_sentence).strip()
            if sentence_str and not sentence_str.endswith("."):
                current_sentence.append(".")

        cv2.putText(frame_out, "Waiting for sign...",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 255), 2)

    full_display = "".join(current_sentence) + "".join(current_word)
    letter_display = f"{predicted_letter.upper()}  —  {confidence}%" if predicted_letter else "Waiting for sign..."

    return frame_out, letter_display, full_display

def backspace():
    global current_sentence, current_word
    if current_word:
        current_word.pop()
    elif current_sentence:
        while current_sentence and current_sentence[-1] == " ":
            current_sentence.pop()
        if current_sentence:
            last = current_sentence.pop()
            if len(last) > 1:
                current_sentence.append(last[:-1])
    return "".join(current_sentence) + "".join(current_word)

def clear_all():
    global current_sentence, current_word, last_letter, letter_count
    current_sentence = []
    current_word = []
    last_letter = None
    letter_count = 0
    return "Waiting for sign...", ""

CSS = """
* { box-sizing: border-box; }

body, .gradio-container {
    background: #080c1a !important;
    font-family: 'Segoe UI', sans-serif !important;
}

/* Hide gradio footer and unnecessary elements */
footer { display: none !important; }
.built-with { display: none !important; }

/* Top banner */
#top-banner {
    background: #080c1a;
    border-bottom: 1px solid #151a30;
    padding: 16px 40px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0;
}

#logo-text {
    font-size: 1.6em;
    font-weight: 800;
    background: linear-gradient(135deg, #00ff88, #7b61ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

#ai-badge {
    background: #0f1628;
    border: 1px solid #1e2d50;
    border-radius: 20px;
    padding: 6px 16px;
    color: #00ff88;
    font-size: 0.8em;
    letter-spacing: 1px;
}

/* Hero section */
#hero-title {
    text-align: center;
    padding: 36px 20px 20px;
    font-size: 2.4em;
    font-weight: 800;
    color: white;
    line-height: 1.2;
}

#hero-sub {
    text-align: center;
    color: #556;
    font-size: 0.95em;
    padding-bottom: 28px;
}

/* Panel styling */
.panel-label {
    font-size: 0.72em !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
    color: #445 !important;
    padding: 12px 16px !important;
    border-bottom: 1px solid #151a30 !important;
    margin: 0 !important;
}

/* Camera and detection image */
.gradio-image {
    background: #080c1a !important;
    border: 1px solid #151a30 !important;
    border-radius: 14px !important;
    overflow: hidden !important;
}

/* Letter display box */
.letter-display textarea, .letter-display input {
    font-size: 2.8em !important;
    font-weight: 900 !important;
    text-align: center !important;
    background: #0c1020 !important;
    color: #00ff88 !important;
    border: 1px solid #151a30 !important;
    border-radius: 14px !important;
    padding: 20px !important;
}

/* Sentence box */
.sentence-display textarea {
    background: #0c1020 !important;
    color: white !important;
    border: 1px solid #151a30 !important;
    border-radius: 14px !important;
    font-size: 1.2em !important;
    padding: 16px !important;
    line-height: 1.6 !important;
}

/* Buttons */
.btn-clear {
    background: #0c1020 !important;
    color: #ff4444 !important;
    border: 1px solid #ff444422 !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}

.btn-backspace {
    background: #0c1020 !important;
    color: #ff8800 !important;
    border: 1px solid #ff880022 !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}

.btn-clear:hover, .btn-backspace:hover {
    transform: translateY(-1px) !important;
    opacity: 0.85 !important;
}

/* Labels */
label span {
    color: #445 !important;
    font-size: 0.75em !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}
"""

with gr.Blocks(css=CSS, title="PalmScript") as demo:

    gr.HTML('''
        <div id="top-banner">
            <div id="logo-text">PalmScript</div>
            <div id="ai-badge">AI ENGINE LIVE</div>
        </div>
        <div id="hero-title">
            Convert ASL To Text<br>
            <span style="background: linear-gradient(135deg, #00ff88, #7b61ff);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            In Real Time</span>
        </div>
        <div id="hero-sub">
            Trained on 66,272 hand landmarks &nbsp;·&nbsp;
            99.38% accuracy &nbsp;·&nbsp;
            Lighting independent
        </div>
    ''')

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML('<div class="panel-label">Camera Feed</div>')
            webcam = gr.Image(
                sources=["webcam"],
                streaming=True,
                label=""
            )

        with gr.Column(scale=1):
            gr.HTML('<div class="panel-label">Live Detection</div>')
            output_frame = gr.Image(label="")

            letter_box = gr.Textbox(
                label="DETECTED SIGN",
                interactive=False,
                elem_classes=["letter-display"]
            )

            gr.HTML('<div class="panel-label" style="margin-top:12px;">English Output</div>')
            sentence_box = gr.Textbox(
                label="",
                interactive=False,
                lines=3,
                elem_classes=["sentence-display"],
                placeholder="Your sentence will appear here..."
            )

            with gr.Row():
                clear_btn = gr.Button("Clear", elem_classes=["btn-clear"])
                backspace_btn = gr.Button("Backspace", elem_classes=["btn-backspace"])

    webcam.stream(
        predict,
        inputs=[webcam],
        outputs=[output_frame, letter_box, sentence_box],
        time_limit=60,
        stream_every=0.5
    )
    backspace_btn.click(backspace, outputs=[sentence_box])
    clear_btn.click(clear_all, outputs=[letter_box, sentence_box])

demo.launch(show_error=True)