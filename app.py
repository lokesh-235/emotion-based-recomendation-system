from flask import Flask, render_template, request, Response
import cv2
import numpy as np
import pandas as pd
import random
from keras.models import load_model
from keras.preprocessing.image import img_to_array

app = Flask(__name__)

# Load models
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier = load_model('fer.h5')

emotion_labels = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
CSV_FILE = "youtube_playlist_videos.csv"

camera = cv2.VideoCapture(0)
label = None


# ---------------------- VIDEO STREAM ----------------------
def generate_frames():
    global label
    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))

            if np.sum(roi_gray) != 0:
                roi = roi_gray.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                preds = classifier.predict(roi)[0]
                label = emotion_labels[preds.argmax()]

                cv2.putText(frame, label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# ---------------------- CSV RECOMMENDER ----------------------
def get_recommended_videos(emotion, limit=20):
    try:
        df = pd.read_csv(CSV_FILE)
        df['emotion'] = df['emotion'].str.lower().str.strip()

        videos = df[df['emotion'] == emotion.lower()]['video_link'].tolist()
        if not videos:
            return []

        return random.sample(videos, min(len(videos), limit))

    except Exception as e:
        print("CSV Error:", e)
        return []


# ---------------------- ROUTES ----------------------
@app.route('/')
def main():
    return render_template('main.html')


@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/result', methods=['POST'])
def result():
    global label
    option = request.form['btnradio']

    if label is None:
        return "Emotion not detected"

    if option == "video":
        videos = get_recommended_videos(label)
        return render_template('results.html',
                               emotion=label,
                               video_links=videos)

    return "Music option not implemented"


# ---------------------- RUN ----------------------
if __name__ == "__main__":
    app.run(debug=True)
