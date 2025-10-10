from flask import Flask, render_template, request, Response
import cv2, time
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import secret

app = Flask(__name__)
camera = cv2.VideoCapture(0)

# Face and emotion detection
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier = load_model('fer.h5')
emotion_labels = ['Angry','Fear','Happy','Neutral', 'Sad', 'Surprise']

label = None

def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
                if np.sum([roi_gray]) != 0:
                    roi = roi_gray.astype('float')/255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)
                    prediction = classifier.predict(roi)[0]
                    global label
                    label = emotion_labels[prediction.argmax()]
                    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

# Routes
@app.route('/')
def main():
    return render_template('main.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/result', methods=["POST"])
def playlistmanager():
    option = request.form['btnradio']
    global label
    if label is None:
        return "Emotion was not detected properly"
    
    input_emotion = label

    # Mock video/music IDs for interface demo
    video_ids = [f"{input_emotion}_video_{i}" for i in range(1, 6)]
    music_ids = [f"{input_emotion}_music_{i}" for i in range(1, 6)]

    if option == "video":
        return render_template('results.html', input=input_emotion, video_ids=video_ids)
    else:
        return render_template('music.html', input=input_emotion, music_ids=music_ids)

if __name__ == '__main__':
    app.run(debug=True)
