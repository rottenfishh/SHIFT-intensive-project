import os
import torch
from PIL import Image
import numpy as np
import cv2
from flask import Flask, render_template, request
import time
import tempfile
app = Flask(__name__, static_url_path='/static')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    start_time = time.time() 
    model = torch.load('../saved_model/model.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    if request.method == 'POST':
        image_file = request.files.get('image')
        video_file = request.files.get('video')
        if image_file and not video_file:
            image = Image.open(image_file)
            if image.mode=='RGBA':
                image = image.convert("RGB")
            with torch.no_grad():
                image = np.array(image)
                image = cv2.resize(image, (200, 200))
                image = np.transpose(image, (2, 0, 1))
                image = torch.tensor(image).float()
                image = image.unsqueeze(0)
                if torch.cuda.is_available():
                    image = image.to('cuda')
                output = model(image)
                predicted_class = output.argmax(dim=1).item()
                prediction = chr(65 + predicted_class)
                probabilities = (torch.softmax(output, dim=1))
                probability = probabilities[:, predicted_class].item()
            end_time = time.time()
            processing_time = end_time - start_time
            res_time = round(processing_time,2)
            return render_template('index.html', prediction=prediction, processing_time = res_time)
        elif video_file and not image_file:
            temp_dir = tempfile.mkdtemp()
            video_path = os.path.join(temp_dir, video_file.filename)
            video_file.save(video_path)
            video_stream = cv2.VideoCapture(video_path) #здесь читаем видео
            fps = video_stream.get(cv2.CAP_PROP_FPS)
            frames = []
            dif = 120
            ret, prev_frame = video_stream.read()
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            frames.append(prev_frame)
            while True: #делим на фреймы
                ret, frame = video_stream.read()
                if not ret:
                    break
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_diff = cv2.absdiff(prev_gray, gray_frame)
                _, thresh = cv2.threshold(frame_diff, dif, 255, cv2.THRESH_BINARY)
                motion_count = cv2.countNonZero(thresh)
                if motion_count > 0:
                    frames.append(frame)
                prev_gray = gray_frame
            video_stream.release()
            result = ''
            res = ''
            for frame in frames: #обрабатываем
                if frame is None:
                    continue
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with torch.no_grad():
                    image = cv2.resize(image, (200, 200))
                    image = np.transpose(image, (2, 0, 1))
                    image = torch.tensor(image).float()
                    image = image.unsqueeze(0)
                    if torch.cuda.is_available(): 
                        image = image.to('cuda')
                    output = model(image)
                    predicted_class = output.argmax(dim=1).item()
                    if (predicted_class == 28):
                        prediction = ' '
                    elif (predicted_class ==27):
                        prediction =''
                    prediction = chr(65 + predicted_class)
                    probabilities = (torch.softmax(output, dim=1))
                    probability = probabilities[:, predicted_class].item()
                    if (probability>0.5):
                        res = res + prediction
            i=0
            while i<len(res):
                if res.count(res[i])>=5:
                    prev = res[i]
                    result = result + prev
                    while i<len(res) and res[i] == prev:
                        i = i+1
                else:
                    i+=1
            end_time = time.time()
            processing_time = end_time - start_time
            res_time = round(processing_time,2)
            return render_template('index.html', prediction=result, processing_time = res_time)

        return render_template('index.html') 

@app.route("/")
def index():
    return render_template('index.html')
if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    app.run(debug=True, port = 5003)