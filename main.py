import torch
from model_train import create_model
import os
import cv2
from flask import Flask, render_template, request,  session
from torchvision.ops.boxes import nms
import time
from werkzeug.utils import secure_filename


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = create_model(2).to(device)
model.load_state_dict(torch.load('fasterrcnn_resnet50_fpn3.pth'))
model.eval()

UPLOAD_FOLDER = os.path.join('static', 'uploads')

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = '123456'

@app.route('/')
def index():
    return render_template("index.html")

def predict(img_filename):

    start_time = time.time()
    img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], img_filename))
    img_ = img / 255
    img_ = torch.from_numpy(img_).permute(2, 0, 1).unsqueeze(0).to(torch.float).to(device)

    predict = model(img_)
    iou_threshold = 0.1
    scale_percent = 100
    ind = nms(predict[0]['boxes'], predict[0]['scores'], iou_threshold).detach().cpu().numpy()
    max_p = 0
    for i, box in enumerate(predict[0]['boxes'][ind]):
        if predict[0]['scores'][i] > predict[0]['scores'][max_p]:
            max_p = i
    box = predict[0]['boxes'][ind][max_p]
    cv2.rectangle(img,
                  (int(box[0]) - 10, int(box[1]) - 10),
                  (int(box[2]) + 10, int(box[3]) + 10),
                  (0, 0, 255), 2)
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim)
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], img_filename), img)
    end_time = time.time()
    return end_time-start_time

@app.route('/', methods=("POST", "GET"))
def uploadFile():
    if request.method == 'POST':
        uploaded_img = request.files['uploaded-file']
        img_filename = secure_filename(uploaded_img.filename)
        uploaded_img.save(os.path.join(app.config['UPLOAD_FOLDER'], img_filename))

        predict_time = predict(img_filename)
        session['uploaded_img_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
        return render_template('upload_display.html', time=predict_time//0.001/1000)


@app.route('/show_image')
def displayImage():
    img_file_path = session.get('uploaded_img_file_path', None)
    return render_template('show_image.html', image=img_file_path)


if __name__ == "__main__":

#    app.run(host='0.0.0.0', port=80)
    app.run(debug=True)