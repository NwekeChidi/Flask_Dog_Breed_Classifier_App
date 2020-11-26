# Importing Libraries and dependencies
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import model_architect, transformers
from predict import Predict
from PIL import Image, ImageFile
from overlay_filter import OverlayFilter
from werkzeug.utils import secure_filename
from flask import Flask, flash, redirect, request, render_template
from object_detectors import dog_detector, face_detector, detect_image

# Defining and instantiating classes
vgg_model = models.vgg16(pretrained=True)
dog_model = models.vgg16(pretrained=True)

net = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(),
        nn.Dropout(p=0.1),
        nn.Linear(4096, 1000),
        nn.ReLU(),
        nn.Dropout(p=0.15),
        nn.Linear(1000, 133),
        nn.LogSoftmax(dim=1)
        )
for param in dog_model.parameters():
    param.requires_grad = False

breed_predictor = Predict()
dog_model.classifier = net

face_points_model = model_architect.Net()
face_model = cv2.CascadeClassifier('models/haarcascade_frontalface_alt.xml')

face_points_model.load_state_dict(torch.load('models/model.pt', map_location=torch.device('cpu')))
dog_model.load_state_dict(torch.load('models/model_transfer.pt', map_location=torch.device('cpu')))



filter_pth = 'dog_filters/filter3.png'
IMAGE_FOLDER = 'static/'
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']

app = Flask(__name__)  
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')  
def upload():
	return render_template("get_file.html")  
 
@app.route('/success', methods = ['POST'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        # Check if file is the allowed file
        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img_path = img_filename

            # Overlaying filter if image has human face(s)
            if face_detector(img_path=img_path, face_cascade=face_model) == True:
                add_filter = OverlayFilter(img_pth=img_path, filter_pth=filter_pth,
                                            face_points_model=face_points_model, face_model=face_model)

                img_w_filter = add_filter.apply_filter()
                pth = img_path.rsplit('/', 1)[1]
                temp_file_pth = os.path.join(app.config['UPLOAD_FOLDER'], 'd_F_'+pth)
                cv2.imwrite(temp_file_pth, img_w_filter)
                img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'd_F_'+pth)

        #print(image_ext.shape)
        breed = breed_predictor.predict_breed(img_path=img_path, model=dog_model)
        txt1, txt2 = detect_image(img_path=img_path, dog_model=vgg_model,
                                model=dog_model, face_cascade=face_model, breed=breed)
		#result = predict_image(img_path, model)
		#txt = result
        final_text = "Okie Dokey! Let's see what breed we have..."
        return render_template("success.html", name=final_text, greet=txt1, img=img_filename, out_1=txt2)


@app.route('/info', methods = ['POST'])  
def info():
    return render_template("info.html")

if __name__ == '__main__':
    app.run(debug=True)