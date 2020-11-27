# Import libraries
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from transformers import DuplicateArray, ToTensor
from torch.autograd import Variable

class OverlayFilter:

    def __init__(self, img_pth, filter_pth, face_points_model, face_model):
        self.img_pth = img_pth
        self.filter_pth = filter_pth
        self.model = face_points_model
        self.face_cascade = face_model
        self.transform = transforms.Compose([
            DuplicateArray(),
            ToTensor(),
            transforms.Normalize(mean=np.array([0.485, 0.456, 0.406]),
            std=np.array([0.229, 0.224, 0.225]))
        ])


    def transparentOverlay(self, src , overlay , pos=(0,0),scale = 1):

        overlay = cv2.resize(overlay,(0,0),fx=scale,fy=scale)
        h,w,_ = overlay.shape  # Size of foreground
        rows,cols,_ = src.shape  # Size of background Image
        y,x = pos[0],pos[1]    # Position of foreground/overlay image
        
        #loop over all pixels and apply the blending equation
        for i in range(h):
            for j in range(w):
                if x+i >= rows or y+j >= cols:
                    continue
                alpha = float(overlay[i][j][3]/255.0) # read the alpha channel 
                src[x+i][y+j] = alpha*overlay[i][j][:3]+(1-alpha)*src[x+i][y+j]
        return src


    def get_filter_spots(self, pred_res, scale_x, scale_y, scale, dog_filter):
        #scale_x and y for scale of 96x96 image and scale for  overall filter scaling
        filter_nose = dog_filter['nose']
        filter_right_ear = dog_filter['ear_right']

        #Hyper-Paramaters
        y_padding = 5
        ear_padding = 6

        #Add nose
        nose_x = int(pred_res[20]*48+48*scale_x - filter_nose.shape[1]*scale/2)
        nose_y = int( (pred_res[21]*48+48 + y_padding)*scale_y - filter_nose.shape[0]*scale/2)
        

        
        left_ear_x = 0 - ear_padding
        left_ear_y = 0 - ear_padding*2
        
        right_ear_x = int( (96 + ear_padding*2)*scale_x - filter_right_ear.shape[0]*scale )
        right_ear_y = (0 - ear_padding)*scale_y
        
        return [nose_x, nose_y],[left_ear_x*scale_x, left_ear_y*scale_y],[right_ear_x, right_ear_y]


    # Method to scale images and filters
    def get_best_scaling(self, w):
        filter_width = 420
        return 1.1*(w/filter_width)



    # Method to predict facial keypoints
    def predict_face_points(self, img):

        face_model = self.model
        img = Variable(img.float())
        


        face_model.eval()

        prediction = face_model(img)
        prediction = prediction.detach().numpy()

        if len(prediction) > 0:
            return prediction



    # Method to apply dog filter
    def add_dog_filter(self, dog_filter):

        img_path = self.img_pth
        face_cascade = self.face_cascade

        # load color (BGR) image
        img = cv2.imread( img_path )
        # convert BGR image to grayscale
        gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )

        # find faces in image
        faces = face_cascade.detectMultiScale( gray )

        filter_nose = dog_filter['nose']
        filter_left_ear = dog_filter['ear_left']
        filter_right_ear = dog_filter['ear_right']
        
        if len(faces)==0:
            print("No faces Detected in the image")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        for (x,y,w,h) in faces:        
            # add bounding box to color image
            #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            
            img_crop = img[y:y+h, x:x+w]
            
            scale_x = img_crop.shape[0]/96
            scale_y = img_crop.shape[1]/96
            
            img2 = cv2.resize(img_crop, (96,96))
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            #print(img2.shape)

            #for model
            img2 = np.vstack(img2)
            img2 = img2 / 255.  # scale pixel values to [0, 1]
            img2 = img2.astype(np.float32)
            img2 = img2.reshape(-1, 96, 96)
            #print(img2.shape)
            img2 = self.transform(img2)
            img2 = img2.unsqueeze(0)
            #print(img2.shape)

            
            #Predict using CNN model
            pred_res = self.predict_face_points(img2)[0]

            scale = self.get_best_scaling(w)
            
            nose, left_ear,right_ear = self.get_filter_spots(pred_res=pred_res, scale_x=scale_x,
                                                            scale_y=scale_y, scale=scale, dog_filter=dog_filter)  
            
            #Add images
            result = self.transparentOverlay(img.copy(),filter_nose,( int(nose[0]+x), int(nose[1]+y)), scale)
            result = self.transparentOverlay(result.copy(),filter_left_ear,( int(left_ear[0]+x), int(left_ear[1]+y)), scale)
            result = self.transparentOverlay(result.copy(),filter_right_ear,( int(right_ear[0]+x), int(right_ear[1]+y)), scale)
            
            img = result
        #Change to RGB
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        #and return
        return result

    
    # Method to overlay filter
    def apply_filter(self):

        filter_full = cv2.imread(self.filter_pth, cv2.IMREAD_UNCHANGED)
    
        dog_filter = {  'nose' : filter_full[302:390,147:300],
                        'ear_left' : filter_full[55:195,0:160],
                        'ear_right' : filter_full[55:190,255:420],
                     }

        final_result = self.add_dog_filter(dog_filter=dog_filter)
        
        # Change to BGR
        final_result = cv2.cvtColor(final_result, cv2.COLOR_RGB2BGR)
        return final_result