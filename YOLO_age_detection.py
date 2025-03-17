import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image
import cv2
import numpy as np
import torchvision.transforms as T
from ultralytics import YOLO
from torchvision import transforms
import torch.nn as nn
import time

class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1) # 3 x 128 x 128 => 64 x 128 x 128 + pooling
        self.conv_bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) # 64 x 64 x 64 => 128 x 64 x 64 + pooling
        self.conv_bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1) # 128 x 32 x 32 => 256 x 32 x 32 + final pooling = 256 x 16 x 16
        self.conv_bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv_dropout = nn.Dropout2d(0.15)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 16 * 16, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 1)
        self.fc_dropout = nn.Dropout(0.3)

        #init.kaiming_uniform_(self.fc1.weight)
        #init.kaiming_uniform_(self.fc2.weight)
        #init.kaiming_uniform_(self.fc3.weight)
        #init.kaiming_uniform_(self.fc4.weight)

    def forward(self, x):
        x = self.conv_bn1(nn.functional.leaky_relu(self.pool(self.conv1(x))))
        x = self.conv_bn2(nn.functional.leaky_relu(self.pool(self.conv2(x))))
        x = self.conv_dropout(self.conv_bn3(nn.functional.leaky_relu(self.pool(self.conv3(x)))))
        x = self.flatten(x)
        x = self.fc_dropout(self.bn1(nn.functional.leaky_relu(self.fc1(x))))
        x = self.fc_dropout(self.bn2(nn.functional.leaky_relu(self.fc2(x))))
        x = self.fc_dropout(self.bn3(nn.functional.leaky_relu(self.fc3(x))))
        x = self.fc4(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
network = CNN_Model().to(device)
network.load_state_dict(torch.load("best_model.pth"))
network.eval()

model = YOLO('yolov8n-face.pt')
model = model.to(device)
model.eval()

transform_for_age = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

def get_human_boxes(prediction, treshold = 0.6):
    boxes = prediction[0].boxes.cpu().numpy()
    size = boxes.conf.size
    
    cls = np.array(boxes.cls, dtype = int)
    conf = np.array(boxes.conf)
    xyxy = np.array(boxes.xyxy, dtype = int)
    
    # print(conf)
    # print(cls)
    # print(xyxy)

    result = []

    for i in range(size):
        if i>9:
            break
        if(conf[i] > treshold and cls[i] == 0):
            result.append(xyxy[i])

    return result

def show_camera_stream():

    def restart_camera():
        """Properly release and restart the camera."""
        print("Restarting camera...")
        stream.release()
        time.sleep(1)  # Wait before restarting
        return cv2.VideoCapture(0)
    
    stream = cv2.VideoCapture(0)
    stream.set(cv2.CAP_PROP_FPS, 15)
    stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not stream.isOpened():
        print('Camera not found :(')
        exit()
    
    transform = T.ToTensor()
    last_success_time = time.time()
    while(True):
        
        ret, BGR_frame = stream.read()
        if ret:
            last_success_time = time.time()
            frame = cv2.cvtColor(BGR_frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame)
            frame_tensor = transform(frame).unsqueeze(0).to(device)
            prediction = model(frame_tensor, verbose=False)
        
            boxes = get_human_boxes(prediction)
        
            for box in boxes:
                box[0] = max(box[0]-40, 0)
                box[1] = max(box[1]-40, 0)
                box[2] = min(box[2]+40, 640)
                box[3] = min(box[3]+40, 480)
                cv2.rectangle(BGR_frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
                
                isolated_object_frame = BGR_frame[box[1]:box[3], box[0]:box[2]]
                RGB_frame = cv2.cvtColor(isolated_object_frame, cv2.COLOR_BGR2RGB)
                frame_img = Image.fromarray(RGB_frame)
                frame_tensor = transform_for_age(frame_img).unsqueeze(0).to(device)
                result = int(network(frame_tensor).cpu().detach().numpy().item())
                cv2.putText(BGR_frame, f"Age: {result}", (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                #cv2.imshow('Isolated object', isolated_object_frame)
            cv2.imshow('Webcam', BGR_frame)
        else:
            print('Camera Read Failed!')

        if time.time() - last_success_time > 1.5:
            stream = restart_camera()
            last_success_time = time.time()  # Reset timer
                
        if cv2.waitKey(1) == ord('q'):
            break
                
    
    stream.release()
    cv2.destroyAllWindows()

show_camera_stream()
