import cv2 as cv
import torch
from torchvision import transforms
from structure import *

model = torch.load('mood-detector.pth', map_location=torch.device('cpu'))

class_labels = ['angry', 'fear', 'happy', 'neutral', 'sad', 'shock']

def preprocess_image(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    inverted = 255 - gray
    blurred = cv.GaussianBlur(inverted, (21, 21), 0)
    drawing = cv.divide(gray, 255 - blurred, scale=256)
    drawing_resized = cv.resize(drawing, (48, 48))
    img_arr = drawing_resized.astype('float32')
    return img_arr

def getClass(model, img_arr, class_names):
    img_tensor = torch.tensor(img_arr).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    logit = model(img_tensor)
    pred = torch.softmax(logit.squeeze(), dim=0)
    class_ = torch.argmax(pred).item()
    class_name = class_names[class_]
    return class_name

def getTextDimension(top_left, bottom_right, text_size):
    text_X = top_left[0] + (bottom_right[0] - top_left[0] - text_size[0]) // 2
    text_Y = top_left[1] + (bottom_right[1] - top_left[1] + text_size[1]) // 2
    dimension = (text_X, text_Y)
    return dimension

def displayText(text, top_left, bottom_right, img):
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 255, 0)
    font_thickness = 2
    text_size = cv.getTextSize(text, font, font_scale, font_thickness)[0]
    dimension = getTextDimension(top_left, bottom_right, text_size)
    cv.putText(img, text, dimension, font, font_scale, font_color, font_thickness)

face_cascade = cv.CascadeClassifier('haar_cascades.xml')
capture = cv.VideoCapture(0)

while True:
    isTrue, frame = capture.read()
    if not isTrue:
        break

    faces = face_cascade.detectMultiScale(frame, 1.1, 4)

    for (x, y, w, h) in faces:
        face_region = frame[y:y+h, x:x+w]
        img_arr = preprocess_image(face_region)
        mood_prediction = getClass(model, img_arr, class_labels)
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        displayText(mood_prediction, (x, y), (x+w, y+h), frame)

    cv.imshow('Face Detection', frame)

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

capture.release()
cv.destroyAllWindows()
