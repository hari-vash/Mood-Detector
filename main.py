import cv2 as cv
import torch
from torchvision import transforms
from structure import * 


model = torch.load('mood_detector.pth', map_location=torch.device('cpu'))
model.eval()


class_labels = ['angry', 'fear', 'happy', 'neutral', 'sad', 'shock']


# Define the expected input size for the model
input_size = (48, 48)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def getClass(model, img):
    img = transform(img).unsqueeze(0)  
    with torch.no_grad():
        pred_logit = model(img)
        pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)
        predicted_class = torch.argmax(pred_prob).item()
        text = class_labels[predicted_class]
    return text

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

    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.1, 4)

    for (x, y, w, h) in faces:
        face_region = frame[y:y+h, x:x+w]
        mood_prediction = getClass(model, face_region)
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        displayText(mood_prediction, (x, y), (x+w, y+h), frame)

    cv.imshow('Face Detection', frame)


    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

capture.release()
cv.destroyAllWindows()
