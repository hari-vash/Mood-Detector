import streamlit as st
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from model2 import Face_Emotion_CNN

# Load pre-trained model
@st.cache(allow_output_mutation=True)
def load_trained_model(model_path):
    model = Face_Emotion_CNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
    model.eval()
    return model

# Function for predicting emotion on a single frame
def predict_emotion(model, face_img, emotion_dict):
    transform = transforms.Compose([transforms.ToTensor()])
    img = Image.fromarray(face_img)
    img = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        log_ps = model(img)
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)
        pred = emotion_dict[int(top_class.numpy())]
    return pred

# Streamlit app definition
def main():
    st.title("Real-Time Facial Emotion Recognition")
    st.write("Click 'Start' to capture frames from your webcam for emotion detection.")

    model_path = './FER_trained_model.pt'
    model = load_trained_model(model_path)
    
    # Emotion labels
    emotion_dict = {0: 'neutral', 1: 'happiness', 2: 'surprise', 3: 'sadness', 4: 'anger', 5: 'disgust', 6: 'fear'}
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    start_button = st.button("Start Camera")
    stop_button = st.button("Stop Camera")

    frame_placeholder = st.empty()  # Placeholder for displaying video frames

    # Load face detector
    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

    if start_button:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image from webcam.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face_img = gray[y:y + h, x:x + w]
                face_img_resized = cv2.resize(face_img, (48, 48))
                emotion = predict_emotion(model, face_img_resized, emotion_dict)

                # Draw bounding box and emotion label on the frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Convert frame to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_image = Image.fromarray(frame_rgb)

            # Display frame in Streamlit
            frame_placeholder.image(frame_image)

            # Stop if the stop button is pressed
            if stop_button:
                break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
