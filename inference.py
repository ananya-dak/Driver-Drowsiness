import cv2
import mediapipe as mp
import math
import numpy as np
import os
import time
import torch
import led_blink
import RPi.GPIO as GPIO

# Setup GPIO for Raspberry Pi LED
GPIO.setmode(GPIO.BCM)
GPIO.setup(26, GPIO.OUT)

# Initialize video capture
cap = cv2.VideoCapture(0)  # camera port 0

# Eye and mouth landmarks (MediaPipe IDs)
right_eye = [[33, 133], [160, 144], [159, 145], [158, 153]]
left_eye = [[263, 362], [387, 373], [386, 374], [385, 380]]
mouth = [[61, 291], [39, 181], [0, 17], [269, 405]]
states = ['alert', 'drowsy']

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.3, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Load the PyTorch LSTM model
model_lstm_path = 'lstmmodelgpu.pth'
model = torch.jit.load(model_lstm_path)
model.eval()

# ----------------- Utility Functions ------------------

def distance(p1, p2):
    return (((p1[:2] - p2[:2]) ** 2).sum()) ** 0.5

def eye_aspect_ratio(landmarks, eye):
    N1 = distance(landmarks[eye[1][0]], landmarks[eye[1][1]])
    N2 = distance(landmarks[eye[2][0]], landmarks[eye[2][1]])
    N3 = distance(landmarks[eye[3][0]], landmarks[eye[3][1]])
    D = distance(landmarks[eye[0][0]], landmarks[eye[0][1]])
    return (N1 + N2 + N3) / (3 * D)

def eye_feature(landmarks):
    return (eye_aspect_ratio(landmarks, left_eye) + eye_aspect_ratio(landmarks, right_eye)) / 2

def mouth_feature(landmarks):
    N1 = distance(landmarks[mouth[1][0]], landmarks[mouth[1][1]])
    N2 = distance(landmarks[mouth[2][0]], landmarks[mouth[2][1]])
    N3 = distance(landmarks[mouth[3][0]], landmarks[mouth[3][1]])
    D = distance(landmarks[mouth[0][0]], landmarks[mouth[0][1]])
    return (N1 + N2 + N3) / (3 * D)

def pupil_circularity(landmarks, eye):
    perimeter = sum([
        distance(landmarks[eye[0][0]], landmarks[eye[1][0]]),
        distance(landmarks[eye[1][0]], landmarks[eye[2][0]]),
        distance(landmarks[eye[2][0]], landmarks[eye[3][0]]),
        distance(landmarks[eye[3][0]], landmarks[eye[0][1]]),
        distance(landmarks[eye[0][1]], landmarks[eye[3][1]]),
        distance(landmarks[eye[3][1]], landmarks[eye[2][1]]),
        distance(landmarks[eye[2][1]], landmarks[eye[1][1]]),
        distance(landmarks[eye[1][1]], landmarks[eye[0][0]])
    ])
    area = math.pi * ((distance(landmarks[eye[1][0]], landmarks[eye[3][1]]) * 0.5) ** 2)
    return (4 * math.pi * area) / (perimeter ** 2)

def pupil_feature(landmarks):
    return (pupil_circularity(landmarks, left_eye) + pupil_circularity(landmarks, right_eye)) / 2

def run_face_mp(image):
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        landmarks_positions = []
        for data_point in results.multi_face_landmarks[0].landmark:
            landmarks_positions.append([data_point.x, data_point.y, data_point.z])
        landmarks_positions = np.array(landmarks_positions)
        landmarks_positions[:, 0] *= image.shape[1]
        landmarks_positions[:, 1] *= image.shape[0]

        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

        ear = eye_feature(landmarks_positions)
        mar = mouth_feature(landmarks_positions)
        puc = pupil_feature(landmarks_positions)
        moe = mar / ear
    else:
        ear = mar = puc = moe = -1000

    return ear, mar, puc, moe, image

def calibrate(calib_frame_count=25):
    ears, mars, pucs, moes = [], [], [], []
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        ear, mar, puc, moe, image = run_face_mp(image)

        if ear != -1000:
            ears.append(ear)
            mars.append(mar)
            pucs.append(puc)
            moes.append(moe)

        cv2.putText(image, "Calibration", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
        cv2.imshow('Calibration Window', image)

        if cv2.waitKey(5) & 0xFF == ord("q") or len(ears) >= calib_frame_count:
            break

    cap.release()
    cv2.destroyAllWindows()

    return [np.mean(ears), np.std(ears)], [np.mean(mars), np.std(mars)], \
           [np.mean(pucs), np.std(pucs)], [np.mean(moes), np.std(moes)]

def get_classification(input_data):
    model_input = [input_data[i:i+5] for i in range(0, len(input_data), 3)]
    model_input = torch.FloatTensor(np.array(model_input))
    preds = torch.sigmoid(model(model_input)).gt(0.5).int().data.numpy()
    print(preds)
    return int(preds.sum() >= 5)

def infer(ears_norm, mars_norm, pucs_norm, moes_norm):
    ear_main = mar_main = puc_main = moe_main = -1000
    decay = 0.9
    input_data = []
    label = None

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        ear, mar, puc, moe, image = run_face_mp(image)

        if ear != -1000:
            ear = (ear - ears_norm[0]) / ears_norm[1]
            mar = (mar - mars_norm[0]) / mars_norm[1]
            puc = (puc - pucs_norm[0]) / pucs_norm[1]
            moe = (moe - moes_norm[0]) / moes_norm[1]

            if ear_main == -1000:
                ear_main, mar_main, puc_main, moe_main = ear, mar, puc, moe
            else:
                ear_main = decay * ear_main + (1 - decay) * ear
                mar_main = decay * mar_main + (1 - decay) * mar
                puc_main = decay * puc_main + (1 - decay) * puc
                moe_main = decay * moe_main + (1 - decay) * moe

            input_data.append([ear_main, mar_main, puc_main, moe_main])
            if len(input_data) > 20:
                input_data.pop(0)

            if len(input_data) == 20:
                label = get_classification(input_data)

        # Display info
        cv2.putText(image, f"EAR: {ear_main:.2f}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(image, f"MAR: {mar_main:.2f}", (220, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(image, f"PUC: {puc_main:.2f}", (410, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(image, f"MOE: {moe_main:.2f}", (600, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        if label is not None:
            color = (0, 255, 0) if label == 0 else (0, 0, 255)
            cv2.putText(image, states[label], (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)
            if label == 1:
                led_blink.bb()

        cv2.imshow('Driver Drowsiness Detector', image)
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# ----------------- Main Execution ------------------

if __name__ == "__main__":
    print('Starting calibration. Please be in a neutral state.')
    time.sleep(1)
    ears_norm, mars_norm, pucs_norm, moes_norm = calibrate()
    print('Starting main application')
    time.sleep(1)
    infer(ears_norm, mars_norm, pucs_norm, moes_norm)
    face_mesh.close()
    GPIO.cleanup()
