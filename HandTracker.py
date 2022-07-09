import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model

CLASSES_NAMES_PATH = r'/Users/itayshukrun/Desktop/Python/HandTracking/hand-gesture-recognition-code/gesture.names'
MODEL_PATH = r"/Users/itayshukrun/Desktop/Python/HandTracking/hand-gesture-recognition-code/mp_hand_gesture"
CAMERA_NUMBER = 0
LEFT_HAND = "Left"
RIGHT_HAND = "Right"
FLIP_VERTICALLY = 1
WAIT_TIME = 1


class HandTracker:

    def __init__(self, model, class_names):
        self.model = model
        self.class_names = class_names
        self.mp_hands = mp.solutions.hands
        self.cap = cv2.VideoCapture(CAMERA_NUMBER)
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands()
        self.left_hand_location = None
        self.right_hand_location = None

    def run(self):
        while True:
            success, img = self.cap.read()
            img = cv2.flip(img, FLIP_VERTICALLY)
            x, y, c = img.shape
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(imgRGB)
            if results.multi_hand_landmarks:
                landmarks = []
                for handslms in results.multi_hand_landmarks:
                    for lm in handslms.landmark:
                        self.mp_draw.draw_landmarks(img, handslms, self.mp_hands.HAND_CONNECTIONS)
                        lmx = int(lm.x * x)
                        lmy = int(lm.y * y)
                        landmarks.append([lmx, lmy])
                    prediction = self.model.predict([landmarks])
                    classID = np.argmax(prediction)
                    class_name = self.class_names[classID]
                    print(class_name)
            if results.multi_handedness:
                if RIGHT_HAND in str(results.multi_handedness[0]):
                    self.left_hand_location = results.multi_hand_landmarks
                    print(RIGHT_HAND)
                else:
                    self.right_hand_location = results.multi_hand_landmarks
                    print(LEFT_HAND)
            cv2.imshow("Image", img)
            cv2.waitKey(WAIT_TIME)


def main():
    model = load_model(MODEL_PATH)
    class_names = get_class_names()
    hand_tracker = HandTracker(model, class_names)
    hand_tracker.run()


def get_class_names():
    f = open(CLASSES_NAMES_PATH, 'r')
    class_names = f.read().split('\n')
    f.close()
    return class_names


if __name__ == '__main__':
    main()
