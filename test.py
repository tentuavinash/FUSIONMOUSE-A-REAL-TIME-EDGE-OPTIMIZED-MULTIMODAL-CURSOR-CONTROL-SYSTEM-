# Imports

from email.mime import image
from email.mime import image
import cv2
import mediapipe as mp
import pyautogui
import math
from enum import IntEnum
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from google.protobuf.json_format import MessageToDict
import screen_brightness_control as sbcontrol

pyautogui.FAILSAFE = False
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


# ------------------- ACCURACY VARIABLES -------------------
total_predictions = 0
correct_predictions = 0
# ----------------------------------------------------------


# Gesture Encodings 
class Gest(IntEnum):
    FIST = 0
    PINKY = 1
    RING = 2
    MID = 4
    LAST3 = 7
    INDEX = 8
    FIRST2 = 12
    LAST4 = 15
    THUMB = 16    
    PALM = 31
    V_GEST = 33
    TWO_FINGER_CLOSED = 34
    PINCH_MAJOR = 35
    PINCH_MINOR = 36


class HLabel(IntEnum):
    MINOR = 0
    MAJOR = 1


# ---------------- HAND RECOGNITION CLASS ----------------
class HandRecog:
    def __init__(self, hand_label):
        self.finger = 0
        self.ori_gesture = Gest.PALM
        self.prev_gesture = Gest.PALM
        self.frame_count = 0
        self.hand_result = None
        self.hand_label = hand_label
    
    def update_hand_result(self, hand_result):
        self.hand_result = hand_result

    def get_signed_dist(self, point):
        sign = -1
        if self.hand_result.landmark[point[0]].y < self.hand_result.landmark[point[1]].y:
            sign = 1
        dist = (self.hand_result.landmark[point[0]].x - self.hand_result.landmark[point[1]].x)**2
        dist += (self.hand_result.landmark[point[0]].y - self.hand_result.landmark[point[1]].y)**2
        dist = math.sqrt(dist)
        return dist*sign
    
    def get_dist(self, point):
        dist = (self.hand_result.landmark[point[0]].x - self.hand_result.landmark[point[1]].x)**2
        dist += (self.hand_result.landmark[point[0]].y - self.hand_result.landmark[point[1]].y)**2
        return math.sqrt(dist)
    
    def get_dz(self,point):
        return abs(self.hand_result.landmark[point[0]].z - self.hand_result.landmark[point[1]].z)
    
    def set_finger_state(self):
        if self.hand_result == None:
            return

        points = [[8,5,0],[12,9,0],[16,13,0],[20,17,0]]
        self.finger = 0
        self.finger = self.finger | 0

        for idx,point in enumerate(points):
            dist = self.get_signed_dist(point[:2])
            dist2 = self.get_signed_dist(point[1:])
            
            try:
                ratio = round(dist/dist2,1)
            except:
                ratio = 0

            self.finger = self.finger << 1
            if ratio > 0.5 :
                self.finger = self.finger | 1
    

    def get_gesture(self):
        if self.hand_result == None:
            return Gest.PALM

        current_gesture = Gest.PALM

        if self.finger in [Gest.LAST3,Gest.LAST4] and self.get_dist([8,4]) < 0.05:
            current_gesture = Gest.PINCH_MINOR if self.hand_label == HLabel.MINOR else Gest.PINCH_MAJOR

        elif Gest.FIRST2 == self.finger:
            point = [[8,12],[5,9]]
            dist1 = self.get_dist(point[0])
            dist2 = self.get_dist(point[1])
            ratio = dist1/dist2

            if ratio > 1.7:
                current_gesture = Gest.V_GEST
            else:
                if self.get_dz([8,12]) < 0.1:
                    current_gesture = Gest.TWO_FINGER_CLOSED
                else:
                    current_gesture = Gest.MID
        else:
            current_gesture = self.finger
        
        if current_gesture == self.prev_gesture:
            self.frame_count += 1
        else:
            self.frame_count = 0

        self.prev_gesture = current_gesture

        if self.frame_count > 4 :
            self.ori_gesture = current_gesture

        return self.ori_gesture


# ---------------- CONTROLLER CLASS ----------------
class Controller:
    prev_hand = None

    def get_position(hand_result):
        point = 9
        position = [hand_result.landmark[point].x ,hand_result.landmark[point].y]
        sx,sy = pyautogui.size()
        x = int(position[0]*sx)
        y = int(position[1]*sy)
        return (x,y)

    def handle_controls(gesture, hand_result):
        if gesture != Gest.PALM:
            x,y = Controller.get_position(hand_result)
            pyautogui.moveTo(x, y, duration = 0.05)


# ---------------- MAIN CLASS ----------------
class GestureController:
    gc_mode = 0
    cap = None
    hr_major = None
    hr_minor = None
    dom_hand = True

    def __init__(self):
        GestureController.gc_mode = 1
        GestureController.cap = cv2.VideoCapture(0)

    def classify_hands(results):
        left , right = None,None
        try:
            handedness_dict = MessageToDict(results.multi_handedness[0])
            if handedness_dict['classification'][0]['label'] == 'Right':
                right = results.multi_hand_landmarks[0]
            else:
                left = results.multi_hand_landmarks[0]
        except:
            pass
        
        GestureController.hr_major = right
        GestureController.hr_minor = left

    def start(self):
        global total_predictions, correct_predictions
        prediction_locked = False


        handmajor = HandRecog(HLabel.MAJOR)
        handminor = HandRecog(HLabel.MINOR)

        with mp_hands.Hands(max_num_hands=2,min_detection_confidence=0.5,min_tracking_confidence=0.5) as hands:
            while GestureController.cap.isOpened() and GestureController.gc_mode:
                success, image = GestureController.cap.read()
                if not success:
                    continue
                
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = hands.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.multi_hand_landmarks:
                    GestureController.classify_hands(results)
                    handmajor.update_hand_result(GestureController.hr_major)
                    handmajor.set_finger_state()
                    gest_name = handmajor.get_gesture()

                    # -------- ACCURACY COUNTING --------
                    if not prediction_locked:
                        total_predictions += 1
                        prediction_locked = True

                    try:
                        gesture_text = Gest(gest_name).name
                    except:
                        gesture_text = str(gest_name)

                    cv2.putText(image, f"Prediction: {gesture_text}", (10,40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)


                    # -----------------------------------

                cv2.imshow('Gesture Controller', image)

                key = cv2.waitKey(5) & 0xFF

                # Press Q if prediction is correct
                if key == ord('q'):
                    if prediction_locked:
                        correct_predictions += 1
                        prediction_locked = False
                        print("Marked Correct")

                elif key == ord('w'):
                    if prediction_locked:
                        prediction_locked = False
                        print("Marked Wrong")


                # Press Enter to exit
                elif key == 13:
                    break

        GestureController.cap.release()
        cv2.destroyAllWindows()

        # -------- FINAL ACCURACY --------
        if total_predictions > 0:
            accuracy = (correct_predictions / total_predictions) * 100
            print("\nTotal Predictions:", total_predictions)
            print("Correct Predictions:", correct_predictions)
            print("Accuracy:", round(accuracy,2), "%")
        else:
            print("No predictions made.")


# Run Program
gc1 = GestureController()
gc1.start()
