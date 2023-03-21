import cv2
import numpy as np
import mediapipe as mp 
from mediapipe.python.solutions.drawing_utils import draw_landmarks
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface,POINTER(IAudioEndpointVolume))
minVol, maxVol = volume.GetVolumeRange()[:2]

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                #print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        if lmList:
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]

            cv2.circle(img, (x1,y1), 15, (255,0,255), cv2.FILLED)
            cv2.circle(img, (x2,y2), 15, (255,0,255), cv2.FILLED)
            cv2.line(img, (x1,y1), (x2,y2), (255,0,0), 3)

            z1, z2 = (x1+x2)//2 , (y1+y2)//2
            length = hypot(x2-x1 , y2-y1)
            if length < 50 :
                cv2.circle(img, (z1,z2), 10, (255,255,255), cv2.FILLED) 

        vol = np.interp(length, [50,300],[minVol,maxVol])
        volBar = np.interp(length, [50,300], [400,150])
        volPer = np.interp(length, [50,300], [0,100])
        volume.SetMasterVolumeLevel(vol, None)
        # length = 50 ===> volPer = 0
        # length = 300 ===> volPer = 100
        # print(vol)
        # print(int(length))
        # print(minVol, maxVol)
        cv2.rectangle(img, (50,150), (85,400), (123,213,122), 3)
        cv2.rectangle(img, (50, int(volBar)), (85,400), (0,231,23), cv2.FILLED)
        cv2.putText(img, str(int(volPer)), (40,450), cv2.FONT_HERSHEY_PLAIN, 4, (123,213,122), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)   
    # length of line === 50,300
    # range of sound === -21,20
    # range of actual volume === 0,100