import cv2
import mediapipe as mp
import time
import PoseModule as pm


# 0 To read video
cap = cv2.VideoCapture('Pose Videos/1.mp4')
pTime = 0
while True:
    success, img = cap.read()
    detector = pm.poseDetector()
    img = detector.findPose(img)
    lmList = detector.findPosition(img, draw=False)
    # To print and draw landmark 14
    if len(lmList) != 0:
        print(lmList[14])
        cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (255, 0, 0), cv2.FILLED)

    # 1 To check the frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
