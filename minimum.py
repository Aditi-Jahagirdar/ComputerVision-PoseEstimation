import cv2
import mediapipe as mp
import time

#2 To create the object for the model
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

#0 To read video
cap = cv2.VideoCapture('Pose Videos/1.mp4')
pTime = 0

while True:
    success, img = cap.read()
#3 Converting bgr image (img) from cv2 to rgb image required for mediapipe library
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)    #notice the decrease in frame rate after processing
    #print(results.pose_landmarks)
#4 To draw connections when landmarks are detected
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img,results.pose_landmarks,mpPose.POSE_CONNECTIONS )
#5 To extract landmark id wise
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            #print(id, lm)
#6 To get pixel value of landmark
            cx, cy = int(lm.x*w), int(lm.y*h)
            cv2.circle(img, (cx,cy), 10, (255,0,0), cv2.FILLED )
            print(id, cx,cy)

#1 To check the frame rate
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
