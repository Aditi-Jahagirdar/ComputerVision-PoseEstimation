import cv2
import mediapipe as mp
import time

#Creating class with method to detect pose and extract landmark
class poseDetector():

    def __init__(self, mode= False, modelComplexity= 1,smoothLm= True,
                 enableSeg= False,smoothSeg= True, detCon= 0.5, trackCon= 0.5):
        self.mode = mode
        self.modelComplexity = modelComplexity
        self.smoothLm = smoothLm
        self.enableSeg = enableSeg
        self.smoothSeg = smoothSeg
        self.detCon = detCon
        self.trackCon = trackCon

        #2 To create the object for the model
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode,self.modelComplexity,self.smoothLm,
                                     self.enableSeg,self.smoothSeg,self.detCon,
                                     self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, img, draw = True):
        # 3 Converting bgr image (img) from cv2 to rgb image required for mediapipe library
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)  # notice the decrease in frame rate after processing
        #4 To draw connections when landmarks are detected
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS )
        return img

    def findPosition(self, img , draw = True):
        #5 To extract landmark id wise
        lmList = []
        if self.results.pose_landmarks:
                for id, lm in enumerate(self.results.pose_landmarks.landmark):
                    h, w, c = img.shape
                    #print(id, lm)
        #6 To get pixel value of landmark
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    lmList.append([id,cx,cy])
                    if draw:
                        cv2.circle(img, (cx,cy), 10, (255,0,0), cv2.FILLED )
                    #print(id, cx,cy)
        return lmList

def main():
    # 0 To read video
    cap = cv2.VideoCapture('Pose Videos/1.mp4')
    pTime = 0
    while True:
        success, img = cap.read()
        detector = poseDetector()
        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw= False)
        #To print and draw landmark 14
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

if __name__ == "__main__":
    main()
