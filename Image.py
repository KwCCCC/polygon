import numpy as np
import cv2

class Image:
    
    def __init__(self):
        self.image = None
        self.dir = None
        self.contourCenterX = 0
        self.MainContour = None
        
    def Process(self):
        imgray = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY) #Convert to Gray Scale
        # ret, thresh = cv2.threshold(imgray,120, 255,cv2.THRESH_BINARY_INV) #Get Threshold

        # 임계값 설정 (반사된 부분을 구분할 기준)
        threshold_value = 120
        _, mask = cv2.threshold(imgray, threshold_value, 255, cv2.THRESH_BINARY_INV)

        # 마스크 다듬기 - 잡음 제거
        kernel = np.ones((7, 7), np.uint8)  # 5x5 크기의 커널 생성
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # MORPH_OPEN: 작은 점을 제거

        # 마스크 다듬기 - 구멍 메우기
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)  # MORPH_CLOSE: 구멍 메우기
        cv2.imshow("Mask Cleaned", mask_cleaned)

        self.contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE) #Get contour ####CHAIN_APPROX_SIMPLE로 바꿔야함
        
        self.prev_MC = self.MainContour
        if self.contours:
            self.MainContour = max(self.contours, key=cv2.contourArea)
        
            self.height, self.width  = self.image.shape[:2]

            self.middleX = int(self.width/2) #Get X coordenate of the middle point
            self.middleY = int(self.height/2) #Get Y coordenate of the middle point
            
            self.prev_cX = self.contourCenterX
            if self.getContourCenter(self.MainContour) != 0:
                self.contourCenterX = self.getContourCenter(self.MainContour)[0]
                if abs(self.prev_cX-self.contourCenterX) > 5: #이전 중심과 x좌표 차이가 5보다 크면, 5보다 작은놈 있으면 대체.
                    self.correctMainContour(self.prev_cX)
            else:
                self.contourCenterX = 0
            
            self.dir =  int((self.middleX-self.contourCenterX) * self.getContourExtent(self.MainContour)) # 곡률에 비례해 dir 설정
            
            cv2.drawContours(self.image,self.MainContour,-1,(0,255,0),3) #Draw Contour GREEN
            #print(self.MainContour)
            cv2.circle(self.image, (self.contourCenterX, self.middleY), 7, (255,255,255), -1) #Draw dX circle WHITE
            cv2.circle(self.image, (self.middleX, self.middleY), 3, (0,0,255), -1) #Draw middle circle RED
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.image,str(self.middleX-self.contourCenterX),(self.contourCenterX+20, self.middleY), font, 1,(200,0,200),2)
            cv2.putText(self.image,"Weight:%.3f"%self.getContourExtent(self.MainContour),(self.contourCenterX+20, self.middleY+35), font, 0.5,(200,0,200),1)
        
    def getContourCenter(self, contour):
        M = cv2.moments(contour)
        
        if M["m00"] == 0:
            return 0
        
        x = int(M["m10"]/M["m00"])
        y = int(M["m01"]/M["m00"])
        
        return [x,y]
        
    def getContourExtent(self, contour): #bounding box Area / contour area
        area = cv2.contourArea(contour)
        x,y,w,h = cv2.boundingRect(contour)
        rect_area = w*h
        if rect_area > 0:
            return 1-(float(area)/rect_area)
            
    def Aprox(self, a, b, error): #차이 작니
        if abs(a - b) < error:
            return True
        else:
            return False
            
    def correctMainContour(self, prev_cx): #너무 많이 바뀌었어 다른놈 데려와
        if abs(prev_cx-self.contourCenterX) > 5:
            for i in range(len(self.contours)):
                if self.getContourCenter(self.contours[i]) != 0:
                    tmp_cx = self.getContourCenter(self.contours[i])[0]
                    if self.Aprox(tmp_cx, prev_cx, 5) == True:
                        self.MainContour = self.contours[i]
                        if self.getContourCenter(self.MainContour) != 0:
                            self.contourCenterX = self.getContourCenter(self.MainContour)[0]
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
