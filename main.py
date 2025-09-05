import cv2
import numpy as np
cap = cv2.VideoCapture('http://192.168.1.3:8080/video')

def empty(a):
   pass
cv2.namedWindow("parameters")
cv2.createTrackbar("Threshold 1","parameters",150,255,empty)
cv2.createTrackbar("Threshold 2","parameters",255,255,empty)
cv2.createTrackbar("Area", "parameters", 5000, 30000, empty)


def get_limits(color):
    c = np.uint8([[color]])
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    hue = hsvC[0][0][0]
    if hue >= 165:  
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([180, 255, 255], dtype=np.uint8)
    elif hue <= 15:
        lowerLimit = np.array([0, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)
    else:
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)

    return lowerLimit, upperLimit


lr,ur = get_limits([0,0,255])
lg,ug = get_limits([0,255,0])
lb,ub = get_limits([255,0,0])
colordict = {'red':(lr ,ur) , 'green': (lg , ug) , 'blue':(lb, ub)}

classification = {
    ('triangle', 'red'): 'Dangerous obstacle',
    ('square', 'blue'): 'Boundary marker',
    ('circle', 'green'): 'Safe zone'
}

def detect_color(hsv_image, contour):
    mask = np.zeros(hsv_image.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [contour], -1, 255, -1)
    
    max_pixels = 0
    dominant_color = 'unknown'

    for color, (lower, upper) in colordict.items():
        color_mask = cv2.inRange(hsv_image, lower, upper)        
        combined_mask = cv2.bitwise_and(mask, color_mask)
        pixels = cv2.countNonZero(combined_mask)
        if pixels > max_pixels:
            max_pixels = pixels
            dominant_color = color
            
    return dominant_color


while(True):
   _, img = cap.read()
   img = cv2.flip(img , 1)
   hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
   threshold1 = cv2.getTrackbarPos("Threshold 1","parameters")
   threshold2 = cv2.getTrackbarPos("Threshold 2","parameters")
   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   blurred = cv2.GaussianBlur(gray, (5, 5), 0)
   imgcanny = cv2.Canny(gray,threshold1,threshold2)
   kernel = np.ones((5, 5))
   imgDil = cv2.dilate(imgcanny, kernel, iterations=1)
   contours,hierarchy = cv2.findContours(imgDil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
   for cnt in contours:
      x1,y1 = cnt[0][0]
      shape = "Unknown"
      area = cv2.contourArea(cnt)
      areaMin = cv2.getTrackbarPos("Area", "parameters")
      if area > areaMin:
         approx = cv2.approxPolyDP(cnt, 0.04*cv2.arcLength(cnt, True), True)
         x, y, w, h = cv2.boundingRect(approx)
         cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
         if len(approx) == 4:
            ratio = float(w)/h
            if ratio >= 0.9 and ratio <= 1.1:
               cv2.putText(img, 'Square', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
               shape = "square"
            else:
               cv2.putText(img, 'Rectangle', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
               shape = "rectangle"
         elif len(approx)== 3:
            shape = "triangle"
            cv2.putText(img, 'Triangle', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
         else :
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
               cv2.putText(img, 'Unknown', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            circularity = 4 * np.pi * (area / (perimeter ** 2))
            if circularity > 0.7:
               cv2.putText(img, 'Circle', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
               shape = "circle"
         
      color = detect_color(hsv,cnt)
      label = classification.get((shape, color))
      M = cv2.moments(cnt)
      if M["m00"] != 0:
         cX = int(M["m10"] / M["m00"])
         cY = int(M["m01"] / M["m00"])
         cv2.putText(img, label, (cX - 50, cY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

   cv2.imshow("Shapes", img)
   if cv2.waitKey(1) == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()