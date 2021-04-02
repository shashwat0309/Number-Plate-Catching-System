import cv2 as cv
from datetime import datetime
noplate_cascade = cv.CascadeClassifier('resources\haarcascade_russian_plate_number.xml')
cap = cv.VideoCapture(0)
cap.set(3,1080)
cap.set(4,740)
cap.set(10,150)
count = 1
while True:
    success, img = cap.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    noplate = noplate_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in noplate:
        area = w*h
        img = cv.rectangle(img, (x, y), (x + w, y + h), (36,255,12), 2)
        cv.putText(img, 'Number Plate', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    if cv.waitKey(1) & 0xFF == ord('s'):
        cv.imwrite("scanned/noplate_"+str(count)+".jpg",img)
        img = cv.rectangle(img, (x, y), (x + w, y + h), (0,0,255), 2)
        cv.putText(img, 'Img Saved', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        cv.imshow('Number Plate',img)
        cv.waitKey(5000)
        count+=1

    cv.imshow('Number Plate',img)