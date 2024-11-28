import cv2
from ultralytics import YOLO

model = YOLO("model.pt")
cap = cv2.VideoCapture(0)

while cap.isOpened():
    
    success, frame = cap.read()

    if success:
       
        results = model.predict(frame, conf=0.5)

       
        annotated_frame = results[0].plot()
      
        cv2.imshow('Realtime Detection', annotated_frame)

       
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
       
        break


cap.release()
cv2.destroyAllWindows()