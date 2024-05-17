import cv2
import cvlib as cv

cap = cv2.VideoCapture("test.mov")

result = cv2.VideoWriter("cvlib.mov", 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         30, (1620,1080))

while True:
    
    # Read the frame
    _, img = cap.read()

    # Detect the faces
    faces, confidences = cv.detect_face(img) 

    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (w, h), (255, 0, 0), 2)
        cv2.putText(img, str(confidences), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Display
    result.write(img)
    cv2.imshow('cvlib', img)
    
    # Stop if escape key is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
        
# Release the VideoCapture object
cap.release()
result.release()
cv2.destroyAllWindows()