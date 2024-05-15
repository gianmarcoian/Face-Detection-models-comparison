import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

result = cv2.VideoWriter('haar.mov', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         30, (1620,1080))

cap = cv2.VideoCapture("test.mov")

while True:
    
    #Read the frame
    _, img = cap.read()

    #Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw the "box" around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    result.write(img)
    # Display images
    cv2.imshow('haar', img)

    # Stop if escape key is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
        
#release the VideoCapture object
cap.release()
result.release()
cv2.destroyAllWindows()
