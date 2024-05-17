import cv2
import numpy as np

# The network is loaded using cv2.dnn.readNetFromCaffe and the model's layers and weights as passed its arguments.
modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "models/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

result = cv2.VideoWriter('dnn.mov', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         30, (1620,1080))

cap = cv2.VideoCapture("test.mov")

while True:
    
    # Read the frame
    _, img = cap.read()

    # Height and width of the image are extracted
    h, w = img.shape[:2]

    # To achieve the best accuracy I ran the model on BGR images resized to 300x300 
    # applying mean subtraction of values (104, 177, 123) for each blue, green and red channels correspondingly.
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,(300, 300), (104.0, 117.0, 123.0))
    net.setInput(blob)

    # Face detection is performed
    faces = net.forward()

    # For each face detected we draw a bounding box to localize it
    for j in range(faces.shape[2]):
        confidence = faces[0, 0, j, 2]
        if confidence > 0.5:
            box = faces[0, 0, j, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            cv2.rectangle(img, (x, y), (x1, y1), (0,0,255), 2)
            
    # Display
    cv2.imshow('dnn', img)
    result.write(img)

    # Stop if escape key is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
        
# Release the VideoCapture object
cap.release()
result.release()
cv2.destroyAllWindows()