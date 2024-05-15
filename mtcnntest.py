import cv2
from mtcnn.mtcnn import MTCNN
detector = MTCNN()

resultvideo = cv2.VideoWriter('mtcnn.mov', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         30, (1620,1080))

cap = cv2.VideoCapture("test.mov")

while True: 
    
    #Capture frame-by-frame
    __, frame = cap.read()
    
    #Use MTCNN to detect faces
    result = detector.detect_faces(frame)
    if result != []:
        for person in result:
            bounding_box = person['box']
            keypoints = person['keypoints']
    
            cv2.rectangle(frame,
                          (bounding_box[0], bounding_box[1]),
                          (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                          (0,0,255),
                          2)
            
            cv2.circle(frame,(keypoints['left_eye']), 2, (0,155,255), 2)
            cv2.circle(frame,(keypoints['right_eye']), 2, (0,155,255), 2)
            cv2.circle(frame,(keypoints['nose']), 2, (0,155,255), 2)
            cv2.circle(frame,(keypoints['mouth_left']), 2, (0,155,255), 2)
            cv2.circle(frame,(keypoints['mouth_right']), 2, (0,155,255), 2)
            
    # Display resulting frame
    resultvideo.write(frame)
    cv2.imshow('mtcnn',frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# When everything's done, release capture
cap.release()
resultvideo.release()
cv2.destroyAllWindows()