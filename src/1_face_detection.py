import cv2
import mediapipe as mp
import time

faceDetector = mp.solutions.face_detection
drawing = mp.solutions.drawing_utils


# For webcam input:
cap = cv2.VideoCapture(1)

with faceDetector.FaceDetection(min_detection_confidence=0.7) as face_detection:

  while cap.isOpened():
    # reading the image frame wise from the webcam
    success, image = cap.read()

    start = time.time()

    if not success:
      print("check video capture input")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Convert the BGR image to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    
    # finding faces in the images
    results = face_detection.process(image)

    # converting the image color from RGB to BGR so that it can be displayed
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
 
    if results.detections:
      for id, detection in enumerate(results.detections):
        drawing.draw_detection(image, detection)
        print(id, detection)

        bBox = detection.location_data.relative_bounding_box

        h, w, c = image.shape

        boundingBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)

        cv2.putText(image, f'{int(detection.score[0]*100)}%', (boundingBox[0], boundingBox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)

    end = time.time()
    totalTime = end - start

    fps = 1 / totalTime
    print("FPS: ", fps)

    cv2.putText(image, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)

    cv2.imshow('MediaPipe Face Detection', image)

    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()