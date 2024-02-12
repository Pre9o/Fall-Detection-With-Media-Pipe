#@markdown To better demonstrate the Pose Landmarker API, we have created a set of visualization tools that will be used in this colab. These will draw the landmarks on a detect person, as well as the expected connections between those markers.

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import time


def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image


import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Cria um objeto VideoCapture para a câmera padrão.
cap = cv2.VideoCapture("video.mp4")

# Define a resolução da captura.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

person_fallen = False
first_fall = True

# Cria um objeto PoseLandmarker.
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = pose.process(image)

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # When everything done, text on the video window.
        cv2.putText(image, "Press 'ESC' to close", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        # Calculate the angle between the torso and the floor for people falling detection and put on the video window if the person is falling.
        if results.pose_landmarks is not None:
            landmark_list = [landmark for landmark in results.pose_landmarks.landmark]
            min_x = max(0, min(int(landmark.x * image.shape[1]) for landmark in landmark_list)) - 25
            min_y = max(0, min(int(landmark.y * image.shape[0]) for landmark in landmark_list)) - 25
            max_x = min(image.shape[1], max(int(landmark.x * image.shape[1]) for landmark in landmark_list)) + 25
            max_y = min(image.shape[0], max(int(landmark.y * image.shape[0]) for landmark in landmark_list)) + 25

            # Draw the bounding box.
            cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 0, 0), 2)


            right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
            angle_right = np.arctan2(right_shoulder.y - right_hip.y, right_shoulder.x - right_hip.x)
            angle_deg_right = np.degrees(angle_right)
            angle_deg_right = 90 - angle_deg_right  # Calculate the angle between the vertical line and the line formed by the shoulder and hip points.

            left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            angle_left = np.arctan2(left_shoulder.y - left_hip.y, left_shoulder.x - left_hip.x)
            angle_deg_left = np.degrees(angle_left)
            angle_deg_left = 90 - angle_deg_left  # Calculate the angle between the vertical line and the line formed by the shoulder and hip points.

            # cv2.putText(image, f'Angle: {angle_deg_right:.2f}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            # cv2.putText(image, f'Angle Left: {angle_deg_left:.2f}', (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Check if the person has fallen.
            if angle_deg_right < 90 or angle_deg_left < 90 and person_fallen is False:
                if first_fall:
                  start_time = time.time()
                  first_fall = False
                  print('Start time:', start_time)

                # If fall duration is greater than 1 second, the person has fallen.
                if time.time() - start_time >= 1:
                  print('Person has fallen')
                  person_fallen = True
                  
            if person_fallen:
                if angle_deg_right > 90 and angle_deg_left > 90:
                    person_fallen = False
                    first_fall = True
                    print('Person is standing')

                else:
                  # Apply a red filter to the video.
                  image[:, :, 2] = 255

                  # Add a large warning on the screen.
                  text = 'WARNING:'
                  font_scale = 3
                  thickness = 5
                  font = cv2.FONT_HERSHEY_SIMPLEX
                  text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                  text_x = (image.shape[1] - text_size[0]) // 2
                  text_y = (image.shape[0] + text_size[1]) // 2 - 100

                  text_2 = 'Person has fallen'
                  text_size_2, _ = cv2.getTextSize(text_2, font, font_scale, thickness)
                  text_x_2 = (image.shape[1] - text_size_2[0]) // 2
                  text_y_2 = text_y + 100

                  cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA, False)
                  cv2.putText(image, text_2, (text_x_2, text_y_2), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA, False)

  

        cv2.imshow('MediaPipe Pose', image) 
        if cv2.waitKey(5) & 0xFF == 27:
            break

    
cap.release()