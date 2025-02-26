import cv2
import numpy as np
import mediapipe as mp

camera = cv2.VideoCapture(0)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5, max_num_faces=3)
LEFT_EYE = [474, 475, 476, 477]
RIGHT_EYE = [469, 470, 471, 472]
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361,
             288, 397, 365, 379, 378, 400, 377, 152, 148, 176,
             149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

if not camera.isOpened():
    print("Camera is not opened")
    exit()

while True:
    flag, frame = camera.read()
    if not flag:
        print("Camera is not opened")
        break

    resCamera = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(resCamera, cv2.COLOR_BGR2RGB)

    resFaces = face_mesh.process(rgb)

    if resFaces.multi_face_landmarks:
        img_height, img_width, _ = resCamera.shape

        for faceLandmark in resFaces.multi_face_landmarks:
            Coord = np.array([np.multiply([p.x, p.y], [img_width, img_height]).astype(int)
                                for p in faceLandmark.landmark])

            for i in range(len(FACE_OVAL)-1):
                cv2.line(resCamera, tuple(Coord[FACE_OVAL[i+1]]),
                         tuple(Coord[FACE_OVAL[i]]), (0, 0, 255), 4)
                cv2.line(resCamera, tuple(Coord[FACE_OVAL[-1]]),
                         tuple(Coord[FACE_OVAL[0]]), (0, 0, 255), 4)

            if len(Coord) >= 478:
                (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(Coord[LEFT_EYE])
                center_left = np.array([l_cx, l_cy], dtype=np.int32)
                cv2.circle(resCamera, center_left, int(l_radius), (1, 28, 60), 2, cv2.LINE_AA)

                (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(Coord[RIGHT_EYE])
                center_right = np.array([l_cx, l_cy], dtype=np.int32)
                cv2.circle(resCamera, center_right, int(l_radius), (1, 28, 60), 2, cv2.LINE_AA)

    cv2.imshow('Face Mesh', resCamera)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()