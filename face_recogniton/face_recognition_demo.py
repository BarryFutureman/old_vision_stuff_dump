import math
import os
import sys

import cv2
import face_recognition
import numpy as np

DOWNSCALE_FACTOR = 6


def face_confidence(face_distance, face_match_threshold=0.6):
    r = (1 - face_match_threshold)
    linear_val = (1 - face_distance) / (r * 2)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + "%"
    else:
        v = (linear_val + ((1 - linear_val) * math.pow((linear_val-0.5)*2, 0.2))) * 100
        return str(round(v, 2)) + "%"


class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True

    def __init__(self):
        self.encode_faces()

    def encode_faces(self):
        for image in os.listdir("images"):
            face_image = face_recognition.load_image_file(f"images/{image}")
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append("Barry")

        print(self.known_face_names)

    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            sys.exit("Cam not opened...")

        process_frame = False
        frames_per_recognition = 2
        fpr_count = 0
        while True:
            ret, frame = video_capture.read()

            if fpr_count >= frames_per_recognition:
                process_frame = True
                fpr_count = 0
            else:
                process_frame = False
                fpr_count += 1

            if process_frame:
                # print("Process")
                resized_frame = cv2.resize(frame, (0, 0), fx=1/DOWNSCALE_FACTOR, fy=1/DOWNSCALE_FACTOR)

                self.face_locations = face_recognition.face_locations(resized_frame)
                self.face_encodings = face_recognition.face_encodings(resized_frame, self.face_locations)

                self.face_names = []
                for encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, encoding)

                    name = "???"
                    confidence = "???"

                    face_distances = face_recognition.face_distance(self.known_face_encodings, encoding)
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = face_confidence(face_distances[best_match_index])

                    self.face_names.append(f"{name} ({confidence})")

            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                top *= DOWNSCALE_FACTOR
                right *= DOWNSCALE_FACTOR
                bottom *= DOWNSCALE_FACTOR
                left *= DOWNSCALE_FACTOR

                cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 225), 1)
                # cv2.rectangle(frame, (left, bottom - 20), (right, bottom), (225, 225, 225), -1)
                cv2.putText(frame, name, (left + 6, top - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (225, 225, 225), 1)

            cv2.imshow("Face Recognition", frame)

            if cv2.waitKey(1) == ord("d"):
                return

        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()