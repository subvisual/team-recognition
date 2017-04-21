#!/usr/bin/env python2

import face_recognition
import os
import sys
import cv2
import numpy as np
from joblib import Parallel, delayed
import multiprocessing

# import time
# from tempfile import NamedTemporaryFile
# import wave
# from gtts import gTTS
# from pygame import mixer
# mixer.init()

# tts = gTTS(text="Hello World", lang="en", slow=True)

# tts.save("hello.mp3")
# f = NamedTemporaryFile()
# tts.write_to_fp(f)
# mixer.music.load('hello.mp3')
# mixer.music.play()
# time.sleep(5)
# f.close()


team = []

num_cores = multiprocessing.cpu_count()

def load_team_member(name):
    files = list(os.walk("team/" + name))[0][2]
    pretty_name = name.replace("-", " ").title(),
    files = list(map(lambda f: "team/" + name + "/" + f, files))
    print(pretty_name)

    known_faces = []
    for file in files:
        print("opening " + file)
        image = face_recognition.load_image_file(file)
        known_faces = face_recognition.face_encodings(image)
        if len(known_faces) > 1:
            print("Warning: more than one face found on seed data: " + file)
        elif len(known_faces) == 0:
            print("Warning: no faces found on seed data: " + file)
        else:
            known_faces.append(known_faces[0])

    return {
            'name': pretty_name[0],
            'files': files,
            'faces': known_faces,
    }


team = Parallel(n_jobs=num_cores)(delayed(load_team_member)(i) for i in os.listdir("team"))

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # input_image = face_recognition.load_image_file(arg)
    input_locations = face_recognition.face_locations(small_frame)
    input_faces = face_recognition.face_encodings(small_frame, input_locations)
# input_face = input_faces[1]
# input_location = input_locations[1]
    # frame = cv2.imread(arg, cv2.IMREAD_UNCHANGED)

    print("Found " + str(len(input_faces)) + " face(s)")

    for idx, (input_face, input_location) in enumerate(zip(input_faces, input_locations)):
        print("\n-- Matching face #" + str(idx))
        matches = []

        best_match_ratio = 0
        best_match = None

        for member in team:
            results = face_recognition.compare_faces(member["faces"], input_face)
            matches = results.count(True)
            misses = results.count(False)

            ratio = matches / float(matches + misses);

            print("Match for " + member["name"] + ": " + str(ratio * 100))

            if ratio > best_match_ratio:
                best_match_ratio = ratio
                best_match = member

        if best_match:
            print("Best match: " + best_match["name"] + " (" + str(best_match_ratio * 100) + "%)")

            (top, right, bottom, left) = input_location
            top -= 20
            left -= 20
            right += 20
            bottom += 20
            top *= 4
            left *= 4
            right *= 4
            bottom *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 15), (right, bottom), (0, 0, 255), -1)
            cv2.putText(frame, best_match["name"], (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
        else:
            print("No match :(")

    cv2.imshow('image', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
