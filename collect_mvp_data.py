import cv2
import mediapipe as mp
import numpy as np
import os
import time

# Phrases for the MVP set
PHRASES = ["HELLO", "YES", "NO", "STOP", "HELP"]
SAMPLES_PER_PHRASE = 10 
RECORDING_DURATION = 2.0 # seconds

# MediaPipe FaceMesh indices for Lips
# Outer lips contours
LIP_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78,
    185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191
]

def collect_data():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
    cap = cv2.VideoCapture(0)
    
    if not os.path.exists("mvp_data"):
        os.makedirs("mvp_data")

    for phrase in PHRASES:
        for i in range(SAMPLES_PER_PHRASE):
            print(f"\nPREPARING: '{phrase}' (Sample {i+1}/{SAMPLES_PER_PHRASE})")
            print("Press 'S' to start recording...")
            
            while True:
                ret, frame = cap.read()
                cv2.putText(frame, f"SAY: {phrase} ({i+1})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Press 'S' to Start", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                cv2.imshow('Data Collector', frame)
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    break
            
            # Start Recording
            sequence = []
            start_time = time.time()
            print("RECORDING...")
            
            while time.time() - start_time < RECORDING_DURATION:
                ret, frame = cap.read()
                results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    # Extract only lip points and normalize relative to mouth center
                    lip_points = np.array([[landmarks[idx].x, landmarks[idx].y] for idx in LIP_INDICES])
                    sequence.append(lip_points)
                
                # Visual countdown
                remaining = RECORDING_DURATION - (time.time() - start_time)
                cv2.putText(frame, f"RECORDING: {remaining:.1f}s", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Data Collector', frame)
                cv2.waitKey(1)
            
            # Save sequence
            if len(sequence) > 0:
                filename = f"mvp_data/{phrase}_{int(time.time())}.npy"
                np.save(filename, np.array(sequence))
                print(f"Saved to {filename}")
            else:
                print("No landmarks detected. Retrying...")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    collect_data()
