import cv2
import mediapipe as mp
import numpy as np
from step2_lip_to_word import LipToWordPredictor
from step3_post_processing import PostProcessor
import time

def run_live_mvp():
    # 1. Initialize our Integrated Modules
    predictor = LipToWordPredictor(model_path="mvp_lip_model.pth")
    post_processor = PostProcessor(confidence_threshold=0.3)
    
    # 2. Setup MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
    
    cap = cv2.VideoCapture(0)
    
    # Storage for a 2-second "speech" window
    sequence = []
    is_recording = False
    
    # Result display variables
    last_result_text = ""
    last_result_time = 0
    
    print("--- LIVE MVP TEST ---")
    print("Hold 'SPACE' to speak, release to recognize.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Mirror for user
        frame = cv2.flip(frame, 1)
        
        # Draw Instructions
        status_color = (0, 0, 255) if is_recording else (0, 255, 0)
        status_text = "RECORDING (SILENTLY)" if is_recording else "READY - HOLD SPACE"
        cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # Draw Last Result (persists until new recording starts)
        if last_result_text:
            cv2.putText(frame, f"RESULT: {last_result_text}", (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)

        # Process Face
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        current_landmarks = None
        if results.multi_face_landmarks:
            current_landmarks = results.multi_face_landmarks[0]
            # (Optional) Draw lip dots on preview
            # for idx in [61, 291, 13, 14]: # just a few key points
            #     point = current_landmarks.landmark[idx]
            #     cv2.circle(frame, (int(point.x * frame.shape[1]), int(point.y * frame.shape[0])), 2, (255, 0, 0), -1)

        # Handle Keyboard Interaction
        key = cv2.waitKey(1) & 0xFF
        
        # Using waitKey for Space logic (Note: real integration should use pynput for background hotkeys)
        if key == ord(' '): # Press Space
            if not is_recording:
                is_recording = True
                sequence = []
                last_result_text = "" # Clear previous result when starting new recording
        
        if is_recording:
            if current_landmarks:
                # Convert landmarks to numpy array
                points = np.array([[lm.x, lm.y] for lm in current_landmarks.landmark])
                sequence.append(points)
            
            # Auto-stop if sequence gets too long (3 seconds approx)
            if len(sequence) > 50 or (key != ord(' ') and key != 255): 
                is_recording = False
                
                # --- THIS IS THE INTEGRATION PART ---
                
                # A. Send to Step 2 (The Predictor)
                candidates = predictor.predict(sequence)
                
                # B. Send to Step 3 (The Post-Processor)
                best_match = post_processor.process_prediction(candidates)
                
                # C. Final Output
                if best_match:
                    print(f"\nRecognized: {best_match['text']} ({best_match['confidence']:.2f})")
                    last_result_text = best_match['text']
                    last_result_time = time.time()
                else:
                    print("\nCould not recognize clearly.")
                
                sequence = []

        cv2.imshow('Chaplin MVP Demo', frame)
        if key == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_mvp()
