import cv2
import mediapipe as mp
import numpy as np
import time

#  MediaPipe Face Mesh Initialization
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Drawing sketch
mp_draw_face = mp.solutions.drawing_utils
drawing_face_spec = mp_draw_face.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0)) # Green drawing spec

# Replacement Images
try:
    face_replacement_img2 = cv2.imread('Kitty.png', cv2.IMREAD_UNCHANGED)
    face_replacement_img1 = cv2.imread('Kong.png', cv2.IMREAD_UNCHANGED)

    if face_replacement_img1 is None or face_replacement_img2 is None:
        print("Error: Either one of 'em or both aren't loading.")
        print("Check the Path of them. Make sure they are in same directory.")
        exit()

    # Alpha channel to make it transparent so we can exactly replace it with the face.
    def extract_alpha(img):
        if img.shape[2] == 4: # BGRA image
            return img[:, :, 3] / 255.0, img[:, :, :3] # Alpha, BGR
        else: # BGR image (no alpha)
            return np.ones(img.shape[:2], dtype=np.float32), img # Opaque alpha, BGR

    alpha_img1, face_replacement_img1_bgr = extract_alpha(face_replacement_img1)
    alpha_img2, face_replacement_img2_bgr = extract_alpha(face_replacement_img2)

    print("Replacement images successfully loaded.")

except Exception as e:
    print(f"Error: Got error {e} while replacing images.")
    print("Check the Path of them. Make sure they are in same directory.")
    exit()

# Active Image (0 for img1, 1 for img2)
current_face_img_idx = 0
last_switch_time = time.time()

#  Video Capture Initialization
cap = cv2.VideoCapture(0) # Default camera or webCam is (0), but I'll use 1 to take input from phone(PhoneLink)
if not cap.isOpened():
    print("The Camera is getting opened.")
    exit()

ptime = 0 # Previous time for FPS calculation

#  Main Loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Not able to read the camera.")
        break

    # OpenCV takes default as BGR format, MediaPipe takes RGB so conversion is necessary
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the face
    results_face = face_mesh.process(frame_rgb)

    # Frame's dimensions
    h, w, _ = frame.shape

    #  Face Replacement Logic
    if results_face.multi_face_landmarks:

        face_landmarks = results_face.multi_face_landmarks[0]

        # Conversion of landmarks into pixel cuz these will help in swapping
        mesh_points = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark], dtype=np.int32)

        # Calculating bound of Face
        x_min, y_min = np.min(mesh_points, axis=0)
        x_max, y_max = np.max(mesh_points, axis=0)

        # Adding some padding so it covers most of the face
        # I have a diamond face so 0.15 and 0.25 covered face, but you can use it according to your face shape
        padding_x = int((x_max - x_min) * 0.15) # 15%
        padding_y = int((y_max - y_min) * 0.25) # 25%
        #now change these initially calculated values
        x_min = max(0, x_min - padding_x)
        y_min = max(0, y_min - padding_y)
        x_max = min(w, x_max + padding_x)
        y_max = min(h, y_max + padding_y)

        # Target dimensions for replacement image
        target_width = x_max - x_min
        target_height = y_max - y_min

        # Safety Check
        if target_width > 0 and target_height > 0:
            # Current active image
            if current_face_img_idx == 0:
                current_img_bgr = face_replacement_img1_bgr
                current_alpha = alpha_img1
            else:
                current_img_bgr = face_replacement_img2_bgr
                current_alpha = alpha_img2

            # Resizing the replacement image according to the face in camera
            resized_face_img = cv2.resize(current_img_bgr, (target_width, target_height), interpolation=cv2.INTER_AREA)
            resized_alpha = cv2.resize(current_alpha, (target_width, target_height), interpolation=cv2.INTER_AREA)


            # ROI (Region of Interest) where the face
            roi = frame[y_min:y_max, x_min:x_max]

            # safety check between ROI and targeted dimensions
            if roi.shape[0] == target_height and roi.shape[1] == target_width:
                # main part where the replacement actually occurs
                for c in range(0, 3): # Iterate over B, G, R channels
                    roi[:, :, c] = roi[:, :, c] * (1 - resized_alpha) + \
                                   resized_face_img[:, :, c] * resized_alpha
            else:
                print(f"⚠️ Warning: ROI shape {roi.shape} does not match target {target_height}x{target_width}. Skipping blend.")


        if  cv2.waitKey(1) & 0xFF == ord('s'): # press 's' to switch sticker
            if time.time() - last_switch_time > 0.5: # we can only switch image after 0.5 second is passed,to prevent rapid switching
                current_face_img_idx = 1 - current_face_img_idx # to switch the index of image
                print(f"🖼️ Switched to Image {current_face_img_idx + 1}")
                last_switch_time = time.time() # Update last switch time

    #  FPS Calculation
    cTime = time.time()
    fps = 1 / (cTime - ptime)
    ptime = cTime
    # Add FPS Text on Output
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Sticker on face
    cv2.imshow("Face Replacement Filter", frame)

    # This is for to Quit my Video Stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()