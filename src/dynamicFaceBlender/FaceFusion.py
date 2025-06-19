import cv2 as cv2
import insightface
from insightface.app import FaceAnalysis

# Model loading
app = FaceAnalysis(name='buffalo_l')

# IMP -> ctx_id when -1 CPU is used for calculation if you have a good CPU then go for it
# and when it is 0 GPU is used, but you will need to download (CUDA Toolkit) to run it, so the choice is yours
app.prepare(ctx_id = 0)

# you have to download model
# here is the link below.
# https://drive.google.com/file/d/1krOLgjW2tAPaqV-Bw4YALz0xT5zlb5HF/view
swapper = insightface.model_zoo.get_model(r"C:\Users\priya\.insightface\models\inswapper_128.onnx")

src_img = cv2.imread("RDJ.jpeg")

# we might have more faces, but for safety we will ensure we have only one
src_Faces = app.get(src_img)
if not src_Faces :
    raise Exception("No face detected in Source Image.")

src_face = src_Faces[0]

# opening the webcam
cap = cv2.VideoCapture(1)  # again, it's up to you what you choose take as input source I'll take external input.
while cap.isOpened():
    ret, frame = cap.read()
    if ret is None:
        print("Error in taking input")
        break
    faces = app.get(frame)
    for face in faces:
        try:
            frame = swapper.get(frame,face, src_face, paste_back=True)
        except Exception as e:
            print(e)

    cv2.imshow("Swapped window",cv2.flip(frame,1))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Resources release
cap.release()
cv2.destroyAllWindows()
