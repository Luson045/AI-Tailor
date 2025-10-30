import cv2
import numpy as np
import mediapipe as mp
# import tensorflow as tf
# import tensorflow_hub as hub
import matplotlib.pyplot as plt

# ---- CONFIG ----
IMAGE_PATH = "dataset\image1.jpg"   # path to your image
MODEL_PATH_OPENPOSE = "pose_deploy_linevec_faster_4_stages.prototxt"
WEIGHTS_PATH_OPENPOSE = "pose_iter_160000.caffemodel"

# ---- Load image ----
image = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
h, w, _ = image.shape

# ---------------------------------------------------
# 🧍‍♂️ 1. MediaPipe BlazePose
# ---------------------------------------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose_mp = mp_pose.Pose(static_image_mode=True)
results_mp = pose_mp.process(image_rgb)

mp_points = []
if results_mp.pose_landmarks:
    for lm in results_mp.pose_landmarks.landmark:
        mp_points.append((int(lm.x * w), int(lm.y * h)))

# Draw MediaPipe keypoints in GREEN
for x, y in mp_points:
    cv2.circle(image, (x, y), 4, (0, 255, 0), -1)

# ---------------------------------------------------
# 🕺 2. OpenPose (via cv2.dnn)
# ---------------------------------------------------
BODY_PARTS = {0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
              5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "RHip", 9: "RKnee",
              10: "RAnkle", 11: "LHip", 12: "LKnee", 13: "LAnkle",
              14: "REye", 15: "LEye", 16: "REar", 17: "LEar"}

net = cv2.dnn.readNetFromCaffe(MODEL_PATH_OPENPOSE, WEIGHTS_PATH_OPENPOSE)
inp = cv2.dnn.blobFromImage(image, 1.0 / 255, (368, 368), (0, 0, 0), swapRB=False, crop=False)
net.setInput(inp)
out = net.forward()

points_openpose = []
for i in range(len(BODY_PARTS)):
    heatMap = out[0, i, :, :]
    _, conf, _, point = cv2.minMaxLoc(heatMap)
    x = int(w * point[0] / out.shape[3])
    y = int(h * point[1] / out.shape[2])
    if conf > 0.2:
        points_openpose.append((x, y))
        cv2.circle(image, (x, y), 4, (0, 0, 255), -1)  # RED
    else:
        points_openpose.append(None)

# ---------------------------------------------------
# 💃 3. MoveNet (TF Hub)
# ---------------------------------------------------
# model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
# movenet = model.signatures['serving_default']

# input_image = tf.image.resize_with_pad(tf.expand_dims(image_rgb, axis=0), 192, 192)
# input_image = tf.cast(input_image, dtype=tf.int32)
# outputs = movenet(input_image)
# keypoints = outputs['output_0'].numpy()[0, 0, :, :]

# # Draw MoveNet keypoints in BLUE
# for kp in keypoints:
#     x = int(kp[1] * w)
#     y = int(kp[0] * h)
#     cv2.circle(image, (x, y), 4, (255, 0, 0), -1)

# ---------------------------------------------------
# 🖼️ Show Combined Result
# ---------------------------------------------------
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("2D Keypoints: MediaPipe (Green), OpenPose (Red), MoveNet (Blue)")
plt.axis("off")
plt.show()
