import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def distance(p1, p2):
    """Calculate Euclidean distance between two 2D points"""
    return np.linalg.norm(np.array(p1) - np.array(p2))


def determine_shirt_size(chest_cm, shoulder_cm, waist_cm):
    """
    Estimate shirt size using approximate Indian/International standards.
    """
    size_chart = {
        "S": {"chest": (86, 91), "shoulder": (43, 45), "waist": (76, 81)},
        "M": {"chest": (92, 99), "shoulder": (45, 47), "waist": (82, 89)},
        "L": {"chest": (100, 107), "shoulder": (47, 49), "waist": (90, 97)},
        "XL": {"chest": (108, 115), "shoulder": (49, 51), "waist": (98, 105)},
        "XXL": {"chest": (116, 123), "shoulder": (51, 53), "waist": (106, 113)},
    }

    scores = {}
    for size, limits in size_chart.items():
        chest_score = abs(np.mean(limits["chest"]) - chest_cm)
        shoulder_score = abs(np.mean(limits["shoulder"]) - shoulder_cm)
        waist_score = abs(np.mean(limits["waist"]) - waist_cm)
        total = chest_score * 0.5 + shoulder_score * 0.3 + waist_score * 0.2
        scores[size] = total

    best_size = min(scores, key=scores.get)
    return best_size


def estimate_measurements(image_path, person_height_cm=170):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read image.")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
        results = pose.process(image_rgb)
        if not results.pose_landmarks:
            print("No person detected.")
            return

        h, w, _ = image.shape
        landmarks = results.pose_landmarks.landmark

        # Extract key landmarks
        l_shoulder = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w,
                      landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h)
        r_shoulder = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h)
        l_hip = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * w,
                 landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * h)
        r_hip = (landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * w,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * h)
        l_chest = (landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x * w,
                   landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h)
        r_chest = (landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x * w,
                   landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h)
        l_ankle = (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x * w,
                   landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y * h)
        r_ankle = (landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x * w,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y * h)

        # Calculate distances in pixels
        shoulder_width_px = distance(l_shoulder, r_shoulder)
        chest_width_px = distance(l_chest, r_chest)
        hip_width_px = distance(l_hip, r_hip)
        height_px = (landmarks[mp_pose.PoseLandmark.NOSE].y * h -
                     (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y * h))

        # Convert to cm
        px_to_cm = person_height_cm / height_px
        shoulder_cm = shoulder_width_px * px_to_cm
        chest_cm = chest_width_px * px_to_cm * 2.0  # approximate circumference
        hip_cm = hip_width_px * px_to_cm * 2.0
        waist_cm = (shoulder_cm + hip_cm) / 2 * 0.8

        # Determine size
        size = determine_shirt_size(chest_cm, shoulder_cm, waist_cm)

        # Draw landmarks
        annotated = image.copy()
        mp_drawing.draw_landmarks(annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(8, 10))
        plt.imshow(annotated_rgb)
        plt.axis('off')
        plt.title(f"Detected Size: {size}")
        plt.show()

        print("\n📏 Estimated Measurements:")
        print(f"Height (input): {person_height_cm:.2f} cm")
        print(f"Shoulder Width: {shoulder_cm:.2f} cm")
        print(f"Chest Circumference: {chest_cm:.2f} cm")
        print(f"Waist Width: {waist_cm:.2f} cm")
        print(f"Hip Circumference: {hip_cm:.2f} cm")
        print(f"\n👕 Recommended Shirt Size: {size}")

        return {
            "height": person_height_cm,
            "shoulder_width": shoulder_cm,
            "chest_circumference": chest_cm,
            "waist_width": waist_cm,
            "hip_circumference": hip_cm,
            "recommended_size": size
        }


# Example usage
if __name__ == "__main__":
    image_path = "luson.jpg"  # replace with your image path
    estimate_measurements(image_path, person_height_cm=168)
