import json

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def extract_measurements(data):
    """
    Normalizes both file formats into:
    {
        "Pose": {...},
        "Shape": {...}
    }
    """
    if "Pose" in data and "Shape" in data:
        # Earlier format
        return {
            "Pose": data["Pose"],
            "Shape": data["Shape"]
        }
    else:
        # New format
        return {
            "Pose": data["pose_measurements_mm"],
            "Shape": data["shape_measurements_mm"]
        }

def compute_difference(file1_data, file2_data):
    diff = {"Pose": {}, "Shape": {}}

    for category in ["Pose", "Shape"]:
        for key in file1_data[category]:
            diff[category][key] = round(
                file2_data[category][key] - file1_data[category][key], 2
            )

    return diff

if __name__ == "__main__":
    file1_path = "synthetic_data_10_items\\synthetic_data\\measures\\measure_model_10.json"
    file2_path = "./anthropometric_results.json"

    raw1 = load_json(file1_path)
    raw2 = load_json(file2_path)

    data1 = extract_measurements(raw1)
    data2 = extract_measurements(raw2)

    difference = compute_difference(data1, data2)

    print("\nOutput (File2 - File1):")
    print(json.dumps(difference, indent=4))
