from ultralytics import YOLO
import os
import sys

def main():
    try:
        dog_pose_yaml = "dog_attributes.yaml"

        if not os.path.exists(dog_pose_yaml):
            print(f"Error: Dataset YAML file {dog_pose_yaml} not found.")
            sys.exit(1)

        model = YOLO("yolov8s-pose.pt")

        model.train(
        data=dog_pose_yaml,
        epochs=200,
        imgsz=640,
        device="cpu",
        batch=8,
        optimizer="Adam",
        lr0=0.002,
        lrf=0.01,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        scale=0.5,
        translate=0.1,
        )


        results = model("data2/test/images")

        for i, result in enumerate(results):
            result.save(filename=f"runs/detect/output_{i}.jpg")

    except Exception as e:
        print(f"An error has occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
