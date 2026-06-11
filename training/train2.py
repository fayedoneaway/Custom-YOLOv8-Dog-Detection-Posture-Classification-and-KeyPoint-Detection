from ultralytics import YOLO
import os
import sys

def main():
    try:
        dog_posture_yaml = "dog_posture.yaml"

        if not os.path.exists(dog_posture_yaml):
            print(f"Error: Dataset YAML file {dog_posture_yaml} not found.")
            sys.exit(1)

        model = YOLO("runs/detect/train/weights/best.pt")

        model.train(
        data=dog_posture_yaml,
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


        results = model("data/test2 /images")
        for i, result in enumerate(results):
            print(f"\nImage {i} detections:")

            boxes = result.boxes
            if boxes is None:
                print("No detections")
                continue

            for box in boxes:
                cls = int(box.cls)
                conf = float(box.conf)
                coords = box.xyxy.tolist()[0]
                names = model.names
                cls_name = names[cls]

                print(f"Class: {cls_name}, Confidence: {conf:.2f}, Box: {coords}")

        for i, result in enumerate(results):
            result.save(f"runs/detect/output_{i}.jpg")

    except Exception as e:
        print(f"An error has occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
