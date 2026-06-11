from ultralytics import YOLO
import os

def main():
    model_path = "runs/pose/train/weights/best.pt"

    if not os.path.exists(model_path):
        print("Error trained not found.")
        return

    model = YOLO(model_path)
    results = model.predict(
        source="raw_images",
        conf=0.6,
        save=True,
        show=True,
        show_conf=False,
        show_labels=True
    )


if __name__ == "__main__":
    main()