from ultralytics import YOLO
import os

def main():
    model_path = "runs/detect/train-2/weights/best.pt"

    if not os.path.exists(model_path):
        print("Error trained not found.")
        return

    model = YOLO(model_path)
    results = model.predict(
        source="data/test2/images",
        conf=0.5,
        save=True,
        show=True,
        show_conf=False,
        show_labels=True
    )


if __name__ == "__main__":
    main()