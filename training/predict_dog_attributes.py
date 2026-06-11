from ultralytics import YOLO
import os

def main():
    model_path = "runs/pose/train-3/weights/best.pt"

    if not os.path.exists(model_path):
        print("Error trained not found.")
        return

    model = YOLO(model_path)
    results = model.predict(
        source="raw_images",
        conf=0.5,
        save=True,
        show=True,
        show_conf=False,
        show_labels=True
    )

    for r in results:
        r.show()

if __name__ == "__main__":
    main()