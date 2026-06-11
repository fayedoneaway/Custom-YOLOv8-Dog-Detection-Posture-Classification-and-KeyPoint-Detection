import json, os
from PIL import Image

# print(os.path.exists(r"C:\Users\fayed\PycharmProjects\workone\data2\test"))
# print(os.path.exists(r"C:\Users\fayed\PycharmProjects\workone\data2\test\f6db0e21-4TOIZTXMSLK9.jpg"))

json_path = "project3_export.json"
output_dir = "labels3_yolo_pose"
image_dir = r"C:\Users\fayed\PycharmProjects\workone\data2\train\images"
# for f in os.listdir(r"C:\Users\fayed\PycharmProjects\workone\data2\test\images"):
#     print(f"this is filename {f}")
os.makedirs(output_dir, exist_ok=True)

with open(json_path, "r") as f:
    data = json.load(f)


# print(data[0].get("meta", {}))
#
# for item in data:
#     print("ITEM:", item.get("id"))
#     print("META KEYS:", item.get("meta", {}).keys())
#     print("META CONTENT:", item.get("meta", {}))
#     print("----------------")


for item in data:
    data_keys = list(item.get("data", {}).keys())
    if not data_keys:
        continue

    img_key = data_keys[0]
    json_path_value = item["data"][img_key]
    raw_name = os.path.basename(json_path_value)

    if "-" in raw_name:
        real_name = raw_name.split("-", 1)[1]
    else:
        real_name = raw_name


    real_path = os.path.join(image_dir, real_name)
    # print("Python is trying to open", real_path)

    try:
        with Image.open(real_path) as im:
            w, h = im.size
    except:
        print("Could not open image.", real_path)
        continue

    # meta = item.get("meta", {})
    # w = meta.get("width") or meta.get("image_width") or meta.get("original_width") or meta.get("img_width")
    # h = meta.get("height") or meta.get("image_height") or meta.get("original_height") or meta.get("img_height")
    #
    # if w is None or h is None:
    #     print("No width/height found for", img_name)
    #     continue

    ann = item["annotations"][0]["result"]

    # # bounding box
    # bbox_items = [r for r in ann if r["type"] == "rectanglelabels"]
    # if not bbox_items:
    #     print("No bbox for ", img_name)
    #     continue
    #
    # bbox = bbox_items[0]["value"]
    #
    # x = bbox["value"]["x"] / 100
    # y = bbox["value"]["y"] / 100
    # bw = bbox["value"]["width"] / 100
    # bh = bbox["value"]["height"] / 100

    # # convert to YOLO center format
    # xc = x + bw / 2
    # yc = y + bh / 2

    # keypoints
    kps = []
    kp_items = [r for r in ann if r["type"] == "keypointlabels"]

    for kp in kp_items:
        v = 1
        px = kp["value"]["x"] / 100
        py = kp["value"]["y"] / 100
        kps.extend([px, py, v])

    # write YOLO pose line

    out_name = os.path.splitext(real_name)[0] + ".txt"
    out_path = os.path.join(output_dir, out_name)
    with open(out_path, "w") as f:
        f.write(f"0 0 0 0 0 " + " ".join(map(str, kps)))
