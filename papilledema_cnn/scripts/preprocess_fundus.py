import os

try:
    import cv2
except ImportError:
    raise ImportError("opencv-python is required. Install it with: pip install opencv-python")

input_dir = "test_images"
output_dir = "test_images_processed"

os.makedirs(output_dir, exist_ok=True)

classes = os.listdir(input_dir)

for cls in classes:
    cls_in = os.path.join(input_dir, cls)
    cls_out = os.path.join(output_dir, cls)

    if not os.path.isdir(cls_in):  # Add this check to skip non-directories like .DS_Store
        continue

    os.makedirs(cls_out, exist_ok=True)

    for img_name in os.listdir(cls_in):
        img_path = os.path.join(cls_in, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        coords = cv2.findNonZero(thresh)

        x, y, w, h = cv2.boundingRect(coords)

        crop = img[y:y+h, x:x+w]

        resized = cv2.resize(crop, (512,512))

        save_path = os.path.join(cls_out, img_name)

        cv2.imwrite(save_path, resized)

print("Preprocessing complete.")
print("INPUT DIR:", "test_images")
print("Files:", "test_images")