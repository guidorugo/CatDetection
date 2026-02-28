#!/usr/bin/env python3
"""Prepare training data by running YOLO on raw images and cropping detected cats."""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))


def prepare_data(data_dir: str, output_dir: str, yolo_model: str = "yolov8s.pt"):
    from ultralytics import YOLO

    data_path = Path(data_dir)
    output_path = Path(output_dir)
    CAT_CLASS = 15
    PAD_RATIO = 0.2
    TARGET_SIZE = 256

    model = YOLO(yolo_model)

    cat_dirs = [d for d in data_path.iterdir() if d.is_dir() and d.name != "processed"]
    print(f"Found {len(cat_dirs)} cat directories: {[d.name for d in cat_dirs]}")

    for cat_dir in cat_dirs:
        cat_name = cat_dir.name
        cat_output = output_path / cat_name
        cat_output.mkdir(parents=True, exist_ok=True)

        # Collect image paths, skip zip files
        image_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            image_paths.extend(cat_dir.glob(ext))

        # Also search subdirectories (but skip zip-extracted folders with .zip in name)
        for subdir in cat_dir.iterdir():
            if subdir.is_dir() and ".zip" not in subdir.name.lower():
                for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
                    image_paths.extend(subdir.glob(ext))

        print(f"\n{cat_name}: {len(image_paths)} images")
        crop_count = 0

        for img_path in image_paths:
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                results = model.predict(img, conf=0.3, classes=[CAT_CLASS], verbose=False)

                for result in results:
                    for i, box in enumerate(result.boxes):
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        h_img, w_img = img.shape[:2]

                        # Add padding
                        bw = x2 - x1
                        bh = y2 - y1
                        pad_x = int(bw * PAD_RATIO)
                        pad_y = int(bh * PAD_RATIO)
                        x1 = max(0, x1 - pad_x)
                        y1 = max(0, y1 - pad_y)
                        x2 = min(w_img, x2 + pad_x)
                        y2 = min(h_img, y2 + pad_y)

                        crop = img[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue

                        crop = cv2.resize(crop, (TARGET_SIZE, TARGET_SIZE))
                        out_name = f"{img_path.stem}_crop{i}.jpg"
                        cv2.imwrite(str(cat_output / out_name), crop)
                        crop_count += 1

            except Exception as e:
                print(f"  Error processing {img_path.name}: {e}")

        print(f"  Cropped {crop_count} cat images")

    # Split into train/val/test (80/10/10)
    print("\nSplitting into train/val/test...")
    for cat_dir in output_path.iterdir():
        if not cat_dir.is_dir():
            continue

        images = sorted(list(cat_dir.glob("*.jpg")))
        if len(images) < 3:
            print(f"  Skipping {cat_dir.name}: too few images ({len(images)})")
            continue

        # Split
        train_imgs, test_imgs = train_test_split(images, test_size=0.2, random_state=42)
        val_imgs, test_imgs = train_test_split(test_imgs, test_size=0.5, random_state=42)

        cat_name = cat_dir.name
        for split, split_imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            split_dir = output_path / split / cat_name
            split_dir.mkdir(parents=True, exist_ok=True)
            for img_path in split_imgs:
                dest = split_dir / img_path.name
                img_path.rename(dest)

        # Remove now-empty cat directory
        for f in cat_dir.glob("*"):
            f.unlink()
        cat_dir.rmdir()

        print(f"  {cat_name}: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")

    print("\nData preparation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare cat training data")
    parser.add_argument("--data-dir", default="data", help="Raw data directory")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    parser.add_argument("--yolo-model", default="yolov8s.pt", help="YOLO model path")
    args = parser.parse_args()

    prepare_data(args.data_dir, args.output_dir, args.yolo_model)
