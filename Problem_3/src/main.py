from pathlib import Path
import cv2
import numpy as np
from typing import Tuple
import os



def read_image_safe(path: Path):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print("Ошибка чтения:", path, e)
        return None



def save_image_safe(path: Path, image: np.ndarray) -> bool:
    try:
        ext = path.suffix
        if ext == "":
            ext = ".png"
            path = path.with_suffix(ext)

        ok, buffer = cv2.imencode(ext, image)
        if not ok:
            return False

        buffer.tofile(str(path))
        return True
    except Exception as e:
        print("Ошибка сохранения:", path, e)
        return False



def convert_image_spaces(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return hsv, gray



def find_all_images(root_dir: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted([p for p in root_dir.rglob("*") if p.suffix.lower() in exts])



def build_useful_mask(bgr: np.ndarray) -> np.ndarray:
    hsv, gray = convert_image_spaces(bgr)
    _, s, v = cv2.split(hsv)

    v = v.astype(np.float32) / 255.0
    s = s.astype(np.float32) / 255.0

    
    cloud = ((v > 0.65) & (s < 0.35)) | (v > 0.82)

    mask = np.ones_like(gray, dtype=np.uint8)
    mask[cloud] = 0

    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cloud255 = ((mask == 0).astype(np.uint8) * 255)
    cloud255 = cv2.dilate(cloud255, kernel, iterations=1)

    mask = np.ones_like(gray, dtype=np.uint8)
    mask[cloud255 > 0] = 0

    return mask


def build_final_map(images, masks):
    h, w, _ = images[0].shape

    result = np.ones((h, w, 3), dtype=np.uint8) * 255
    filled = np.zeros((h, w), dtype=bool)

    for y in range(h):
        for x in range(w):
            pixels = []

            for img, mask in zip(images, masks):
                if mask[y, x] == 1:
                    pixels.append(img[y, x])

            if len(pixels) > 0:
                pixels = np.array(pixels, dtype=np.uint8)
                result[y, x] = np.median(pixels, axis=0).astype(np.uint8)
                filled[y, x] = True

    return result, filled



def group_by_shape(items):
    groups = {}
    for item in items:
        shape = item["image"].shape[:2]
        groups.setdefault(shape, []).append(item)
    return groups



def main():
    base_dir = Path(__file__).resolve().parent.parent

    input_dir = base_dir / "problem_data"
    output_dir = base_dir / "results" / "final_maps"

    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = find_all_images(input_dir)

    print("Найдено изображений:", len(image_paths))
    print("Папка результатов:", output_dir)

    items = []

    for img_path in image_paths:
        img = read_image_safe(img_path)

        if img is None:
            print("Пропущено (не читается):", img_path)
            continue

        hsv, gray = convert_image_spaces(img)
        mask = build_useful_mask(img)
        mask_vis = (mask * 255).astype(np.uint8)

        name = f"{img_path.parent.name}_{img_path.stem}"

        ok_original = save_image_safe(output_dir / f"{name}_original.png", img)
        ok_gray = save_image_safe(output_dir / f"{name}_gray.png", gray)
        ok_hsv = save_image_safe(output_dir / f"{name}_hsv.png", hsv)
        ok_mask = save_image_safe(output_dir / f"{name}_mask.png", mask)
        ok_mask_vis = save_image_safe(output_dir / f"{name}_mask_vis.png", mask_vis)

        print(
            f"{name}: "
            f"original={ok_original}, gray={ok_gray}, hsv={ok_hsv}, "
            f"mask={ok_mask}, mask_vis={ok_mask_vis}"
        )

        items.append({
            "image": img,
            "mask": mask
        })

    if not items:
        print("Нет данных для сборки карты")
        return

    groups = group_by_shape(items)
    print("Найдено групп по размеру:", len(groups))

    for i, (shape, group) in enumerate(groups.items(), 1):
        images = [x["image"] for x in group]
        masks = [x["mask"] for x in group]

        final_map, filled = build_final_map(images, masks)

        coverage = np.zeros(filled.shape, dtype=np.uint8)
        coverage[filled] = 1
        coverage_vis = (coverage * 255).astype(np.uint8)

        h, w = shape
        group_name = f"group_{i}_{h}x{w}"

        ok_final = save_image_safe(output_dir / f"{group_name}_final_map.png", final_map)
        ok_cov = save_image_safe(output_dir / f"{group_name}_coverage.png", coverage)
        ok_cov_vis = save_image_safe(output_dir / f"{group_name}_coverage_vis.png", coverage_vis)

        print(
            f"{group_name}: "
            f"final_map={ok_final}, coverage={ok_cov}, coverage_vis={ok_cov_vis}"
        )

    print("Готово")
    print("Содержимое папки результатов:")

    for p in output_dir.iterdir():
        print(" -", p.name)

    os.startfile(str(output_dir))


if __name__ == "__main__":
    main()