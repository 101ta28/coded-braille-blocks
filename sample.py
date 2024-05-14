import cv2
import numpy as np
import sys
import json


# グレースケール画像を読み込む
def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return image


# 点字ブロックの外枠を検出
def detect_outer_rectangle(image):
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 1:
        raise ValueError("Not enough contours found")
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4:
            return approx.reshape(-1, 2)
    raise ValueError("No rectangle detected")


# 射影変換で画像を正方形に補正
def rectify_perspective(image, points):
    # 頂点を左上、右上、右下、左下の順序に並べ替える
    rect = np.zeros((4, 2), dtype="float32")

    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]

    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]

    side_length = max(
        np.linalg.norm(rect[0] - rect[1]),
        np.linalg.norm(rect[1] - rect[2]),
        np.linalg.norm(rect[2] - rect[3]),
        np.linalg.norm(rect[3] - rect[0]),
    )
    dst = np.array(
        [
            [0, 0],
            [side_length - 1, 0],
            [side_length - 1, side_length - 1],
            [0, side_length - 1],
        ],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(rect, dst)
    rectified_image = cv2.warpPerspective(
        image, M, (int(side_length), int(side_length))
    )
    return rectified_image


# 三角形の検出
def detect_triangle(image):
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    triangles = []
    for cnt in contours:
        epsilon = 0.04 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 3:
            triangles.append((cv2.contourArea(cnt), approx.reshape(-1, 2)))
    if triangles:
        triangles.sort(reverse=True, key=lambda x: x[0])  # 面積でソート
        return triangles[0][1]  # 最大の三角形を返す
    return None


# 三角形の方向を決定
def determine_direction(triangle_coords, image_shape):
    centroid = triangle_coords.mean(axis=0)
    mid_x = image_shape[1] / 2
    mid_y = image_shape[0] / 2
    if centroid[0] < mid_x and centroid[1] < mid_y:
        return 0
    elif centroid[0] < mid_x and centroid[1] > mid_y:
        return 1
    elif centroid[0] > mid_x and centroid[1] > mid_y:
        return 2
    elif centroid[0] > mid_x and centroid[1] < mid_y:
        return 3
    else:
        return None


# 画像と三角形の座標を回転
def rotate_image_and_coords(image, coords, direction):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    if direction == 1:
        M = cv2.getRotationMatrix2D(center, -90, 1.0)
    elif direction == 2:
        M = cv2.getRotationMatrix2D(center, -180, 1.0)
    elif direction == 3:
        M = cv2.getRotationMatrix2D(center, 90, 1.0)
    else:
        return image, coords

    rotated_image = cv2.warpAffine(image, M, (w, h))
    coords = np.hstack([coords, np.ones((coords.shape[0], 1))]).dot(M.T)
    coords = coords[:, :2].astype(int)
    return rotated_image, coords


# 内枠の平均輝度を計算
def calculate_inner_rectangle_mean_intensity(image):
    mask = np.ones_like(image, dtype=np.uint8) * 255  # 全て白で初期化
    mask[image < 127] = 0  # 黒を設定
    mean_intensity = np.mean(image[mask == 255])
    return mean_intensity


# 三角形の領域をマスク
def mask_triangle_area(image, triangle_coords, background_color):
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [triangle_coords], -1, 255, thickness=cv2.FILLED)
    if background_color == "white":
        image[mask == 255] = 255
    else:
        image[mask == 255] = 0
    return image


# グリッドの解析
def analyze_grid(image, grid_size=5):
    h, w = image.shape[:2]
    step_x = w // grid_size
    step_y = h // grid_size
    grid_data = []
    for i in range(grid_size):
        row_data = []
        for j in range(grid_size):
            tl_x = j * step_x
            tl_y = i * step_y
            br_x = tl_x + step_x
            br_y = tl_y + step_y
            cell = image[tl_y:br_y, tl_x:br_x]
            ratio = np.sum(cell < 127) / cell.size
            row_data.append(ratio)
        grid_data.append(row_data)
    return grid_data


# 点字ブロックの番号を計算
def classify_cells(grid_data, threshold=0.1):
    classified_data = []
    for row in grid_data:
        classified_row = []
        for ratio in row:
            if ratio >= threshold:
                classified_row.append(1)
            else:
                classified_row.append(0)
        classified_data.append(classified_row)
    return classified_data


# メイン処理
def main(image_path):
    image = load_image(image_path)

    outer_rectangle = detect_outer_rectangle(image)
    if outer_rectangle is None:
        raise ValueError("No outer rectangle detected in the image.")

    # トリミングして射影変換で補正
    rectified_image = rectify_perspective(image, outer_rectangle)

    # 内枠の平均輝度を計算
    mean_intensity = calculate_inner_rectangle_mean_intensity(rectified_image)
    background_color = "white" if mean_intensity > 127 else "black"

    triangle_coords = detect_triangle(rectified_image)
    if triangle_coords is None:
        raise ValueError("No triangle detected in the image.")

    direction = determine_direction(triangle_coords, rectified_image.shape)
    rotated_image, triangle_coords_rotated = rotate_image_and_coords(
        rectified_image, triangle_coords, direction
    )

    # 三角形の領域をマスク
    image_masked = mask_triangle_area(
        rotated_image, triangle_coords_rotated, background_color
    )

    grid_data = analyze_grid(image_masked)
    classified_cells = classify_cells(grid_data)

    NUMBERS_SEQUENCE = [
        [16777216, 8388608, 4194304, 2097152, 1048576],
        [524288, 262144, 131072, 65536, 32768],
        [16384, 8192, 4096, 2048, 1024],
        [512, 256, 128, 64, 32],
        [16, 8, 4, 2, 1],
    ]

    result_numbers = []
    for i in range(5):
        for j in range(5):
            if classified_cells[i][j] == 1:
                result_numbers.append(NUMBERS_SEQUENCE[i][j])
    sum_result = sum(result_numbers)

    return classified_cells, sum_result, direction


if __name__ == "__main__":
    if len(sys.argv) != 2 or not (
        sys.argv[1].endswith(".jpg") or sys.argv[1].endswith(".png")
    ):
        print("Usage: python detect_tactile_blocks.py <filename.jpg or filename.png>")
        sys.exit(1)

    image_path = sys.argv[1]
    try:
        classified_cells, sum_result, direction = main(image_path)
        print("Classified Cells:", json.dumps(classified_cells))
        print("Sum:", sum_result)
        print("Direction:", direction)
    except Exception as e:
        print(str(e))
        sys.exit(1)
