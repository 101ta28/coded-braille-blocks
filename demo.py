import json
import sys
import cv2
import numpy as np


def detect_triangle(image, threshold_value=127):
    ret, thresh = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    triangles = []
    for cnt in contours:
        epsilon = 0.04 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 3:
            triangles.append((cv2.contourArea(cnt), approx.reshape(-1, 2)))
    if triangles:
        triangles.sort(reverse=True, key=lambda x: x[0])  # Sort by area
        return triangles[0][1]  # Return the largest triangle
    return None


def detect_outer_rectangle(image, threshold_value=127):
    ret, thresh = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 2:
        raise ValueError("Not enough contours found")
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contour = contours[1]
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    return approx.reshape(-1, 2)


def compute_black_ratio(cell, threshold=127):
    black_pixels = np.sum(cell < threshold)
    total_pixels = cell.size
    return black_pixels / total_pixels


def analyze_grid(image, outer_rectangle, grid_size=5):
    sorted_rect = np.array(sorted(outer_rectangle, key=lambda x: x[1]))
    top_points = sorted(sorted_rect[:2], key=lambda x: x[0])
    bottom_points = sorted(sorted_rect[2:], key=lambda x: x[0])
    sorted_rect = np.vstack([top_points, bottom_points])
    step_x = (sorted_rect[1][0] - sorted_rect[0][0]) / grid_size
    step_y = (sorted_rect[3][1] - sorted_rect[0][1]) / grid_size
    grid_data = []
    for i in range(grid_size):
        row_data = []
        for j in range(grid_size):
            tl_x = int(sorted_rect[0][0] + j * step_x)
            tl_y = int(sorted_rect[0][1] + i * step_y)
            br_x = int(tl_x + step_x)
            br_y = int(tl_y + step_y)
            cell = image[tl_y:br_y, tl_x:br_x]
            ratio = compute_black_ratio(cell)
            row_data.append(ratio)
        grid_data.append(row_data)
    return grid_data


def fill_triangle(image, triangle_coords):
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [triangle_coords], 0, 255, -1)
    image_filled = np.where(mask == 255, 255, image)
    return image_filled


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


def rotate_to_direction_0(image, direction):
    if direction == 1:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif direction == 2:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif direction == 3:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return image


def detect_circle(image):
    blurred = cv2.GaussianBlur(image, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=30,
        param1=50,
        param2=30,
        minRadius=5,
        maxRadius=30,
    )
    if circles is not None:
        circles = np.uint16(np.around(circles))
        sorted_circles = sorted(circles[0, :], key=lambda x: x[2], reverse=True)
        return sorted_circles[0][:2]
    return None


NUMBERS_SEQUENCE = [
    [16777216, 8388608, 4194304, 2097152, 1048576],
    [524288, 262144, 131072, 65536, 32768],
    [16384, 8192, 4096, 2048, 1024],
    [512, 256, 128, 64, 32],
    [16, 8, 4, 2, 1],
]

if __name__ == "__main__":
    if len(sys.argv) != 2 or not (
        sys.argv[1].endswith(".jpg") or sys.argv[1].endswith(".png")
    ):
        print("Usage: python demo.py <filename.jpg or filename.png>")
        sys.exit(1)

    image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Failed to load image: {sys.argv[1]}")
        sys.exit(1)

    triangle_coords = detect_triangle(image)
    if triangle_coords is None:
        print("No triangle detected in the image.")
        sys.exit(1)

    direction = determine_direction(triangle_coords, image.shape)
    image_rotated = rotate_to_direction_0(image, direction)
    triangle_coords_rotated = detect_triangle(image_rotated)
    if triangle_coords_rotated is None:
        print("No triangle detected after rotation.")
        sys.exit(1)

    outer_rectangle = detect_outer_rectangle(image_rotated)
    if outer_rectangle is None:
        print("No outer rectangle detected in the image.")
        sys.exit(1)

    image_without_triangle = fill_triangle(
        image_rotated.copy(), triangle_coords_rotated.reshape(-1, 1, 2)
    )
    grid_data = analyze_grid(image_without_triangle, outer_rectangle)
    classified_cells = classify_cells(grid_data)
    print(classified_cells)

    result_numbers = []
    for i in range(5):
        for j in range(5):
            if classified_cells[i][j] == 1:
                result_numbers.append(NUMBERS_SEQUENCE[i][j])
    sum_result = sum(result_numbers)

    print("Sum:", sum_result)
