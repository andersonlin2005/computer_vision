import cv2
import numpy as np
import os
import glob

def find_keypoints_and_matches(img1, img2):#偵測特徵點並計算匹配
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    return kp1, kp2, good_matches

def compute_homography(kp1, kp2, matches):#計算單應矩陣
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    return H

def get_overlap_coordinates_sorted(left_img, right_img, H):
    h_l, w_l = left_img.shape[:2]
    h_r, w_r = right_img.shape[:2]
    
    corners_l = np.float32([
        [0, 0], [0, h_l], [w_l, h_l], [w_l, 0]
    ]).reshape(-1, 1, 2)
    
    corners_r = np.float32([
        [0, 0], [0, h_r], [w_r, h_r], [w_r, 0]
    ]).reshape(-1, 1, 2)
    corners_r_transformed = cv2.perspectiveTransform(corners_r, H)
    
    all_corners = np.vstack((corners_l, corners_r_transformed)).reshape(-8, 2)
    
    x_coords = np.sort(all_corners[:, 0])
    y_coords = np.sort(all_corners[:, 1])
    
    x_min = x_coords[3]
    x_max = x_coords[4]
    y_min = y_coords[3]
    y_max = y_coords[4]
    
    if x_max <= x_min or y_max <= y_min:
        return None
    
    return [
        (x_min, y_min),  # Top-left
        (x_max, y_min),  # Top-right
        (x_max, y_max),  # Bottom-right
        (x_min, y_max)   # Bottom-left
    ]

def stitch_images(left_img, right_img, H, output_shape):#拼接圖片
    # Warp the right image
    warped_right = cv2.warpPerspective(right_img, H, output_shape)
    
    # Create a mask for the left image
    result = np.zeros_like(warped_right)
    h_l, w_l = left_img.shape[:2]
    # 只複製 left_img 的有效高度範圍
    result[:h_l, :w_l] = left_img
    
    # Blend where right image contributes
    seam_x = min(w_l, warped_right.shape[1])
    
    # Simple blending: use left image up to seam, right image after
    result[:h_l, :seam_x] = left_img[:, :seam_x]
    right_contrib = warped_right[:h_l, seam_x:]
    result[:h_l, seam_x:] = np.where(right_contrib > 0, right_contrib, result[:h_l, seam_x:])
    
    return result

def crop_black_border(img):#裁切黑邊
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    coords = cv2.findNonZero(gray)
    if coords is None:
        return img  
    
    x, y, w, h = cv2.boundingRect(coords)
    
    return img[y:y+h, x:x+w]

def process_image_pair(left_path, right_path, output_dir, filename):#image stitching
    left_img = cv2.imread(left_path)
    right_img = cv2.imread(right_path)
    
    if left_img is None or right_img is None:
        print(f"Failed to load images: {left_path} or {right_path}")
        return None
    
    # Convert to grayscale for keypoint detection
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    
    # Find keypoints and matches
    kp1, kp2, matches = find_keypoints_and_matches(left_gray, right_gray)
    
    if len(matches) < 4:
        print(f"Not enough matches for {filename}")
        return None
    
    # Compute homography
    H = compute_homography(kp1, kp2, matches)
    
    # Determine output size (extend to accommodate warped right image)
    h_l, w_l = left_img.shape[:2]
    h_r, w_r = right_img.shape[:2]
    corners_r = np.float32([[0, 0], [0, h_r], [w_r, h_r], [w_r, 0]]).reshape(-1, 1, 2)
    corners_r_transformed = cv2.perspectiveTransform(corners_r, H)
    x_max = int(max(np.max(corners_r_transformed[:, :, 0]), w_l))
    y_max = int(max(np.max(corners_r_transformed[:, :, 1]), h_l))
    output_shape = (x_max, y_max)
    
    # Stitch images
    result = stitch_images(left_img, right_img, H, output_shape)
    
    # 裁切黑邊
    result = crop_black_border(result)
    
    # Get overlap coordinates
    overlap_coords = get_overlap_coordinates_sorted(left_img, right_img, H)
    
    # Save result
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, result)
    
    return overlap_coords

def main():
    student_id = "411285016"  
    left_dir = "left"
    right_dir = "right"
    output_dir = student_id
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all image pairs
    left_images = sorted(glob.glob(os.path.join(left_dir, "*.jpg")))
    right_images = sorted(glob.glob(os.path.join(right_dir, "*.jpg")))
    
    overlap_lines = []
    
    for left_path, right_path in zip(left_images, right_images):
        filename = os.path.basename(left_path)
        overlap_coords = process_image_pair(left_path, right_path, output_dir, filename)
        
        if overlap_coords:
            x1, y1 = map(int, overlap_coords[0])
            x2, y2 = map(int, overlap_coords[1])
            x3, y3 = map(int, overlap_coords[2])
            x4, y4 = map(int, overlap_coords[3])
            line = f"{filename.split('.')[0]} {y1} {x1} {y2} {x2} {y3} {x3} {y4} {x4}"
            overlap_lines.append(line)
    
    # Write overlap.txt
    with open(os.path.join(output_dir, "overlap.txt"), "w") as f:
        for line in overlap_lines:
            f.write(line + "\n")

if __name__ == "__main__":
    main()