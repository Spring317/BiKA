import cv2, os, numpy as np

def process_image(img_path):
    img = cv2.imread(img_path)
    if img is None: return
    
    # 1. Edge map from HSV (for painted lanes, specifically yellow and white)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 80, 200])
    upper_yellow = np.array([40, 255, 255])
    lower_white = np.array([0, 0, 245])
    upper_white = np.array([180, 20, 255])
    mask_hsv = cv2.bitwise_or(cv2.inRange(hsv, lower_white, upper_white), cv2.inRange(hsv, lower_yellow, upper_yellow))
    edges_hsv = cv2.Canny(mask_hsv, 50, 150)
    
    # 2. Edge map from Grayscale (for non-painted boundaries like the right curb)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges_gray = cv2.Canny(blur, 50, 150)
    
    # 3. Combine both edge maps
    combined_edges = cv2.bitwise_or(edges_hsv, edges_gray)
    
    # 4. Strict ROI masking to prevent catching sidewalk texture and ego vehicle hood
    height, width = combined_edges.shape
    roi_vertices = np.array([[(250, 650), (480, 480), (800, 480), (1050, 650)]], dtype=np.int32)
    roi_mask = np.zeros_like(combined_edges)
    cv2.fillPoly(roi_mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(combined_edges, roi_mask)
    
    # 5. Hough Lines with slope filtering
    lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=150)
    
    line_img = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1 != x2:
                slope = abs((y2 - y1) / (x2 - x1))
                # Reject horizontal edges (like the back of cars)
                if slope < 0.4:
                    continue
            cv2.line(line_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 6)
            
    result = cv2.addWeighted(img, 0.8, line_img, 1.0, 0)
    
    name, ext = os.path.splitext(img_path)
    cv2.imwrite(f"{name}_lane{ext}", result)

for f in ['t1_sp20.png', 't1_sp30.png', 't2_sp20.png', 't3_sp20.png', 't5_sp1.png', 't5_sp20.png']:
    process_image(f'/mnt/storage/pylot/report/images/{f}')
