import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tkinter import filedialog, Tk

PTS_SCALE = 1000

if __name__ == "__main__":
    Tk().withdraw()
    image_path = filedialog.askopenfilename(title="Select an image file")
    img_raw = cv2.imread(image_path)    
    
    # Crop image
    plt.imshow(img_raw)
    plt.title("Select four points in order: top-left, top-right, bottom-right, bottom-left.")
    pts_add = np.float32([[pt[0], pt[1]] for pt in plt.ginput(4)])
    plt.close()
    
    cm_size_x = int(input("Enter the width of the square in cm."))
    cm_size_y = int(input("Enter the height of the square in cm."))
    
    calibration_pts_cm = np.float32([[0, 0], [cm_size_x, 0], [cm_size_x, cm_size_y], [0, cm_size_y]])
    pts_tg = calibration_pts_cm * PTS_SCALE / calibration_pts_cm.max()
    tg_size = tuple(np.max(pts_tg, axis = 0).astype(int))
    
    M = cv2.getPerspectiveTransform(pts_add, pts_tg)
    img_warped = cv2.warpPerspective(img_raw, M, tg_size)
    
    plt.figure(figsize = (20,10))
    plt.subplot(121)
    plt.imshow(img_raw)
    plt.title('Raw')
    plt.scatter(pts_add[:,0], pts_add[:,1], color = 'red')
    for i in range(pts_add.shape[0]):
        plt.text(pts_add[i,0] + 50, pts_add[i, 1] - 50, f"({','.join(calibration_pts_cm[i, :].astype(int).astype(str))})")
        
    plt.subplot(122)
    plt.imshow(img_warped)
    plt.title('Warped')
    plt.waitforbuttonpress()
    plt.close()
    
    # Modified detection section
    gray = cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to merge nearby pixels
    gray = cv2.GaussianBlur(gray, (15, 15), 0)  # Kernel size (5,5), adjust based on dot size
    
    # Consider using adaptive threshold or Otsu's method if needed
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    # Alternative with Otsu's:
    # _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    coordinates = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = M["m10"] / M["m00"]
            cY = M["m01"] / M["m00"]
            coordinates.append((cX, cY))
    coordinates = np.array(coordinates)

    # Rest of the code remains the same...
    plt.figure(figsize = (10,10))
    plt.imshow(img_warped)
    plt.scatter(coordinates[:,0], coordinates[:,1], facecolors='none', edgecolors='r', s=80)
    plt.title("Remove points")
    pts_remove = np.float32([[pt[0], pt[1]] for pt in plt.ginput(0)])
    plt.close()
    for pt in pts_remove:
        closest_pt_index = sorted(range(len(coordinates)), key = lambda c_i: np.linalg.norm(coordinates[c_i, :] - pt))[0]
        coordinates = np.delete(coordinates, closest_pt_index, 0)
    
    plt.figure(figsize = (10,10))
    plt.imshow(img_warped)
    plt.scatter(coordinates[:,0], coordinates[:,1], facecolors='none', edgecolors='r', s=80)
    plt.title("Add points.")
    pts_add = np.float32([[pt[0], pt[1]] for pt in plt.ginput(0)])
    plt.close()
    if pts_add.shape[0] > 0:
        coordinates = np.concatenate((coordinates, pts_add), axis = 0)
    
    plt.figure(figsize = (10,10))
    plt.imshow(img_warped)
    plt.scatter(coordinates[:,0], coordinates[:,1], facecolors='none', edgecolors='r', s=80)
    plt.title("Result")
    plt.waitforbuttonpress()
    plt.close()
    
    out_file_path = image_path.split('.')[0] + ".xlsx"
    coords_cm = [(x / tg_size[0] * cm_size_x, cm_size_y - (y / tg_size[1] * cm_size_y)) for x, y in coordinates]
    df = pd.DataFrame(coords_cm, columns=["X (cm)", "Y (cm)"])
    df.to_excel(out_file_path, index=False)
    print(f"Coordinates saved to {out_file_path}")