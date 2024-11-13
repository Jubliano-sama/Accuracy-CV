import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tkinter import filedialog, Tk

PTS_SCALE = 1000


# Run the program
if __name__ == "__main__":
    Tk().withdraw()
    
    # Select and load image file
    image_path = filedialog.askopenfilename(title="Select an image file")
    img_raw = cv2.imread(image_path)    
    
    
    # Crop image
    # Display the image and let the user select polygon points
    plt.imshow(img_raw)
    plt.title("Select four points in order: top-left, top-right, bottom-right, bottom-left.")
    pts_add = np.float32([[pt[0], pt[1]] for pt in plt.ginput(4)])  # Capture as many points as the user clicks
    plt.close()
    
    
    # Input min and max
    cm_size_x = int(input("Enter the width of the square in cm."))
    cm_size_y = int(input("Enter the height of the square in cm."))
    
    
    # np.float32([[ -40, -38], [  40, -38], [  40,   30], [-40,   30]])
    # pts_px = np.float32([(calibration_pts.loc['x_px', col], calibration_pts.loc['y_px', col]) for col in calibration_pts.columns])
    calibration_pts_cm = np.float32([[0, 0], [cm_size_x, 0], [cm_size_x, cm_size_y], [0, cm_size_y]])
    pts_tg = calibration_pts_cm * PTS_SCALE / calibration_pts_cm.max()
    tg_size = tuple(np.max(pts_tg, axis = 0).astype(int))
    
    
    # Transform image according to calibration points
    M = cv2.getPerspectiveTransform(pts_add, pts_tg)
    img_warped = cv2.warpPerspective(img_raw, M, tg_size)
    plt.figure(figsize = (20,10))
    
    # Left
    plt.subplot(121)
    plt.imshow(img_raw)
    plt.title('Raw')
    plt.scatter(pts_add[:,0], pts_add[:,1], color = 'red')
    for i in range(pts_add.shape[0]):
        plt.text(pts_add[i,0] + 50, pts_add[i, 1] - 50, f"({','.join(calibration_pts_cm[i, :].astype(int).astype(str))})")
        
    # Right
    plt.subplot(122)
    plt.imshow(img_warped)
    plt.title('Warped')
    plt.waitforbuttonpress()
    plt.close()
    
    
    # Detect points automatically
    gray = cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    coordinates = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = M["m10"] / M["m00"]
            cY = M["m01"] / M["m00"]
            coordinates.append((cX, cY))
    coordinates = np.array(coordinates)


    # Remove points manually
    plt.figure(figsize = (10,10))
    plt.imshow(img_warped)
    plt.scatter(coordinates[:,0], coordinates[:,1], facecolors='none', edgecolors='r', s=80)
    plt.title("Remove points")
    # plt.show()
    pts_remove = np.float32([[pt[0], pt[1]] for pt in plt.ginput(0)])  # Capture as many points as the user clicks
    plt.close()
    for pt in pts_remove:
        closest_pt_index = sorted(range(len(coordinates)), key = lambda c_i: np.linalg.norm(coordinates[c_i, :] - pt))[0]
        coordinates = np.delete(coordinates, closest_pt_index, 0)
    
    
    # Add points manually
    plt.figure(figsize = (10,10))
    plt.imshow(img_warped)
    plt.scatter(coordinates[:,0], coordinates[:,1], facecolors='none', edgecolors='r', s=80)
    plt.title("Add points.")
    pts_add = np.float32([[pt[0], pt[1]] for pt in plt.ginput(0)])  # Capture as many points as the user clicks
    plt.close()
    if pts_add.shape[0] > 0:
        coordinates = np.concatenate((coordinates, pts_add), axis = 0)
    
    
    # Show result, wait for key press
    plt.figure(figsize = (10,10))
    plt.imshow(img_warped)
    plt.scatter(coordinates[:,0], coordinates[:,1], facecolors='none', edgecolors='r', s=80)
    plt.title("Result")
    plt.waitforbuttonpress()
    plt.close()
    
    
    # Transform to cm values and save to Excel
    out_file_path = image_path.split('.')[0] + ".xlsx"
    coords_cm = [(x / tg_size[0] * cm_size_x, cm_size_y - (y / tg_size[1] * cm_size_y)) for x, y in coordinates]
    df = pd.DataFrame(coords_cm, columns=["X (cm)", "Y (cm)"])
    df.to_excel(out_file_path, index=False)
    print(f"Coordinates saved to {out_file_path}")