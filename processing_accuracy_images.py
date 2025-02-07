import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tkinter import filedialog, Tk
from matplotlib.widgets import Button  # For the GUI button

PTS_SCALE = 1000

def wait_for_continue():
    """
    Display a 'Continue' button in the current figure.
    The function will block until the button is clicked.
    """
    fig = plt.gcf()  # Get current figure
    # Create a new axes for the button; adjust the position as needed.
    button_ax = fig.add_axes([0.45, 0.01, 0.1, 0.05])
    # Use a mutable container to store the state from the callback.
    clicked = [False]
    
    def on_click(event):
        clicked[0] = True
        plt.close(fig)
    
    btn = Button(button_ax, 'Continue')
    btn.on_clicked(on_click)
    # plt.show() will block until the window is closed (via the button)
    plt.show()

if __name__ == "__main__":
    Tk().withdraw()
    image_path = filedialog.askopenfilename(title="Select an image file")
    img_raw = cv2.imread(image_path)    
    
    # === Crop image ===
    plt.imshow(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB))
    plt.title("Select four points in order: top-left, top-right, bottom-right, bottom-left.")
    # Set timeout=0 so it waits indefinitely for the 4 points.
    pts_add = np.float32([[pt[0], pt[1]] for pt in plt.ginput(4, timeout=0)])
    plt.close()
    
    cm_size_x = int(input("Enter the width of the square in cm: "))
    cm_size_y = int(input("Enter the height of the square in cm: "))
    
    calibration_pts_cm = np.float32([[0, 0], [cm_size_x, 0], [cm_size_x, cm_size_y], [0, cm_size_y]])
    pts_tg = calibration_pts_cm * PTS_SCALE / calibration_pts_cm.max()
    tg_size = tuple(np.max(pts_tg, axis=0).astype(int))
    
    M = cv2.getPerspectiveTransform(pts_add, pts_tg)
    img_warped = cv2.warpPerspective(img_raw, M, tg_size)
    
    # === Show Raw and Warped images with a "Continue" button ===
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    
    # Raw image with points and labels
    axs[0].imshow(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Raw')
    axs[0].scatter(pts_add[:, 0], pts_add[:, 1], color='red')
    for i in range(pts_add.shape[0]):
        label = f"({','.join(calibration_pts_cm[i, :].astype(int).astype(str))})"
        axs[0].text(pts_add[i, 0] + 50, pts_add[i, 1] - 50, label)
    
    # Warped image
    axs[1].imshow(cv2.cvtColor(img_warped, cv2.COLOR_BGR2RGB))
    axs[1].set_title('Warped')
    
    # Wait until the user clicks the Continue button
    wait_for_continue()
    
    # === Modified detection section ===
    gray = cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (15, 15), 0)
    
    # Binary thresholding (adjust the threshold value if necessary)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    coordinates = []
    for contour in contours:
        M_contour = cv2.moments(contour)
        if M_contour["m00"] != 0:
            cX = M_contour["m10"] / M_contour["m00"]
            cY = M_contour["m01"] / M_contour["m00"]
            coordinates.append((cX, cY))
    coordinates = np.array(coordinates)

    # === Remove points interactively ===
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img_warped, cv2.COLOR_BGR2RGB))
    plt.scatter(coordinates[:, 0], coordinates[:, 1], facecolors='none', edgecolors='r', s=80)
    plt.title("Remove points (click and then press Enter when done)")
    # Set timeout=0 so that it waits indefinitely until Enter is pressed.
    pts_remove = np.float32([[pt[0], pt[1]] for pt in plt.ginput(0, timeout=0)])
    plt.close()
    
    for pt in pts_remove:
        # Remove the closest point to the clicked location.
        closest_pt_index = sorted(range(len(coordinates)),
                                    key=lambda i: np.linalg.norm(coordinates[i] - pt))[0]
        coordinates = np.delete(coordinates, closest_pt_index, 0)
    
    # === Add points interactively ===
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img_warped, cv2.COLOR_BGR2RGB))
    plt.scatter(coordinates[:, 0], coordinates[:, 1], facecolors='none', edgecolors='r', s=80)
    plt.title("Add points (click and then press Enter when done)")
    pts_new = np.float32([[pt[0], pt[1]] for pt in plt.ginput(0, timeout=0)])
    plt.close()
    
    if pts_new.shape[0] > 0:
        coordinates = np.concatenate((coordinates, pts_new), axis=0)
    
    # === Show final result with a "Continue" button ===
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img_warped, cv2.COLOR_BGR2RGB))
    plt.scatter(coordinates[:, 0], coordinates[:, 1], facecolors='none', edgecolors='r', s=80)
    plt.title("Result")
    wait_for_continue()
    
    # === Save results ===
    out_file_path = image_path.rsplit('.', 1)[0] + ".xlsx"
    # Convert coordinates to cm based on tg_size and calibration dimensions.
    coords_cm = [(x / tg_size[0] * cm_size_x, cm_size_y - (y / tg_size[1] * cm_size_y))
                 for x, y in coordinates]
    df = pd.DataFrame(coords_cm, columns=["X (cm)", "Y (cm)"])
    df.to_excel(out_file_path, index=False)
    print(f"Coordinates saved to {out_file_path}")
