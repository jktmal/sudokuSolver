import pandas as pd
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path

def order_points(pts):
    # Order: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]    # top-left
    rect[2] = pts[np.argmax(s)]    # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # top-right
    rect[3] = pts[np.argmax(diff)] # bottom-left
    return rect

def divide_cells(img):
    width, height = img.shape
    for x_div in np.arange(0, 252, 28):
        for y_div in np.arange(0, 252, 28):
            cell_arr = img[x_div:x_div+28, y_div:y_div+28]
            cell_id = "cell_" + str(x_div//28) + "_" + str(y_div//28) 
            cv.imwrite(cell_id + '.jpg', cell_arr)

def standardize_cell(img):
    # Wczytaj obraz w odcieniach szarości
    gray = cv.imread('cell_8_2.jpg', cv.IMREAD_GRAYSCALE)

    # 1. Skalowanie do 0–1
    gray_norm = gray.astype(np.float32) / 255.0

    # 2. Binaryzacja
    binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv.THRESH_BINARY, 3, 2)

    # 3. Czyszczenie krawędzi (np. zamalowanie ramek)
    binary[:4, :] = 255
    binary[-4:, :] = 255
    binary[:, :4] = 255
    binary[:, -4:] = 255

    # (opcjonalnie) Czyszczenie przez morfologię
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
    
    #print(cleaned)
    
    
    cv.imwrite('cell_8_2_cleaned.jpg', cleaned)
    
    if np.count_nonzero(cleaned) < 10:
        print("Blank or nearly blank")
    else:
        print("Not blank")

def process_image(file_path, matrix):
    """
    Process the image to extract and standardize cells.

    Parameters:
    - file_path: str, path to the image file.

    Returns:
    - None
    """
    
    # Load the image
    img = cv.imread(file_path)
    
    
    
    img = cv.imread('data/image1072.jpg', 0)

    width, height = img.shape

    #gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    blur = cv.GaussianBlur(img, (5,5), 0)

    plt.imshow(blur)

    #th = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
    #            cv.THRESH_BINARY,11,2)
    th = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv.THRESH_BINARY_INV, 11, 2)

    #np.where(th=255)

    contours, _ = cv.findContours(th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    max_area = 0
    biggest_square = None
    size = 0

    for cnt in contours:
        peri = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
        area = cv.contourArea(approx)
        if len(approx) == 4 and area > max_area:
            # Optionally check if it's "square enough" (aspect ratio ~1)
            x, y, w, h = cv.boundingRect(approx)
            aspect_ratio = float(w)/h
            if 0.8 < aspect_ratio < 1.2:  # adjust as needed
                max_area = area
                biggest_square = approx
                size = np.max([w,h])

    mask = np.zeros_like(blur)
    cv.drawContours(mask, [biggest_square], -1, (255), thickness=cv.FILLED)
    masked = cv.bitwise_and(blur, blur, mask=mask)

    plt.imshow(masked)

    pts_src = biggest_square.reshape(4, 2).astype(np.float32)
    ordered_pts_src = order_points(pts_src)

    width = 252
    height = 252
    #width = np.max(w,h)
    pts_dst = np.float32([
        [0, 0],
        [width-1, 0],
        [width-1, height-1],
        [0, height-1]
    ])


    M = cv.getPerspectiveTransform(ordered_pts_src, pts_dst)
    sudoku_board = cv.warpPerspective(masked, M, (252, 252))

    plt.imshow(sudoku_board)

    divide_cells(sudoku_board)
    
    
    divide_cells(img)
    # Process each cell
    for i in range(9):
        for j in range(9):
            cell_id = f'cell_{i}_{j}.jpg'
            if Path(cell_id).exists():
                standardize_cell(cell_id)
                # Here you can add code to extract features from the standardized cell
                # and append them to the matrix if needed.
                #features = extract_features(cell_id)
                #matrix.append(features)

def prepare_data(file_path, test_size=0.2, random_state=42):
    """
    Load and prepare the dataset for training and testing.

    Parameters:
    - file_path: str, path to the CSV file containing the dataset.
    - test_size: float, proportion of the dataset to include in the test split.
    - random_state: int, controls the shuffling applied to the data before applying the split.

    Returns:
    - X_train: DataFrame, training features.
    - X_test: DataFrame, testing features.
    - y_train: Series, training labels.
    - y_test: Series, testing labels.
    """
    
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Separate features and target variable
    X = data.drop(columns=['target'])
    y = data['target']
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test


# Specify the directory
directory = Path("/home/jasiek/Dokumenty/sudokuSolver/train")
# Iterate through all files in the directory
for file in directory.iterdir():
    if file.suffix == ".png":
        process_image(file, [])