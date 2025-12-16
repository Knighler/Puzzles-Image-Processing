import cv2
import numpy as np
import os

INPUT_ROOT = 'E:\\ASU\\Fall 25\\Image\\Project\\Raw Images\\Gravity Falls'
OUTPUT_ROOT = 'E:\\ASU\\Fall 25\\Image\\Project\\Raw Images\\Gravity Falls\\preprocessing_results'

SUBFOLDERS = ['puzzle_2x2', 'puzzle_4x4', 'puzzle_8x8']

def save_piece(image, folder_stage, subfolder, filename):
    target_dir = os.path.join(OUTPUT_ROOT, folder_stage, subfolder)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    save_path = os.path.join(target_dir, filename)
    cv2.imwrite(save_path, image)

def slice_image_grid(image, grid_n):
    height, width, _ = image.shape
    step_y = height // grid_n
    step_x = width // grid_n
    pieces = []
    for row in range(grid_n):
        for col in range(grid_n):
            y_start = row * step_y
            y_end = (row + 1) * step_y
            x_start = col * step_x
            x_end = (col + 1) * step_x
            piece = image[y_start:y_end, x_start:x_end]
            pieces.append(piece)
    return pieces

def apply_denoising_NLM(piece):
    return cv2.fastNlMeansDenoisingColored(piece, None, 2, 1, 9, 21)

def apply_denoising_gaussian(piece):
    return cv2.GaussianBlur(piece, (5, 5), 0)

def apply_denoising_bilateral(piece):
    return cv2.bilateralFilter(piece, d=9, sigmaColor=75, sigmaSpace=75)

def apply_denoising_median(piece):
    return cv2.medianBlur(piece, 3)

def apply_sharpening_laplacian(piece):
    lab = cv2.cvtColor(piece, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    kernel = np.array([[0, -1, 0], 
                       [-1, 4, -1], 
                       [0, -1, 0]])
    
    l_sharpened = cv2.filter2D(l, -1, kernel)
    merged = cv2.merge((l_sharpened, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def apply_enhancement_HE(piece):
    lab = cv2.cvtColor(piece, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = cv2.equalizeHist(l)
    merged = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def apply_enhancement(piece):
    lab_image = cv2.cvtColor(piece, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(7,7))
    enhanced_l = clahe.apply(l)
    merged_lab = cv2.merge((enhanced_l, a, b))
    return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

def run_preprocessing():
    if not os.path.exists(INPUT_ROOT):
        print(f"Error: Input path not found: {INPUT_ROOT}")
        return

    for subfolder in SUBFOLDERS:
        input_folder_path = os.path.join(INPUT_ROOT, subfolder)
        if not os.path.exists(input_folder_path):
            print(f"Skipping {subfolder},not found")
            continue

        print(f"///////////// Processing {subfolder} ")
        try:
            before_x = subfolder.split('x')[0]
            if '_' in before_x:
                number_str = before_x.split('_')[-1]
            else:
                number_str = before_x
            grid_n = int(number_str)
        except:
            print(f"Error parsing size for {subfolder}")
            continue

        files = [f for f in os.listdir(input_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for file in files:
            img_path = os.path.join(input_folder_path, file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            base_name = os.path.splitext(file)[0]
            base_name+=".png"
            #raw_pieces = slice_image_grid(img, grid_n)

            #for idx, piece in enumerate(raw_pieces):
            #piece_name = f"{base_name}.png"
                
            #save_piece(img, "1_sliced", subfolder, base_name)

            denoised_piece = apply_denoising_NLM(img)
            save_piece(denoised_piece, "2_denoised", subfolder, base_name)

            enhanced_piece = apply_enhancement(denoised_piece)
            save_piece(enhanced_piece, "3_enhanced", subfolder, base_name)

            enhanced_piece = apply_enhancement(img)
            save_piece(enhanced_piece, "4_enhanced_only", subfolder, base_name)

            #print(f"Processed {file} ,{len(raw_pieces)} pieces generated")

    print("\nProcessing Complete")

if __name__ == "__main__":
    run_preprocessing()