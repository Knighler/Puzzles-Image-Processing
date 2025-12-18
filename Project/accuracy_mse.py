import cv2
import numpy as np
import os
import csv


corrected_folder = 'E:\\ASU\\Fall 25\\Image\\Project\\Raw Images\\Gravity Falls\\correct'

# The folder containing YOUR SOLVER'S output
result_folder = 'E:\\ASU\\Fall 25\\Image\\Project\\Raw Images\\Gravity Falls\\Paper_Results'


subfolders = ['puzzle_2x2', 'puzzle_4x4', 'puzzle_8x8']


output_csv_file = 'mse_results.csv'

def run_comparison():

    with open(output_csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Folder', 'Image Name', 'MSE Score'])

        for folder_name in subfolders:
       
            path_to_results = os.path.join(result_folder, folder_name)
            if not os.path.exists(corrected_folder) or not os.path.exists(path_to_results):
                print(f"Skipping {folder_name} (Folder not found)")
                continue

            image_files = [f for f in os.listdir(path_to_results) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

            for file_name in image_files:
                name_no_ext = os.path.splitext(file_name)[0]
                original_filename = name_no_ext + ".png"
                original_path = os.path.join(corrected_folder, original_filename)
 
                result_path = os.path.join(path_to_results, file_name)

                original_img = cv2.imread(original_path)
                result_img = cv2.imread(result_path)

                if original_img is None:
                    print(f"Warning: Original image missing for {file_name}")
                    continue

                if original_img.shape != result_img.shape:
                    writer.writerow([folder_name, file_name, "Size Mismatch"])
                    continue

                diff = original_img.astype("float") - result_img.astype("float")
                squared_diff = diff ** 2
                mse_value = np.mean(squared_diff)
                if mse_value <500:
                    mse_value="accurate"
                else:
                    mse_value="not accurate"
                writer.writerow([folder_name, file_name, mse_value])
                
if __name__ == "__main__":
    run_comparison()