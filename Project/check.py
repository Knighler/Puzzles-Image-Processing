import os

path = r'E:\ASU\Fall 25\Image\Project\Raw Images\Gravity Falls\preprocessing_results\5_laplacian'

print(f"Checking: {path}")
if os.path.exists(path):
    print(" Base path exists!")
    print("Contents:")
    for item in os.listdir(path):
        print(f" - {item}")
else:
    print("Base path does NOT exist.")