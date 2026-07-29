import os
import glob
import shutil

def copy_samples():
    src_dir = r"d:\BrailleLens\archive\Braille Dataset\Braille Dataset"
    dst_dir = r"d:\BrailleLens\braille_lens_flutter\assets\samples"
    os.makedirs(dst_dir, exist_ok=True)
    
    # Pick 1 sample image for each letter a..z
    for char_code in range(ord('a'), ord('z') + 1):
        letter = chr(char_code)
        pattern = os.path.join(src_dir, f"{letter}1*.jpg")
        matches = glob.glob(pattern)
        if matches:
            src_file = matches[0]
            dst_file = os.path.join(dst_dir, f"sample_{letter}.jpg")
            shutil.copy(src_file, dst_file)
            print(f"Copied {src_file} -> {dst_file}")

if __name__ == "__main__":
    copy_samples()
