import os
import os.path as osp
import numpy as np
from PIL import Image

def generate_image_mask_from_parse(input_parse_path, output_mask_path):
    """
    Generates a binary image mask from a semantic segmentation parse map.

    Args:
        input_parse_path (str): Full path to the input image-parse PNG file.
        output_mask_path (str): Full path where the generated image-mask PNG file will be saved.
    """
    if not osp.exists(input_parse_path):
        print(f"Warning: Input parse file not found: {input_parse_path}. Skipping.")
        return

    try:
        # 1. Load the segmentation map as a grayscale image (Luminosity mode)
        # As per cp_dataset.py, image-parse is loaded with .convert('L')
        im_parse = Image.open(input_parse_path).convert('L')
        parse_array = np.array(im_parse)

        # 2. Create the binary mask:
        # Pixels with value > 0 are considered foreground (part of the person/clothing),
        # while 0 is background. This creates a boolean array.
        binary_mask_boolean = (parse_array > 0)

        # 3. Convert the boolean array to uint8 (0 or 1) and scale to 0-255
        # This makes it a proper grayscale image (black for background, white for foreground)
        binary_mask_array_255 = binary_mask_boolean.astype(np.uint8) * 255

        # 4. Convert the numpy array back to a PIL Image and save it
        # Save in 'L' mode (8-bit pixels, black and white)
        Image.fromarray(binary_mask_array_255, mode='L').save(output_mask_path)
        print(f"Generated mask for {osp.basename(input_parse_path)}")

    except Exception as e:
        print(f"Error processing {input_parse_path}: {e}")

def process_all_parse_images(data_root, datamode='train'):
    """
    Processes all image-parse files in a specified datamode directory
    to generate corresponding image-mask files.

    Args:
        data_root (str): The root directory of your dataset (e.g., 'data' in your repo).
        datamode (str): The specific data mode to process (e.g., 'train', 'test').
    """
    # Assuming 'image-parse-new' is the source as per your cp_dataset.py preference
    input_parse_dir = osp.join(data_root, datamode, 'image-parse-new')
    output_mask_dir = osp.join(data_root, datamode, 'image-mask')

    # Create the output directory if it doesn't exist
    os.makedirs(output_mask_dir, exist_ok=True)

    print(f"--- Starting mask generation for {datamode} mode ---")
    print(f"Reading parse maps from: {input_parse_dir}")
    print(f"Saving generated masks to: {output_mask_dir}")

    processed_count = 0
    if not osp.exists(input_parse_dir):
        print(f"Error: Input parse directory not found: {input_parse_dir}")
        print("Please ensure your 'image-parse-new' (or 'image-parse') folder exists and is populated.")
        return

    # Iterate through all PNG files in the input_parse_dir
    for parse_file_name in os.listdir(input_parse_dir):
        if parse_file_name.lower().endswith('.png'): # Case-insensitive check
            input_path = osp.join(input_parse_dir, parse_file_name)
            output_path = osp.join(output_mask_dir, parse_file_name) # Save with the same filename

            generate_image_mask_from_parse(input_path, output_path)
            processed_count += 1

    print(f"--- Finished processing {datamode} mode. Total masks generated: {processed_count} ---")


if __name__ == "__main__":
    
    data_directory_root = 'data/' 

    # process_all_parse_images(data_directory_root, datamode='train')
    process_all_parse_images(data_directory_root, datamode='test')