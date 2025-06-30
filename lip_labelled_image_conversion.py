from PIL import Image
import numpy as np
import os

# --- STEP 1: DEFINE YOUR RGB-TO-LIP MAPPING ---
# YOU MUST CUSTOMIZE THIS PART!
# This dictionary maps the exact RGB color tuples found in your
# input (RGB) parsing images to the standard LIP integer labels (0-20).
#
# How to get your RGB tuples:
# 1. Open one of your RGB parsing images (e.g., from your 'image-parse' folder)
#    in an image editor (like GIMP, Photoshop, Paint.net, or even MS Paint's eyedropper tool).
# 2. Use the eyedropper tool to pick colors from different semantic regions (e.g., face, hair, upper clothes, background).
# 3. Note down the exact RGB (R, G, B) values for each unique color.
# 4. Map these RGB values to the corresponding LIP integer IDs (0-20)
#    from the LIP legend in cp_dataset.py's comments (e.g., 13 for Face, 5 for UpperClothes).
#
# Example (YOU NEED TO REPLACE THESE WITH YOUR ACTUAL VALUES):
your_rgb_to_lip_map = {
    # --- Standard LIP labels (refer to cp_dataset.py comments) ---
    (0, 0, 0): 0,      # Background
    (128, 0, 0): 1,    # Hat
    (254, 0, 0): 2,    # Hair
    (0, 85, 0): 3,     # Glove
    (170, 0, 51): 4,   # Sunglasses
    (254, 85, 0): 5,   # UpperClothes
    (0, 0, 85): 6,     # Dress
    (0, 119, 221): 7,  # Coat
    (85, 85, 0): 8,    # Socks
    (0, 85, 85): 9,    # Pants
    (170, 170, 50): 10,   # Jumpsuits
    (52, 86, 128): 11, # Scarf
    (0, 128, 0): 12,   # Skirt
    (0, 0, 254): 13,   # Face
    (51, 169, 220): 14, # LeftArm
    (0, 254, 254): 15, # RightArm
    (85, 255, 170): 16, # LeftLeg
    (169, 254, 85): 17, # RightLeg
    (255, 255, 0): 18, # LeftShoe
    (255, 170, 0): 19, # RightShoe
    (85, 51, 0): 20, # Skin/Neck/Chest

}

your_rgb_parse_input_dir = 'data/test/image-parse-v3/'
#your_rgb_parse_input_dir = 'data/train/image-parse-v3/'

lip_grayscale_output_dir = 'data/test/image-parse-new/'
#lip_grayscale_output_dir = 'data/train/image-parse-new/'

# --- Create the output directory if it doesn't exist ---
if not os.path.exists(lip_grayscale_output_dir):
    os.makedirs(lip_grayscale_output_dir)

# --- STEP 3: RUN THE CONVERSION ---
print(f"Starting conversion from '{your_rgb_parse_input_dir}' to '{lip_grayscale_output_dir}'...")

processed_count = 0
error_count = 0

for filename in os.listdir(your_rgb_parse_input_dir):
    # Process only image files (PNGs are common for segmentation masks)
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        input_filepath = os.path.join(your_rgb_parse_input_dir, filename)
        # Ensure output is always a PNG for consistency
        output_filename = os.path.splitext(filename)[0] + '.png'
        output_filepath = os.path.join(lip_grayscale_output_dir, output_filename)

        try:
            # Open the image in RGB mode
            img_rgb = Image.open(input_filepath).convert('RGB')
            img_array = np.array(img_rgb) # Convert to NumPy array (H, W, 3)

            # Create an empty array for the new grayscale LIP-compatible labels
            # Initialize with background (0) or a default value
            lip_label_array = np.zeros(img_array.shape[:2], dtype=np.uint8)

            # Iterate through your defined RGB-to-LIP mapping
            for rgb_tuple, lip_id in your_rgb_to_lip_map.items():
                # Find all pixels in the image that match the current RGB color
                target_rgb_np = np.array(rgb_tuple).reshape(1, 1, 3)
                matches = np.all(np.abs(img_array - target_rgb_np) <= 3, axis=2)
                # Assign the corresponding LIP ID to those matching pixels
                lip_label_array[matches] = lip_id

            # Save the new grayscale image with LIP integer labels
            # 'L' mode signifies a single-channel 8-bit grayscale image
            Image.fromarray(lip_label_array, mode='L').save(output_filepath)
            print(f"Converted: {filename} -> {output_filename}")
            processed_count += 1

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            error_count += 1

print(f"\nConversion complete! Processed {processed_count} images, {error_count} errors.")
print(f"LIP-compatible grayscale images are saved in: {lip_grayscale_output_dir}")