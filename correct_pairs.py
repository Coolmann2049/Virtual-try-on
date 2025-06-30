'''
The provided test_pairs.txt and train_pairs.txt in the dataset do not match with this repo's needs
So this script allows us to pair the 2. 
'''

import os

def generate_simple_image_pairs_file(
    source_images_dir: str,
    output_filepath: str,
    image_extensions=('.jpg', '.jpeg', '.png')
):
    """
    Generates a text file listing image pairs where each image is paired with itself.
    This is suitable when the cloth image has the same filename as the person image,
    and both exist in their respective directories.

    Args:
        source_images_dir (str): Path to the directory containing the images to list (e.g., 'data/train/image/').
                                 These filenames will be used for both parts of the pair.
        output_filepath (str): Path and filename for the output .txt file (e.g., 'data/train_pairs.txt').
        image_extensions (tuple): A tuple of valid image file extensions to include.
    """

    pairs = []
    
    # Get a list of all relevant image files in the source directory
    image_filenames = [
        f for f in os.listdir(source_images_dir)
        if os.path.isfile(os.path.join(source_images_dir, f)) and f.lower().endswith(image_extensions)
    ]
    image_filenames.sort() # Ensure consistent order

    for img_name in image_filenames:
        # Each image is paired with itself, as per your dataset structure
        pairs.append(f"{img_name} {img_name}")

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    # Write the pairs to the output file
    with open(output_filepath, 'w') as f:
        for pair_line in pairs:
            f.write(pair_line + '\n')

    print(f"\nGenerated {len(pairs)} pairs and saved to '{output_filepath}'")

# --- Configuration ---
# You need to set these paths to match your actual data directory structure

# For generating train_pairs.txt:
PERSON_IMAGES_TRAIN_DIR = 'data/train/image/' # Your directory with person images (e.g., 000003_0.jpg)
OUTPUT_TRAIN_PAIRS_FILE = 'data/train_pairs.txt' # The output .txt file

# For generating test_pairs.txt (if needed, assuming similar structure):
PERSON_IMAGES_TEST_DIR = 'data/test/image/'
OUTPUT_TEST_PAIRS_FILE = 'data/test_pairs.txt'

# For generating test_pairs_same.txt (this is a common use case for this exact pairing):
# If your test_pairs_same.txt also follows this pattern, you would use this.
PERSON_IMAGES_TEST_SAME_DIR = 'data/test/image/' # Often the same as test/image
OUTPUT_TEST_PAIRS_SAME_FILE = 'data/test_pairs_same.txt'


# --- Run the script ---
if __name__ == "__main__":
    print("--- Generating train_pairs.txt ---")
    generate_simple_image_pairs_file(
        source_images_dir=PERSON_IMAGES_TRAIN_DIR,
        output_filepath=OUTPUT_TRAIN_PAIRS_FILE
    )



    print("\n--- Generating test_pairs_same.txt ---")
    generate_simple_image_pairs_file(
        source_images_dir=PERSON_IMAGES_TEST_SAME_DIR,
        output_filepath=OUTPUT_TEST_PAIRS_SAME_FILE
    )

    print("\nScript finished.")