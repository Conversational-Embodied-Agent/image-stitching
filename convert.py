import os
from PIL import Image

def batch_convert_ppm_to_jpg(input_folder, output_folder, nth_image):
    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all .ppm files in the folder and sort them in canonical order
    ppm_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith('.ppm')])

    # Process every nth file in the sorted list
    for i in range(0, len(ppm_files), nth_image):
        filename = ppm_files[i]
        input_path = os.path.join(input_folder, filename)

        # Open and convert the image to RGB, then save as .jpg
        with Image.open(input_path) as img:
            img = img.convert("RGB")
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.jpg")
            img.save(output_path, "JPEG")
            print(f"Converted {filename} to {output_path}")

# Paths to the input and output folders
input_folder = '/home/g/gajdosech2/icup/sequence3ppm'
output_folder = '/home/g/gajdosech2/icup/sequence3'

# Specify the nth image to convert (e.g., every 3rd image)
nth_image = 1

# Run the batch conversion
batch_convert_ppm_to_jpg(input_folder, output_folder, nth_image)