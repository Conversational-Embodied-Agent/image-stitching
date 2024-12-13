import os
from PIL import Image


def batch_convert_ppm_to_jpg(input_folder, output_folder, nth_image):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    ppm_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith('.ppm')])

    for i in range(0, len(ppm_files), nth_image):
        filename = ppm_files[i]
        input_path = os.path.join(input_folder, filename)

        with Image.open(input_path) as img:
            img = img.convert('RGB')
            output_path = os.path.join(output_folder, f'{os.path.splitext(filename)[0]}.jpg')
            img.save(output_path, 'JPEG')
            print(f'Converted {filename} to {output_path}')


if __name__ == '__main__':
    input_folder = '/home/g/gajdosech2/icup/sequence3ppm'
    output_folder = '/home/g/gajdosech2/icup/sequence3'
    # Specify the nth image to convert (e.g., every 3rd image)
    M = 1
    batch_convert_ppm_to_jpg(input_folder, output_folder, M)