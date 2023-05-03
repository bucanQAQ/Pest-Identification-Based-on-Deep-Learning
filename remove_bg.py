import os
from PIL import Image

from rembg.bg import remove


def remove_background(input_path,output_path):

    with open(input_path, 'rb') as i:
        with open(output_path, 'wb') as o:
            input = i.read()
            output = remove(input)
            o.write(output)

    # Convert background to white
    img = Image.open(output_path)
    background = Image.new('RGBA', img.size, (255, 255, 255))
    if img.mode == 'RGBA':
        img = Image.alpha_composite(background, img)
        img = img.convert('RGB')
    img.save(output_path)

