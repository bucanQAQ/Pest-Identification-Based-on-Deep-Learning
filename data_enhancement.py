
from tqdm import tqdm


def flip(root_path,img_name):   #Flip Image
    img = Image.open(os.path.join(root_path, img_name))
    background = Image.new('RGBA', img.size, (255, 255, 255))

    # Turn background white
    if img.mode == 'RGBA':
        img = Image.alpha_composite(background, img)
        img = img.convert('RGB')

    filp_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # filp_img.save(os.path.join(root_path,img_name.split('.')[0] + '_flip.jpg'))
    return filp_img

def rotation_90(root_path, img_name):
    img = Image.open(os.path.join(root_path, img_name))
    background = Image.new('RGBA', img.size, (255, 255, 255))

    # Turn background white
    if img.mode == 'RGBA':
        img = Image.alpha_composite(background, img)
        img = img.convert('RGB')
    rotation_img = img.rotate(90)
    # rotation_img.save(os.path.join(root_path,img_name.split('.')[0] + '_rotation.jpg'))
    return rotation_img

def rotation_180(root_path, img_name):
    img = Image.open(os.path.join(root_path, img_name))
    background = Image.new('RGBA', img.size, (255, 255, 255))

    # Turn background white
    if img.mode == 'RGBA':
        img = Image.alpha_composite(background, img)
        img = img.convert('RGB')
    rotation_img = img.rotate(180)
    # rotation_img.save(os.path.join(root_path,img_name.split('.')[0] + '_rotation.jpg'))
    return rotation_img

def rotation_270(root_path, img_name):
    img = Image.open(os.path.join(root_path, img_name))
    background = Image.new('RGBA', img.size, (255, 255, 255))

    # Turn background white
    if img.mode == 'RGBA':
        img = Image.alpha_composite(background, img)
        img = img.convert('RGB')
    rotation_img = img.rotate(270)
    # rotation_img.save(os.path.join(root_path,img_name.split('.')[0] + '_rotation.jpg'))
    return rotation_img


def contrastEnhancement(root_path, img_name):
    img = Image.open(os.path.join(root_path, img_name))
    background = Image.new('RGBA', img.size, (255, 255, 255))

    # Turn background white
    if img.mode == 'RGBA':
        img = Image.alpha_composite(background, img)
        img = img.convert('RGB')
    enh_con = ImageEnhance.Contrast(img)
    contrast = 1

    img_contrasted = enh_con.enhance(contrast)
    return img_contrasted

def brightnessEnhancement(root_path,img_name):#亮度增强
    img = Image.open(os.path.join(root_path, img_name))
    background = Image.new('RGBA', img.size, (255, 255, 255))

    # Turn background white
    if img.mode == 'RGBA':
        img = Image.alpha_composite(background, img)
        img = img.convert('RGB')
    enh_bri = ImageEnhance.Brightness(img)
    brightness = 1
    image_brightened = enh_bri.enhance(brightness)
    return image_brightened

def colorEnhancement(root_path,img_name):
    img = Image.open(os.path.join(root_path, img_name))
    background = Image.new('RGBA', img.size, (255, 255, 255))

    # Turn background white
    if img.mode == 'RGBA':
        img = Image.alpha_composite(background, img)
        img = img.convert('RGB')
    enh_col = ImageEnhance.Color(img)
    color = 1.0
    image_colored = enh_col.enhance(color)
    return image_colored



from PIL import Image
from PIL import ImageEnhance
import os

def data_enhance(imagepath,savepath):
    class_names = os.listdir(imagepath)

    for class_name in tqdm(class_names,desc='Data augmentation is in progress'):

        a = 1
        img_path = os.path.join(imagepath, class_name)
        save_path = os.path.join(savepath, class_name)
        file_names=os.listdir(img_path)
        for name in file_names:

            saveName = str(a) + "id_" + class_name + ".jpeg"
            image = Image.open(os.path.join(img_path, name))

            background = Image.new('RGBA',image.size,(255,255,255))

            #Turn background white
            if image.mode == 'RGBA':
                image = Image.alpha_composite(background, image)
                image = image.convert('RGB')
            image.save(os.path.join(save_path, saveName))

            #
            # saveName = str(a) + "be_"+class_name+".png"
            # saveImage = brightnessEnhancement(img_path, name)
            # saveImage.save(os.path.join(save_path,saveName))

            # Rotate 90°
            saveName = str(a) + "rot90_" + class_name + ".jpeg"
            saveImage = rotation_90(img_path, name)
            if saveImage.mode == 'P' or saveImage.mode == 'RGBA':
                saveImage = saveImage.convert('RGB')
            saveImage.save(os.path.join(save_path, saveName))

            # Rotate 180°
            saveName = str(a) + "rot180_" + class_name + ".jpeg"
            saveImage = rotation_180(img_path, name)
            if saveImage.mode == 'P' or saveImage.mode == 'RGBA':
                saveImage = saveImage.convert('RGB')
            saveImage.save(os.path.join(save_path, saveName))

            # Rotate 270°
            saveName = str(a) + "rot270_" + class_name + ".jpeg"
            saveImage = rotation_270(img_path, name)
            if saveImage.mode == 'P' or saveImage.mode == 'RGBA':
                saveImage = saveImage.convert('RGB')
            saveImage.save(os.path.join(save_path, saveName))

            # flip
            saveName = str(a) + "fl_" + class_name + ".jpeg"
            saveImage = flip(img_path, name)

            saveImage.save(os.path.join(save_path, saveName))

            # contrast enhancement
            # saveName = str(a) + "con_"+class_name+".jpg"
            # saveImage = contrastEnhancement(img_path, name)
            # saveImage.save(os.path.join(save_path,saveName))

            a = a + 1

    print("Data augmentation complete")

def test(imagepath,savepath):
    class_names = os.listdir(imagepath)

    for class_name in tqdm(class_names,desc='Data augmentation is in progress'):


        img_path = os.path.join(imagepath, class_name)
        save_path = os.path.join(savepath, class_name)
        file_names=os.listdir(img_path)

        for name in file_names:
            saveImage = colorEnhancement(img_path, name)
            saveImage.save(os.path.join(save_path, name))
            saveImage = contrastEnhancement(save_path, name)
            saveImage.save(os.path.join(save_path, name))

