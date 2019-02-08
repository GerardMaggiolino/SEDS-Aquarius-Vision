'''
Converts all non RGB images to RGB through PIL when run. 
'''
from PIL import Image
from os import listdir

def main(): 
    dir_names = ['safe_1', 'safe_2', 'safe_3', 'safe_4', 'safe_5']
    # Get number of images per subdirectory 
    num_images = [len([n for n in listdir(f'labelled/{name}')]) - 1 
        for name in dir_names]

    for num_img, dir_name in zip(num_images, dir_names): 
        for num in range(num_img):
            f = f'labelled/{dir_name}/{num}.tiff'
            img = Image.open(f)
            if img.mode != 'RGB': 
                img.convert('RGB').save(f)
                print(f'Converted {f} to RGB from {img.mode}')

if __name__ == '__main__': 
    main()
