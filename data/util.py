''' 
Utilities for labelling data. 
'''

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
from os import rename
from shutil import copyfile
from shutil import copy2
from torchvision import transforms
Image.MAX_IMAGE_PIXELS = None


def main(): 
    # extract_portions('raw/raw_1.tiff', 250, 'regions/raw_1', 55)
    # extract_portions('raw/raw_2.tiff', 250, 'regions/raw_2', 1)
    # extract_portions('raw/raw_3.tiff', 250, 'regions/raw_3', 20)
    # extract_portions('raw/raw_5.tiff', 250, 'regions/raw_5', 35)   

    # partition_images(250)
    greyscale_transform()

def greyscale_transform(): 
    num = 0
    while True: 
        try: 
            img = Image.open(f'labelled/original/{num}.tiff').convert('L')
            img.save(f'labelled/processed/{num}.tiff')
            num += 1
        except: 
            break

def change_to_csv(): 
    from_dir = 'labelled/safe'
    to_dir = 'labelled/original'
    csv = 'labelled/labels.csv'
    dirs = [1, 2, 3, 4, 5]
    put_ind = 0
    take_ind = 0
    with open(csv, 'a') as lab: 
        for d in dirs: 
            while True:
                try: 
                    pic_name = f'{from_dir}_{d}/{take_ind}.tiff'
                    copyfile(pic_name, f'{to_dir}/{put_ind}.tiff')
                    lab.write(f'{put_ind}.tiff,{d}\n')
                    take_ind += 1
                    put_ind += 1
                except FileNotFoundError: 
                    take_ind = 0
                    break
        


def partition_images(pixels): 
    filename = 'regions/raw'
    dest = 'labelled/safe'
    pic = [2]
    accepted = [3, 4, 5]
    discard = [0, 1, 2]
    disp = plt.imshow(np.zeros((pixels, pixels)))

    for targ_num in pic: 
        val = 0
        for _ in range(500):
            pic_name = f'{filename}_{targ_num}_{val}.tiff'
            try: 
                img = Image.open(pic_name)
                disp.set_data(img)
                plt.draw()
                plt.pause(0.01)

                while True: 
                    try: 
                        choice = int(input()) 
                        if choice in accepted or choice in discard: 
                            break
                    except ValueError:
                        pass
                    except:
                        exit()
                    print('Incorrect Choice')

                if choice in accepted: 
                    print(f'Moving {pic_name} to {dest}_{choice}')
                    rename(pic_name, f'{dest}_{choice}/{targ_num}_{val}.tiff')
                val += 1
            except FileNotFoundError: 
                val += 1

def extract_portions(filename, region_size, dest_name, drop=10): 
    print(f'Extracting {filename}')
    img = Image.open(filename) #.convert('L')
    w, h = img.size
    saved = 0
    skipped = 0
    for sw in range(0, w - region_size, region_size): 
        for sh in range(0, h - region_size, region_size): 
            if random.randint(1, drop) != 1:
                skipped += 1
                continue

            bounds = [
                (sw, sh), 
                (sw + region_size - 1, sh), 
                (sw, sh + region_size - 1),
                (sw + region_size - 1 , sh + region_size - 1)
            ]
            pix = [img.getpixel(bound) for bound in bounds]
            if (0,0,0) in pix: 
                skipped += 1
                continue
            
            # print(f'Saving {saved} ... at {sw}, {sh}')
            part = img.crop((sw, sh, sw+region_size, sh+region_size))
            part.save(f'{dest_name}_{saved}.tiff')
            saved += 1

    print(f'Total pictures saved: {saved}')
    print(f'Total pictures skipped: {skipped}')

if __name__ == '__main__': 
    main()

