import torch
import numpy as np
import matplotlib.pyplot as plt
from cnn import LanderCNN
import matplotlib.pyplot as plt
from torchvision import transforms 
from matplotlib import colors
from PIL import Image


def main(): 
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    model = LanderCNN()
    model.load_state_dict(torch.load('281213-model.pt'))
    
    image_names = [f'data/test/{i}.tiff' for i in range(3)]
    for name in image_names: 
        classify(model, Image.open(name), transform)


def classify(model, img, transform): 
    ''' 
    Performs visualized classification over given image with model. 

    Only works with single channel images.
    '''
    model.eval()

    sub_pix = 125
    width, height = img.size

    # Number of strides 
    left_strides = int(width / sub_pix) 
    down_strides = int(height / sub_pix) 
    # Sauce for change of width from anchor points
    if width % sub_pix != 0: 
        addin = sub_pix % 2
        left_pix_delta = (width - sub_pix) / (width // sub_pix) 
        left_pixel_anchors = [int(sub_pix // 2 + left_pix_delta * i + 0.5) 
            for i in range(left_strides + 1)] 
        left_bounds = [(anchor - int(sub_pix // 2), anchor + 
            int(sub_pix // 2 + addin)) for anchor in left_pixel_anchors]
        left_strides += 1
    else: 
        left_bounds = [(sub_pix * i, sub_pix * (i + 1)) for i in 
            range(int(width / sub_pix))]
    if height % sub_pix != 0: 
        addin = sub_pix % 2
        down_pix_delta = (height - sub_pix) / (height // sub_pix)
        down_pixel_anchors = [int(sub_pix // 2 + down_pix_delta * i + 0.5) 
            for i in range(down_strides + 1)] 
        down_bounds = [(anchor - int(sub_pix // 2), anchor + 
            int(sub_pix // 2 + addin)) for anchor in down_pixel_anchors]
        down_strides += 1
    else: 
        down_bounds = [(sub_pix * i, sub_pix * (i + 1)) for i in 
            range(int(height / sub_pix))]

    # Get image prepped for forward pass and labels
    labels = np.zeros((left_strides, down_strides))

    img = transform(img)
    img = img.view(-1, *img.shape)

    for l_ind, left in enumerate(left_bounds):
        for d_ind, down in enumerate(down_bounds): 
            # This indexing needs to change for multi channel images
            logit = (model.forward(img[0][0][down[0]:down[1],left[0]:left[1]].
                view(1, 1, img.shape[0], img.shape[1])))
            model_label = 0
            for neuron in torch.sigmoid(log[0]):
                if neuron > 0.5: 
                    model_label += 1
                else: 
                    break
            labels[d_ind][l_ind] = model_label 

    # Visual display 
    cmap = colors.ListedColormap(['r', 'xkcd:orange', 'y', 'xkcd:aqua', 
        'xkcd:darkgreen'])
    trans = transforms.ToPILImage()        
    img = trans(img[0])
    extent = (0, img.size[0], 0, img.size[1])
    fig = plt.figure(frameon=False)        
    plt.imshow(img, extent=extent)
    plt.imshow(labels,alpha=.15, extent=extent, cmap=cmap, vmin=0, vmax=5)
    plt.colorbar()
    plt.show()


if __name__ == '__main__': 
    main()

