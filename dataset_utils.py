import os
import numpy as np
from PIL import Image
import glob
from regions import Regions
from astropy.io import fits
from astropy.visualization import ZScaleInterval, ImageNormalize, LogStretch, SqrtStretch
import matplotlib.pyplot as plt
from tqdm import tqdm
from shapely.geometry import Polygon, Point
from shapely import box
import random
import shutil
from IPython.display import display, clear_output



#  Define global directory paths
global_dir = ""
ann_train_path = ""
ann_val_path = ""
img_train_path = ""
img_val_path = ""




def init(name='reggie', scale_factor=3, shift=False, shuffle=None, radio_factor=1, halpha_factor=1):
    """
    Input: 
        name: The name of the dataset to be created.
        scale_factor: The 'radius' of the cropped image is proportional to the radius of its corresponding region by this factor.
        shift: If true, it will create duplicate images of each region with the region shifted in different positions in the crop.
               Increases the size of dataset by a factor of 5.
        shuffle: Integer. Uses the seed to shuffle the order of the dataset images.
        radio_factor: Scales the data of the radio image by this factor before any crops are made.
        halpha_factor: Scales the data of the halpha image by this factor before any crops are made.

    Output: None
    """

    abort = create_dirs(name)
    if abort:
        return

    #  Generate combined image
    img = gen_img(radio_factor,  halpha_factor)

    #  Retrieve list of Polygons and corresponding labels
    polygons, labels = get_polygons_and_labels()
      
    #  Generate list of bounding boxes for dataset generation
    bboxes = gen_bboxes(polygons, scale_factor, shift, img.shape[:2])

    #  Load or generate & save annotation array
    ann = gen_ann(polygons, labels, img.shape)

    #  Generate dataset
    gen_basesem(img, ann, bboxes, shuffle=shuffle)




def create_dirs(name):
    """
    Input: The name of the parent folder of the to-be created dataset. This method creates the directory structure based off the dataset name.

    Output: The global directory path
    """
    
    global global_dir, ann_train_path, ann_val_path, img_train_path, img_val_path

    global_dir = os.path.join('DATASET', name)
    ann_train_path = os.path.join(global_dir,'data','my_dataset','ann','train')
    ann_val_path = os.path.join(global_dir,'data','my_dataset','ann','val')
    img_train_path = os.path.join(global_dir,'data','my_dataset','img','train')
    img_val_path = os.path.join(global_dir,'data','my_dataset','img','val')

    #  Check if dataset already exists. Prompt user to overwrite or abort
    if os.path.exists(global_dir):
        print(global_dir + ' exists!')
        proceed = input('Overwrite? [y/n] ')
        if proceed == 'n':
            print('Exiting...')
            return True
        print('Overwriting directories...')
        shutil.rmtree(global_dir)
    print('Creating directories...')
    os.makedirs(global_dir)
    os.makedirs(ann_train_path)
    os.makedirs(ann_val_path)
    os.makedirs(img_train_path)
    os.makedirs(img_val_path)
    return False




def gen_img(radio_factor, halpha_factor):
    """
    Input: The radio and halpha factors to scale the pixel data.

    Output: The combined 2-channel full size image data
    """

    print('Generating image...')

    # Halpha image
    img_h = cleanup_data(fits.open('./LMC/lmc_ha_csub.fits')[0].data)
    
    # Clip upper bound to 1500
    img_h = np.clip(img_h, a_min=None, a_max=1500)

    # Radio image
    img_r = cleanup_data(fits.getdata('./LMC/lmc_askap.fits')[0][0])
    
    # Scale radio by factor
    img_r = img_r * radio_factor

    # Scale halpha by factor
    img_h = img_h * halpha_factor
    
    # Merge images and return
    return merge_images(img_r, img_h)




def get_polygons_and_labels():
    """
    Input: None
    
    Output: Returns the list of region polygons, and a list of their corresponding labels (1 for HII, 2 for SNR)
    """

    print('Generating polygons and labels...')

    #  Define the HII and SNR region files
    HII_reg_files = glob.glob(os.path.join('./LMC/HII_boundaries', '*.reg'))
    SNR_reg_files = glob.glob(os.path.join('./LMC/SNR_boundaries', '*.reg'))

    #  Remove bad files
    HII_reg_files.remove('./LMC/HII_boundaries/mcels-l381.reg')
    HII_reg_files.remove('./LMC/HII_boundaries/mcels-l279.reg')

    # List of shapely Polygon objects
    polygons = []

    # Their corresponding labels
    labels = []

    #  Background is 0

    #  HII Regions -> Label 1
    for file in HII_reg_files:
        curr_region = Regions.read(file, format='ds9')
        Xs = curr_region[0].vertices.x
        Ys = curr_region[0].vertices.y
        polygons.append(Polygon(list(zip(Xs, Ys))))
        labels.append(1)

    #  SNR Regions -> Label 2
    for file in SNR_reg_files:
        curr_region = Regions.read(file, format='ds9')
        Xs = curr_region[0].vertices.x
        Ys = curr_region[0].vertices.y
        polygons.append(Polygon(list(zip(Xs, Ys))))
        labels.append(2)

    return polygons, labels




def gen_bboxes(polygons, scale_factor, shift, imgshape):
    """
    Input: List of Polygons, scale factor, shift, and image shape for corrections

    Output: The list of (corrected) bounding boxes
    """

    print('Generating bounding boxes...')

    #  Define the list to hold the bounding boxes. If shift=True, creating 5 bounding boxes per polygon
    #  Each bbox is (min_x, min_y, max_x, max_y)
    bboxes = []

    for poly in polygons:
        min_x, min_y, max_x, max_y = poly.bounds
        width = max_x - min_x
        height = max_y - min_y
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        #  Calculate the new larger bounding box
        new_min_x = min_x - width * (scale_factor - 1) / 2
        new_max_x = max_x + width * (scale_factor - 1) / 2
        new_min_y = min_y - height * (scale_factor - 1) / 2
        new_max_y = max_y + height * (scale_factor - 1) / 2
        bboxes.append((int(new_min_x), int(new_min_y), int(new_max_x), int(new_max_y)))

        if shift:
            #  Create shifted bboxes
            new_width = new_max_x - new_min_x
            new_height = new_max_y - new_min_y

            #  Shift North West
            new_max_x = center_x
            new_min_x = new_max_x - new_width
            new_max_y = center_y
            new_min_y = new_max_y - new_height
            bboxes.append((int(new_min_x), int(new_min_y), int(new_max_x), int(new_max_y)))

            #  Shift North East
            new_min_x = center_x
            new_max_x = new_min_x + new_width
            new_max_y = center_y
            new_min_y = new_max_y - new_height
            bboxes.append((int(new_min_x), int(new_min_y), int(new_max_x), int(new_max_y)))

            #  Shift South East
            new_min_x = center_x
            new_max_x = new_min_x + new_width
            new_min_y = center_y
            new_max_y = new_min_y + new_height
            bboxes.append((int(new_min_x), int(new_min_y), int(new_max_x), int(new_max_y)))

            # Shift South West
            new_max_x = center_x
            new_min_x = new_max_x - new_width
            new_min_y = center_y
            new_max_y = new_min_y + new_height
            bboxes.append((int(new_min_x), int(new_min_y), int(new_max_x), int(new_max_y)))

    return correct_bounding_boxes(bboxes, imgshape)




def gen_ann(polygons, labels, img_shape):
    """ 
    Input: Polygons and their corresponding labels. Also the image shape
    
    Output: The annotation array. Also saves the annotation array to file for future use
    """

    if os.path.exists('ann.npy'):
        print('Loading the annotation array...')
        ann = np.load('ann.npy')

    else:
        print('Generating the annotation array...')
        ann = np.zeros(img_shape[:2], dtype=np.uint8)
        
        #  Sort polygons largest to smallest by area
        sorted_polygons = sorted(polygons, key=(lambda polygon: polygon.area), reverse=True)
        for i, poly in enumerate(tqdm(sorted_polygons)):
            min_x, min_y, max_x, max_y = poly.bounds
            label = labels[i]

            for x in range(int(min_x), int(max_x) + 1):
                for y in range(int(min_y), int(max_y) + 1):
                    point = Point(x, y)
                    if poly.contains(point):
                        #  Point is either HII or SNR
                        ann[y, x] = label
        
        #  Save ann.npy to file
        file_path = 'ann.npy'
        np.save(file_path, ann)
        print('Saved ' + file_path)

    #  Save ann.png to file
    ground_truth_image = Image.fromarray(ann)
    file_path = 'ann.png'
    ground_truth_image.save(file_path)
    print('Saved ' + file_path)

    return ann



def gen_basesem(img, ann, bboxes, shuffle=None):
    """
    Input: The full img and ann images, and the list of bounding boxes for crop images. Also the optional shuffle seed to shuffle the dataset order

    Output: None. Just creates and saves the base semantic dataset.
    """

    #  Shuffle if necessary
    if shuffle is not None:
        r = random.Random(shuffle)
        r.shuffle(bboxes)

    #  Divide bboxes into train and val lists
    val_ratio = 0.2
    split_point = int(len(bboxes) * val_ratio)
    train_bboxes = bboxes[split_point:]
    val_bboxes = bboxes[:split_point]
    
    #  Loop training boxes
    print('Looping training bboxes...')
    for i, bbox in enumerate(tqdm(train_bboxes)):
        min_x, min_y, max_x, max_y = bbox
        data = img[min_y:max_y,min_x:max_x,:]
        img_save = os.path.join(img_train_path, str(i)+'.npy')
        np.save(img_save, data)

        cur_ann = Image.fromarray(ann[min_y:max_y,min_x:max_x])
        ann_save = os.path.join(ann_train_path, str(i)+'.png')
        cur_ann.save(ann_save)

    #  Loop validation boxes
    print('Looping validation bboxes...')
    for i, bbox in enumerate(tqdm(val_bboxes)):
        min_x, min_y, max_x, max_y = bbox
        data = img[min_y:max_y,min_x:max_x,:]
        img_save = os.path.join(img_val_path, str(i)+'.npy')
        np.save(img_save, data)

        cur_ann = Image.fromarray(ann[min_y:max_y,min_x:max_x])
        ann_save = os.path.join(ann_val_path, str(i)+'.png')
        cur_ann.save(ann_save)





#  Helper functions

def printImgStats(img):
    """
    Input: An image

    Output: A string containing statistics about the image data
    """

    v = list(np.percentile(img, [10, 25, 50, 75, 90, 99]))
    s = f"avg:\t{np.mean(img)}"
    s += f"\nstd:\t{np.std(img)}"
    s += f"\nmin:\t{np.min(img)}"
    s += f"\n10:\t{v[0]}"
    s += f"\n25:\t{v[1]}"
    s += f"\n50:\t{v[2]}"
    s += f"\n75:\t{v[3]}"
    s += f"\n90:\t{v[4]}"
    s += f"\n99:\t{v[5]}"
    s += f"\nmax:\t{np.max(img)}\n"
    print(s)



def merge_images(img0, img1):
    """
    Input: Two 1-channel images
    
    Output: One two channel image, with image1 in the zeroth channel and image2 in the first channel
    """

    # Create a 3-channel numpy array filled with zeros
    result_image = np.zeros((img0.shape[0], img0.shape[1], 2), dtype=np.float32)
    # Set the zeroth channel of the result image to be the values from image1
    result_image[:, :, 0] = img0
    # Set the first channel of the result image to be the values from image2
    result_image[:, :, 1] = img1
    return result_image.astype(np.float32)




def cleanup_data(img):
    """
    Input: An image

    Output: The image with all nans and -10000s set to img.min()
    """

    # Define the minimum value (excluding NaNs and -10000s)
    min_value = np.nanmin(img[img != -10000])
    # Replace NaN values and -10000 with the minimum value
    img = np.where(np.logical_or(np.isnan(img), img == -10000), min_value, img)
    return img




def correct_bounding_boxes(bounding_boxes, image_shape):
    """
    Corrects bounding boxes to ensure they are within the bounds of the image.

    Parameters:
    - bounding_boxes (list): List of bounding boxes in the format [x_min, y_min, x_max, y_max].
    - image_shape (tuple): Shape of the image array in the format (height, width).

    Returns:
    - corrected_boxes (list): List of corrected bounding boxes, ensuring they are within bounds of the image.
    """
    
    #print('Correcting bounding boxes...')

    corrected_boxes = []

    for box in bounding_boxes:
        x_min, y_min, x_max, y_max = box

        # Ensure x_min is within bounds
        x_min = max(0, x_min)
        # Ensure y_min is within bounds
        y_min = max(0, y_min)
        # Ensure x_max is within bounds
        x_max = min(image_shape[1], x_max)
        # Ensure y_max is within bounds
        y_max = min(image_shape[0], y_max)

        # Check if the bounding box is valid (non-empty)
        if x_min < x_max and y_min < y_max:
            corrected_boxes.append([x_min, y_min, x_max, y_max])

    return corrected_boxes




