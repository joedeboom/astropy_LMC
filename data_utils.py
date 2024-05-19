import os
import json
import numpy as np
from PIL import Image
from astropy.visualization import ZScaleInterval, ImageNormalize, LogStretch, SqrtStretch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from IPython.display import display, clear_output
from dataset_utils import merge_images
from tabulate import tabulate



def create_mask(annotation, region):
    """
    Create a mask for a specific region based on the annotation.

    Parameters:
    - annotation: numpy array containing the annotation mask
    - region: integer representing the region of interest (1 for HII, 2 for SNR)

    Returns:
    - mask: binary mask where 1 indicates the region of interest
    """
    mask = np.zeros_like(annotation)
    mask[annotation == region] = 1
    return mask



def compute_avg(image, mask):
    """
    Compute the average pixel value of the image within the masked region.

    Parameters:
    - image: numpy array representing the image
    - mask: binary mask indicating the region of interest

    Returns:
    - average_value: average pixel value within the masked region
    """
    masked_image = image[mask == 1]
    if len(masked_image) == 0:
        return np.nan  # Return NaN if the region is empty
    else:
        return float(np.mean(masked_image))




def getRoiStats(img, ann):
    """
    Input: An image and its corresponding annotation

    Output: A string containing statistics of the region of interest
    """

    s = {}
    s['Radio'] = {'HII':compute_avg(img[:,:,0], create_mask(ann,1)), 'SNR':compute_avg(img[:,:,0], create_mask(ann,2))}
    s['Halpa'] = {'HII':compute_avg(img[:,:,1], create_mask(ann,1)), 'SNR':compute_avg(img[:,:,1], create_mask(ann,2))}
    return json.dumps(s, indent=4)




def getImgStats2(img):
    """
    Input: An image
    
    Output: A formatted table containing statistics about both channels of the image data
    """

    channels = ['Radio', 'Halpha']
    statistics = ["Avg", "Std", "Min", "10", "25", "50", "75", "90", "99", "Max"]
    
    # Precompute statistics for each channel
    channel_stats = {}
    for ch, channel_name in enumerate(channels):
        channel_data = img[:, :, ch]
        percentiles = np.percentile(channel_data, [10, 25, 50, 75, 90, 99])
        channel_stats[channel_name] = {
            "Avg": np.mean(channel_data),
            "Std": np.std(channel_data),
            "Min": np.min(channel_data),
            "10": percentiles[0],
            "25": percentiles[1],
            "50": percentiles[2],
            "75": percentiles[3],
            "90": percentiles[4],
            "99": percentiles[5],
            "Max": np.max(channel_data)
        }
    
    # Assemble the data into the desired format
    stats = [[stat] + [channel_stats[channel][stat] for channel in channels] for stat in statistics]
    headers = ["Statistic"] + channels
    return tabulate(stats, headers=headers, tablefmt="grid")




def getImgStats(img):
    """
    Input: An image
    
    Output: A string containing statistics about the image data
    """

    v = list(np.percentile(img, [10, 25, 50, 75, 90, 99]))
    s = f"\navg:\t{np.mean(img)}"
    s += f"\nstd:\t{np.std(img)}"
    s += f"\nmin:\t{np.min(img)}"
    s += f"\n10:\t{v[0]}"
    s += f"\n25:\t{v[1]}"
    s += f"\n50:\t{v[2]}"
    s += f"\n75:\t{v[3]}"
    s += f"\n90:\t{v[4]}"
    s += f"\n99:\t{v[5]}"
    s += f"\nmax:\t{np.max(img)}\n"
    return s




#  NOTE: This function is currently depreciated and not in use
def inspect_stats(dataset, idx=0):
    """
    Input: Name of dataset and image index to be inspected
    Output: None, just prints the stats
    """
    dr = os.path.join('DATASET',dataset,'data','my_dataset','img','train')
    entry = f"{dr}/{idx}.npy"
    print(entry)
    img = np.load(entry)
    img0 = img[:,:,0]
    img1 = img[:,:,1]
    print(getImgStats(img0))
    print(getImgStats(img1))
    return




def inspect_dataset(dataset, transforms=[], idx=0):
    """ 
    This function should be called from a jupyter notebook cell
    
    Input: The name of the dataset to inspect, and the list of transformations to apply to the images. Optionally include an initial image index

    Output: None, just displays the figures and data
    """

    dr = os.path.join('DATASET',dataset,'data','my_dataset','img','train')
    while True:
        entry = f"{dr}/{idx}.npy"
        print(entry)
        image, norm_bounds = apply_transforms(entry, transforms)
        ax1,ax2 = inspect_imgs(image, entry)
        ax3,ax4 = inspect_channels(image, norm_bounds)
        ax5,ax6 = inspect_hists(np.load(entry), norm_bounds)
        plt.show()
        action = input('Press enter for next image, \'p\' for previous image, or enter image id. \'q\' to quit')
        if action.isdigit():
            idx = int(action)
        elif action == 'p':
            idx = idx-1
        elif action == 'q':
            break
        else:
            idx = idx+1
        clear_output(wait=True)
    plt.close()
    return




def inspect_hists(image, norm_bounds):
    """
    Input: The image and the applied normalization bounds

    Output: Returns the histogram plt axis for each channel for display
    """

    img = np.array(image, copy=True, order='K')
    img_r = img[:,:,0]
    img_h = img[:,:,1]
    
    hist_range_r = (norm_bounds['radio']['min'],norm_bounds['radio']['max'])
    hist_range_h = (norm_bounds['halpha']['min'],norm_bounds['halpha']['max'])

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.hist(img_r.flatten(), bins=500, range=hist_range_r, log=True, color='blue', alpha=0.7)
    ax1.set_title('Histogram of raw Radio Data')
    ax1.set_xlabel('Pixel Value')
    ax1.set_ylabel('Frequency')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.hist(img_h.flatten(), bins=500, range=hist_range_h, log=True, color='blue', alpha=0.7)
    ax2.set_title('Histogram of raw H-Alpha Data')
    ax2.set_xlabel('Pixel Value')
    ax2.set_ylabel('Frequency')
    return ax1, ax2




def inspect_channels(image, norm_bounds):
    """
    Input: The image and the applied normalization bounds
    
    Output: Returns plt axis for each channel
    """

    img = np.array(image, copy=True, order='K')
    height, width, depth = img.shape
    img_r = np.zeros((height, width, 3), dtype=img.dtype)
    img_r[:,:,0] = img[:,:,0]
    img_h = np.zeros((height, width, 3), dtype=img.dtype)
    img_h[:,:,2] = img[:,:,1]
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title('Radio Data')
    ax1.imshow(img_r, vmin=norm_bounds['radio']['min'], vmax=norm_bounds['radio']['max'])
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title('H-Alpha Data')
    ax2.imshow(img_h, vmin=norm_bounds['halpha']['min'], vmax=norm_bounds['halpha']['max'])
    return ax1, ax2




def inspect_imgs(image, entry):
    """
    Input: The image and the image path entry to retrieve the corresponding annotation
    
    Output: Returns plt axis for the image (converted to a three-channel image) and the annotation
    """

    img = np.array(image, copy=True, order='K')
    img = get3channel(img)
    ann = Image.open(get_companion(entry))
    colors = ['black', 'yellow', 'green']  # Define colors for background, HII (1), and SNR (2) regions
    cmap = mcolors.ListedColormap(colors)
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(ann, cmap=cmap, vmin=0, vmax=2)
    ax1.set_title('Annotation')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(img)
    ax2.set_title('Image')
    return ax1, ax2





def apply_transforms(entry, transforms):
    """
    Input: The image filepath entry and the list of transformations to apply.

    Output: The transformed image and the applied normalization bounds if applicable. 
            This function also prints statistics after each transformation
    """

    img = np.load(entry)
    print('\n\nInitial data stats')
    print(getImgStats2(img))
    
    norm_bounds = {}
    
    #  Apply transforms on channels separately 
    if 'separate' in transforms:
        img_r = img[:,:,0]
        img_h = img[:,:,1]
        if 'clip99' in transforms:
            print('\nApplying \'clip99\' on channels separately')
            img_r = clip99(img_r)
            img_h = clip99(img_h)
            img[:,:,0] = img_r
            img[:,:,1] = img_h
            print(getImgStats2(img))
        if 'normalize' in transforms:
            print('\nApplying \'normalize\' on channels separately')
            img_r, norm_bounds['radio'] = normalize(img_r)
            img_h, norm_bounds['halpha'] = normalize(img_h)
            img[:,:,0] = img_r
            img[:,:,1] = img_h
            print(getImgStats2(img))
        if 'logstretch' in transforms:
            print('\nApplying \'logstrectch\' on channels separately')
            img_r = logstretch(img_r)
            img_h = logstretch(img_h)
            img[:,:,0] = img_r
            img[:,:,1] = img_h
            print(getImgStats2(img))
        if 'sqrtstretch' in transforms:
            print('\nApplying \'sqrtstretch\' on channels separately')
            img_r = sqrtstretch(img_r)
            img_h = sqrtstretch(img_h)
            img[:,:,0] = img_r
            img[:,:,1] = img_h
            print(getImgStats2(img))
        if 'zscale' in transforms:
            print('\nApplying \'zscale\' on channels separately')
            img_r = newZScale(img_r)
            img_h = newZScale(img_h)
            img[:,:,0] = img_r
            img[:,:,1] = img_h
            print(getImgStats2(img))

    #  Apply transforms on channels together
    else:
        if 'clip99' in transforms:
            print('\nApplying \'clip99\' on channels together')
            img = clip99(img)
            print(getImgStats2(img))
        if 'normalize' in transforms:
            print('\nApplying \'normalize\' on channels together')
            img, norm_bounds['radio'] = normalize_together(img)
            norm_bounds['halpha'] = norm_bounds['radio']
            print(getImgStats2(img))
        if 'logstretch' in transforms:
            print('\nApplying \'logstrectch\' on channels together')
            img = logstretch(img)
            print(getImgStats2(img))
        if 'sqrtstretch' in transforms:
            print('\nApplying \'sqrtstretch\' on channels together')
            img = sqrtstretch(img)
            print(getImgStats2(img))
        if 'zscale' in transforms:
            print('\nApplying \'zscale\' on channels together')
            img = newZScale(img)
            print(getImgStats2(img))

    #  Print normalization bounds if applicable
    if 'normalize' in transforms:
        print('\nNormilzation bounds:')
        print(json.dumps(norm_bounds, indent=4))

    #  Print stats regarding ROI
    print('\nRegion of interest average pixel value:')
    print(getRoiStats(img, np.array(Image.open(get_companion(entry)))))
    
    return img, norm_bounds




#  NOTE: This function needs fixing
def newZScale(img):
    """
    Input: An image

    Output: The image with zscale transformation applied
    """

    print('computing new zscale')
    zscale = ZScaleInterval(contrast=0.3)
    vmin, vmax = zscale.get_limits(img)
    norm = ImageNormalize(vmin=vmin, vmax=vmax)
    return norm(img)



def logstretch(img):
    """
    Input: An image

    Output: The image with a logstretch transformation applied
    """

    stretch = LogStretch()
    return stretch(img)




def sqrtstretch(img):
    """
    Input: An image

    Output: The image with a square root transformation applied
    """
    
    stretch = SqrtStretch()
    return stretch(img)




def normalize(img):
    """
    Input: An image

    Output: The image normalized to the range (0,1). Also returns the computed normaliztion bounds for reference
    """

    min_val = np.min(img)
    max_val = np.max(img)
    return (img - np.min(img)) / (np.max(img) - np.min(img)), {'min':float(min_val), 'max':float(max_val)}




def normalize_together(img):
    """
    Input: A two channel image

    Output: The two channel image normalized to the range (0,1). Also returns the computed normaliztion bounds for reference
    """    

    img_r = img[:,:,0]
    img_h = img[:,:,1]
    min_val_r = np.min(img_r)
    max_val_r = np.max(img_r)
    min_val_h = np.min(img_h)
    max_val_h = np.max(img_h)
    norm_max = max(max_val_r, max_val_h)
    norm_min = min(min_val_r, min_val_h)
    norm_r = (img_r - norm_min) / (norm_max - norm_min)
    norm_h = (img_h - norm_min) / (norm_max - norm_min)
    return merge_images(norm_r, norm_h), {'min':float(norm_min), 'max':float(norm_max)}




def clip99(img):
    """
    Input: An image

    Output: The image with the upper bound of its data clipped to the pixel value of its 99.9th percentile
    """

    b_max = np.percentile(img,99.9)
    return np.clip(img, a_min=None, a_max=b_max)




def get3channel(image):
    """
    Input: A numpy image
               
    Output: If image is 1 channel: duplicate channel to RGB
            If image is 2 channel: make R0B (green channel is 0)
            If image is 3 channel: return image
    """
    img_dtype = image.dtype

    if image.ndim == 2:
        # Single channel image. Make grayscale
        result_image = np.stack([image,image,image], axis=-1)
    elif image.shape[2] == 2:
        # Two channel. R0B
        result_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)
        result_image[:,:,0] = image[:,:,0]
        result_image[:,:,2] = image[:,:,1]
    else:
        # Three or more channels. Return image
        result_image = image
    result_image = result_image.astype(img_dtype)
    return result_image




def get_companion(file_path):
    """
    Input: The img or ann file path

    Output: If file path is img/train/xxx.npy, it will find ann/train/xxx.png.
            If file path is ann/val/yyy.npy, it will find img/val/yyy.npy. 
    """
    comp_file = ''
    if 'img' in file_path:
        # File is img. Replace img with ann and .npy with .png
        comp_file = file_path.replace('img', 'ann')
        comp_file = comp_file.replace('npy', 'png')
    else:
        # File is ann. Replace ann with img and png with npy
        comp_file = file_path.replace('ann', 'img')
        comp_file = comp_file.replace('npy', 'png')
    return comp_file




def float_to_int(img):
    """
    Input: A floating point img.
    
    Output: The floating point data will be clipped to (0,1) and then scaled to (0,255) and converted to uint8
    """
    new_channels = []
    for channel in range(img.shape[2]):
        channel_data = img[:,:,channel]  # Get data for the current channel
        if np.all(channel_data == 0):
            new_channel = channel_data
        else:
            new_channel = (np.clip(channel_data,0,1) * 255).astype(np.uint8)
        new_channels.append(new_channel)
    ret_img = np.stack(new_channels, axis=-1)
    ret_img = ret_img.astype(np.uint8)
    return ret_img

