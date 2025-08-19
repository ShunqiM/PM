import math
import torch
def calculate_entropy(probabilities):
    # Normalize the probabilities to sum to 1
    total = sum(probabilities)
    normalized_probabilities = [p / total for p in probabilities]
    
    # Calculate entropy
    entropy = -sum(p * math.log2(p) for p in normalized_probabilities if p > 0)
    return entropy

def calculate_uncertainties(probabilities_list):
    M = len(probabilities_list)  # Number of models/samples

    # Calculate the average entropy (u_al)
    entropies = [calculate_entropy(probs) for probs in probabilities_list]
    u_al = sum(entropies) / M

    # Calculate the total entropy (H)
    avg_probabilities = [sum(probs) / M for probs in zip(*probabilities_list)]
    H = calculate_entropy(avg_probabilities)

    # Calculate epistemic uncertainty (u_ep)
    u_ep = H - u_al

    return u_al, u_ep

from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import numpy as np

def dynamic_patchify(img, num_patches):
        # img is a PIL Image
        # num_patches is a tuple (num_horizontal_patches, num_vertical_patches)
        
        w, h = img.size
        num_horizontal_patches, num_vertical_patches = num_patches

        # Lists to hold the dimensions and starting points of each patch
        widths = [w // num_horizontal_patches + (1 if x < w % num_horizontal_patches else 0) for x in range(num_horizontal_patches)]
        heights = [h // num_vertical_patches + (1 if x < h % num_vertical_patches else 0) for x in range(num_vertical_patches)]

        patches = []
        y_start = 0

        # Loop through the image and extract patches
        for height in heights:
            x_start = 0
            for width in widths:
                # Define the box to crop
                box = (x_start, y_start, x_start + width, y_start + height)
                # Crop and append the patch to list
                patch = img.crop(box)
                patches.append(patch)
                x_start += width
            y_start += height

        return patches


def resize_attention(attention, target_size):
    attention = np.array(attention, dtype=np.float64)
    # Calculate zoom factors
    zoom_factor = np.array(target_size) / np.array(attention.shape)
    # Resize using zoom (interpolation)
    resized_attention = zoom(attention, zoom_factor, order=1)  # order=1 for bilinear interpolation
    return resized_attention

def overlay_attention_on_image(image, attention):
    # Normalize the attention matrix to be between 0 and 1
    attention_normalized = (attention - np.min(attention)) / (np.max(attention) - np.min(attention))
    
    # Create a colormap
    cmap = plt.get_cmap('viridis')
    
    # Convert the attention matrix to an RGBA image
    attention_colored = cmap(attention_normalized)
    
    # Convert the attention matrix to a PIL image
    attention_image = Image.fromarray((attention_colored[:, :, :3] * 255).astype(np.uint8))
    
    # Blend with the original image
    blended_image = Image.blend(image.convert('RGBA'), attention_image.convert('RGBA'), alpha=0.5)
    return blended_image

def find_max_attention_region(attention, region_size=(50, 50)):
    # Assuming 'attention' is a 2D numpy array and region_size is the size of the rectangle
    max_avg = 0
    max_pos = (0, 0)
    
    # Slide over the attention map with a window of size 'region_size'
    for i in range(attention.shape[0] - region_size[0]):
        for j in range(attention.shape[1] - region_size[1]):
            # Compute the average attention in this window
            current_avg = np.mean(attention[i:i+region_size[0], j:j+region_size[1]])
            if current_avg > max_avg:
                max_avg = current_avg
                max_pos = (j, i)  # (x, y) coordinates
    
    return max_pos, region_size

from PIL import Image, ImageDraw

def draw_attention_rectangle(image, attention, region_size=(50, 50)):
    # Resize the attention map to match the image size (if not already matched)
    resized_attention = resize_attention(attention, image.size)
    
    # Find the region with the highest attention
    top_left, region_size = find_max_attention_region(resized_attention, region_size)
    
    # Create a draw object
    draw = ImageDraw.Draw(image)
    
    # Calculate bottom right coordinate of the rectangle
    bottom_right = (top_left[0] + region_size[0], top_left[1] + region_size[1])
    
    # Draw the rectangle
    draw.rectangle([top_left, bottom_right], outline="red", width=3)
    
    return image

import cv2
import matplotlib.patches as patches
def apply_threshold(heatmap, threshold=0.8):
    # Normalize the heatmap to be between 0 and 1
    normalized_heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
    # Apply threshold
    thresholded_heatmap = (normalized_heatmap >= threshold).astype(int)
    return thresholded_heatmap


def find_most_intense_contour(heatmap, thresholded_heatmap, mute = False):
    contours, _ = cv2.findContours(thresholded_heatmap.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    max_intensity = 0
    most_intense_contour = None
    found = False

    for contour in contours:
        area = cv2.contourArea(contour)
        # print(area)
        if area < 1600:
            found = area if area > found else found
            continue 
        # Create a mask for the contour
        mask = np.zeros_like(heatmap, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, color=255, thickness=cv2.FILLED)

        # Calculate the mean intensity within the contour
        mean_intensity = np.mean(heatmap[mask == 255])

        # Calculate "weighted area" as area * mean intensity
        area = cv2.contourArea(contour)
        weighted_area = area * mean_intensity
        
        if weighted_area > max_intensity:
            max_intensity = weighted_area
            most_intense_contour = contour

    # print(cv2.contourArea(most_intense_contour))
    if found is False and not mute:
        print(f'contou too small: {found}')
    return most_intense_contour


def find_intense_contours(heatmap, thresholded_heatmap):
    contours, _ = cv2.findContours(thresholded_heatmap.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    max_intensity = 0
    intense_contours = []
    found = False

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 1600:
            found = area if area > found else found
            continue 
        # Create a mask for the contour
        mask = np.zeros_like(heatmap, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, color=255, thickness=cv2.FILLED)

        # Calculate the mean intensity within the contour
        mean_intensity = np.mean(heatmap[mask == 255])

        # Calculate "weighted area" as area * mean intensity
        area = cv2.contourArea(contour)
        weighted_area = area * mean_intensity
        
        intense_contours.append(contour)

    if found is False:
        print(f'contou too small: {found}')
    return intense_contours

def draw_bounding_box(heatmap, contour):
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(heatmap, cmap='viridis')

    # Enclosing rectangle for the largest contour
    if contour is not None:
        x, y, w, h = cv2.boundingRect(contour)
        # Create a Rectangle patch
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    else:
        print("No contour found.")

    plt.show()

def draw_bounding_box_on_image(image, contour, draw_arrow = False):
    # Convert PIL image to an array if not already an array
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Create a figure and axis for drawing
    image_to_draw = image.copy()

    if image.ndim == 2:  # If the image is grayscale, convert to RGB for coloring
        image_to_draw = cv2.cvtColor(image_to_draw, cv2.COLOR_GRAY2RGB)
    # Enclosing rectangle for the contour
    if contour is not None:
        x, y, w, h = cv2.boundingRect(contour)
        # Draw the rectangle directly on the image
        cv2.rectangle(image_to_draw, (x, y), (x+w, y+h),  (255, 0, 0), 4)  # Red color, 2 px thickness

        if draw_arrow:
            # Image dimensions
            img_height, img_width, _ = image_to_draw.shape

            # Determine the optimal arrow start and end points
            if x + w/2 < img_width / 2:  # If bbox is on the left half of the image
                arrow_start = (x + w + 50, y + h // 2)
                arrow_end = (x + w, y + h // 2)
            else:  # If bbox is on the right half of the image
                arrow_start = (x - 50, y + h // 2)
                arrow_end = (x, y + h // 2)

            # Draw the arrow
            cv2.arrowedLine(image_to_draw, arrow_start, arrow_end, (255, 0, 0), 3, tipLength=0.3)  # Green color, smaller tip

    return image_to_draw

def expand2square_with_mask(pil_img, background_color):
    width, height = pil_img.size
    # Determine the new size and the top-left position for pasting the original image
    if width == height:
        return pil_img, Image.new('1', pil_img.size, 1)  # The mask is all ones since no expansion needed
    elif width > height:
        new_size = (width, width)
        top_left = (0, (width - height) // 2)
    else:
        new_size = (height, height)
        top_left = ((height - width) // 2, 0)

    # Create the new image and the corresponding mask
    result = Image.new(pil_img.mode, new_size, background_color)
    mask = Image.new('1', new_size, 0)  # Initialize mask with zeros (False)

    # Paste the original image into the result image at the calculated position
    result.paste(pil_img, top_left)
    mask.paste(Image.new('1', pil_img.size, 1), top_left)  # Paste '1's where the original image goes

    return result, mask


def process_images_return_pad(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image_pad, mask = expand2square_with_mask(image, tuple(int(x*255) for x in image_processor.image_mean))
            # image.show()
            # print(image)
            image = image_processor.preprocess(image_pad, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)

    elif image_aspect_ratio == "anyres":
        raise NotImplementedError
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images, image_pad, mask