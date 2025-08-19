import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
from experiments.llava.mm_utils import process_images
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates
from PIL import ImageFilter


def create_pseudo_attention_map(image, focus_points, sigma=100):
    # Load the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    # Initialize an empty attention map
    attention_map = np.zeros((h, w), dtype=np.float32)

    # Add Gaussian blobs around each focus point
    for point in focus_points:
        y, x = point
        # Create a Gaussian overlay
        y_grid, x_grid = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        gaussian = np.exp(-((x_grid - x) ** 2 + (y_grid - y) ** 2) / (2 * sigma ** 2))
        attention_map += gaussian

    # Normalize the attention map to the range [0, 1]
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

    
    return attention_map

def calculate_marginals(attention_map):
    # Calculate marginal distributions along the x and y axes
    marginal_x = np.max(attention_map, axis=0)  # Max over rows (height)
    marginal_y = np.max(attention_map, axis=1)  # Max over columns (width)
    return marginal_x, marginal_y

def cumulative_distribution(marginal):
    # Calculate the cumulative distribution function (CDF) of a marginal
    cdf = np.cumsum(marginal)
    cdf = cdf / cdf[-1]  # Normalize to range [0, 1]
    return cdf

def inverse_sampling(cdf, num_samples):
    # Generate samples from the inverse CDF
    uniform_samples = np.linspace(0, 1, num_samples)
    inverse_function = interp1d(cdf, np.arange(len(cdf)), bounds_error=False, fill_value="extrapolate")
    sampled_indices = inverse_function(uniform_samples)
    return sampled_indices  # Return float indices for bilinear sampling

def attention_sampling(image, attention_map, out_size, method="nearest"):
    """
    Perform attention-based sampling on an image.

    Parameters:
        image (numpy.ndarray): Input image array of shape (H, W, C).
        attention_map (numpy.ndarray): Attention map of shape (H, W).
        out_size (tuple): Desired output size (height, width).
        method (str): Interpolation method, either 'nearest' or 'bilinear'.

    Returns:
        numpy.ndarray: The attention-sampled image.
    """
    h, w = attention_map.shape

    # Step 1: Decompose attention map to marginal distributions
    marginal_x, marginal_y = calculate_marginals(attention_map)

    # Step 2: Calculate cumulative distribution functions for both axes
    cdf_x = cumulative_distribution(marginal_x)
    cdf_y = cumulative_distribution(marginal_y)

    # Step 3: Use inverse transform sampling to get sampled indices for x and y
    sampled_x_indices = inverse_sampling(cdf_x, out_size[1])  # Output width
    sampled_y_indices = inverse_sampling(cdf_y, out_size[0])  # Output height

    # Check method and perform the corresponding sampling
    if method == "nearest":
        # Nearest sampling
        sampled_x_indices = np.round(sampled_x_indices).astype(int)
        sampled_y_indices = np.round(sampled_y_indices).astype(int)
        
        # Clip indices to be within image bounds
        sampled_x_indices = np.clip(sampled_x_indices, 0, w - 1)
        sampled_y_indices = np.clip(sampled_y_indices, 0, h - 1)

        # Create the sampled image by selecting pixels according to sampled indices
        sampled_image = image[sampled_y_indices[:, np.newaxis], sampled_x_indices]

    elif method == "bilinear":
        # Bilinear sampling
        # Generate a mesh grid for coordinate pairs
        sampled_x, sampled_y = np.meshgrid(sampled_x_indices, sampled_y_indices)

        # Use bilinear sampling with map_coordinates
        sampled_image = np.zeros((out_size[0], out_size[1], image.shape[2]))
        for c in range(image.shape[2]):  # For each color channel
            sampled_image[:, :, c] = map_coordinates(image[:, :, c], [sampled_y, sampled_x], order=1)

    else:
        raise ValueError("Invalid method. Choose 'nearest' or 'bilinear'.")

    return sampled_image


def direct_sampling(cdf, num_samples):
    """
    Directly applies the CDF to uniformly spaced indices to perform inverse sampling.
    This 'compresses' high-attention areas by mapping them back to a uniform space.

    Args:
        cdf (numpy.ndarray): CDF of marginal distribution.
        num_samples (int): Number of samples in the inverse-mapped image.

    Returns:
        numpy.ndarray: Sampled indices mapping back to the uniform space.
    """
    uniform_samples = np.linspace(0, len(cdf) - 1, num_samples)
    direct_function = interp1d(np.arange(len(cdf)), cdf, bounds_error=False, fill_value="extrapolate")
    mapped_indices = direct_function(uniform_samples)
    return np.clip(mapped_indices, 0, len(cdf) - 1)  # Ensure indices are in range


def inverse_attention_sampling(image, attention_map, out_size, method="nearest"):
    """
    Perform inverse attention-based sampling, compressing high-attention regions.

    Args:
        image (numpy.ndarray): Input resampled image (H', W', C).
        attention_map (numpy.ndarray): Attention map (H, W), same as used in forward sampling.
        out_size (tuple): Desired output size (H_orig, W_orig) for restoring the original resolution.
        method (str): Interpolation method, either 'nearest' or 'bilinear'.

    Returns:
        numpy.ndarray: The inversely sampled image, effectively "undoing" the magnification.
    """
    h, w = attention_map.shape

    # Compute marginal distributions
    marginal_x, marginal_y = calculate_marginals(attention_map)

    # Compute cumulative distribution functions (CDFs)
    cdf_x = cumulative_distribution(marginal_x)
    cdf_y = cumulative_distribution(marginal_y)

    # Directly apply the CDF to the uniform grid (instead of taking the inverse)
    mapped_x_indices = direct_sampling(cdf_x, out_size[1])
    mapped_y_indices = direct_sampling(cdf_y, out_size[0])

    if method == "nearest":
        # Nearest sampling
        mapped_x_indices = np.round(mapped_x_indices).astype(int)
        mapped_y_indices = np.round(mapped_y_indices).astype(int)

        # Clip indices to valid range
        mapped_x_indices = np.clip(mapped_x_indices, 0, w - 1)
        mapped_y_indices = np.clip(mapped_y_indices, 0, h - 1)

        # Select pixels according to mapped indices
        inversely_sampled_image = image[mapped_y_indices[:, np.newaxis], mapped_x_indices]

    elif method == "bilinear":
        # Bilinear sampling
        mapped_x, mapped_y = np.meshgrid(mapped_x_indices, mapped_y_indices)

        # Apply bilinear interpolation
        inversely_sampled_image = np.zeros((out_size[0], out_size[1], image.shape[2]))
        for c in range(image.shape[2]):  # Process each channel separately
            inversely_sampled_image[:, :, c] = map_coordinates(image[:, :, c], [mapped_y, mapped_x], order=1)

    else:
        raise ValueError("Invalid method. Choose 'nearest' or 'bilinear'.")

    return inversely_sampled_image



import torch 
from experiments.utils.helper_functions import *

def aggregate_layer_attention(attentions, target_size, attention_mode = 'max', vis_start=35, deep_layer = 12, 
                             map_constructor = 'norm_average', single_layer = False):
    vis_end = vis_start + 24 * 24

    shallow_attns = []
    deep_attns = []
    attn_layers = range(1, len(attentions[0])+1)

    with torch.inference_mode():
        for l in attn_layers:

            layer_attention =  attentions[0][l-1].detach().cpu()
            visual_attention = layer_attention[0, :, -1, vis_start:vis_end]
            visual_attention_mean = visual_attention.mean(dim=0)
            if attention_mode == 'mean':
                visual_attention_aggregated = visual_attention_mean
            elif attention_mode == 'max':
                visual_attention_aggregated, _ = visual_attention.max(dim=0)
            else:
                raise NotImplementedError
            
            # Reshape to 2D (assuming visual tokens can be reshaped into a square matrix)
            reshaped_attention = visual_attention_aggregated.reshape(24, 24)
            
            if map_constructor == 'original':
                resized_attention = reshaped_attention
            else:
                # Overlay the attention NOTE: Here the width and heights are replaced
                width, height = target_size
                target_size = (height, width)  # Replace with actual image dimensions
                resized_attention = resize_attention(reshaped_attention, target_size)
                if map_constructor == 'norm_average':
                    resized_attention = (resized_attention - np.min(resized_attention)) / (np.max(resized_attention) - np.min(resized_attention))
                elif map_constructor == 'sum_scale':
                    pass

            if l < deep_layer:
                shallow_attns.append(resized_attention)
            else:
                deep_attns.append(resized_attention)
                if single_layer: break

    if map_constructor == 'sum_scale':
        stacked_arrays = np.stack(shallow_attns, axis=0)
        shallow_aggregate = np.sum(stacked_arrays, axis=0)
        stacked_arrays = np.stack(deep_attns, axis=0)
        deep_aggregate = np.sum(stacked_arrays, axis=0)
    elif map_constructor == 'norm_average':
        stacked_arrays = np.stack(shallow_attns, axis=0)
        shallow_aggregate = np.mean(stacked_arrays, axis=0)
        stacked_arrays = np.stack(deep_attns, axis=0)
        deep_aggregate = np.mean(stacked_arrays, axis=0)
    elif map_constructor == 'original':
        stacked_arrays = np.stack(shallow_attns, axis=0)
        shallow_aggregate = np.sum(stacked_arrays, axis=0)
        stacked_arrays = np.stack(deep_attns, axis=0)
        deep_aggregate = np.sum(stacked_arrays, axis=0)
    else:
        raise NotImplementedError
    
    return shallow_aggregate, deep_aggregate


def magnify_attention_region(attentions, image, pad_mask, attention_mode = 'max', vis_start=35, deep_layer = 12, 
                             map_constructor = 'norm_average', regularization=0.01):

    shallow_aggregate, deep_aggregate = aggregate_layer_attention(attentions, target_size=image.size, attention_mode=attention_mode, vis_start=vis_start, deep_layer=deep_layer, map_constructor=map_constructor)

    cd_attn = deep_aggregate 
    heatmap = cd_attn * pad_mask
    image_np = np.array(image)
    out_size = image.size
    tensor = attention_sampling(image_np, heatmap + regularization, out_size, method = 'bilinear')
    tensor = tensor.astype(np.uint8)
    return tensor


def get_highlight_mask(attention_map, method = 'cluster'):
    if method == 'cluster':
        flat_map = attention_map.flatten()

        # Initialize cluster centers with the minimum and maximum values
        min_val, max_val = flat_map.min(), flat_map.max()
        centers = np.array([min_val, max_val])

        # Run K-means clustering with early exit
        for _ in range(10):
            # Calculate distances to each cluster center
            distances = np.abs(flat_map[:, np.newaxis] - centers)

            # Assign each point to the nearest cluster
            cluster_assignments = np.argmin(distances, axis=1)

            # Store the current centers for comparison
            previous_centers = centers.copy()

            # Update each center by computing the mean of the points assigned to it
            for i in range(2):
                if np.sum(cluster_assignments == i) > 0:
                    centers[i] = flat_map[cluster_assignments == i].mean()

            # Check for early exit if the centers have not moved significantly
            center_shift = np.abs(centers - previous_centers)
            if np.all(center_shift < 1e-4):
                break

        # Reshape cluster assignments back to the original 2D shape
        clusters = cluster_assignments.reshape(attention_map.shape)

        # Identify the highlighted cluster (the one with the higher center)
        highlighted_cluster = np.argmax(centers)
        highlighted_regions = (clusters == highlighted_cluster)

        return highlighted_regions

def prepare_attention_mask(total_length, highlighted_regions, vis_start_tokens=0):
    """
    Convert a 24x24 attention map into an attention mask matching the length of `input_ids`.
    
    Args:
        input_ids (list or np.ndarray): The input token IDs for the model.
        highlighted_regions (np.ndarray): A 2D boolean array (24x24) indicating highlighted regions.
        vis_start_tokens (int): Number of visual start tokens to add at the front of the attention map.

    Returns:
        np.ndarray: The attention mask aligned with `input_ids` length, with the middle part inverted from `highlighted_regions`.
    """
    # Flatten and invert the highlighted regions (24x24 -> 576, then invert)
    flat_highlight = ~highlighted_regions.flatten()

    # Calculate padding needed at the end
    end_padding_length = total_length - vis_start_tokens - len(flat_highlight)
    # Construct the attention map with visual start tokens, middle inverted section, and end padding
    attention_map = np.concatenate([
        np.ones(vis_start_tokens, dtype=bool),       # Front visual start tokens
        flat_highlight,                              # Middle inverted highlight section
        np.ones(end_padding_length, dtype=bool)      # End padding
    ])


    return attention_map

from PIL import ImageFilter
def merge_with_blur(mask, image, blur_radius=100, threshold=0.5):
    """
    Applies a blur effect to the unmasked region of an image.

    Parameters:
        mask (PIL.Image.Image): The mask image in grayscale (0-255).
        image (PIL.Image.Image): The original image.
        blur_radius (int): The radius of the blur effect.
        threshold (int): The threshold to determine the unmasked region.

    Returns:
        PIL.Image.Image: The modified image with blur applied to unmasked regions.
    """
    # Convert images to NumPy arrays
    image_np = np.array(image).astype(np.float32)[..., :3]
    mask_np = np.array(mask).astype(np.float32)

    # Normalize the mask and apply threshold
    mask_np = (mask_np > threshold * 255).astype(np.float32)

    # Create the blurred version of the image
    blurred_image = image.filter(ImageFilter.GaussianBlur(blur_radius))
    blurred_image_np = np.array(blurred_image).astype(np.float32)[..., :3]

    # Combine original and blurred images based on the mask
    blended_np = image_np * mask_np[:, :, None] + blurred_image_np * (1 - mask_np[:, :, None])

    # Convert the result back to a PIL image
    blended_image = Image.fromarray(blended_np.astype(np.uint8))
    return blended_image

def merge_with_bbox(mask, image, threshold=0.5, mute=False):
    # Convert images to NumPy arrays
    image_np = np.array(image).astype(np.float32)[..., :3]
    mask_np = np.array(mask).astype(np.float32)
    mask_ = mask_np

    # Normalize the mask and apply threshold
    # print(mask_np)
    mask_np = (mask_np > threshold * 255).astype(np.float32)
    # thresholded_heatmap = thresholded_heatmap.astype(np.uint8)

    # Perform closing
    kernel = np.ones((10,10), np.uint8)
    mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)

    # Find the most intense contour
    most_intense_contour = find_most_intense_contour(mask_, mask_np, mute=mute)
    bb_image = draw_bounding_box_on_image(image, most_intense_contour, draw_arrow=True)
    bb_image_pil = Image.fromarray(bb_image)
    return bb_image_pil

def crop_with_mask(mask, image, threshold=0.5, background_color=(0, 0, 0)):
    # Convert images to NumPy arrays
    image_np = np.array(image).astype(np.uint8)[..., :3]
    mask_np = np.array(mask).astype(np.float32)

    # Normalize the mask and apply the threshold
    mask_np = (mask_np > threshold * 255).astype(np.uint8)  # Binary mask (1 for unmasked region, 0 for the rest)

    # Apply the mask: keep the image where mask is 1, set background color where mask is 0
    cropped_np = image_np * mask_np[:, :, None] + (1 - mask_np[:, :, None]) * np.array(background_color, dtype=np.uint8)

    # Convert the result back to a PIL image
    cropped_image = Image.fromarray(cropped_np)
    return cropped_image


if __name__ == '__main__':
    # Example usage
    image = cv2.imread("./b1.jpg")
    image = cv2.resize(image, (500, 500))
    # attention_map = np.random.uniform(0, 1, (image.shape[0], image.shape[1]))  # Replace with actual attention map
    # # Define a few hotspot locations and set high attention values around them
    hotspots = [(180, 100), (150, 200), (50, 50)]
    
    attention_map = create_pseudo_attention_map(image, hotspots, 100)
    intensity = 0.6
    # Apply the attention map to the image with transparency
    heatmap = cv2.applyColorMap((attention_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 1 - intensity, heatmap, intensity, 0)

    # Display the result
    plt.figure(figsize=(8, 8))
    plt.imshow(overlay)
    plt.axis('off')
    plt.show()



    print(attention_map.max(), attention_map.mean(), attention_map.min())

    out_size = (500, 500)
    print(image.shape, attention_map.shape, out_size)
    print(type(image), type(attention_map), out_size)
    sampled_image = attention_sampling(image, attention_map, out_size, method='bilinear')

    import matplotlib.pyplot as plt
    plt.imshow(sampled_image)
    plt.title("Attention-Sampled Image")
    plt.show()