import streamlit as st
from skimage import io, measure, morphology, filters
from PIL import Image
import numpy as np
import pandas as pd
import cv2

def segment_and_quantify(channel, threshold, min_size, color_name):
    # Apply Gaussian filter to smooth out noise
    smoothed = filters.gaussian(channel, sigma=1)

    # Apply threshold
    binary_mask = smoothed > threshold

    # Remove small objects
    cleaned_mask = morphology.remove_small_objects(binary_mask, min_size=min_size)

    # Label connected components
    labeled_mask = measure.label(cleaned_mask)
    properties = measure.regionprops(labeled_mask)

    # Collect data
    count = len(properties)
    areas = [prop.area for prop in properties]

    # Prepare DataFrame
    data = {
        "Color": [color_name] * count,
        "Object ID": range(1, count + 1),
        "Area (pixels)": areas
    }
    return pd.DataFrame(data), cleaned_mask

def apply_color_mask_with_contours(mask, color, contour_thickness):
    """Converts a binary mask to a colored mask and adds yellow contours for display."""
    # Create a colored mask
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i in range(3):  # Fill color channels
        colored_mask[:, :, i] = mask * color[i]

    if contour_thickness > 0:
        # Find contours
        contours = measure.find_contours(mask, level=0.5)
        # Draw yellow contours on the colored mask
        for contour in contours:
            contour = np.flip(contour, axis=1).astype(np.int32)  # Swap and convert to integer
            cv2.polylines(colored_mask, [contour], isClosed=True, color=(255, 255, 0), thickness=contour_thickness)  # Yellow contour

    return colored_mask

def subtract_background_from_channel(image, background_color, sigma):
    """Applies Gaussian background subtraction to the specified color channel in the image."""
    # Separate channels
    red_channel, green_channel, blue_channel = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    # Apply background subtraction to the selected channel
    if background_color == "Green":
        background = filters.gaussian(green_channel, sigma=sigma)
        green_channel = np.clip(green_channel - background, 0, 1)
    elif background_color == "Red":
        background = filters.gaussian(red_channel, sigma=sigma)
        red_channel = np.clip(red_channel - background, 0, 1)
    elif background_color == "Blue":
        background = filters.gaussian(blue_channel, sigma=sigma)
        blue_channel = np.clip(blue_channel - background, 0, 1)

    # Reconstruct the image with the background-subtracted channel
    return np.stack([red_channel, green_channel, blue_channel], axis=-1)

def create_colocalization_masks(red_mask, green_mask, blue_mask):
    """Creates colocalization masks for different combinations of colors."""
    rg_mask = red_mask & green_mask  # Red-Green overlap
    gb_mask = green_mask & blue_mask  # Green-Blue overlap
    rb_mask = red_mask & blue_mask  # Red-Blue overlap
    rgb_mask = red_mask & green_mask & blue_mask  # Red-Green-Blue overlap

    return rg_mask, gb_mask, rb_mask, rgb_mask

def quant_and_display_colocalized_mask(mask, color, contour_color, contour_thickness, label):
    """Quantifies and adds contours to a colocalized mask."""
    data, _ = segment_and_quantify(mask, threshold=0.5, min_size=1, color_name=label)  # Threshold and min_size set for binary
    colored_display = apply_color_mask_with_contours(mask, color, contour_thickness)
    return data, colored_display

st.title("Image Segmentation and Quantification Tool with Colocalization")

# Image upload field
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])

if uploaded_image:
    # Check if uploaded image is in webp format, and handle it with Pillow
    if uploaded_image.type == "image/webp":
        image = Image.open(uploaded_image).convert("RGB")  # Convert to RGB if needed
        image = np.array(image)  # Convert to NumPy array for processing
    else:
        # Use skimage.io.imread for other formats
        image = io.imread(uploaded_image)

    # Normalize image to [0, 1] for processing
    image = image / 255.0

    st.image(image, caption="Original Image", use_column_width=True)

    # Add sliders and controls for thresholding and background subtraction parameters
    st.sidebar.header("Segmentation Parameters")

    # Background subtraction parameters
    apply_bg_subtraction = st.sidebar.checkbox("Apply Background Subtraction", value=True)
    background_color = st.sidebar.selectbox("Background Color", ["Green", "Red", "Blue"])
    sigma = st.sidebar.slider("Background Subtraction Sigma", 1, 100, 50)

    # Apply background subtraction if enabled
    if apply_bg_subtraction:
        image = subtract_background_from_channel(image, background_color, sigma)

    # Separate color channels after background subtraction
    blue_channel = image[:, :, 2]
    green_channel = image[:, :, 1]
    red_channel = image[:, :, 0]

    # Threshold sliders
    blue_threshold = st.sidebar.slider("Blue Threshold", 0.0, 1.0, 0.5)
    green_threshold = st.sidebar.slider("Green Threshold", 0.0, 1.0, 0.5)
    red_threshold = st.sidebar.slider("Red Threshold", 0.0, 1.0, 0.5)

    # Minimum size slider
    min_size = st.sidebar.slider("Minimum Object Size (pixels)", 1, 100, 20)
    contour_thickness = st.sidebar.slider("Contour Thickness (pixels)", 0, 5, 2)

    # Segment and quantify each color
    blue_data, blue_mask = segment_and_quantify(blue_channel, blue_threshold, min_size, "Blue")
    green_data, green_mask = segment_and_quantify(green_channel, green_threshold, min_size, "Green")
    red_data, red_mask = segment_and_quantify(red_channel, red_threshold, min_size, "Red")

    # Generate color displays for segmented images
    blue_display = apply_color_mask_with_contours(blue_mask, [0, 0, 255], contour_thickness)  # Blue with yellow contours
    green_display = apply_color_mask_with_contours(green_mask, [0, 255, 0], contour_thickness)  # Green with yellow contours
    red_display = apply_color_mask_with_contours(red_mask, [255, 0, 0], contour_thickness)  # Red with yellow contours

    # Display segmented images with contours for each channel
    st.subheader("Segmented Images with Contours")
    st.image(blue_display, caption="Blue Channel Segmentation with Contours", use_column_width=True)
    st.image(green_display, caption="Green Channel Segmentation with Contours", use_column_width=True)
    st.image(red_display, caption="Red Channel Segmentation with Contours", use_column_width=True)

    # Generate colocalization masks
    rg_mask, gb_mask, rb_mask, rgb_mask = create_colocalization_masks(red_mask, green_mask, blue_mask)

    # Quantify and visualize each colocalization mask
    rg_data, rg_display = quant_and_display_colocalized_mask(rg_mask, [0, 255, 255], (255, 0, 255), contour_thickness, "Red-Green")  # Cyan
    gb_data, gb_display = quant_and_display_colocalized_mask(gb_mask, [255, 0, 255], (0, 255, 255), contour_thickness, "Green-Blue")  # Magenta
    rb_data, rb_display = quant_and_display_colocalized_mask(rb_mask, [255, 255, 0], (0, 255, 255), contour_thickness, "Red-Blue")  # Yellow
    rgb_data, rgb_display = quant_and_display_colocalized_mask(rgb_mask, [255, 165, 0], (255, 255, 255), contour_thickness, "Red-Green-Blue")  # White

    # Display colocalized images with contours for colocalization
    st.subheader("Colocalized Regions with Contours")
    st.image(rg_display, caption="Red-Green Colocalization", use_column_width=True)
    st.image(gb_display, caption="Green-Blue Colocalization", use_column_width=True)
    st.image(rb_display, caption="Red-Blue Colocalization", use_column_width=True)
    st.image(rgb_display, caption="Red-Green-Blue Colocalization", use_column_width=True)

    # Combine results
    results = pd.concat([blue_data, green_data, red_data, rg_data, gb_data, rb_data, rgb_data], ignore_index=True)

    # Display segmentation results
    st.subheader("Segmentation and Colocalization Results")
    st.dataframe(results)

    # Summarized results with colocalization percentage
    st.subheader("Summary")
    # Total area of all segmented regions
    total_area = results["Area (pixels)"].sum()

    # Grouped summary with colocalization percentage
    summary = results.groupby("Color").agg(
        Count=("Object ID", "count"),
        Total_Area=("Area (pixels)", "sum"),
        Mean_Area=("Area (pixels)", "mean")
    )

    # Calculate the percentage of total area for each colocalization type
    summary["Percentage of Total Area (%)"] = (summary["Total_Area"] / total_area) * 100

    st.table(summary)
