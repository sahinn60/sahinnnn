import cv2
import numpy as np

def enhance_low_light(image_path: str, output_path: str = "enhanced_output.jpg") -> None:
    # Load image
    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError(f"Cannot load image from: {image_path}")
    
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # Step 1: Detect low-light regions using gray-level thresholding
    threshold_value = 80  # Adjust if needed
    _, low_light_mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # Step 2: Apply histogram equalization to grayscale
    hist_eq = cv2.equalizeHist(gray)

    # Step 3: Apply adaptive spatial filtering (Gaussian Blur to reduce noise)
    smoothed = cv2.GaussianBlur(hist_eq, (5, 5), 0)

    # Step 4: Sharpening using Laplacian and unsharp masking
    laplacian = cv2.Laplacian(smoothed, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    sharpened = cv2.addWeighted(smoothed, 1.5, laplacian, -0.5, 0)

    # Step 5: Merge enhanced regions back into original
    enhanced_gray = np.where(low_light_mask == 255, sharpened, gray)

    # Convert back to BGR (3-channel) and apply to original color image
    enhanced_gray_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

    # Use the mask to replace only low-light regions in the original image
    mask_3ch = cv2.merge([low_light_mask] * 3)
    result = np.where(mask_3ch == 255, enhanced_gray_bgr, original)

    # Save output
    cv2.imwrite(output_path, result)
    print(f"[✓] Enhanced image saved to: {output_path}")

# Example usage
if __name__ == "__main__":
    enhance_low_light("your_photo.jpg")
