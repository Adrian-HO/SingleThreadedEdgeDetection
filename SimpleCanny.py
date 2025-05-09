import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

def detect_edges(image_path, low_threshold=100, high_threshold=200):
    
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None, None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    
    # For display purposes, convert BGR to RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return rgb_img, edges

def display_results(original, edges, title="Canny Edge Detection"):
    """Display original image and detected edges side by side"""
    plt.figure(figsize=(12, 6))
    plt.subplot(121), plt.imshow(original)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def save_results(original, edges, output_path="edges_output.jpg"):
    """Save the edge detection results"""
    # Create a combined image
    h, w = original.shape[:2]
    combined = np.zeros((h, 2*w, 3), dtype=np.uint8)
    
    # Convert edges to BGR for concatenation
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Convert original back to BGR for saving
    original_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    
    # Place images side by side
    combined[:, :w, :] = original_bgr
    combined[:, w:, :] = edges_bgr
    
    # Save the combined image
    cv2.imwrite(output_path, combined)
    print(f"Results saved to {output_path}")

def main():
    # Check if image path is provided
    if len(sys.argv) < 2:
        print("Usage: python script.py <image_path> [low_threshold] [high_threshold] [output_path]")
        return
    
    # Get parameters from command line
    image_path = sys.argv[1]
    
    # Optional parameters
    low_threshold = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    high_threshold = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    output_path = sys.argv[4] if len(sys.argv) > 4 else "edges_output.jpg"
    
    # Detect edges
    original, edges = detect_edges(image_path, low_threshold, high_threshold)
    
    if original is not None and edges is not None:
        # Display results
        display_results(original, edges)
        
        # Save results
        save_results(original, edges, output_path)

if __name__ == "__main__":
    main()