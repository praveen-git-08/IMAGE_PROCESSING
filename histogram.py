import cv2
import matplotlib.pyplot as plt
import os

# Step 1: Load image
img = cv2.imread('../sample/nature.png', cv2.IMREAD_GRAYSCALE)

# Step 2: Global Histogram Equalization
equalized = cv2.equalizeHist(img)

# Step 3: CLAHE (Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe_img = clahe.apply(img)

# Step 4: Display Results
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(equalized, cmap='gray')
plt.title("Histogram Equalization")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(clahe_img, cmap='gray')
plt.title("CLAHE")
plt.axis('off')

plt.tight_layout()
plt.show()

# Step 5: Save Results
os.makedirs('../results', exist_ok=True)
cv2.imwrite('../results/equalized.png', equalized)
cv2.imwrite('../results/clahe.png', clahe_img)
print("Results saved in results folder.")

# Load image
img = cv2.imread('../data/sample.png', cv2.IMREAD_GRAYSCALE)

# Apply Histogram Equalization
equalized = cv2.equalizeHist(img)

# Ensure results folder exists
os.makedirs('../results', exist_ok=True)

# Save result
cv2.imwrite('../results/equalized.png', equalized)

print("Enhanced image saved in results/equalized.png")