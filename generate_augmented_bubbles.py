import cv2
import numpy as np
import os
import random

# Output directories
os.makedirs("dataset/filled", exist_ok=True)
os.makedirs("dataset/empty", exist_ok=True)

num_samples = 500
img_size = 28

def add_noise(img):
    noise = np.random.randint(0, 30, (img_size, img_size), dtype='uint8')
    return cv2.add(img, noise)

def random_transform(img):
    # Random rotation
    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((img_size//2, img_size//2), angle, 1)
    img = cv2.warpAffine(img, M, (img_size, img_size), borderValue=255)

    # Random blur
    if random.random() > 0.7:
        k = random.choice([1,3])
        img = cv2.GaussianBlur(img, (k,k), 0)

    # Random brightness/contrast
    alpha = random.uniform(0.8, 1.2)  # contrast
    beta = random.randint(-20, 20)    # brightness
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    return img

for i in range(num_samples):
    # Empty bubble
    empty = np.ones((img_size, img_size), dtype='uint8') * 255
    cv2.circle(empty, (img_size//2, img_size//2), img_size//2 - 2, 0, 1)
    empty = add_noise(empty)
    empty = random_transform(empty)
    cv2.imwrite(f"dataset/empty/empty_{i}.jpg", empty)

    # Filled bubble
    filled = np.ones((img_size, img_size), dtype='uint8') * 255
    cv2.circle(filled, (img_size//2, img_size//2), img_size//2 - 2, 0, -1)
    filled = add_noise(filled)
    filled = random_transform(filled)
    cv2.imwrite(f"dataset/filled/filled_{i}.jpg", filled)

print("✅ Augmented bubble dataset generated!")