import numpy as np
import math 
import os 
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
sample = cv2.imread('/home/iiticos/Desktop/College/8th sem/CV/Project/finger/socofing/SOCOFing/Altered/Altered-Hard/101__M_Left_index_finger_CR.BMP')


best_score = counter = 0
filename = image = kp1 = kp2 = mp = None

read_files = os.listdir('/home/iiticos/Desktop/College/8th sem/CV/Project/finger/socofing/SOCOFing/Real')
for file in tqdm(read_files,desc='Matching'):
    counter += 1
    fingerprint_img = cv2.imread('/home/iiticos/Desktop/College/8th sem/CV/Project/finger/socofing/SOCOFing/Real/' + file)
    sift = cv2.SIFT_create()
    keypoints_1, des1 = sift.detectAndCompute(sample, None)
    keypoints_2, des2 = sift.detectAndCompute(fingerprint_img, None)
    matches = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 10}, {}).knnMatch(
        des1, des2, k=2
    )

    match_points = []
    for p, q in matches:
        if p.distance < 0.1 * q.distance:
            match_points.append(p)

    keypoints = 0
    if len(keypoints_1) <= len(keypoints_2):
        keypoints = len(keypoints_1)
    else:
        keypoints = len(keypoints_2)
    if len(match_points) / keypoints * 100 > best_score:
        best_score = len(match_points) / keypoints * 100
        filename = file
        image = fingerprint_img
        kp1, kp2, mp = keypoints_1, keypoints_2, match_points

print("Best match:  " + filename)
print("Best score:  " + str(best_score))

result = cv2.drawMatches(sample, kp1, image, kp2, mp, None)
result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
plt.figure(figsize=(10, 10))
plt.imshow(result)
plt.axis('off')
plt.title('Best Match')
plt.savefig('best_match_plot.png', bbox_inches='tight')  # Save the plot

