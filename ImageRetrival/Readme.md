<h1> Understanding Bag of Visual Words and Implementing Image Retrieval</h1>

## Overview
This Python code implements a Bag of Visual Words (BoVW) model for image classification. It performs the following tasks:

1. Data exploration and visualization
2. Dataset preprocessing
3. Keypoint extraction using SIFT
4. Building a codebook using K-means clustering
5. Generating sparse vectors using vector quantization and tf-idf weighting
6. Image search functionality based on cosine similarity

## Requirements
- Python 3.x
- OpenCV (`opencv-python` and `opencv-contrib-python`)
- NumPy
- Matplotlib
- scikit-learn

### 1. Dataset Exploration 
Functions like `display_sample_images`, `plot_image_histogram`, `plot_image_sizes`, `plot_average_color_distribution`, `plot_class_distribution`, and `plot_image_sharpness_distribution` are defined to explore and visualize the dataset.

### 2. Data Preprocessing
The images are gray-scaled, and also resized to lower dimension to decrease the computational cost.

### 3. Visualization of Keypoints
- Keypoints from images are visualized using the SIFT (Scale-Invariant Feature Transform) algorithm.
- SIFT descriptors are extracted from images to capture local features.
- Keypoints are overlaid on the original images to visualize their distribution.

### 4. Building CodeBook
- SIFT descriptors extracted from images are used to build a codebook.
- K-means clustering is performed on the descriptors to group them into clusters.
- Each cluster centroid represents a visual word in the codebook.

### 5. Saving and Loading CodeBook
- The codebook obtained from K-means clustering is saved using joblib for later use.
- This allows the codebook to be reused without recomputation, saving time and resources.

### 6. Building Sparse Vectors
- Vector quantization maps visual feature descriptors to visual words based on the codebook.
- Frequency vectors are created for each image by counting the occurrences of visual words.
- Tf-idf (Term Frequency-Inverse Document Frequency) weighting is applied to the frequency vectors to adjust their importance.

### 7. Image Search Functionality
- A search function calculates cosine similarity between tf-idf weighted vectors to perform an image search.
- Top-K similar images are identified based on their cosine similarity scores to the search image.

## Example
To perform an image search:
```python
search(index)
```
where,
- `index` : index of the image in the dataset that you want to search.
