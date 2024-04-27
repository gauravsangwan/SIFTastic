#!/usr/bin/env python
# coding: utf-8

import os
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from scipy.cluster.vq import kmeans, vq
from scipy.linalg import norm
import joblib


# Constants
TRAIN_PATH = 'dataset/train'
TEST_PATH = 'dataset/test'
NO_CLUSTERS = 100
KERNEL_TYPE = 'linear'


def display_sample_images(path, num_samples=3):
    """
    Display sample images from the specified path.

    Args:
    path (str): The path to the dataset.
    num_samples (int): Number of sample images to display.

    Returns:
    None
    """
    classes = os.listdir(path)
    fig, axs = plt.subplots(len(classes), num_samples, figsize=(num_samples * 4, len(classes) * 3))

    for i, cls in enumerate(classes):
        images = [f for f in listdir(join(path, cls)) if isfile(join(path, cls, f))]
        for j in range(num_samples):
            img_path = join(path, cls, images[j])
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if len(classes) == 1:
                ax = axs[j]
            else:
                ax = axs[i, j]
            ax.imshow(img)
            ax.set_title(f"Class: {cls}")
            ax.axis('off')

    plt.tight_layout()
    plt.show()


def plot_image_histogram(image_path):
    """
    Plot histogram for the specified image.

    Args:
    image_path (str): The path to the image file.

    Returns:
    None
    """
    img = cv2.imread(image_path)
    color = ('b', 'g', 'r')
    plt.figure(figsize=(10, 5))
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.title('Histogram for color scale picture')
    plt.show()


def plot_average_color_distribution(path):
    """
    Plot average color distribution for each class in the dataset.

    Args:
    path (str): The path to the dataset.

    Returns:
    None
    """
    classes = os.listdir(path)
    avg_color_per_class = {cls: np.zeros(3, dtype=np.float64) for cls in classes}
    count_per_class = {cls: 0 for cls in classes}

    for cls in classes:
        images = [f for f in listdir(join(path, cls)) if isfile(join(path, cls, f))]
        for img_file in images:
            img = cv2.imread(join(path, cls, img_file))
            avg_color = img.mean(axis=0).mean(axis=0)
            avg_color_per_class[cls] += avg_color
            count_per_class[cls] += 1

    for cls in classes:
        avg_color_per_class[cls] /= count_per_class[cls]

    # Plotting
    plt.figure(figsize=(10, 4))
    bars = plt.bar(range(len(classes)), [1] * len(classes),
                   color=[(avg_color_per_class[cls][2] / 255, avg_color_per_class[cls][1] / 255,
                           avg_color_per_class[cls][0] / 255) for cls in classes])
    plt.xlabel('Classes')
    plt.ylabel('Average Color')
    plt.title('Average Color Distribution by Class')
    plt.xticks(range(len(classes)), classes, rotation=45)
    plt.show()


def plot_class_distribution(path):
    """
    Plot the distribution of images per class in the dataset.

    Args:
    path (str): The path to the dataset.

    Returns:
    None
    """
    class_counts = {}
    classes = os.listdir(path)
    for cls in classes:
        class_counts[cls] = len([name for name in listdir(join(path, cls)) if isfile(join(path, cls, name))])

    # Plotting
    plt.figure(figsize=(10, 4))
    plt.bar(class_counts.keys(), class_counts.values(), color='grey')
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.title('Number of Images per Class')
    plt.xticks(rotation=45)
    plt.show()


def plot_image_sharpness_distribution(path):
    """
    Plot the distribution of image sharpness in the dataset.

    Args:
    path (str): The path to the dataset.

    Returns:
    None
    """
    sharpness_values = []
    classes = os.listdir(path)
    for cls in classes:
        images = [f for f in listdir(join(path, cls)) if isfile(join(path, cls, f))]
        for image in images:
            img = cv2.imread(join(path, cls, image), 0)
            laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
            sharpness_values.append(laplacian_var)

    plt.figure(figsize=(10, 5))
    plt.hist(sharpness_values, bins=50, color='blue', alpha=0.7)
    plt.xlabel('Sharpness (Variance of Laplacian)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Image Sharpness')
    plt.show()

def get_files(train, path):
    """
    Get file paths from the specified directory.

    Args:
    train (bool): Whether to shuffle the file paths for training.
    path (str): The path to the directory.

    Returns:
    List: A list of file paths.
    """
    images = []
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        for file in os.listdir(folder_path):
            images.append(os.path.join(folder_path, file))
    if train:
        np.random.shuffle(images)
    return images


def read_image(img_path):
    """
    Read and resize the image from the specified path.

    Args:
    img_path (str): The path to the image file.

    Returns:
    np.array: The resized image.
    """
    img = cv2.imread(img_path, 0)
    return cv2.resize(img, (64, 64))


def extract_descriptors(sift, img):
    """
    Extract keypoints and descriptors from the image.

    Args:
    sift: The SIFT extractor.
    img: The image.

    Returns:
    tuple: Keypoints and descriptors.
    """
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints, descriptors


def vstack_descriptors(descriptor_list):
    """
    Stack all descriptors vertically in a numpy array.

    Args:
    descriptor_list (list): List of descriptors.

    Returns:
    np.array: Stacked descriptors.
    """
    descriptors = np.array(descriptor_list[0])
    for descriptor in descriptor_list[1:]:
        descriptors = np.vstack((descriptors, descriptor))
    return descriptors


def cluster_descriptors(descriptors, no_clusters):
    """
    Cluster descriptors using KMeans.

    Args:
    descriptors (np.array): Descriptors.
    no_clusters (int): Number of clusters.

    Returns:
    KMeans: KMeans model.
    """
    kmeans_model = KMeans(n_clusters=no_clusters).fit(descriptors)
    return kmeans_model


def plot_confusion_matrix(true, predictions):
    """
    Plot confusion matrix for true and predicted labels.

    Args:
    true (array-like): True labels.
    predictions (array-like): Predicted labels.

    Returns:
    None
    """
    np.set_printoptions(precision=2)

    class_names = ["city", "face", "green", "house_building", "house_indoor", "office", "sea"]
    plot_confusion_matrix(true, predictions, classes=class_names,
                          title='Confusion matrix, without normalization')

    plot_confusion_matrix(true, predictions, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()


def find_accuracy(true, predictions):
    """
    Print accuracy score.

    Args:
    true (array-like): True labels.
    predictions (array-like): Predicted labels.

    Returns:
    None
    """
    print('accuracy score: %0.3f' % accuracy_score(true, predictions))


def plot_image_sharpness_distribution(path):
    """
    Plot the distribution of image sharpness in the dataset.

    Args:
    path (str): The path to the dataset.

    Returns:
    None
    """
    sharpness_values = []
    classes = os.listdir(path)
    for cls in classes:
        images = [f for f in listdir(join(path, cls)) if isfile(join(path, cls, f))]
        for image in images:
            img = cv2.imread(join(path, cls, image), 0)
            laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
            sharpness_values.append(laplacian_var)

    plt.figure(figsize=(10, 5))
    plt.hist(sharpness_values, bins=50, color='blue', alpha=0.7)
    plt.xlabel('Sharpness (Variance of Laplacian)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Image Sharpness')
    plt.show()


def build_codebook(descriptors, k):
    """
    Build codebook using KMeans clustering.

    Args:
    descriptors (np.array): Descriptors.
    k (int): Number of clusters.

    Returns:
    tuple: Tuple containing the number of clusters and the codebook.
    """
    codebook, _ = kmeans(descriptors, k)
    return k, codebook


def save_codebook(k, codebook, filename):
    """
    Save codebook to a file.

    Args:
    k (int): Number of clusters.
    codebook (np.array): Codebook.
    filename (str): Name of the file to save the codebook.

    Returns:
    None
    """
    joblib.dump((k, codebook), filename, compress=3)


def load_codebook(filename):
    """
    Load codebook from a file.

    Args:
    filename (str): Name of the file containing the codebook.

    Returns:
    tuple: Tuple containing the number of clusters and the codebook.
    """
    return joblib.load(filename)


def vector_quantization(descriptors, codebook):
    """
    Perform vector quantization using codebook.

    Args:
    descriptors (np.array): Descriptors.
    codebook (np.array): Codebook.

    Returns:
    np.array: Visual words.
    """
    visual_words, _ = vq(descriptors, codebook)
    return visual_words


def build_sparse_vectors(visual_words, k):
    """
    Build sparse vectors using visual words.

    Args:
    visual_words (list): List of visual words.
    k (int): Number of clusters.

    Returns:
    np.array: Frequency vectors.
    """
    frequency_vectors = []
    for img_visual_words in visual_words:
        img_frequency_vector = np.zeros(k)
        for word in img_visual_words:
            img_frequency_vector[word] += 1
        frequency_vectors.append(img_frequency_vector)
    return np.stack(frequency_vectors)


def calculate_tfidf(frequency_vectors, N):
    """
    Calculate TF-IDF values.

    Args:
    frequency_vectors (np.array): Frequency vectors.
    N (int): Number of images.

    Returns:
    np.array: TF-IDF values.
    """
    df = np.sum(frequency_vectors > 0, axis=0)
    idf = np.log(N / df)
    return frequency_vectors * idf


def search(i, top_k=5):
    """
    Perform similarity search.

    Args:
    i (int): Index of the search image.
    top_k (int): Number of top similar images to retrieve.

    Returns:
    None
    """
    print("Search image:")
    # show the search image
    plt.imshow(bw_images[i], cmap='gray')
    plt.show()
    print("-----------------------------------------------------")
    # get search image vector
    a = tfidf[i]
    # get the cosine distance for the search image `a`
    cosine_similarity = np.dot(a, b.T) / (norm(a) * norm(b, axis=1))
    print("Min cosine similarity:", round(np.min(cosine_similarity), 1))
    print("Max cosine similarity:", np.max(cosine_similarity))
    # get the top k indices for most similar vecs
    idx = np.argsort(-cosine_similarity)[:top_k]
    # display the results
    for i in idx:
        print(f"{i}: {round(cosine_similarity[i], 4)}")
        plt.imshow(bw_images[i], cmap='gray')
        plt.show()


# Main function
def main():
    # Display sample images from training dataset
    display_sample_images(TRAIN_PATH, num_samples=5)

    # Plot histogram for an example image
    example_image_path = join(TRAIN_PATH, os.listdir(TRAIN_PATH)[0], os.listdir(join(TRAIN_PATH, os.listdir(TRAIN_PATH)[0]))[0])
    plot_image_histogram(example_image_path)

    # Plot average color distribution by class
    plot_average_color_distribution(TRAIN_PATH)

    # Plot distribution of images per class
    plot_class_distribution(TRAIN_PATH)

    # Plot distribution of image sharpness
    plot_image_sharpness_distribution(TRAIN_PATH)

    # Read and process images
    image_paths = get_files(True, TRAIN_PATH)
    images_training = [read_image(img_path) for img_path in image_paths]

    # SIFT extractor
    extractor = cv2.SIFT_create()

    # Extract keypoints and descriptors for each image
    keypoints_list = []
    descriptors_list = []
    for img in images_training:
        kp, des = extract_descriptors(extractor, img)
        keypoints_list.append(kp)
        descriptors_list.append(des)

    # Stack descriptors
    stacked_descriptors = vstack_descriptors(descriptors_list)

    # Cluster descriptors
    k, codebook = build_codebook(stacked_descriptors, NO_CLUSTERS)

    # Save codebook
    save_codebook(k, codebook, "bovw-codebook.pkl")

    # Load codebook
    k, codebook = load_codebook("bovw-codebook.pkl")

    # Vector quantization
    visual_words = [vector_quantization(des, codebook) for des in descriptors_list]

    # Build sparse vectors
    frequency_vectors = build_sparse_vectors(visual_words, k)

    # Calculate TF-IDF
    N = len(image_paths)
    tfidf = calculate_tfidf(frequency_vectors, N)

    # Search for similar images
    search(10)


if __name__ == "__main__":
    main()
