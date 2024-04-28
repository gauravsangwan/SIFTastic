# SIFTastic: Unveiling Reality: SIFT Applications for Visual Classification, Adversarial Detection, 3D Reconstruction, and Fingerprint Matching

**Github Repository for CSL7360 Computer Vision Course**

Our project is based on SIFT and how it can be used in multiple ways. Main Target problems are
1.  **SIFT for visual classification** : Using bag of visual words classfication technique 
2. **SIFT for adversarial Detection** : Using SIFT to detect if the input image is adversarially attacked or not. 
3. **3D Reconstruction and Localization** of monuments using uncaliberated stereo and image localization of the same monuments using their 3d reconstruction on any new image.
4. **SIFT for Fingerprint matching** ([dataset](https://www.kaggle.com/datasets/ruizgara/socofing)) 
5. **SIFT for live webcam matching** 

> Note: Save the bag of visual works to be reused again and again.
## Setup

Before running any part of the application, ensure you have Python installed along with the necessary libraries:

```
pip install -r requirements.txt
```
## Running the application

### Dash Application
To run the Dash application that integrates all the above functionalities, execute the provided run.sh script, ensure you are in the correct directory
```
cd <path_to_script>
chmod +x run.sh
./run.sh
```
The project is structured into multiple scripts, each corresponding to a different functionality. Here's how you can run each component:
### Visual Classification and Adversarial detection
To start the visual classification module, run the following script:
```
python3 app.py
```
### Live Webcam Feature Matching
For real-time feature matching using your webcam, execute:
```
python3 app_webcam.py

```
### Fingerprint Matching
To perform fingerprint matching, use the following command:
```
python3 app_fingerprint.py
```
### 3D Reconstruction
For 3D reconstruction tasks:
```
python3 app_3d.py
```


## Results 

### Adversarial Detection 
CIFAR 10 Dataset 

| Attack    | Detector Accuracy | 0       | 0.1   | 0.2   | 0.3   | 0.4   | 0.5   | 0.6   | 0.7   | 0.8    | 0.9   | 1      |
|-----------|-------------------|---------|-------|-------|-------|-------|-------|-------|-------|--------|-------|--------|
| FGSM      | 96.14             | 77.97   | 74.26 | 70.56 | 66.86 | 63.15 | 59.45 | 55.75 | 52.04 | 48.344 | 44.64 | 40.936 |
| CW        | 55.23             | 61.54   | 61.12 | 60.7  | 60.28 | 59.87 | 59.45 | 59.04 | 58.62 | 58.21  | 57.78 | 57.36  |
| BIM       | 75                | 69.4342 | 67.44 | 65.44 | 63.44 | 61.45 | 59.45 | 57.45 | 55.46 | 53.46  | 51.47 | 49.47  |
| Deep Fool | 50.14             | 59.51   | 59.49 | 59.48 | 59.47 | 59.46 | 59.46 | 59.44 | 59.45 | 59.42  | 59.41 | 59.39  |
| ALL       | 74.1225           | 69.08   | 67.15 | 65.23 | 63.3  | 61.38 | 59.45 | 57.52 | 55.6  | 53.67  | 51.75 | 49.82  |

CIFAR 100 Dataset

| Attack | Detector Accuracy | 0     | 0.1   | 0.2   | 0.3   | 0.4   | 0.5    | 0.6   | 0.7   | 0.8   | 0.9   | 1     |
|--------|-------------------|-------|-------|-------|-------|-------|--------|-------|-------|-------|-------|-------|
| FGSM   | 89.19             | 75.09 | 71.96 | 68.83 | 65.71 | 62.58 | 59.45  | 56.32 | 53.19 | 50.07 | 46.94 | 43.81 |
| ALL    | 70.56             | 67.66 | 66.01 | 64.37 | 62.73 | 61.09 | 59.455 | 57.81 | 56.17 | 54.53 | 52.89 | 51.24 |

Attack generalisation

| Trained on FGSM | CIFAR10 | Cifar100 |
|-----------------|---------|----------|
| FGSM            | 96.4    | 59.19    |
| Deep Fool       | 73      | 58       |
| BIM             | 96.2    | 88.16    |
| CW              | 64      | 78       |
| Total           | 79.38   | 70.566   |

Dataset generalisation

| Trained On (down)  Tested On (side) | CIFAR 10 | CIFAR 100 |
|-------------------------------------|----------|-----------|
| CIFAR 10                            | 74.1225  | 69.32     |
| CIFAR 100                           | 70.566   | 73.63     |

### Visual Classification
- Feature Detection Algorithm = SIFT
- Clustering Algorithm = KMeans
- Classifier = SVM

Accuracy

| k | kernel_type | Accuracy |
|---|-------------|----------|
| 50 | linear | 54.3 |
| 50 | precomputed | 52.4 |
| 100 | linear | 54.3 |
| 100 | precomputed | 56.7 |

## Work Distribution: 

| Name | Tasks | 
| --- | --- | 
| Gaurav Sangwan | Adversarial, fingerprint and live webcam matching, Dash App, Video | 
| Mukul Shingwani | 3D reconstruction and Localization, Dash App, Video |
| Shashank Asthana | Visual Classification, Report |
| Anushkaa Ambuj | BoVW & Image Retrieval |
