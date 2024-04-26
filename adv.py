import torch
import cv2
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.autograd import Variable
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import joblib 




def sift_keypoints(tensor):
    EDGE_THRESHOLD = 6
    sift = cv2.SIFT_create()
    sift.setEdgeThreshold(EDGE_THRESHOLD)
    array = tensor.numpy().transpose((1,2,0))*255
    array = array.astype(np.uint8)
    gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    key_points_sift, descriptor = sift.detectAndCompute(array,None)
    image = array.astype(np.float64)
    mean_img, std_img = cv2.meanStdDev(image)
    mean = np.mean(mean_img)
    std = np.mean(std_img)
    return len(key_points_sift), mean, std



def adv_predict(image):
    in_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    image = in_transform(image)
    columns = ['KP', 'mean', 'std']
    clf = LogisticRegression()
    filename = 'finalized_model.sav'
    clf = joblib.load(filename)
    kp, mean, std = sift_keypoints(image)
    new_data = pd.DataFrame([[kp, mean, std]], columns= columns)
    prediction = clf.predict(new_data)
    print(f'Prediction: {prediction[0]}')
    probability = clf.predict_proba(new_data)[:, 1]
    print(f'Probability of positive class: {probability[0]:.4f}')



image = plt.imread('image.jpg')
adv_predict(image)