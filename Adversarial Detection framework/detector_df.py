import concurrent.futures
import torch
import cv2
import numpy as np
import pandas as pd


from config import *
import data_loader
from torchvision import transforms
from torch.autograd import Variable
from tqdm import tqdm

# Defining the detector and its threshold

sift = cv2.SIFT_create()
sift.setEdgeThreshold(EDGE_THRESHOLD)

def sift_keypoints(tensor):
    array = tensor.numpy().transpose((1,2,0))*255
    array = array.astype(np.uint8)
    gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    key_points_sift, descriptor = sift.detectAndCompute(array,None)
    image = array.astype(np.float64)
    mean_img, std_img = cv2.meanStdDev(image)
    mean = np.mean(mean_img)
    std = np.mean(std_img)
    return len(key_points_sift), mean, std

def sift_all_parallel(test_data, batch_size = BATCH_SIZE):
    total = 0
    keypoint_len_array = []
    mean_array = []
    std_array = []
    for data_index in tqdm(range(int(np.floor(test_data.size(0)/batch_size))), desc='Creating Features'):
            target = test_label[total : total + batch_size]
            data = test_data[total : total + batch_size]
            total += batch_size
            data, target = Variable(data, requires_grad = True), Variable(target)
            data = data.detach()
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = list(executor.map(sift_keypoints, data))
            for res in results:
                keypoint_len_array.append(res[0])
                mean_array.append(res[1])
                std_array.append(res[2])
    return keypoint_len_array, mean_array, std_array
     


####################Define the detector's dataloader #################################

in_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])


print("Loading Dataset: ", DATASET)

train_loader, _ = data_loader.getTargetDataSet(DATASET, BATCH_SIZE, in_transform, DATAROOT)
test_clean_data = torch.load(OUTF+ 'clean_data_%s_%s_%s.pth' % (MODEL, DATASET, ADVERSARY))
test_adv_data = torch.load(OUTF+ 'adv_data_%s_%s_%s.pth' % (MODEL, DATASET, ADVERSARY))
test_noisy_data = torch.load(OUTF+ 'noisy_data_%s_%s_%s.pth' % (MODEL, DATASET, ADVERSARY))
test_label = torch.load(OUTF+ 'label_%s_%s_%s.pth' % (MODEL, DATASET, ADVERSARY))

keypoint_len_array, mean_array, std_array = sift_all_parallel(test_adv_data)
keypoint_len_array_org , mean_array_org, std_array_org = sift_all_parallel(test_clean_data)

keypoint_len_array = np.array(keypoint_len_array)
keypoint_len_array_org = np.array(keypoint_len_array_org)

mean_adv = np.array(mean_array)
mean_org = np.array(mean_array_org)
std_adv = np.array(std_array)
std_org = np.array(std_array_org)


keypoint_len_array = keypoint_len_array.flatten()
keypoint_len_array_org = keypoint_len_array_org.flatten()
mean_adv = mean_adv.flatten()
mean_org = mean_org.flatten()
std_adv = std_adv.flatten()
std_org = std_org.flatten()
        


data = {
    'KP':keypoint_len_array,
    'mean': mean_adv,
    'std': std_adv
}
df1 = pd.DataFrame(data)
df1['target'] = 1
data2 = {
    'KP':keypoint_len_array_org,
    'mean': mean_org,
    'std': std_org
}
df2 = pd.DataFrame(data2)
df2['target'] = -1
df_merged = pd.concat([df1, df2], axis=0)
# df_merged.info()

# final_df = df_merged.sample(frac=1).reset_index(drop=True)
final_df = df_merged
final_df['Attack'] = ADVERSARY
final_df['Dataset'] = DATASET

final_df.to_csv(OUTF + 'detector_df_%s_%s.csv' % (DATASET, ADVERSARY), index = False)

print("Dataframe saved as detector_df_%s_%s.csv" % (DATASET, ADVERSARY))
#######################################################################################