# Visual Classification using Bag of Visual Words Approach
## Train an SVM classifier using BoVW technique.

<b> How to Run? </b> <br>
To run the image classifier, execute the following from the command line: 
```
python3 BoVWC.py --train path [TRAIN PATH] --test path [TEST PATH] --no clusters [NO CLUSTERS] --kernel [KERNEL] 
```
where,
1. `TRAIN PATH ` : Path of train dataset.
2. `TEST PATH` : Path of test dataset.
3. `NO CLUSTERS` : Number of clusters, default 50.
4. `KERNEL` : Type of kernel, linear or precomputed, default linear.

For example, if you want to classify images with the 'precomputed' kernel, where the train path is 'dataset/train' and the test path is 'dataset/test' by choosing '100' cluster centres. You have to execute the following command:
```
python3 BoVWC.py --train path dataset/train --test path dataset/test --no clusters 100 --kernel precomputed
```

<b> Dataset Used </b> <br>
You can use images inside the 'dataset' folder as the dataset to run the classifier. <br>
Note: To visualize the key points/features extracted, you can refer to the `ImageRetrival/BoVW_Code.ipynb` file.

<b> Comparative Study </b> <br>
The classifier is evaluated based on experimentation over various sets of no. of clusters and kernel type (linear & precomputed) of the SVM classifier. You can refer to the `BoWC.ipynb` file. 

<b> Key Observations </b> <br>
1. When we resize images to smaller dimensions, the number of descriptors for each image and testing accuracy decreases.
2. Higher cluster size means higher training accuracy.
3. Experiments with `kernel_type = precomputed` are more accurate than `kernel_type = linear` since precomputed Chi-Squared kernels are useful for handling discrete features such as bag-of-features.
