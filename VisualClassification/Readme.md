<b> How to Run? </b> <br>
To run image classifier, execute the following from the command line: 
```
python3 BoWC.py --train path [TRAIN PATH] --test path [TEST PATH] --no clusters [NO CLUSTERS] --kernel [KERNEL] 
```
where,
1. TRAIN PATH : Path of train dataset.
2. TEST PATH : Path of test dataset.
3. NO CLUSTERS : Number of clusters, default 50.
4. KERNEL : Type of kernel, linear or precomputed, default linear.

For example, if you want to classify images with precomputed kernel, where train path is dataset/train and test path is dataset/test by choosing 100 cluster centers. You have to execute following command:
```
python3 BoWC.py --train path dataset/train --test path dataset/test --no clusters 100 --kernel precomputed
```

<b> Dataset Used </b> <br>
You can use images inside the 'dataset' folder as the dataset for running the classifier.

<b> Comparative Study </b> <br>
The classifier is evaluated based on experimentation over various sets of no. of clusters and kernel type of the SVM classifier. You can refer to the `BoWC.ipynb` file.
