# [SHREC 2021, Protein Shape Retrieval and Classification Challenge](http://shrec.ge.imati.cnr.it/shrec21_protein/)

This repository contain implementation  of two different 3D shape retrieval algorithms (methods) applied on the SHREC 2021 datasets on Retrieval and classification of protein surfaces equipped with physical & chemical properties.
* Further details regarding this retrieval track can be found **[here](http://shrec.ge.imati.cnr.it/shrec21_protein/)**.

### Our Implementation
Considering that two different datasets are presented for [this retrieval and classification challenge](http://shrec.ge.imati.cnr.it/shrec21_protein/), as highlighted below, this repository, therefore, presnts two different retrieval methods, which are:
* ***[HAPPS](https://github.com/KoksiHub/Methods-For-SHREC-2021-Protein-Retrieval-and-Classification-Datasets/blob/main/method%20HAPPS%20for%20Geometry%20Data/HAPPS_3Dretrievalmethod.py)*** (Hybrid Augmented Point-pair Signature): This is a 3D shape descriptor which is applicable for 3D triangular meshes (or point cloud) datasets (Training and Testing).
* ***[HP4-EDA](https://github.com/KoksiHub/Methods-For-SHREC-2021-Protein-Retrieval-and-Classification-Datasets/tree/main/method%20HP4-EDA%20for%20Chemical%20Data)*** (Histogram of Processed Physicochemical Properties of Protein, following Exploratory Data Analysis): This is a simple deescriptive statistic (DS) based descriptor, developed for the Physicochemical protein datasets (Training and Testing).

### [Dataset, Ground Truth and Evaluation](http://shrec.ge.imati.cnr.it/shrec21_protein/)
"A dataset of approximately 5000 protein surfaces and corresponding properties will be provided. Each model will be represented by an OFF file and a TXT file. Each row of the TXT file corresponds to a vertex of the triangulation in the OFF file (in the same order); each row in the TXT file contains the physicochemical properties evaluated in the corresponding vertex in the OFF file. The dataset will be subdivided into a training and a test set (in the proportion 70%-30%). An example of the 3D models we will use in this track is shown in Figure 1. The example OFF file can be downloaded here, while the TXT file with the corresponding physicochemical properties is available here."

Each model in the test set will be used in turn as a query against the remaining part of the test set. To compare the performance of the methods equipped of physicochemical properties against the simple geometric models we will ask the participants two runs:

**Run A:** only the OFF files of the models are considered (i.e., only the geometry is considered);
**Run B:** in addition to the geometry, the participant is asked to also consider the text file (texture matching).
For a given query, the goal of the track is twofold: to retrieve the most similar objects and (optionally) to classify the query itself. The closeness of the retrieved structures with the ground truth might be evaluated a-priori on the basis of their SCOPe classification or of their sequence similarity.
See reference for the above **[here](http://shrec.ge.imati.cnr.it/shrec21_protein/)**.

### Additional notes
- The ***[HAPPS method](https://github.com/KoksiHub/Methods-For-SHREC-2021-Protein-Retrieval-and-Classification-Datasets/blob/main/method%20HAPPS%20for%20Geometry%20Data/HAPPS_3Dretrievalmethod.py)*** produces results for Run A (Run-1a, Run-2a, and Run-3a).
- The ***[HP4-EDA methods](https://github.com/KoksiHub/Methods-For-SHREC-2021-Protein-Retrieval-and-Classification-Datasets/tree/main/method%20HP4-EDA%20for%20Chemical%20Data)*** produce results for Run B ([Run-1b](https://github.com/KoksiHub/Methods-For-SHREC-2021-Protein-Retrieval-and-Classification-Datasets/blob/main/method%20HP4-EDA%20for%20Chemical%20Data/HP4EDA_physicochemical_descriptorRun1b.py), [Run-2b](https://github.com/KoksiHub/Methods-For-SHREC-2021-Protein-Retrieval-and-Classification-Datasets/blob/main/method%20HP4-EDA%20for%20Chemical%20Data/HP4EDA_physicochemical_descriptorRun2b.py), and [Run-3b](https://github.com/KoksiHub/Methods-For-SHREC-2021-Protein-Retrieval-and-Classification-Datasets/blob/main/method%20HP4-EDA%20for%20Chemical%20Data/HP4EDA_physicochemical_descriptorRun3b.py.py)).

Both of our methods are completely implemeted in Python 3.6. We strictly adopt the FOP (Functional Oriented Programming) coding style for all functions and algorithms presented here.
