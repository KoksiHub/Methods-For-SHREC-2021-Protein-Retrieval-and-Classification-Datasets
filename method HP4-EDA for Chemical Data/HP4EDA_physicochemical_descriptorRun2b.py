
'''This Script Can Run Alone.
Author: Ekpo Otu (e-mail: eko @aber.ac.uk)


Shape Retrieval Contest (SHREC 2021 Protein Shape Benchmark Dataset): Track 3: Retrieval and Classification of Protein Surfaces Equiped with Physical and Chemical Properties
Dataset Type: Chemical (i.e. with 3 columns of Physicochemical Properties)

TRAININD DATASET: The training set countains 3,585 models. Divided into 144 Classes.
TESTING DATASET: The test set contains 1,543 models.
'''
# ---------------------------------------------------------------------------------------- #
#Import the needed library.
import numpy as np  
import pandas as pd 
import os  

import utilities02 as fnc
import ntpath

#Suppress the printing of numbers in Scientific Notation or Exponential Form.
np.set_printoptions(suppress = True)

import time

# ---------------------------------------------------------------------------------------- #
# SEED, in order to produce a stable output from this notebook, across different runs
np.random.seed(28)

# TRAINING and TESTING datasets, in the Chemical category (i.e. .TXT property files)
dataset_training = "../dataset2_chemical/training/"
dataset_testing = "../dataset2_chemical/testing/"
file_extension = ".txt"

#Path to the folder to LOAD or SAVE in results or data during/after processing
outputdir = "../outputdir/run2b/"

# ---------------------------------------------------------------------------------------- #
# USING lowIQR = 10% or 25% and highIQR = 75% or 90% IQR
lowIQR = 0.10
highIQR = 0.90
nBins = 150

benchmark = "shrec21proteinChem"
metric = 'KLD' 																	#Similarity Metric used to compute Square-Distance-Matrix
saveMatrixAs = 'HP4EDA_run2b_' + benchmark + '_{}'.format(metric)				#Name to call Computed Square-Distance-Matrix for all models in Db.

# ---------------------------------------------------------------------------------------- #
#Function to compute Global DECRIPTOR for PHYSICOCHEMICAL Properties of Proteins.
def compute_HP4EDA_forAlldata_run2b(dataset_source, file_extension, lowIQR, highIQR):
	'''
	INTRODUCTION:
	In this implementation, we would remove OUTLIERS from each of the 3 variables (f1, f2, and f3) using lower(lowIRR = 10% or 25%) and higher(highIQR = 75% or 90%) technique.
		We would then NORMALIZE the final extracted and concatenated values that form the final feature-vector. Normalize between the range [0, 1]
		
	Identifying Outliers with Interquartile Range (IQR)
		The interquartile range (IQR) is a measure of statistical dispersion and is calculated as the difference between the $highIQR^{th}$ and $lowIQR^{th}$ percentiles. 
		It is represented by the formula $IQR = Q3 − Q1$. The lines of code below calculate and print the interquartile range for each of the variables in the dataset.

	INPUTS: 
	(i) dataset_source: Path to the dataset. Example: 'c:/ekpo/shrec21proteins/dataset_training/'
	(ii) file_extension:  Extension to the data files in the dataset. Example: '.txt' 
	(iii) lowIQR: Float value, Lower Interquartile range. Default = 10%
	(iv) highIQR: Float value, Higher Interquartile range. Default = 90%

	OUTPUT:
	(i) allDescriptors: [N x D] array of computed descriptors. Where N is the number of files in dataset and D is dimension of each descriptor.
	
	Author: Ekpo Otu (e-mail: eko @aber.ac.uk)
	'''
	sorted_filepaths = fnc.sort_directoryfiles(dataset_source, file_extension)
	D = len(sorted_filepaths)
	allDescriptors = []
	
	for i in np.arange(0, len(sorted_filepaths)):
		fn = ntpath.basename(sorted_filepaths[i])
		item = pd.read_csv(sorted_filepaths[i], sep = " ", header = None, names = ["f1", "f2", "f3"])
		print("...Computing Shape-Descriptor for Data {} of {}".format(i+1, D))
		startTime = time.time()
		
		Q1 = item.quantile(lowIQR)
		Q3 = item.quantile(highIQR)
		IQR = Q3 - Q1
		
		item = item[~((item < (Q1 - 1.5 * IQR)) | (item > (Q3 + 1.5 * IQR))).any(axis = 1)]
		
		mea = np.array(item.mean())
		stv = np.array(item.std())
		
		quant = np.array(item.quantile([lowIQR, highIQR], axis = 0))
		corr_matrix = item.corr()
		compare = corr_matrix["f1"].sort_values(ascending = False)
		descriptor = np.hstack((mea, stv, quant[:,0], quant[:,1], quant[:,2], compare[1]))
		print('\nFINAL DESCRIPTOR:\n', descriptor)
		allDescriptors.append(descriptor)

		stopTime = time.time()
		duration = stopTime - startTime
		print("...Processing Time for {}:\t".format(fn), str(duration) + 'secs.')
		print('\n')
	return allDescriptors 


# ---------------------------------------------------------------------------------------- #
#COMPUTE DESCRIPTORS FOR ALL MODELS
allDb_Descriptors = compute_HP4EDA_forAlldata_run2b(dataset_testing, file_extension, lowIQR, highIQR) 
print('\nallDb_Descriptors First 3:\n', allDb_Descriptors[:3])
print('\nallDb_Descriptors.shape:\n', allDb_Descriptors.shape)
 
# ---------------------------------------------------------------------------------------- #
'''
Compute Square 'Distance Matrix' (Similarity Matrix) for All Descriptors, using 1D EMD Metric
		 DESCRIPTOR COMPARISONS    -    RETURN NxN DISTANCE METRIX, D.              
'''
matrixTitle = '{} Similarity Matrix For'.format(metric) + benchmark
KLD_Matrix = fnc.compare_allDescriptors_kld(allDb_Descriptors, outputdir, saveMatrixAs)
print("Shape of KLD_Matrix:\n", KLD_Matrix.shape)
 
# ---------------------------------------------------------------------------------------- #
