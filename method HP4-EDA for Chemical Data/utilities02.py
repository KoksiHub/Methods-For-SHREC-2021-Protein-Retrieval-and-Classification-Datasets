"""
Utility functions needed to compute the HP4-EDA descriptors

@author: Ekpo Ekpo Otu (eko@aber.ac.uk)
"""
# ------------------------------------------------------------------------------------------------- #
#Import the needed library.
import numpy as np
import glob, re
from scipy.spatial.distance import pdist, squareform 

# ------------------------------------------------------------------------------------------------- #
#Generic Function that returns a SORTED list of ALL filenames in a given directory(path), in ascending order.
def sort_directoryfiles(path, ext):
    '''
    INPUTS:
		(i) path: (Path to the directory containing files. Example: 'c:/ekpo/dataset/')
		(ii) ext: (Extension of the files we intend to sort. Example: '.txt')
    OUTPUTS:
		(i) sortedf: (List of ONLY filenames in 'path'. Example: '1.txt, 2.txt, ..., 9.txt, 10.txt, 11.txt ..., 99.txt, 100.txt,')
   
    AUTHOR: Ekpo Otu (eko@aber.ac.uk) 
    '''
    filepaths = glob.glob1(path,'*'+ ext)
    sortedf = sorted(filepaths, key=lambda x:float(re.findall("(\d+)",x)[0]))
    sortedpath = []
    for i in np.arange(0, len(sortedf)):
        fullpath = path + sortedf[i]
        sortedpath.append(fullpath)
    return sortedpath


# ------------------------------------------------------------------------------------------------- #
def compare_allDescriptors_EMD(allDescriptors, outputdir, save_matrix_as = True):
	'''
	PURPOSE:
	To compute the EMD (Wasserstein Distance) between a set of shape-descriptors.

	INPUTS:
		(i) allDescriptors: An [M x K] matrix of all descriptors, where M is the total number of models/objects in the database, and K is the length of each descriptor.
		(ii) outputdir: Location or Directory, where the output to this function would be saved in. E.g: outputdir = "c:/myPyGraphics/2018_Research_Implementations/5_May_2018/ekpoMayImplementations/"
		(iii) save_matrix_as: Default(None): If 'save_matrix_as' is given, This MUST be 'STRING' input, and this function adds the '.matrix' extension. E.g: "jaccardDistanceMatrix_spbLSD.txt"
	OUTPUT: 
		(i) Dist_matrix: An N x N matrix, where the ij entry is the wasserstein_distance between the shape-descriptor for point cloud i and point cloud j.
	AUTHOR: Ekpo Otu (eko@aber.ac.uk) 
	'''
	pwdist = pdist(allDescriptors, wasserstein_distance) 
	Dist_matrix = squareform(pwdist)
	if save_matrix_as:
		descrpath = outputdir + save_matrix_as + ".matrix"
		np.savetxt(descrpath, Dist_matrix, fmt='%f')
		
	return Dist_matrix
# ------------------------------------------------------------------------------------------------- #


def compare_allDescriptors_kld(allDescriptors, outputdir, save_matrix_as = True):
	'''
	PURPOSE:
	To compute the Kullback Leibner Divergence Similarity/Distance between a set of shape-descriptors.

	INPUTS:
	allDescriptors: An MxK matrix of all descriptors, where M is the total number of models(3D meshes) in the database, and K is the length of each descriptor.
	outputdir: Location or Directory, where the output to this function would be saved in. E.g: outputdir = "c:/myPyGraphics/2018_Research_Implementations/5_May_2018/ekpoMayImplementations/"
	save_matrix_as: Default(None): If 'save_matrix_as' is given, This MUST be 'STRING' input, and this function adds '.txt' extension. E.g: "kldDistanceMatrix_spbLSD.txt"

	OUTPUT: 
	Dist_matrix: An N x N matrix, where the ij entry is the KLD distance between the shape-descriptor for point cloud i and point cloud j.
	'''
	pw_dist = pdist(allDescriptors, kullback_divergence)
	Dist_matrix = squareform(pw_dist)
	if save_matrix_as:
		descrpath = outputdir + save_matrix_as + ".matrix"
		np.savetxt(descrpath, Dist_matrix, fmt='%f')
		
	return Dist_matrix