
'''
This Script Can Run Alone.

Author: Ekpo Otu (e-mail: eko @aber.ac.uk)


Shape Retrieval Contest (SHREC 2021 Protein Shape Benchmark Dataset): Track 3: Retrieval and Classification of Protein Surfaces Equiped with Physical and Chemical Properties
DATASET: Geometry (i.e. 3D triangular Meshes)
TRAININD DATASET: The training set countains 3,585 models. Divided into 144 Classes.
TESTING DATASET: The test set contains 1,543 models.
'''


#Import the needed library.
import utilities01 as fnc
import numpy as np
from numpy.core.umath_tests import inner1d
np.set_printoptions(suppress = True) #Suppress the printing of numbers in Scientific Notation or Exponential Form.

import matplotlib.pyplot as plt
import open3d
import trimesh
from sklearn import preprocessing

import ntpath
import gc

# ------------------------------------------------------------------------------------------------- #
# TRAINING dataset, in the Geometry category (i.e. .OFF mesh files)
filepath = "../dataset/1.off" 
filename = "1.off" 

# Where to save outputs
outputdir = "../outputdir/"

# Number of Random Points to sample from 3D Triangular Mesh
N = 3500  #4500

# Number of Bins to use for the multi-dimensional histogram
nBins = 8

# Note, the LARGER the value of 'r', the more the number of points in the 'Local Surface Patch(s)' extracted.
# Example if 'r' = 0.17, about 26 points are obtained per LSP, and when 'r'= 0.27, about 65 points are obtained for the same LSP, instead.
r = 0.50    #0.40
vs = 0.30 

# ------------------------------------------------------------------------------------------------- #
# Load-in 3D Model or Shape.
mesh = trimesh.load_mesh(filepath)

# Sample N points from the surface of the MESH - Gives 'pointCloud'.
# Use trimesh's native Function - 'trimesh.sample.sample_surface(mesh, N)' to Generate N-Randomly Sampled points = 'PointCloud'
Ps = trimesh.sample.sample_surface(mesh, N)[0]

# Compute Ns for Ps
Ps, Ns = fnc.Ps_vs_Ns(Ps)
# ------------------------------------------------------------------------------------------------- #


# Function to compute HAPPS (6D-APPFD + HoGD)
def compute_HAPPS(Ps, Ns, filename, outputdir, N, r, vs, nBins):
	'''    
	INPUT: 
	i. Ps: N x 3 array, PointsCloud for a single 3D model.

	ii. Ns:  Normal Vectors correspoinding to every points in the Ps.

	iii. filename: Specific filename of a single 3D model to be read and processed. 
		Example: cat4.off

	iv. outputdir:  Path/directory where output data/values from this function should be saved to. 
		Example: "c:/outputdir/"

	v. N - Number 'Random'/'Uniform' Samples from 3D Triangular Mesh (i.e filename.obj). Default N = 4500

	vi. r (Float: Default = 0.27): Radius param, used by r-nn search to determine the size of Local Surface Patch (LSP) or Region. 

	vii. nBins = 7 or 8     #Number of bins for the multi-dimensional histogram of the Features. Default = 4.

	viii. vs(Float, Default = 0.15):  Parameter to be used by the Voxel Down-Sampling function of Open3D.

	OUTPUT:
	i. HAPPS = [...]: which becomes the final SD for the given 3D model for a single 3D input model.
	We have scaled our input model or 'Ps' before Features Extraction - such that its RMS distance from origin = 1.

	Author: Ekpo Otu (eko@aber.ac.uk)
	'''
	basename = ntpath.basename(filename)[:-4] 
	pcd.points = open3d.Vector3dVector(Ps)
	dsc = fnc.ds_Open3d(pcd, vs)
	print("\nOUTPUT:        Downsampled Cloud Size:\t", len(dsc))

	adsc, adscn = fnc.getActual(dsc, Ps, Ns)
	accummulated_final_appFeats = []
	for pnt in range(0, len(adsc)):
		ip = adsc[pnt]
		ipn = adscn[pnt]
		nn, nNs = fnc.rnn_normals_skl(Ps, Ns, ip, r, leafSize = 30)
		patchCentre = np.mean(nn, axis = 0)
		location = ip - patchCentre
		lsp = fnc.gsp(nn, nNs) 
		lsp_pairs = fnc.gpe(lsp, comb = 2) 
		p1 = lsp_pairs[:, 0, 0, :] 
		p2 = lsp_pairs[:, 1, 0, :] 
		n1 = lsp_pairs[:, 0, 1, :] 
		n2 = lsp_pairs[:, 1, 1, :]
		
		p2_p1 = lsp_pairs[:, 1, 0, :] - lsp_pairs[:, 0, 0, :] 
		p1_p2 = lsp_pairs[:, 0, 0, :] - lsp_pairs[:, 1, 0, :] 
		
		lhs = abs(np.einsum('ij,ij->i', lsp_pairs[:, 0, 1, :], (lsp_pairs[:, 1, 0, :] - lsp_pairs[:, 0, 0, :])))    #Left-Hand-Side
		lhs[np.isnan(lhs)] = 0. 
		rhs = abs(np.einsum('ij,ij->i', lsp_pairs[:, 1, 1, :], (lsp_pairs[:, 1, 0, :] - lsp_pairs[:, 0, 0, :])))    #Right-Hand-Side
		rhs[np.isnan(rhs)] = 0. 
 
		vecs1 = p1 - patchCentre
		vecs2 = p1 - location
		
		lhs_angles1 = fnc.angrw(p1_p2, vecs1) 
		lhs_angles2 = fnc.angrw(p1_p2, vecs2)
		
		crossP1 = np.cross(p2_p1, n1)
		crossP1[np.isnan(crossP1)] = 0.
		V1 = fnc.div0(crossP1, np.sqrt(inner1d(crossP1, crossP1))[:, None])
	 
		W1 = np.cross(n1, V1)
		W1[np.isnan(W1)] = 0.
		x = np.einsum('ij,ij->i', W1, lsp_pairs[:, 1, 1, :]) 
		x[np.isnan(x)] = 0. 
		y = np.einsum('ij,ij->i', n1, lsp_pairs[:, 1, 1, :]) 
		y[np.isnan(y)] = 0. 
		alpha1 = np.arctan2(x, y) 
		beta1 = np.einsum('ij,ij->i', V1, lsp_pairs[:, 1, 1, :]) 
		 
		normedP1 = fnc.div0(p2_p1, np.sqrt(inner1d(p2_p1, p2_p1))[:, None])                 
		gamma1 = np.einsum('ij,ij->i', n1, normedP1)
		rheo1 = np.sqrt(inner1d(p2_p1, p2_p1))
		
		rppf_lhs = np.column_stack((lhs_angles1, lhs_angles2, alpha1, beta1, gamma1, rheo1))
		indx = np.asarray(np.nonzero(lhs <= rhs))
		final_rppf_lhs = np.squeeze(rppf_lhs[indx], axis = 0) 
		
		vecs1x = p2 - patchCentre
		vecs2x = p2 - location
		rhs_angles1 = fnc.angrw(p2_p1, vecs1x)
		rhs_angles2 = fnc.angrw(p2_p1, vecs2x)
		crossP2 = np.cross(p1_p2, n2)
		crossP2[np.isnan(crossP2)] = 0.
		
		V2 = fnc.div0(crossP2, np.sqrt(inner1d(crossP2, crossP2))[:, None])
		W2 = np.cross(n2, V2)
		W2[np.isnan(W2)] = 0.
		x2 = np.einsum('ij,ij->i', W2, lsp_pairs[:, 0, 1, :])
		x2[np.isnan(x2)] = 0.
		y2 = np.einsum('ij,ij->i', n2, lsp_pairs[:, 0, 1, :])
		y2[np.isnan(y2)] = 0.
		
		alpha2 = np.arctan2(x2, y2)
		beta2 = np.einsum('ij,ij->i', V2, lsp_pairs[:, 0, 1, :])
		normedP2 = fnc.div0(p1_p2, np.sqrt(inner1d(p1_p2, p1_p2))[:, None])
		gamma2 = np.einsum('ij,ij->i', n2, normedP2)
		rheo2 = np.sqrt(inner1d(p1_p2, p1_p2))
		
		rppf_rhs = np.column_stack((rhs_angles1, rhs_angles2, alpha2, beta2, gamma2, rheo2))
		indxx = np.asarray(np.nonzero(lhs > rhs))
		final_rppf_rhs = np.squeeze(rppf_rhs[indxx], axis = 0)

		full_final_fppf = np.vstack((final_rppf_lhs, final_rppf_rhs))
		
		columns_1to5 = preprocessing.minmax_scale(full_final_fppf[:, 0:5])
		column_6 = full_final_fppf[:, 5]
		normalizedfeats = np.column_stack((columns_1to5, column_6))
		
		accummulated_final_appFeats.append(normalizedfeats)
	accummulated_final_appFeats = np.vstack(accummulated_final_appFeats)
	appfd = fnc.multi_dim_hist(accummulated_final_appFeats, nBins)  #APPFD - Augmented Point-Pairs Features Descriptors
	
	#Compute HoGD
	ct = np.mean(Ps, axis = 0)
	dist = np.sqrt(np.sum((ct - Ps)**2,axis = 1))
	ldata = Ps.shape[0]
	bins2 = 65
	histo, _ = np.histogram(dist, bins = bins2, density = False)
	norm_hist = histo.astype(np.float32) / histo.sum()
	
	# HAPPS
	happs = np.hstack((appfd, norm_hist))
	happs = happs.astype(np.float32) / happs.sum()
	d = len(happs)
	gc.collect()
	plt.plot(happs, color = 'darkred', label = "$Descr.$")
	plt.xlabel('{}-Dim. Descr - File: {}'.format(d, basename))
	plt.ylabel('Frequency of Descriptor Data (PDF)')
	plt.title("HAPPS (APPFD+HoDD)-{}vs{} Bins".format(nBins, bins2))
	plt.legend()
	plt.savefig(outputdir + '{}_happs'.format(basename) + str(N) + 'pts.pdf')
	plt.close()

	return happs
	
# ------------------------------------------------------------------------------------------------- #
# Compute, save and return HAPPS for a single 3D mesh.
startTime = time.time()

happs_descr = compute_HAPPS(Ps, Ns, filename, outputdir, N, r, vs, nBins)

stopTime = time.time()
duration = stopTime - startTime
# ------------------------------------------------------------------------------------------------- #

print('HAPPS successfully computed!!!\n Descriptor Dimension:\t', len(happs_descr))
print("Total Computation Time for {}:\t".format(filename), str(duration) + 'secs.')