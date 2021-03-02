"""
Utility functions needed to compute the APPFD, HoGD and HAPPS descriptors

@author: Ekpo Ekpo Otu (eko@aber.ac.uk)
"""
# ------------------------------------------------------------------------------------------------- #
#Import the needed library.
import numpy as np
from scipy.spatial import cKDTree
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from itertools import combinations
import open3d


# ------------------------------------------------------------------------------------------------- #
def div0(a, b):
	with np.errstate(divide = 'ignore', invalid = 'ignore'):
		c = np.true_divide(a, b)
		c[ ~ np.isfinite(c)] = 0. 
	return c
	

# Function that computed Alpha and Thetha features for LSP
def angrw(vecs1, vecs2):
	p1 = np.einsum('ij,ij->i', vecs1, vecs2)
	p2 = np.einsum('ij,ij->i', vecs1, vecs1)
	p3 = np.einsum('ij,ij->i', vecs2, vecs2)
	p4 = div0(p1, np.sqrt(p2 * p3))
	return np.arccos(np.clip(p4, -1.0, 1.0))
# ------------------------------------------------------------------------------------------------- #

#Function to Normalize 1D array (Min-Max Scaling)
def norm1D(fv):
	'''
	NOTE: Adding together the values returned will NOT SUM to 1.
	fv - Must be an [N x 1] or [1 x N] array.
	
	Author: Ekpo Otu (eko@aber.ac.uk)
	'''
	fvMin = fv.min()
	return (fv - fvMin) / (fv.max() - fvMin) 
	
# ------------------------------------------------------------------------------------------------- #
#Function to find the nearest neighbours within a sphere radius 'r', to the interest point, (using Scikit-Learn) and return those 'neighbours' and 'normals' corresponding to the neighbours.
def rnn_normals_skl(scaledPs, Ns, ip, r, leafSize = 30):
	'''
	INPUT:
	-----
	scaledPs(N x 3 array): Point Cloud data 
	ip(1 x 3 array): A single point from the input 'scaledPs' array.
	r(Float: Default, r=0.17 for Normal Vector Estimation, r=0.35 for Local Surface Patch Selection.): 
		Sphere radius around which nearest points to the 'Interest Points' are obtained. 
	
	OUTPUT:
	-----
	neighbours(N x 3 array): Coordinates of the points neighbourhood within the distance givien by the radius, 'r'.
	dist(1 x N array): Contains the distances to all points which are closer than 'r'. N is the size or number of neighbouring points returned.
	
	Author: Ekpo Otu(eko@aber.ac.uk) 
	'''	
	neigh = NearestNeighbors(radius = r)
	neigh.fit(scaledPs)
	NearestNeighbors(algorithm = 'auto', leaf_size = leafSize) 
	rng = neigh.radius_neighbors([ip]) 
	dist = np.asarray(rng[0][0]) 
	ind = np.asarray(rng[1][0]) 
	if len(ind) < 5:
		k = 15 
		tree = cKDTree(scaledPs, leafsize = 2) 
		distn, indx = tree.query(ip, k) 
		kcc = scaledPs[indx] 
		nNs = Ns[indx]	 
		return kcc, nNs
	else:
		nn = scaledPs[ind] 
		nNs = Ns[ind] 
		return nn, nNs

# ------------------------------------------------------------------------------------------------- #
#Function that uses scipy library's 'FASTER' cKDTree to find Nearest neighbour within a radius r, to point p of interest
#With the condition that the minimum number of r-Neighbours to any given 'ip' MUST be greater than 5. Else k-NN,
#where k = 9, is used instead.
def rnn_conditional(Ps, ip, rr = 0.17, kk = 9):
	'''
	INPUT:
		- Ps: N x 3 Array of pointCloud.
		- ip: 1x3 vector or coordinate, which is EACH point from the 'sub-sampled points' or 'keypoints'.
		- rr (Floating Value): Radius, 'rr', around which Neighbouring points to 'ip' are to be determined.
		- kk (Integer Value): If the 'number of r-neigbours' to 'ip' is < 5, then USE k-NN search on 'Ps', where 'k' = kk.
	OUTPUT:
		- rcc: r-Nearest points to interest point (i), if len(rcc) >= 5
		- kcc: k-Nearest points to interest point (i), if len(rcc) < 5
	'''
	tree = cKDTree(Ps, leafsize = 2)
	indx = tree.query_ball_point(ip, rr)
	rcc = Ps[indx]
	if rcc.shape[0] < 5:
		_, indx = tree.query(ip, kk)
		kcc = Ps[indx] 
		return kcc
	else:
		return rcc	

# ------------------------------------------------------------------------------------------------- #
#Function to find the principal axes of a 3D point cloud.
def computePCA(Ps):
	'''
	Author: Ekpo Otu (eko@aber.ac.uk)
	'''
	Ps -= np.mean(Ps, axis=0)
	coords = Ps.T
	CM = np.cov(coords)
	eval, evec = np.linalg.eig(CM)
	order = eval.argsort()[::-1]
	eval = eval[order]
	evec = evec[:, order]
	return eval, evec

# ------------------------------------------------------------------------------------------------- #
def knn2(Ps, Ns, ip, k):
	tree = cKDTree(Ps, leafsize = 2)
	dist, indx = tree.query(ip, k)
	kcc = Ps[indx]
	kccn = Ns[indx]
	return kcc, kccn
	
# ------------------------------------------------------------------------------------------------- #
# Helper functions
def gsp(Ps, Ns):
	surflets = [(Ps[i], Ns[i]) for i in range(0, len(Ps))]
	return np.asarray(surflets)
	
def gpe(val, comb = 2):
	return np.array(list(combinations(val, comb)))
	
# ------------------------------------------------------------------------------------------------- #	
#Given a Down-sampled pointsCloud or sub-Cloud, 'Find and Return' the EXACT points from the 'main pointsCloud' that are
#CLOSEST to the sub-Cloud, and the respective/corresponding subCloud Normals.
def getActual(subCloud, Ps, Ns):
	'''
	INPUTS:
	subCloud(N x 3 array):  minimal points, Downs-sampled from the main pointsCloud. 
	Ps(N x 3 array):  main pointsCloud.
	Ns(N x 3 array):  Normal Vectors to the Ps.
	
	Author: Ekpo Otu(eko@aber.ac.uk) 
	'''
	k = 1
	adc = [] 
	adcn = [] 
	for p in subCloud:
		coord, norm = knn2(Ps, Ns, p, k)
		adc.append(coord)
		adcn.append(norm)
	return np.asarray(adc), np.asarray(adcn)
# ------------------------------------------------------------------------------------------------- #

#Function to compute 'Ns' for every p in Ps.
#This function uses scipy library's cKDTree to find k-Nearest neighbours OR points within a radius rr, to point p of interest
def Ps_vs_Ns(Ps, rr = 0.17, kr = 9):
	'''
	INPUTS:
		Ps:      N x 3 Array of 'pointCloud', sampled from 3D Triangular Mesh.
		rr (Floating Value): Radius, 'rr', around which Neighbouring points to 'ip' are to be determined.

		kk (Integer Value): The number of neighbours to an interest-point, to use for normal calculation. Default/Minimum = between 7 to 15. Really depends on the number of random points sampled for Ps.	
	OUTPUT:
		[Ps], [Ns]: Each, an N x 3 array of data, representing 'Ps'.
		
	Author: Ekpo Otu(eko@aber.ac.uk)
	'''
	pts = []
	Ns = []
	for i in range(0, len(Ps)):
		ip = Ps[i]
		pts.append(ip)
		nn = rnn_conditional(Ps, ip, rr, kk)
		_, evec = computePCA(nn)
		Pn = evec[:, 2]
		Ns.append(Pn)
		
	return np.asarray(pts), np.asarray(Ns)	
	
# ------------------------------------------------------------------------------------------------- #
# Function to Down-sample a Given Point Cloud, using Voxel Grid, with Open3D.
def ds_Open3d(Ps, vs = 0.15):
	'''
	INPUTS:
	Ps - Open3D.PointCloud object (pcd), loaded in with Open3d Python.
	vs(Float) - Size of occupied voxel grid. 
	OUTPUTS:
	dsp - Nx3 array of the down-sampled point cloud. 
 
	Author: Ekpo Otu (eko@aber.ac.uk)
	'''
	dsp = open3d.voxel_down_sample(Ps, vs)
	return np.asarray(dsp.points)
	
# ------------------------------------------------------------------------------------------------- #

#Compute and Flatten a multi-dimensional Histogram for an [M x D] data. M = Observations, D = Dimensions
def multi_dim_hist(data, nBins):
	'''
	INPUTS:
	-------
		- data: An [M x D] or [N x D] array of Numpy data.
		- nBins: (Integer), size of bins in Each dimension of the vector/data.
	OUTPUT:
		- 1D array: Normalized Vector, of the flattened histogram.
	STEPS:
	------
		- Compute a multi-dimensional Histogram, using 'scipy.stats.binned_statistic_dd()' in Python
		- Flatten the Histogram to a 1D vector, of size/length: N^d i.e Number-of-bins raised to power Number-of-dimensions.
		- Replace all zero-values of the bins with lowest non-zero value
		- Finally, Normalize the histogram to the range [0, 1]
	AUTHOR:
	-------
	Ekpo Otu(eko@aber.ac.uk)
	'''
	hist = stats.binned_statistic_dd(data, values = False, statistic = 'count', bins = nBins)[0]
	hf = hist.flatten()
	hn = norm1D(hf)
	lowestNoneZero_min = min(i for i in hn if i > 0)
	lower = 0.98 * lowestNoneZero_min
	finalVector = np.where(hn < lowestNoneZero_min, lower, hn)
	return finalVector

# ------------------------------------------------------------------------------------------------- #