from classifierTraining import classifierTraining
from docNode import docNode
import heapq
import heapq_max
from math import fabs
from nltk.corpus import reuters
import _pickle
from time import time





"""
This module runs the various classifier methods that this project is focused on. The project revolves around refining the choices of nearest
neighbors to compare an unrefined k-NN classifier with a faster (O(n^3) - > O(n)) general inclusion classifier. The project will also compare 
refining the neighbors and applying the k-NN classifier in order to observe the behavior that refining the neighbrs has on another classification
method. This way, we can determine if the refinement method is specific to the general inclusion classifier, or if it is a universal improvement
in the choce of nearest neighbors.

"""


class hyperplaneClassifier:


	"""Compare x^2Filter/getK_Neighbors/KNN_SimScore
			   x^2Filter/getK_Neighbors/nn_inclusion (this is already known to be less accurate than previous)
			   x^2Filter/getRefined_Neighbors/KNN_SimScore
			   x^2Filter/getRefined_Neighbors/nn_inclusion
	"""
	
	@staticmethod
	def __run1__(trainDocs, testDocs):

		"""
		This method takes the filtered&processed train and processed test docs and performs the nearest neighbor retrieve
		with n=1. With this result, the k nearest neighbors of this nearest neighbor are refined using getNeighbors_refinement1,
		after which the test docs are categorized using general inclusion and the required values necessary for F1 evaluation 
		are returned.

		"""

		#testDocs is of form {docId : doc_vec}
		fp = 0
		tp = 0
		fn = 0
		for docId, vec in testDocs.items():
			nnId = hyperplaneClassifier.nearestNeighbors(trainDocs, docId, vec)
			#nearest_neighbors = hyperplaneClassifier.nearestNeighbors(trainDocs,docId,vec)
			refinedNeighbors = hyperplaneClassifier.getNeighbors_refinement1(trainDocs, nnId, vec)
			cats = set(hyperplaneClassifier.categorize_nnInclusion(refinedNeighbors))
			#cats = set(hyperplaneClassifier.categorize_kNNSimScoring(testDocs,trainDocs,docId,nearest_neighbors))
			print(cats)
			actual_cats = set(reuters.categories(docId))
			print(actual_cats)
			print("\n")
			fp += len(cats.difference(actual_cats))
			tp += len(cats.intersection(actual_cats))
			fn += len(actual_cats.difference(cats))
		return fp, tp, fn

		#precision = .753
		#recall = .744
		#F1 = .7486
		#914 2786 958 - nn


	@staticmethod
	def __run2__(trainDocs, testDocs):

		"""
		This method takes the filtered&processed train and processed test docs and performs the nearest neighbor retrieve
		with n=k. With this result, the k nearest neighbors of this nearest neighbor are categorized using kNN similairty scoring
		and the required values necessary for F1 evaluation are returned.

		"""

		#testDocs is of form {docId : doc_vec}
		fp = 0
		tp = 0
		fn = 0
		for docId, vec in testDocs.items():
			nearest_neighbors = hyperplaneClassifier.nearestNeighbors(trainDocs,docId,vec)
			cats = set(hyperplaneClassifier.categorize_kNNSimScoring(testDocs,trainDocs,docId,nearest_neighbors))
			print(cats)
			actual_cats = set(reuters.categories(docId))
			print(actual_cats)
			print("\n")
			fp += len(cats.difference(actual_cats))
			tp += len(cats.intersection(actual_cats))
			fn += len(actual_cats.difference(cats))
		return fp, tp, fn

		#precision = .765
		#recall = .7278
		#F1 = .746
		#836 2725 1019 - knn

	@staticmethod
	def categorize_kNNSimScoring(testDocs, trainDocs, docId, neighbors, fr = 0.75):
		
		"""
		Uses a similarity scoring approach for categorizing the test docs. With the nearest neighbors,
		for each class we assign a score based on which neighboring nodes have that class and assign weightings based 
		on the cosine similarity.
		"""

		#input is a test doc (assume nonempty or else the fr part in line 446 (second to last) will cause issues)

		#O(n^3)
		categories = reuters.categories(neighbors) 
		catgScores = []
		for c in categories:
			score = 0.0
			for n in neighbors:
				if c in reuters.categories(n): score += classifierTraining.dot(testDocs[docId], trainDocs[n].doc_vec) 
			catgScores.append((score, c))
		catgScores =  heapq.nlargest(int(round(len(categories)*fr)), catgScores) #should check if round function is O(1) or O(n)
		return [c for s, c in catgScores] 

	@staticmethod
	def categorize_nnInclusion(neighbors):

		"""
		General inclusion categorization (returns the categories that the nearest neighbors have). 
		Assumes an input of getNeighbors_refinement_, as the point of the experiment is to see if this O(1)
		function can replace the O(n^3) k-NN similarity scoring if we refine the neighbors.

		Based on the assumption that getNeighbors_Refinement1 or getNeighbors_Refinement2 produces a better
		set of nearest neighbors, so that no weighting is needed and we can just take all categories without 
		losing much accuracy.

		This approach probably gives good results if a point is close to the train doc, and probably gives meh
		results if not.

		"""

		#doesnt need docId info since we are using it on testDocs nearest neighbor.
		#O(1)
		return reuters.categories(neighbors)


	@staticmethod
	def getNeighbors_refinement1(trainDocs, nnId, test_vec):
		#use nearest neighbor id as nnId
		#returns a list of the acceptable nearest neighbors

		"""
		Method is to find the nearest neighbor of the test doc. Since the bisecting plane between the test and train docs separates into regions 
		of points closest to the test doc and likewise for the train doc, assume that the pointToPlane(test, train) result serves as a maximal radius
		of the cluster of points surrounding the train doc of interest such that the classification will treat all points within this maximal radius as 
		equivalent to the train Doc.

		Thus, we find the pointToPlanes between the train doc of interest and its neighbors, and only those neighbors whose point to plane value 
		is less than the maximal radius are included in the refinement. Keep in mind that the point to the plane is actually half the euclidean dist.
		"""

		nearestNode = trainDocs[nnId]
		neighbors = nearestNode.k_neighbors
		maxRadius = classifierTraining.euclideanDist(nearestNode.doc_vec, test_vec)/2.0 
		refinedNeighbors = [nnId]
		for docId in neighbors:
			dist_to_plane =  classifierTraining.euclideanDist(nearestNode.doc_vec, trainDocs[docId].doc_vec)
			if dist_to_plane < maxRadius : 
				refinedNeighbors.append(docId)
		return refinedNeighbors


	@staticmethod
	def getNeighbors_refinement2(trainDocs, nnId, test_vec):
		#use nearest neighbor id as nnId
		#returns a list of the acceptable nearest neighbors

		"""
		In this method we first choose the j-th furthest neighbor (minRadius) and assume the bisecting plane between the main training node and its
		neighbor serves as a maximal radius of which bisecting planes are close enough to the train doc. Then, we see the distance from the test doc
		to the bisecting planes of the neighbors, and if it is smaller than the maxRadius, we consider the testDoc to be close enough to the neighbor
		to include this neighbor in the test categoriation.

		Thus, we find the pointToPlanes between the test doc of interest and its neighbors, and only those neighbors whose point to plane value 
		is less than the maximal radius are included in the refinement (including the closest neighbor to the test doc).
		"""

		nearestNode = trainDocs[nnId]
		neighbors = nearestNode.k_neighbors
		minRadius = hyperplaneClassifier.minRadius(trainDocs, nearestNode)# if we dont need the w, b values then sheeit
		refinedNeighbors = [nnId]
		for docId in neighbors:
			dist_to_plane =  hyperplaneClassifier.pointToPlane(test_vec, neighbors[docId][0], neighbors[docId][1])
			if dist_to_plane < minRadius : 
				refinedNeighbors.append(docId)
		return refinedNeighbors

	@staticmethod
	def nearestNeighbors(trainDocs, testId, test_vec, n=1):
		#muse n = 1 for the refined case, use n=k for kNN approach

		#O(n^2)
		#find closest vector to new test node
		#returns the nearest neighbor id
		#optimize (should be ideally linear)
		neighbors = []
		for currId, currNode in trainDocs.items():
			if currId is not testId:# in case we use a train doc
				dist = classifierTraining.euclideanDist(currNode.doc_vec, test_vec)
				if len(neighbors) < n: 
					heapq_max.heappush_max(neighbors, (dist, currId))
				else:
					if neighbors[0][0] > dist: heapq_max.heapreplace_max(neighbors, (dist, currId))

		neighbors = [i for d, i in neighbors]
		if len(neighbors) is 1:
			return neighbors[0]
		else:
			return neighbors
		
	
	@staticmethod
	def pointToPlane(doc_vec, w, b):
		#shortest dist from doc_vec to plane defined by w and b
		#assume w is point from plane to central docNode 
	
		#####I got a zero division error here
		norm = classifierTraining.norm(w)
		if norm is not 0:
			return fabs((classifierTraining.dot(w, doc_vec)-b)/norm)
		else:
			return 0

	@staticmethod
	def minRadius(trainDocs, docNode, j=4):
		#O(n) 
		#finds the jth closest distance if j = 0 then return 0
		#assume j < k -@AM
		rankList = []
		count = 0
		for docId in docNode.k_neighbors:
			dist = classifierTraining.euclideanDist(trainDocs[docId].doc_vec, docNode.doc_vec)/2.0 # should try to remove this and replace with O(1)
			if count < j : 
				heapq_max.heappush_max(rankList, dist)
			else:
				if rankList[0] > dist: heapq_max.heapreplace_max(rankList, dist) # replaces biggest dist in list with smaller one, so if k neighbors, then k-j biggest outside list so j smallest
			count += 1

		if count is not 0:
			return max(rankList)
		else:
			return 0

if __name__ == "__main__":

	with open("processedTrainDocs_Trained.txt", "rb") as f:
		trainDocs = _pickle.load(f)
	with open("processedTestDocs.txt", "rb") as f:
		testDocs = _pickle.load(f)
	
	print(hyperplaneClassifier.__run1__(trainDocs, testDocs))