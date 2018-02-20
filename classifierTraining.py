from docNode import docNode
import heapq_max
from math import log
from math import sqrt
from nltk.corpus import reuters
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import _pickle
from re import VERBOSE
from re import compile
from time import time


"""
Uses the filtered training docs, termFreqDict, and termCategoryDict created in the featureSelection_filter module,
and takes the filtered & processed training docs, and puts them into a docNode while updating its k nearest
neighbors as it goes. 

Initially stores the docIds of the k nearest neighbors of a document vector in a list within the docNodes, but 
after the hyperplanify() method takes the trained set and assigns to the k_neighbors attribute of the docNodes
a dict of the form {docId : (w, b)}, where the current node has a bisecting hyperplane between its nearest neighbor
(specified with the docId) defined by equation w.x + b = 0. (w is a normal vector of the bisecting plane; b is the 
intersect).

This module also processes the test docs (since we need the filtered dicts of interest to correctly process them).

Things to take into consideration:

	1.) We have to take care of words that are in the test docs but not in the train docs.
	
	2.) If we add new words to the pre-existing dictionaries, we'd have recalculate all tf-idf 
		values everytime since the idf values of all documents could be changed, and in addition 
		we'd have to modify termFreqDict, categoryDict, and termCategoryDict as well.

Conclusion:

	1.) We should just ignore new words and assume that the training docs are enough to give decent result.
		So ignore new words when converting test docs to vectors.

"""

#the below three lines only relevant to operations that would be run in this module
word_splitter = compile(r"\w+", VERBOSE)
stemmer=PorterStemmer()
stop_words = set(stopwords.words('english'))


class classifierTraining:


	def __init__(self, trainDocs, testDocs):
		t0 = time()

		self.NumDocs = len(trainDocs) 
		with open("termFreqDict_filtered.txt", "rb") as f:
			self.termFreqDict = _pickle.load(f) 
		with open("categoryDict.txt", "rb") as f:#
			self.categoryDict = _pickle.load(f)#
		with open("termCategoryDict_filtered.txt", "rb") as f:#
			self.termCategoryDict = _pickle.load(f)#

		with open("processedTrainDocs_filtered.txt", "rb") as f:
			self.trainDocs = self.train(_pickle.load(f))
		with open("processedTrainDocs_Trained.txt", "wb") as f:
			_pickle.dump(self.trainDocs, f)

		with open('processedTestDocs.txt', "wb") as f:
			_pickle.dump(self.processTestDocs(testDocs), f)

		print("classifier training takes "+"{:.3}".format(time()-t0))
		#training takes about 2 hours.

	@staticmethod
	def norm(d):
		#for use on doc vecs
		sum_sq = 0.0
		for k in d:
			sum_sq += d[k] * d[k]
		return sqrt(sum_sq)

	@staticmethod
	def dot(q, d):
		#for use on doc vecs
		s=0.0
		for k in q: 
			if k in d: 
				s += q[k] * d[k]
		return s

	@staticmethod
	def euclideanDist(q, d):
		#for use on doc vecs
		s = 0.0
		for k in q:
			if k in d: 
				s += (q[k]-d[k])**2
			else:
				s+= q[k]**2
		for k in d:
			if k not in q: 
				s += d[k]**2
		return sqrt(s)

	@staticmethod
	def hyperplanify(trainDocs):
		"""
		Takes trainDocs, and for each of these, assigns to 
		the k_neighbors attribute a dict of the form {neighborId : (w, b)},
		where w is the normal of the plane bisecting the docNode point and 
		its neighbors (orientation from docNode to neighbors).
		"""
	
		#O((n^2)*k)
		t0 = time()
		for docId, docNode in trainDocs.items():
			temp = {}
			for dist, nId in docNode.k_neighbors:
				temp[nId] = classifierTraining.createHyperplane(docNode.doc_vec, trainDocs[nId].doc_vec)
			docNode.k_neighbors = temp 
		print("hyperplanify completed in "+"{:.3}".format(time()-t0))
		#takes aprox 3 secs
		return trainDocs

	@staticmethod
	def createHyperplane(v1, v2):
		"""
		Returns (w, b), which indicate the hyperplane bisecting v1 and v2 
		with a normal pointing from v1 to v2 (v1 should be docNode and v2 
		one of its nearest neighbors).
		"""

		#optimize
		#O(n)
		w = {}
		for k in v2:
			if k in v1:
				w[k] = v2[k]-v1[k]
			else:
				w[k] = v2[k]
		for k in v1:
			if k not in v2:
				w[k] = -v1[k]
		b = (classifierTraining.dot(v2, v2)-classifierTraining.dot(v1, v1))/2.0 
		return (w, b)

	def doc_to_vec(self, docId, document):
		"""returns a normed tf-idf weighted doc 
		note that different forms of tf-idf exist for 
		taking care of things such as varying 
		doc length (feature selection kNN doc)
		"""
		d = {}
		for w in document:
			if w in self.termFreqDict:
				idf = log(float(self.NumDocs)/len(self.termFreqDict[w])) 
				#below deals with non training vecs
				if docId in self.termFreqDict[w]: 
					if w not in d: d[w] = self.termFreqDict[w][docId]*idf 
				else:
					if w not in d: d[w] = document.count(w)*idf

		norm = classifierTraining.norm(d)
		for k, v in d.items(): # figure out how to use itertools or something here @AM
			d[k] = v/norm
		return d 


	def processTestDocs(self, testDocs):
		t0 = time()
		testVecs = {}
		for docId in testDocs:
			processedDoc = [stemmer.stem(w.lower()) for w in word_splitter.findall(reuters.raw(docId)) if not w in stop_words]
			doc_vec = self.doc_to_vec(docId, processedDoc)
			testVecs[docId] = doc_vec
		print("processTestDocs completed in "+"{:.3}".format(time()-t0))
		#takes aroun 44 secs
		return testVecs


	def train(self, trainDocs):
		"""
		Returns trainDocs, which is a dict of
		the form {docId : docNode}
		"""
		#this is bottleneck now
		#test

		train_Docs = {}
		for docId, processedDoc in trainDocs.items():
			#print(docId)		
			doc_container = docNode(self.doc_to_vec(docId, processedDoc)) # insert with filtered doc
			if train_Docs: 
				t0 = time()
				doc_container.k_neighbors = self.updateNeighbors(docId, doc_container, train_Docs) #Test this @AM
				print("updateNeighbors takes "+"{:.3}".format(time()-t0))
				train_Docs[docId] = doc_container
			else:
				train_Docs[docId] = doc_container
		#takes approx 2 hours to run lol
		return train_Docs

	def updateNeighbors(self, currId, currNode, trainDocs, k=8):
		"""
		Takes the trained docs and updates the nearest neighbors
		within the train() method. k is the number of neighbors 
		you want to consider for the classification.
		"""
		
		#this is part of bottleneck

		currNode_neighbors = []
		for docId, docNode in trainDocs.items():
			dist = classifierTraining.euclideanDist(currNode.doc_vec,docNode.doc_vec) #this is what makes this n^2
			if len(currNode_neighbors) < k: 
				heapq_max.heappush_max(currNode_neighbors, (dist, docId))
			else:
				if currNode_neighbors[0][0] > dist: heapq_max.heapreplace_max(currNode_neighbors, (dist, docId)) 
			if len(docNode.k_neighbors) < k: 
				heapq_max.heappush_max(docNode.k_neighbors, (dist, currId))	
			else: 
				if docNode.k_neighbors[0][0] > dist: heapq_max.heapreplace_max(docNode.k_neighbors, (dist, currId))
		return currNode_neighbors


if __name__ == "__main__":
	pass