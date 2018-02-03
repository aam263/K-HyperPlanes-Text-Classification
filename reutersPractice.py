import re
from nltk.corpus import reuters
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import heapq
import _pickle
from math import log
from math import sqrt

class docNode:
	#should include a way to show which categories doc is in
	def __init__(self,document):
		self.doc_vec = document
		self.k_neighbors = [] # should be a minheap with max k entries


class vocabListings:
	"""Takes processed Docs and creates vocabList, a 
	dictionary of the form {word : {docId : word frequency in doc}}.
	"""

	@staticmethod
	def vocabProcess(documents):
		word_splitter = re.compile(r"\w+", re.VERBOSE)
		stemmer=PorterStemmer()#
		stop_words = set(stopwords.words('english'))
		
		wordDict = {}
		for docId in documents:
			#doc ids are in unicode
			processedDoc = [stemmer.stem(w.lower()) for w in word_splitter.findall(reuters.raw(docId)) if not w in stop_words]
			print(processedDoc)
			for w in processedDoc:
				if w not in wordDict:
					wordDict[w] = {docId : processedDoc.count(w)}
				else:
					wordDict[w][docId] = processedDoc.count(w)
		with open("vocabListings.txt", "wb") as f:
			_pickle.dump(wordDict, f)
	

################################################################################################################################
class docsProcess:
	"""Instead of doing a category dict, just create a 
	dict for each node that has 1 if it has that category,
	and has a 0 if it isnt."""



	def __init__(self, documents):
		with open("vocabListings.txt", "rb") as f:
			self.vocabulary = _pickle.load(f)
		self.trainDocs = {}
		self.testDocs = {}
		self.preProcess(documents)

	@staticmethod
	def norm(d):
		sum_sq = 0.0
		for k in d:
			sum_sq += d[k] * d[k]
		return sqrt(sum_sq)

	@staticmethod
	def dot(q, d):
		s=0.0
		for k in q: 
			if k in d: 
				s += q[k] * d[k]
		return s


	def doc_to_vec(self, docId, document, N):
		#returns a normed tf-idf weighted doc
		d = {}
		for w in document:
			idf = log(float(N)/(len(self.vocabulary[w])+1)) 
			if w not in d: d[w] = self.vocabulary[w][docId]*idf 
		norm = docsProcess.norm(d)
		for k, v in d.items(): # figure out how to use itertools or something here @AM
			d[k] = v/norm
		return d 

	#see if you can make this into a multithreaded fashion
	def preProcess(self, documents): #as is the additon of update neighbors makes this n^3 so we have to use feature selection in the future?
		word_splitter = re.compile(r"\w+", re.VERBOSE)
		stemmer=PorterStemmer()#
		stop_words = set(stopwords.words('english'))
		N = len(documents)
		for docId in documents:
			print(docId)
			processedDoc = [stemmer.stem(w.lower()) for w in word_splitter.findall(reuters.raw(docId)) if not w in stop_words]			
			doc_container = docNode(self.doc_to_vec(docId, processedDoc, N))
			if "train" in docId:
				if self.trainDocs:
					doc_container.k_neighbors = self.updateNeighbors(docId,doc_container) #Test this @AM
					self.trainDocs[docId] = doc_container
				else:
					self.trainDocs[docId] = doc_container
			else:
				self.testDocs[docId] = doc_container
		print("preProcess completed")
		with open("preProcessed_trainDocs.txt", "wb") as f:
			_pickle.dump(self.trainDocs, f)
		with open("preProcessed_testDocs.txt", "wb") as f:
			_pickle.dump(self.testDocs, f)


	def updateNeighbors(self,docId,currNode,k=5): #check if it simultaneously updates dict and returns desired value
		currNode_neighbors = []
		for docId, docNode in self.trainDocs.items():
			dist = docsProcess.dot(currNode.doc_vec,docNode.doc_vec) #this is what makes this n^2, should instead see if feature selection will make this faster
			if len(currNode_neighbors) < k: 
				heapq.heappush(currNode_neighbors, (dist, docId))
			else:
				if currNode_neighbors[0][0] < dist: heapq.heapreplace(currNode_neighbors, (dist, docId)) 
			if len(docNode.k_neighbors) < k: 
				heapq.heappush(docNode.k_neighbors, (dist, docId))	
			else: 
				if docNode.k_neighbors[0][0] < dist: heapq.heapreplace(docNode.k_neighbors, (dist, docId))
		return currNode_neighbors


if __name__ == "__main__":
	documents = reuters.fileids()
	test = docsProcess(documents)
	
	
	
	