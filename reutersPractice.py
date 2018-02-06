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
		self.k_neighbors = []


class vocabListings:
	"""Takes processed Docs and creates vocabList, a 
	dictionary of the form {word : {docId : word frequency in doc}}.
	"""

	#so far everything done is including all docs, but this might be bad since in general we wont know all the data (using solely train docs vs train and test)
	@staticmethod
	def vocabFreqProcess(documents): #check if you should use only training docs for this? @AM
		word_splitter = re.compile(r"\w+", re.VERBOSE)
		stemmer=PorterStemmer()#
		stop_words = set(stopwords.words('english'))
		
		termFreqDict = {}
		#for each term
		for docId in documents:
			processedDoc = [stemmer.stem(w.lower()) for w in word_splitter.findall(reuters.raw(docId)) if not w in stop_words]
			#print(processedDoc)
			for w in processedDoc:
				#take the categories belonging to the docId an put them into the dict for each word
				if w not in freqDict:
					termFreqDict[w] = {docId : processedDoc.count(w)}
				else:
					termFreqDict[w][docId] = processedDoc.count(w)
		print("vocabProcess completed")
		with open("vocabListings.txt", "wb") as f:
			_pickle.dump(termFreqDict, f)
	
	@staticmethod
	def categoryDict(categories): #reuters.categories()
		#test
		#used for fast feature selection (to calc chi^2)
		categoryDict = {}
		for cat in categories:
			categoryDict[cat] = set(reuters.categories(cat))
		return categoryDict

	@staticmethod
	def vocabCategoryProcess(termFreqDict, categoryDict):
		#test
		#used for fast feature selection (to calc chi^2)
		termCategoryDict = {}
		for t in termFreqDict:
			for cat, docs in categoryDict.items():
				catOverlap = len(docs.intersect(termFreqDict[t]))
				if catOverlap > 0: termCategoryDict[t] = {cat : catOverlap}
		return termCategoryDict


class docsProcess:
	"""Instead of doing a category dict, just create a 
	dict for each node that has 1 if it has that category,
	and has a 0 if it isnt."""

	def __init__(self, documents):
		with open("vocabListings.txt", "rb") as f:
			self.vocabulary = _pickle.load(f)
		self.NumDocs = len(documents)
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

	@staticmethod
	def euclideanDist(q, d):
		#test
		s = 0.0
		for k in q:
			if k in d: 
				s += (q[k]-d[k])**2
			else:
				s+= q[k]**2
		for k in d:
			if k not in q: s += d[k]**2
		return sqrt(s)

	@staticmethod
	def cosineMeasure(q, d):
		return float(docsProcess.dot(q,d)/(docsProcess.norm(q)*docsProcess.norm(d)))

	def doc_to_vec(self, docId, document, N):
		#returns a normed tf-idf weighted doc: note that different forms of tf-idf exist for taking car eof things such as varying doc length (feature selection kNN doc)
		d = {}
		for w in document:
			idf = log(float(N)/len(self.vocabulary[w])) 
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


	def updateNeighbors(self,currId,currNode,k=5): #check if it simultaneously updates dict and returns desired value
		currNode_neighbors = []
		for docId, docNode in self.trainDocs.items():
			dist = docsProcess.euclideanDist(currNode.doc_vec,docNode.doc_vec) #this is what makes this n^2, should instead see if feature selection will make this faster
			#dist should be euclidean distance for finding the kNN
			#use cos-sim for category weighting


			#print(currId, docId, dist)
			if len(currNode_neighbors) < k: 
				heapq.heappush(currNode_neighbors, (dist, docId))
			else:
				if currNode_neighbors[0][0] < dist: heapq.heapreplace(currNode_neighbors, (dist, docId)) 
			if len(docNode.k_neighbors) < k: 
				heapq.heappush(docNode.k_neighbors, (dist, currId))	
			else: 
				if docNode.k_neighbors[0][0] < dist: heapq.heapreplace(docNode.k_neighbors, (dist, currId))
		return currNode_neighbors



#to each document assign it a set of categories
#to each document assign each of its nearest neighbors with (w,b), which is what defines the hyperplane between two points
#find how to find the distance between a point and a plane


#so k-nearest neighbors works, now need to store which categories each docId has?
#also need to indicate on which side of each hyperplane that defines the brilloiun zone its on?
if __name__ == "__main__":
	#documents = reuters.fileids()[:10]
	#for docId in documents:
	#	print(reuters.raw(docId))
	#with open("preProcessed_trainDocs.txt", "rb") as f:
	#		d = _pickle.load(f)
	#print(reuters.categories())
	#count = 0
	#for docId, doc_container in d.items():
	#	if count < 10: 
	#		print(len(doc_container.doc_vec))
	#		print(doc_container.doc_vec)
	#		print("\n")
	#	count+=1
	#vocabListings.vocabProcess(documents)
	#obj = docsProcess(documents)
	
	
	