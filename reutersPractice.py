import heapq
from math import log
from math import sqrt
import multiprocessing
from nltk.corpus import reuters
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import _pickle
from re import compile
from re import VERBOSE
from time import time

class docNode:
	#should include a way to show which categories doc is in
	def __init__(self,document):
		self.doc_vec = document
		self.k_neighbors = []


class docsPreProcess:
	"""Takes processed Docs and creates vocabList, a 
	dictionary of the form {word : {docId : word frequency in doc}}.
	"""

	"""
	First problem, should initial termFreqDict and categoryDict be made with solely training docs (yes)?

	Isn't that the point? But then you'd have to concern yourself with terms in the test docs
	that arent in the training docs? Would you just ignore than?

	-----------------------------------------------------------------------------------------------

	To filter properly, first create termFreqDict, categoryDict, and termCategoryDict as usual.
	Filtering requires noise input, so first run must be with unfiltered docs.

	When you preprocess, you have to first filter termFreqDict within the docsProcess class. Then
	remake the termCategoryDict using remade termFreqDict.

	After this you can use doc_to_vec (after filtering) with proper tf-idf values so that values
	are properly normed, and all your training docs have the required, filtered form.

	---------------------------------------------------------------------------------------------------

	Now that training docs will be taken care of, how do we deal with test docs? They won't be filtered
	so we can just filter the termFreqDict and use those values to determine doc vecs, except have to 
	take care of words that are in test docs but not in training docs.

	If we add new words to the pre-existing dictionaries, will we have recalc all tf-idf everytime 
	a new doc is added (yes)? will that result in the need to retrim every doc as you go (yes)? 

	So we should just ignore new words and assume that the training docs are enough to give decent result.
	So ignore new words when converting test docs to vectors.


	"""

	@staticmethod
	def __vocabFreqProcess(trainDocs):
		#put multiprocessing here or multithreading?
		p = multiprocessing.pool()
		word_splitter = compile(r"\w+", VERBOSE)
		stemmer=PorterStemmer()#
		stop_words = set(stopwords.words('english'))
		
		termFreqDict = {}
		processedDocs = {}
		for docId in trainDocs:
			print(docId)
			processedDoc = [stemmer.stem(w.lower()) for w in word_splitter.findall(reuters.raw(docId)) if not w in stop_words]
			#print(processedDoc)
			processedDocs[docId] = processedDoc
			for w in processedDoc:
				if w not in termFreqDict:
					termFreqDict[w] = {docId : processedDoc.count(w)}
				else:
					termFreqDict[w][docId] = processedDoc.count(w)
		print("vocabFreqProcess completed")
		return (termFreqDict, processedDocs)

		#def helper(docId, stemmer, word_splitter, stop_words):
		#	processedDoc = [stemmer.stem(w.lower()) for w in word_splitter.findall(reuters.raw(docId)) if not w in stop_words]

			
	
	@staticmethod
	def __categoryProcess(categories): #reuters.categories()
		#only takes into account train docs
		#used for fast feature selection (to calc chi^2)
		categoryDict = {}
		for cat in categories:
			catDocs = [docId for docId in reuters.fileids(cat) if "train" in docId]
			categoryDict[cat] = set(catDocs)  
		print("categoryProcess completed")
		return categoryDict

	@staticmethod
	def __vocabCategoryProcess(termFreqDict, categoryDict):
		#used for fast feature selection (to calc chi^2)
		#will be in terms of training docs since both termfreq & categoryDict are
		termCategoryDict = {}
		for t in termFreqDict:
			for category, docs in categoryDict.items():
				catOverlap = len(docs.intersection(termFreqDict[t].keys()))
				if t not in termCategoryDict:
					if catOverlap > 0: termCategoryDict[t] = {category : catOverlap}
				else:
					if catOverlap > 0: termCategoryDict[t][category] = catOverlap #typo
		print("vocabCategoryProcess completed")
		return termCategoryDict

	@staticmethod
	def load(trainDocs, categories):
		#stores unfiltered dicts of interest
		t0 = time()
		(termFreqDict, processedTrainDocs) = docsPreProcess.__vocabFreqProcess(trainDocs)
		categoryDict = docsPreProcess.__categoryProcess(categories)
		termCategoryDict = docsPreProcess.__vocabCategoryProcess(termFreqDict, categoryDict)
		with open("processedTrainDocs_unfiltered.txt", "wb") as f:
			_pickle.dump(processedTrainDocs, f)
		with open("termFreqDict_unfiltered.txt", "wb") as f:
			_pickle.dump(termFreqDict, f)
		with open("categoryDict.txt", "wb") as f:
			_pickle.dump(categoryDict, f)
		with open("termCategoryDict_unfiltered.txt", "wb") as f:
			_pickle.dump(termCategoryDict, f)
		print("load method takes "+"{:.3}".format(time()-t0)+"seconds")

	def __init__(self, trainDocs, categories):
		self.NumDocs = len(trainDocs)
		with open("termFreqDict_unfiltered.txt", "rb") as f:
			self.termFreqDict = _pickle.load(f)
		with open("categoryDict.txt", "rb") as f:
			self.categoryDict = _pickle.load(f)
		with open("termCategoryDict_unfiltered.txt", "rb") as f:
			self.termCategoryDict = _pickle.load(f)


	def featureFilter(self, trainDocs):
		
		print("featureFiltering begun")

		termFreqDict = {}
		trainDocs = {} #return trainDocs too
		for docId, doc in trainDocs.items(): #doc should be processed
			#for some reason if you hold down the command prompt with cursor it breaks if 
			#going through this iteration with a "OSError: raw write() returned invalid length"
			#print(docId, len(doc))
			filteredDoc = self.featureFilterChi(docId, doc)
			#print(docId, len(filteredDoc))
			trainDocs[docId] = filteredDoc
			for w in filteredDoc:
				if w not in termFreqDict:
					termFreqDict[w] = {docId : filteredDoc.count(w)}
				else:
					termFreqDict[w][docId] = filteredDoc.count(w)
		termCategoryDict = docsPreProcess.__vocabCategoryProcess(termFreqDict, self.categoryDict)
		return (termFreqDict, termCategoryDict)



		print("reFilter complete")


	def featureFilterChi(self, docId, processedDoc, minLength = 15, k = 10):
		#should change k depending on length of doc.
		#should be used with multiprocessing or naw?

		"""For this multimodal feature selection we assume that
		categories are independent of eachother, so that a word with a 
		high chi^2 value in multiple categories can be assumed to be
		a good indicator of belonging to each individual category (presence of
		categories that are in general not indpt of eachother is ignored). k is
		the amount of terms you want to filter off, and minLength is the min
		length of the prcessed doc allowed to be filtered."""

		#also assumes that short docs are more descriptive, so shorter docs are unchanged (can adapt this though)
		
		#test
		filteredDoc = []
		if len(processedDoc) > minLength:
			for term in processedDoc:
				x = 0.0
				for category in reuters.categories(docId):
					x += self.chiSquared(term, category)
				if len(filteredDoc) < len(processedDoc)-k:
					heapq.heappush(filteredDoc, (x, term))
				else:
					if filteredDoc[0][0] < x: heapq.heapreplace(filteredDoc, (x, term))

		filteredDoc = [tup[1] for tup in filteredDoc] # this part might break
		return filteredDoc

	def chiSquared(self, t, c):
		#test

		#must be used on unfiltered data
		#chi^2 between a term being in a doc and the doc being in a certain category
		#O(1) 
		#check logic for nij values
		n00 = (self.NumDocs-len(self.termFreqDict[t]))-(len(self.categoryDict[c])-self.termCategoryDict[t][c]) 
		n01 = len(self.categoryDict[c]) - self.termCategoryDict[t][c]
		n10 = len(self.termFreqDict[t]) - self.termCategoryDict[t][c]
		n11 = self.termCategoryDict[t][c]
		chi_val = ((n00+n01+n10+n11)*((n11*n00-n10*n01)**2))/((n11+n01)*(n11+n10)*(n10+n00)*(n01+n00)) # horners algorithm?
		return chi_val

########################################################################################################
class docsProcess:
	"""We have filtered training docs, filtered termFreqDict, and filtered termCategoryDict.
	So now we need to process (a.k.a nearest neighbor fixing) -- make docs process solely about
	kNN for the filtered train docs, and about the test docs"""


	#need to add how categories are kept in doc_containers
	#need to add refiltering and remaking of categoryDict and termFreqDict


	def __init__(self, documents):
		with open("termFreqDict.txt", "rb") as f:
			self.termFreqDict = _pickle.load(f) # need to filter beforehand
		with open("categoryDict.txt", "rb") as f:#
			self.categoryDict = _pickle.load(f)#
		with open("termCategoryDict.txt", "rb") as f:#
			self.termCategoryDict = _pickle.load(f)#
		self.NumDocs = len(documents)
		self.trainDocs = {}
		self.testDocs = {}
		#reFilter(termFreqDict)
		#remake(termCategoryDict)

		#self.preProcess(documents)

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
		#test
		#for use on doc vecs
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
		return float(docsProcess.dot(q,d))#/(docsProcess.norm(q)*docsProcess.norm(d))) docs are normalized


	def doc_to_vec(self, docId, document):
		#returns a normed tf-idf weighted doc: note that different forms of tf-idf exist for taking car eof things such as varying doc length (feature selection kNN doc)
		d = {}
		for w in document:
			if w in self.termFreqDict:
				idf = log(float(self.NumDocs)/len(self.termFreqDict[w])) 
				if w not in d: d[w] = self.termFreqDict[w][docId]*idf 
		norm = docsProcess.norm(d)
		for k, v in d.items(): # figure out how to use itertools or something here @AM
			d[k] = v/norm
		return d 

#####################################################################################################
	#see if you can make this into a multithreaded fashion
	#need to fix to take into account that our train docs are already processed
	def preProcess(self, documents): #as is the additon of update neighbors makes this n^3 so we have to use feature selection in the future?
		word_splitter = compile(r"\w+", VERBOSE)
		stemmer=PorterStemmer()#
		stop_words = set(stopwords.words('english'))

		for docId in documents:
			print(docId)
			processedDoc = [stemmer.stem(w.lower()) for w in word_splitter.findall(reuters.raw(docId)) if not w in stop_words]			
			if "train" in docId:
				doc_container = docNode(self.doc_to_vec(docId, featureFilterChi(processedDoc))) # insert with filtered doc
				if self.trainDocs:
					doc_container.k_neighbors = self.updateNeighbors(docId, doc_container) #Test this @AM
					self.trainDocs[docId] = doc_container
				else:
					self.trainDocs[docId] = doc_container
			else:
				doc_container = docNode(self.doc_to_vec(docId, processedDoc, self.NumDocs))
				self.testDocs[docId] = doc_container

		print("preProcess completed")
		with open("preProcessed_trainDocs.txt", "wb") as f:
			_pickle.dump(self.trainDocs, f)
		with open("preProcessed_testDocs.txt", "wb") as f:
			_pickle.dump(self.testDocs, f)

#need to see if filtering works
#################################################################################################
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
	docs = reuters.fileids()
	docs = [ids for ids in docs if 'training' in ids]
	categories = reuters.categories()
	obj = docsPreProcess.load(docs, categories)
	#vocabListings.vocabFreqProcess(documents)
	#vocabListings.categoryProcess(categories)
	#with open("termFreqDict.txt", "rb") as f:
	#		termFreqDict = _pickle.load(f)
	#with open("categoryDict.txt", "rb") as f:
	#		categoryDict = _pickle.load(f)
	#vocabListings.vocabCategoryProcess(termFreqDict, categoryDict)

	
	
	