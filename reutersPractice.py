import re
from nltk.corpus import reuters
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import heapq

class docNode:
	#should include a way to show which categories doc is in
	def __init__(self,document):
		self.doc_vec = document
		self.k_neighbors = [] # should be a minheap with max k entries


class vocabListings:
	"""Takes processed Docs and creates vocabList, a 
	dictionary of the form {word : {docId : word frequency in doc}}.
	"""

	def __init__(self, processedDocs):
		self.vocab = vocabListings.vocabulary(processedDocs)
		self.vocabDict = vocabListings.vocab_Dict(self.vocab, processedDocs)

	@staticmethod
	def vocabulary(processedDocs):
		vocab = {}
		for k, v in processedDocs:
			vocab.union(set(v))
		return vocab

	@staticmethod
	def vocab_Dict(vocabulary, processedDocs):
		vocabDict = {}
		for docId, doc in processedDocs :
			for w in doc:
				if w not in vocabDict: 
					vocabDict[w] = {docId : doc.count(w)}
				else:
					vocabulary[w][docId] = doc.count(w)
		return vocabDict

################################################################################################################################
class docsProcess:
	"""Instead of doing a category dict, just create a 
	dict for each node that has 1 if it has that category,
	and has a 0 if it isnt."""

	def __init__(self, documents):
		self.vocabulary = vocabListings(docsProcess.preProcess(documents)).vocabDict
		self.trainDocs = {}
		self.testDocs = {}
		preProcess(documents)

	@staticmethod
	def norm(d):
		sum_sq = 0.0
		for k in d:
			sum_sq += d[k] * d[k]
		return math.sqrt(sum_sq)

	@staticmethod
	def dot(q, d):
		s=0.0
		for k in q:  
			if k in d: 
				s += q[k] * d[k]
		return s

	def doc_to_vec(self, docId, document):
		#returns a normed tf-idf weighted doc
		d = {}
		for w in document:
			idf = 1.0/(len(self.vocabulary[w])+1) #should ideally use N instead of 1/""
			if w not in d: d[w] = self.vocabulary[w][docId]*idf 
		norm = docsProcess.norm(d)
		for k, v in d:
			d[k] = v/norm
		return d 

	def preProcess(self, documents): #as is the additon of update neighbors makes this n^3 so we have to use feature selection in the future?
		word_splitter = re.compile(r"\w+", re.VERBOSE)
		stemmer=PorterStemmer()#
		stop_words = set(stopwords.words('english'))

		for docId in documents:
			processedDoc = [stemmer.stem(w.lower()) for w in word_splitter.findall(reuters.raw(docId)) if not w in stop_words]			
			doc_container = docNode(doc_to_vec(docId, processedDoc))
			if "train" in docId:
				if self.trainDocs:
					doc_container.k_neighbors = updateNeighbors(docId,doc_container) #Test this @AM
					self.trainDocs[docId] = doc_container
				else:
					self.trainDocs[docId] = doc_container
			else:
				self.testDocs[docId] = doc_container

	def updateNeighbors(self,docId,docNode,k=5): #check if it simultaneously updates dict and returns desired value
		currNode_neighbors = []
		for node in self.trainDocs:
			dist = docsProcess.dot(docNode.doc_vec,node.doc_vec) #this is what makes this n^2, should instead see if feature selection will make this faster
			if len(node.k_neighbors) < k:
				heapq.heappush(currNode_neighbors, (dist, docId)) 
				heapq.heappush(node.k_neighbors, (dist, docId))
			else:
				if node.k_neighbors[0][0] < dist: 
					heapq.heapreplace(currNode_neighbors, (dist, docId)) 
					heapq.heapreplace(node.k_neighbors, (dist, docId))
		return currNode_neighbors


if __name__ == "__main__":
	documents = reuters.fileids()
	
	
	