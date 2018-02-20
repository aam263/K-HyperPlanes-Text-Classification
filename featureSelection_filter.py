import heapq
from nltk.corpus import reuters
import _pickle
from time import time


"""
Using the unfiltered results of the unfilteredDocs_load module, we filter the termFreqDict and termCategoryDict
while also filtering the processed (i.e. stemmed) training vectors. TermFreqDict, termCategoryDict, and categoryDict are used for fast 
chi^2 evaluation.



Things to keep in mind:

		1.) Chi^2 filtering is biased towards rare terms, so to avoid creating a correspondence between rare terms
		    in a doc and the doc being in a certain category, it would probably help to filter by document frequency
		    or something similar.
		
		2.) We are doing multi-category feature selection, and as such have to assume that categories are independent of 
			eachother, so that a word with a high chi^2 value in multiple categories can be assumed to be a good indicator 
			of belonging to each individual category (presence of categories that are in general not indpt of eachother is 
			ignored). 

Questions:

		1.) For single category classification we assume the contiguity hypothesis, which states that a vector space is separable
			into disjoint, contiguous regions (closely bordering eachother) where each disjoint region corresponds to being in a certain 
			category. For practical purposes we have to deal with multimodal categories and multiple categories. Does dealing with multiple
			categories in an any-of scheme (a doc can have multiple classifiers) contradict the contiguity hypothesis? 

		2.) I read in the stanford nlp book that you can create nonseparable spaces by embedding lower dimensional nonseparable spaces
			into higher dimensions. An exercise in the book shows that as the dimensionality of your space increases, the probability
			of having nonseparable spaces decreases. If the contiguity hypothesis doesn't necessarily hold for multiple classes in an
			any-of scheme, does the fact that documents are high dimensional allow us to follow through with categorization techniques 
			used for the one-of schemes?

		Tags: contiguity hypothesis, vector space separability, linear separability.
"""



class featureSelection:


	def __init__(self, trainDocs, categories):
		t0 = time()
		self.NumDocs = len(trainDocs)
		with open("termFreqDict_unfiltered.txt", "rb") as f:
			self.termFreqDict = _pickle.load(f)
		with open("categoryDict.txt", "rb") as f:
			self.categoryDict = _pickle.load(f)
		with open("termCategoryDict_unfiltered.txt", "rb") as f:
			self.termCategoryDict = _pickle.load(f)
		with open("processedTrainDocs_unfiltered.txt", "rb") as f:
			self.trainDocs = _pickle.load(f)

		(self.trainDocs, self.termFreqDict, self.termCategoryDict) = self.featureFilter(self.trainDocs)
		
		with open("processedTrainDocs_filtered.txt", "wb") as f:
			_pickle.dump(self.trainDocs, f)
		with open("termFreqDict_filtered.txt", "wb") as f:
			_pickle.dump(self.termFreqDict, f)
		with open("termCategoryDict_filtered.txt", "wb") as f:
			_pickle.dump(self.termCategoryDict, f)

		print("startup w/ filteringtakes "+"{:.3}".format(time()-t0))
		#takes like 20 secs


	def featureFilter(self, trainDocs):
		#uses processed, unfiltered docs
		print("feature filtering begun")

		termFreqDict = {}
		train_Docs = {} 
		for docId, doc in trainDocs.items(): 
			#for some reason if you hold down the command prompt with cursor it breaks if 
			#going through this iteration with a "OSError: raw write() returned invalid length"
			filteredDoc = self.featureFilterChi(docId, doc)
			train_Docs[docId] = filteredDoc
			for w in filteredDoc:
				if w not in termFreqDict:
					termFreqDict[w] = {docId : filteredDoc.count(w)}
				else:
					termFreqDict[w][docId] = filteredDoc.count(w)
		termCategoryDict = vocabCategoryProcess(termFreqDict, self.categoryDict)
		print("feature filter complete")
		return (train_Docs, termFreqDict, termCategoryDict)


	def featureFilterChi(self, docId, processedDoc, minLength = 15, k = 10):
		#should change k depending on length of doc.
		#should be used with multiprocessing or naw?

		"""k is the amount of terms you want to filter off, 
		and minLength is the min length of the prcessed doc 
		allowed to be filtered."""

		#assumes short docs are more descriptive, so shorter docs unchanged (can adapt this though)
		
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
			filteredDoc = [tup[1] for tup in filteredDoc] 
			return filteredDoc
		else:
			return processedDoc

	def chiSquared(self, t, c):
		#must be used on unfiltered data
		#chi^2 between a term being in a doc and the doc being in a certain category
		#could improve accuracy using horners algorithm

		#test
		#check logic for nij values
		n00 = (self.NumDocs-len(self.termFreqDict[t]))-(len(self.categoryDict[c])-self.termCategoryDict[t][c]) 
		n01 = len(self.categoryDict[c]) - self.termCategoryDict[t][c]
		n10 = len(self.termFreqDict[t]) - self.termCategoryDict[t][c]
		n11 = self.termCategoryDict[t][c]
		chi_val = ((n00+n01+n10+n11)*((n11*n00-n10*n01)**2))/((n11+n01)*(n11+n10)*(n10+n00)*(n01+n00)) # horners algorithm?
		return chi_val

if __name__ == "__main__":
	pass