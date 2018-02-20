import multiprocessing
from nltk.corpus import reuters
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import _pickle
from re import compile
from re import VERBOSE
from time import time

"""
This module creates the unfiltered termFreqDict, categoryDict, and unfiltered termCategoryDict. It also turns the 
training docs into their unfiltered, canonical versions for later filtering.


termFreqDict is of form {terms : {docIds : freq}}, where for a term in the vocab, we have the docIds of the documents it
	is in, in addition to the number of times it occurs in that specific document. This is used for fast tf-idf calculations
	in order to reduce the time complexity of the doc_to_vec conversion. This is made considering only the training documents.

categoryDict is of the form {categories : {docIds}}, where the keys are all the categories of the docs, and the values are the
	set of documents that have that category. This is used for fast computation of the chi^2 values used in the feature selection
	process in the docsPreProcess module. This is made only considering the training documents.

termCategoryDict is of the form {terms : {categories : freq}}, where for a term in the vocab, we have the categories of the 
	documents in which the term is found in addition to the number of documents containing the term with the specified category.
	This is used for fast computation of the chi^2 values used in the feature selection process in the docsPreProcess module. This 
	is made only considering the training docs.

Things to keep in mind: 
		
		1.) termFreqDict and termCategoryDict are created using the unfiltered docs, meaning that after the docs
			feature selection, that these have to be filtered as well, which is done in the featureSelection module.

		2.) everything is made in terms of training docs because we assume we have no knowledge about the test docs.

"""



word_splitter = compile(r"\w+", VERBOSE)
stemmer=PorterStemmer()#
stop_words = set(stopwords.words('english'))
	
def vocabFreqProcess(trainDocs):
	#must have a threadpool p initialized
	#bottleneck
	termFreqDict = {}
	processedDocs = {}
	for docId, processedDoc in p.imap_unordered(vfpHelper, trainDocs):
		processedDocs[docId] = processedDoc
		for w in processedDoc:
			if w not in termFreqDict:
				termFreqDict[w] = {docId : processedDoc.count(w)}
			else:
				termFreqDict[w][docId] = processedDoc.count(w)
	print("vocabFreqProcess completed")
	return (termFreqDict, processedDocs)

def vfpHelper(docId):
	processedDoc = [stemmer.stem(w.lower()) for w in word_splitter.findall(reuters.raw(docId)) if not w in stop_words]
	return (docId, processedDoc)
			
	
	
def categoryProcess(categories):
	#used for fast feature selection (to calc chi^2)
	categoryDict = {}
	for cat in categories:
		catDocs = [docId for docId in reuters.fileids(cat) if "train" in docId]
		categoryDict[cat] = set(catDocs)  
	print("categoryProcess completed")
	return categoryDict

def vocabCategoryProcess(termFreqDict, categoryDict):
	#used for fast feature selection (to calc chi^2)
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

def load(trainDocs, categories):
	#stores unfiltered dicts of interest
	t0 = time()
	(termFreqDict, processedTrainDocs) = vocabFreqProcess(trainDocs)
	categoryDict = categoryProcess(categories)
	termCategoryDict = vocabCategoryProcess(termFreqDict, categoryDict)
	with open("processedTrainDocs_unfiltered.txt", "wb") as f:
		_pickle.dump(processedTrainDocs, f)
	with open("termFreqDict_unfiltered.txt", "wb") as f:
		_pickle.dump(termFreqDict, f)
	with open("categoryDict.txt", "wb") as f:
		_pickle.dump(categoryDict, f)
	with open("termCategoryDict_unfiltered.txt", "wb") as f:
		_pickle.dump(termCategoryDict, f)
	print("load method takes "+"{:.3}".format(time()-t0)+"seconds") # takes around 34 secs

if __name__ == "__main__":
	pass
