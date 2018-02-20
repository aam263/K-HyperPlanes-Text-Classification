"""
These are the containers that will contain the document vectors along with the
k nearest neighbors. Used for training docs.
"""

class docNode:

	def __init__(self, document):
		self.doc_vec = document
		self.k_neighbors = []