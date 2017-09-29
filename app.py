from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import NMF
import numpy as np
import re

import sys
from Tkinter import *

def prepareData(fileName, categories):
	# Prepare dataset for training or testing

	dataset = {'data': [], 'target': [], 'target_names': categories}
	with open(fileName) as file:
	  	dataset['data'] = file.readlines()
	dataset['data'] = [x.strip() for x in dataset['data']]

	numberOfSongs = len(dataset['data']) / len(categories)
	for i in xrange(len(categories)):
		for j in xrange(numberOfSongs):
			dataset['target'].append(i)

	return dataset

def train(categories):
	# Process train data with tfidf
	# Apply lsa to the tfidf model
	# Train SVM with the model
	# Return lsa model, count vector, tfidf transformer, SVM prediction model

	dataset = prepareData("dataset.txt", categories)

	countVector = CountVectorizer()
	XTrainCounts = countVector.fit_transform(dataset['data'])
	tfidfTransformer = TfidfTransformer()
	XTrainTfidf = tfidfTransformer.fit_transform(XTrainCounts)

	lsa = NMF(n_components=40, random_state=90)
	XTrainTfidf = lsa.fit_transform(XTrainTfidf)
	XTrainTfidf = Normalizer(copy=False).fit_transform(XTrainTfidf)

	return lsa, countVector, tfidfTransformer, SVC().fit(XTrainTfidf, dataset['target'])

def testAccuracy(categories, lsa, countVector, tfidfTransformer, svm):
	# Process test data with tfidf
	# Apply lsa to the tfidf model
	# Return accuracy of test set using trained SVM model

	testSet = prepareData("testset.txt", categories)
	documents = testSet['data']
	XTestCounts = countVector.transform(documents)
	XTestTfIdf = tfidfTransformer.transform(XTestCounts)

	XTestTfIdf = lsa.transform(XTestTfIdf)
	XTestTfIdf = Normalizer(copy=False).transform(XTestTfIdf)

	prediction = svm.predict(XTestTfIdf)
	accuracy = np.mean(prediction == testSet['target'])
	prediction = ", ".join(map(str, prediction))

	print(testSet['target'])
	print("[" + prediction + "]")
	return accuracy

def runApplication(categories, lsa, countVector, tfidfTransformer, svm):
	# Create UI with tkinter
	# Use trained SVM model to predict new lyrics

	window = Tk()
	window.geometry("600x400+0+0")
	window.title("Song Genre Classifier")
	output = StringVar()
	newInput = StringVar()

	def parse(lyrics):
		lyrics = re.sub('[^a-zA-Z\n]', ' ', lyrics)
		lyrics = lyrics.lower()
		return lyrics

	def solve():
		lyrics = newInput.get()
		documents = [parse(lyrics)]
		XNewCounts = countVector.transform(documents)
		XNewTfidf = tfidfTransformer.transform(XNewCounts)

		XNewTfidf = lsa.transform(XNewTfidf)
		XNewTfidf = Normalizer(copy=False).transform(XNewTfidf)

		prediction = svm.predict(XNewTfidf)
		output.set("Genre: " + categories[prediction[0]])

	def callback(event):
	    window.after(50, select_all, event.widget)

	def select_all(widget):
	    widget.select_range(0, 'end')
	    widget.icursor('end')

	newline = Label(window, text="\n\n").pack()
	title = Label(window, text="Input Lyrics", font=("Helvetica", 20)).pack()
	newline = Label(window, text="").pack()

	entry = Entry(window, textvariable=newInput, width=50, font=("Helvetica", 20))
	entry.focus()
	entry.bind('<Control-a>', callback)
	entry.pack()
	newline = Label(window, text="\n\n").pack()

	button1 = Button(window, text="Submit", command=solve, fg="black", bg="green", font=("Helvetica", 16)).pack()
	newline = Label(window, text="\n").pack()
	outputLabel = Label(window, textvariable=output, font=("Helvetica", 20)).pack()

	window.mainloop()

def main():
	categories = ['pop', 'dangdut', 'hiphop', 'rock']

	lsa, countVector, tfidfTransformer, svm = train(categories)

	accuracy = testAccuracy(categories, lsa, countVector, tfidfTransformer, svm)
	print('Accuracy: ' + str(accuracy*100) + '%')

	runApplication(categories, lsa, countVector, tfidfTransformer, svm)

main()