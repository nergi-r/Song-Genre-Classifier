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

use_lsa = True
testAccuracy = True

categories = ['pop', 'dangdut', 'hiphop', 'rock']

dataset = {'data': [], 'target': [], 'target_names': categories}
with open("dataset.txt") as f:
  dataset['data'] = f.readlines()
dataset['data'] = [x.strip() for x in dataset['data']]

n_songs = len(dataset['data']) / len(categories)
for i in xrange(len(categories)):
	for j in xrange(n_songs):
		dataset['target'].append(i)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(dataset['data'])
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

if(use_lsa):
	lsa = NMF(n_components=40, random_state=90)
	X_train_tfidf = lsa.fit_transform(X_train_tfidf)
	X_train_tfidf = Normalizer(copy=False).fit_transform(X_train_tfidf)

clf = SVC().fit(X_train_tfidf, dataset['target'])

if(testAccuracy):
	testset = {'data': [], 'target': [], 'target_names': categories}
	with open("testset.txt") as f:
	  testset['data'] = f.readlines()
	testset['data'] = [x.strip() for x in testset['data']]

	n_songs = len(testset['data']) / len(categories)
	for i in xrange(len(categories)):
		for j in xrange(n_songs):
			testset['target'].append(i)

	documents = testset['data']
	X_test_counts = count_vect.transform(documents)
	X_test_tfidf = tfidf_transformer.transform(X_test_counts)

	if(use_lsa):
		X_test_tfidf = lsa.transform(X_test_tfidf)
		X_test_tfidf = Normalizer(copy=False).transform(X_test_tfidf)

	predicted = clf.predict(X_test_tfidf)
	acc = np.mean(predicted == testset['target'])

	print(predicted)
	print('Accuracy: ' + str(acc*100) + '%')

def parse(lyrics):
	lyrics = re.sub('[^a-zA-Z\n]', ' ', lyrics)
	lyrics = lyrics.lower()
	return lyrics

def solve():
	lyrics = newInput.get()
	documents = [parse(lyrics)]
	X_new_counts = count_vect.transform(documents)
	X_new_tfidf = tfidf_transformer.transform(X_new_counts)

	if(use_lsa):
		X_new_tfidf = lsa.transform(X_new_tfidf)
		X_new_tfidf = Normalizer(copy=False).transform(X_new_tfidf)

	predicted = clf.predict(X_new_tfidf)
	output.set("Genre: " + categories[predicted[0]])
	return

def callback(event):
    root.after(50, select_all, event.widget)

def select_all(widget):
    widget.select_range(0, 'end')
    widget.icursor('end')

root = Tk()
root.geometry("600x400+0+0")
root.title("Song Genre Classifier")
output = StringVar()
newInput = StringVar()

newline = Label(root, text="\n\n").pack()
title = Label(root, text="Input Lyrics", font=("Helvetica", 20)).pack()
newline = Label(root, text="").pack()

entry = Entry(root, textvariable=newInput, width=50, font=("Helvetica", 20))
entry.focus()
entry.bind('<Control-a>', callback)
entry.pack()
newline = Label(root, text="\n\n").pack()

button1 = Button(root, text="Submit", command=solve, fg="black", bg="green", font=("Helvetica", 16)).pack()
newline = Label(root, text="\n").pack()
outputLabel = Label(root, textvariable=output, font=("Helvetica", 20)).pack()

root.mainloop()