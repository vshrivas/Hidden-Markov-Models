import sys
import numpy as np
import random
from nltk.corpus import cmudict
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from NaivePoemGeneration import load_Shakespeare_Lines

def get_data():
	# convert lines into X and Y encoded as integers
	lines, rhymingDict = load_Shakespeare_Lines()
	x_len = 3
	X = []

	observations = {}
	indexes = {}
	index = 0
	for x in lines:
		for word in x:
			if word == ' ':
			    print('FOUND SPACE')
			if word not in observations:
				observations[word] = index
				indexes[index] = word
				index += 1

	num_words = len(observations.keys())

	print(num_words)

	Y = []

	for lineID in range(0, len(lines)):
		for wordID in range(0, len(lines[lineID]) - x_len):
			# every x_len words in line are input
			x = lines[lineID][wordID: wordID + x_len]
			# x_len th word is output, y is the output word
			y = lines[lineID][wordID + x_len]

			# convert x to integers
			# each input seq is a list of integers, where integers correspond to words
			x_input = []
			for word in x:
				x_input.append(observations[word])

			X.append(x_input)

			# create one-hot-encoded version of y
			y_one_hot = np.zeros(num_words)
			# find index of word
			word_num = observations[y]
			# put one in index location
			y_one_hot[word_num] = 1

			Y.append(y_one_hot)

	n_inputs = len(X)
	n_outputs = len(Y)
	print n_inputs
	print n_outputs

	return X, Y, n_inputs, observations, indexes

# helper method to calculate the syllable count
# word is the word to calculate the syllable count for, and d is the dictionary
def nsyl(word, d):
    # get syllable counts for all pronunciations of the word
    pronunciationSyllabs = [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]] 
    # return syllable count for first pronunciation, assuming that is the most common one
    return pronunciationSyllabs[0]

def trainRNN():
	d = cmudict.dict()
	defaultSyllabCount = 2

	dataX, dataY, n_patterns, observations, indexes = get_data()
	x_len = 3
	num_words = len(observations.keys())

	X = np.array(dataX)
	X = np.reshape(dataX, (n_patterns, x_len, 1))
	'''# normalize
	X = X / float(num_words)'''

	Y = np.array(dataY)

	# create model
	model = Sequential()
	model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
	model.add(Dropout(0.2))
	'''model.add(LSTM(256, return_sequences=True))
	model.add(Dropout(0.2))'''
	model.add(LSTM(256))
	model.add(Dropout(0.2))
	model.add(Dense(Y.shape[1], activation='softmax'))

	# compile the LSTM model
	model.compile(loss='categorical_crossentropy', optimizer='adam')

	filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
	callbacks_list = [checkpoint]

	# fit the model
	model.fit(X, Y, nb_epoch=20, batch_size=128, callbacks=callbacks_list)
	
def generateRNN():
	d = cmudict.dict()
	defaultSyllabCount = 2

	dataX, dataY, n_patterns, observations, indexes = get_data()
	x_len = 3
	num_words = len(observations.keys())

	X = np.array(dataX)
	X = np.reshape(dataX, (n_patterns, x_len, 1))
	'''# normalize
	X = X / float(num_words)'''

	Y = np.array(dataY)

	# define the LSTM model
	model = Sequential()
	model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
	model.add(Dropout(0.2))
	'''model.add(LSTM(256, return_sequences=True))
	model.add(Dropout(0.2))'''
	model.add(LSTM(256))
	model.add(Dropout(0.2))
	model.add(Dense(Y.shape[1], activation='softmax'))

	# load file with optimal weights
	filename = "weights-improvement-19-5.0980.hdf5"
	model.load_weights(filename)

	model.compile(loss='categorical_crossentropy', optimizer='adam')

	numLines = 14

	# generate words
	for line in range(0, numLines):
		output = ''
		numSyllab = 0
		lastResult = ''

		# pick a random seed
		start = np.random.randint(0, len(dataX)-1)
		pattern = dataX[start]

		# add seed to line
		for value in pattern:
			output += ' ' + indexes[value]
			if indexes[value] not in d:
				numSyllab += defaultSyllabCount
			else:
				numSyllab += nsyl(indexes[value], d)
			lastresult = indexes[value]

	 	while numSyllab < 10:
	 		# generate prediction from seed
			x = np.reshape(pattern, (1, len(pattern), 1))
			prediction = model.predict(x, verbose=0)

			# sample next word from this prediction
			rand_var = random.uniform(0, 1)
			next_obs = 0

			while rand_var > 0:
			    rand_var -= prediction[0][next_obs]
			    next_obs += 1
			next_obs -= 1

			index = next_obs
			result = indexes[index]

			# if getting repeated words, try new seed
			if(result == lastResult):
				#print('result same as lastresult')
				start = np.random.randint(0, len(dataX)-1)
				pattern = dataX[start]
				for value in pattern:
					output += ' ' + indexes[value]
					if indexes[value] not in d:
						numSyllab += defaultSyllabCount
					else:
						numSyllab += nsyl(indexes[value], d)
				lastresult = indexes[value]
				continue

			# word isn't in dictionary
			if result not in d:
			    #print('****', nextWord, 'not in dictionary ****')
			    wordSyllab = defaultSyllabCount
			# get syllable count for most common pronunciation
			else:
			    wordSyllab = nsyl(result, d)

			numSyllab += wordSyllab

			output += ' ' + result

			# add predicted word to pattern, and remove first word from pattern
			pattern.append(index)
			pattern = pattern[1:len(pattern)]

			lastResult = result

		print (output.capitalize())

	print('DONE!!!')

generateRNN()




			


