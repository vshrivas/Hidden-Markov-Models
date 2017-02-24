import string

# this method parses the shakespeare.txt file
# to create lines the HMM can learn from
def load_Shakespeare_Lines():
	# translator used to remove punctuation
	translator = str.maketrans('', '', string.punctuation)
	lines = []
	with open('shakespeare.txt') as f:
		# list which will store all words in a line
		poemLine = []
		# for each line
		for line in f:
			# remove whitespaces
			workLine = line.strip()
			# if line is empty or a digit, skip
			if workLine == '' or workLine.isdigit():
				continue
			else:
				# for each word in line
				for word in workLine.split():
					# append lowercase word with punctuation removed to the word list for the line
					poemLine.append(word.translate(translator).lower())
					# append this line to list of all lines
					lines.append(poemLine)
					# reset list of words in line to empty list for next line
					poemLine = []

	#print(lines)
	return lines




