import string

# this method parses the shakespeare.txt file
# to create lines the HMM can learn from
def load_Shakespeare_Lines():
	# translator used to remove punctuation
	#translator = str.maketrans('', '', string.punctuation)
	lines = []

	with open('shakespeare.txt') as f:
		# list which will store all words in a line
		poemLine = []
		# key is a string, value is list of all words which rhyme with key
		rhymingDict = {}

		firstRhymePair = []
		secondRhymePair = []

		lineNum = 0

		# for each line
		for line in f:
			# remove whitespaces
			workLine = line.strip()
			# if line is empty or a digit, skip
			# indicates start of new sonnet
			if workLine == '' or workLine.isdigit():
				# add last words of final couplet from last sonnet to the rhyming dictionary
				if workLine.isdigit() and workLine != '1':
					# if last sonnet had a final couplet
					if len(firstRhymePair) != 0:
						if firstRhymePair[0] in rhymingDict:
							rhymingDict[firstRhymePair[0]].append(firstRhymePair[1])
						else:
							rhymingDict[firstRhymePair[0]] = [firstRhymePair[1]]

						if firstRhymePair[1] in rhymingDict:
							rhymingDict[firstRhymePair[1]].append(firstRhymePair[0])
						else:
							rhymingDict[firstRhymePair[1]] = [firstRhymePair[0]]
					firstRhymePair = []
					secondRhymePair = []
					lineNum = 0
				continue
			else:
				# for each word in line
				for word in workLine.split():
					# append lowercase word with punctuation removed to the word list for the line
					for p in string.punctuation:
						word = word.replace(p, '')
					poemLine.append(word.lower())
					
				# append this line to list of all lines
				lines.append(poemLine)
				# reset list of words in line to empty list for next line
				poemLine = []

				lastWord = workLine.split()[len(workLine.split()) - 1]
				for p in string.punctuation:
						lastWord = lastWord.replace(p, '')
				lastWord = lastWord.lower()

				# last two lines are a couplet
				if (lineNum == 12 or lineNum == 13):
					firstRhymePair.append(lastWord)
					lineNum += 1
					continue

				# line is first or third in quatrain
				elif((lineNum % 4) % 2 == 0):
					firstRhymePair.append(lastWord)
				# line is second or fourth in quatrain
				else:
					secondRhymePair.append(lastWord)

				#print(firstRhymePair)
				#print (secondRhymePair)

				lineNum += 1
				# starting new quatrain
				if(lineNum % 4 == 0):
					# add rhymes to rhyming dictionary
					if firstRhymePair[0] in rhymingDict:
						rhymingDict[firstRhymePair[0]].append(firstRhymePair[1])
					else:
						rhymingDict[firstRhymePair[0]] = [firstRhymePair[1]]

					if firstRhymePair[1] in rhymingDict:
						rhymingDict[firstRhymePair[1]].append(firstRhymePair[0])
					else:
						rhymingDict[firstRhymePair[1]] = [firstRhymePair[0]]

					if secondRhymePair[0] in rhymingDict:
						rhymingDict[secondRhymePair[0]].append(secondRhymePair[1])
					else:
						rhymingDict[secondRhymePair[0]] = [secondRhymePair[1]]

					if secondRhymePair[1] in rhymingDict:
						rhymingDict[secondRhymePair[1]].append(secondRhymePair[0])
					else:
						rhymingDict[secondRhymePair[1]] = [secondRhymePair[0]]

					firstRhymePair = []
					secondRhymePair = []


	#print(lines)
	#print(rhymingDict)
	#print(lines)
	return lines, rhymingDict

