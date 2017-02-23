import string

def load_Shakespeare_Lines():
	translator = str.maketrans('', '', string.punctuation)
	lines = []
	with open('shakespeare.txt') as f:
		poemLine = []
		for line in f:
			workLine = line.strip()
			if workLine == '' or workLine.isdigit():
				continue
			else:
				for word in workLine.split():
					poemLine.append(word.translate(translator).lower())
					lines.append(poemLine)
					poemLine = []

	#print(lines)
	return lines




