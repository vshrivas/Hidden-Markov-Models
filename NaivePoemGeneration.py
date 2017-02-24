import string

def load_Shakespeare_Lines():
	lines = []
	with open('shakespeare.txt') as f:
		poemLine = []
		for line in f:
			workLine = line.strip()
			if workLine == '' or workLine.isdigit():
				continue
			else:
				for word in workLine.split():
					poemLine.append(word.translate(None, string.punctuation).lower())
			lines.append(poemLine)
			poemLine = []

	#print(lines)
	return lines