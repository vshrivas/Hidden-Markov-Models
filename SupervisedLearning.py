from NaivePoemGeneration import load_Shakespeare_Lines
from HMM import unsupervised_HMM
from HMM import supervised_HMM
from nltk.corpus import cmudict
import numpy as np

# returns the stress state of a given word 
# takes in the state labels of each stress class and a dictionary of all stress patterns
def getStressState(word, dictionary, state1_vals):
    # if the given word is not in d, automatically assign state1 
    if word not in dictionary:
        return state1_vals[np.random.randint(0,len(state1_vals))]
    # if it is, return the appropriate state
    return dictionary[word]   
    
# compiles a dictionary of the stress states of every word in d
def getStressStateDict(state1_vals, state2_vals, state3_vals, state4_vals):
    # dictionary of all words we know stresses of
    d = cmudict.dict()
    state = 0
    statesDict = {}

    for word in d: 

        # arbitrarily select the first pronunciation
        pronunciation = d[word][0]
        firstStress = 0;
        firstStressSet = False
        lastStress = 0;
        for element in pronunciation:
            if "0" in element:
                if not firstStressSet:
                    firstStress = 0
                    firstStressSet = True
                lastStress = 0
            elif "1" in element or "2" in element:
                if not firstStressSet:
                    firstStressSet = True
                    firstStress = 1
                lastStress = 1

        # starts stressed, ends stressed 
        if firstStress == 1 and lastStress == 1:
            state = state1_vals[np.random.randint(0,len(state1_vals))]
        # starts stressed, ends unstressed 
        if firstStress == 1 and lastStress == 0:
            state = state2_vals[np.random.randint(0,len(state2_vals))]
        # starts unstressed, ends stressed 
        if firstStress == 0 and lastStress == 1:
            state = state3_vals[np.random.randint(0,len(state3_vals))]
        # starts unstressed, ends unstressed 
        if firstStress == 0 and lastStress == 0:
            state = state4_vals[np.random.randint(0,len(state4_vals))]

        # build the dictionary entry and insert it into the dictionary of states
        dictEntry = {word: state}
        statesDict.update(dictEntry)
    return statesDict


def supervised_learning():
    '''
    Trains an HMM using supervised learning on the file 'shakespeare.txt' and
    prints the results.
    To do this, it assigns states to the words in the lines in 'shakespeare.txt' 
    based on the stress of the word. 
    '''

    d = cmudict.dict()
    # get the input lines from the shakespeare lines
    lines = load_Shakespeare_Lines()

    # array of possible states; can be split into several cases
    # Case 1: states in which we have a word that begins and ends with a
    # stressed syllable
    states_1 = range(4)
    # Case 2: states in which we have a word that begins stressed, ends
    # unstressed
    states_2 = range(4, 8)
    # Case 3: states in which we have a word that begins unstressed, ends
    # stressed
    states_3 = range(8, 12)
    # Case 4: states in which we have a word that begins unstressed, ends
    # unstressed
    states_4 = range(12, 16)

    # assembles the dictionary of stress patterns given the dictionary of words
    statesDict = getStressStateDict(states_1, states_2, states_3, states_4)

    # the matrix that holds that state sequence for each line in lines 
    Y =[]

    num_lines = len(lines)
    # use the stress of the words to assign it a state
    for line_index in range(len(lines)):
        line = lines[line_index] 
        Y.append([])
        for word_index in range(len(line)):
            word = line[word_index]
            Y[line_index].append(getStressState(word, statesDict, states_1))

    # Train the HMM.
    HMM = supervised_HMM(lines, Y) 
    numLines = 14
    for i in range(0, numLines):
        numSyllables = 10
        emission = HMM.generate_emission(HMM.indexes, numSyllables)
        print(emission)

supervised_learning()