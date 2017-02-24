from NaivePoemGeneration import load_Shakespeare_Lines
from HMM import unsupervised_HMM
import random
from SupervisedLearning import *

def semisupervised_learning(n_states, n_iters):
    '''
    Trains an HMM using supervised learning on the file 'ron.txt' and
    prints the results.

    Arguments:
        n_states:   Number of hidden states that the HMM should have.
    '''
    lines, rhymingDict = load_Shakespeare_Lines()

    # train HMM using supervised learning
    HMM = supervised_learning()

    # Train the HMM using unsupervised learning
    HMM.unsupervised_learning(lines, n_iters, HMM.observations)

    #HMM = unsupervised_HMM(lines, n_states, n_iters, rhymingDict)
    numQuatrains = 3
    numCouplets = 1
    numSyllables = 10

    for i in range(0, numQuatrains):
        endLine1 = random.choice(list(rhymingDict.keys()))
        line1 = HMM.generate_emission(HMM.indexes, numSyllables, endLine1)
        print(line1)

        endLine2 = random.choice(list(rhymingDict.keys()))
        line2 = HMM.generate_emission(HMM.indexes, numSyllables, endLine2)
        print(line2)

        rhyme1 = rhymingDict[endLine1][random.choice(range(0, len(rhymingDict[endLine1])))]

        line3 = HMM.generate_emission(HMM.indexes, numSyllables, rhyme1)
        print(line3)

        rhyme2 = rhymingDict[endLine2][random.choice(range(0, len(rhymingDict[endLine2])))]

        line4 = HMM.generate_emission(HMM.indexes, numSyllables, rhyme2)
        print(line4)

        print()

    for i in range(0, numCouplets):
        endLine1 = random.choice(list(rhymingDict.keys()))
        line1 = HMM.generate_emission(HMM.indexes, numSyllables, endLine1)
        print(line1)

        rhyme1 = rhymingDict[endLine1][random.choice(range(0, len(rhymingDict[endLine1])))]
        line2 = HMM.generate_emission(HMM.indexes, numSyllables, rhyme1)
        print(line2)
       
#numStates currently not being used
numStates = 15
numIter = 10
semisupervised_learning(numStates, numIter)
