from NaivePoemGeneration import load_Shakespeare_Lines
from HMM import unsupervised_HMM

def unsupervised_learning(n_states, n_iters):
    '''
    Trains an HMM using supervised learning on the file 'ron.txt' and
    prints the results.

    Arguments:
        n_states:   Number of hidden states that the HMM should have.
    '''
    lines = load_Shakespeare_Lines()

    # Train the HMM.
    HMM = unsupervised_HMM(lines, n_states, n_iters)
    numLines = 14
    for i in range(0, numLines):
        numSyllables = 10
        emission = HMM.generate_emission(HMM.indexes, numSyllables)
        print(emission)

numStates = 10
numIter = 100
unsupervised_learning(numStates, numIter)