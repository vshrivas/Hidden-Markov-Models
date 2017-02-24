########################################
# CS/CNS/EE 155 2017
# Problem Set 5
#
# Author:       Andrew Kang
# Description:  Set 5 solutions
########################################

import random
import string
from nltk.corpus import cmudict
import numpy as np

class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O, indexes, observations, rhymingDict):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0. 
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state. 

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.

            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.

            D:          Number of observations.
            
            A:          The transition matrix.
            
            O:          The observation matrix.
            
            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for _ in range(self.L)]
        self.indexes = indexes
        self.observations = observations
        self.rhymingDict = rhymingDict


    def forward(self, x, observations, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.

                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''
        M = len(x)      # Length of sequence.
        alphas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        # Note that alpha_j(0) is already correct for all j's.
        # Calculate alpha_j(1) for all j's.
        for curr in range(self.L):
            alphas[1][curr] = self.A_start[curr] * self.O[curr][observations[x[0]]]

        # Calculate alphas throughout sequence.
        for t in range(1, M):
            # Iterate over all possible current states.
            for curr in range(self.L):
                prob = 0

                # Iterate over all possible previous states to accumulate
                # the probabilities of all paths from the start state to
                # the current state.
                for prev in range(self.L):
                    prob += alphas[t][prev] \
                            * self.A[prev][curr] \
                            * self.O[curr][observations[x[t]]]

                # Store the accumulated probability.
                alphas[t + 1][curr] = prob

            if normalize:
                norm = sum(alphas[t + 1])
                for curr in range(self.L):
                    alphas[t + 1][curr] /= norm

        return alphas


    def backward(self, x, observations, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.

                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.

                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''

        M = len(x)      # Length of sequence.
        betas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        # Initialize initial betas.
        for curr in range(self.L):
            betas[-1][curr] = 1

        # Calculate betas throughout sequence.
        for t in range(-1, -M - 1, -1):
            # Iterate over all possible current states.
            for curr in range(self.L):
                prob = 0

                # Iterate over all possible next states to accumulate
                # the probabilities of all paths from the end state to
                # the current state.
                for nxt in range(self.L):
                    if t == -M:
                        prob += betas[t][nxt] \
                                * self.A_start[nxt] \
                                * self.O[nxt][observations[x[t]]]

                    else:
                        prob += betas[t][nxt] \
                                * self.A[curr][nxt] \
                                * self.O[nxt][observations[x[t]]]

                # Store the accumulated probability.
                betas[t - 1][curr] = prob

            if normalize:
                norm = sum(betas[t - 1])
                for curr in range(self.L):
                    betas[t - 1][curr] /= norm

        return betas

    def supervised_learning(self, X, Y, observations):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to D - 1. In other words, a list of
                        lists.
            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to L - 1. In other words, a list of
                        lists.
                        Note that the elements in X line up with those in Y.
        '''
        # Calculate each element of A using the M-step formulas.

        # loop through all of the states in this model; these are the states we
        # are transitioning from 
        for b in range(self.L):
            # loop through all of the states again; these are the states we 
            # are transitioning to
            for a in range(self.L):
                # i and j will loop through each element of A
                # these variables keep track of the numerator and denominator 
                # in the given formula
                numer_count = 0.0
                denom_count = 0.0
                # the next two loops will actually perform the summations
                for sequence in Y:
                    for state_index in range(len(sequence) - 1):
                        state = sequence[state_index]
                        next_state = sequence[state_index + 1]
                        if state == b and next_state == a:
                            numer_count+=1.0
                            denom_count+=1.0
                        elif state == b:
                            denom_count+=1.0
                self.A[b][a] = numer_count/denom_count


        # Calculate each element of O using the M-step formulas.
        # loop through the sequences of the model (the genre)
        for w in range(self.D):
            # loop through the states of the model (the mood)
            for z in range(self.L):
                # these variables keep track of the numerator and denominator 
                # in the given formula
                numer_count = 0.0
                denom_count = 0.0
                # the next two loops will actually perform the summations
                for stateSequence_index in range(len(Y)):
                    for state_index in range(len(Y[stateSequence_index])):
                        word = X[stateSequence_index][state_index]
                        state = Y[stateSequence_index][state_index]
                        if w == observations[word] and state == z:
                            numer_count += 1
                            denom_count += 1
                        elif state == z:
                            denom_count += 1
                self.O[z][w] = numer_count/denom_count
                    

        pass

    def unsupervised_learning(self, X, iters, observations):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.
        '''

        # Note that a comment starting with 'E' refers to the fact that
        # the code under the comment is part of the E-step.

        # Similarly, a comment starting with 'M' refers to the fact that
        # the code under the comment is part of the M-step.

        for iteration in range(iters):
            print("Iteration: " + str(iteration))

            # Numerator and denominator for the update terms of A and O.
            A_num = [[0. for i in range(self.L)] for j in range(self.L)]
            O_num = [[0. for i in range(self.D)] for j in range(self.L)]
            A_den = [0. for i in range(self.L)]
            O_den = [0. for i in range(self.L)]

            # For each input sequence:
            for x in X:
                M = len(x)
                # Compute the alpha and beta probability vectors.
                alphas = self.forward(x, observations, normalize=True)
                betas = self.backward(x, observations, normalize=True)

                # E: Update the expected observation probabilities for a
                # given (x, y).
                # The i^th index is P(y^t = i, x).
                for t in range(1, M + 1):
                    P_curr = [0. for _ in range(self.L)]
                    
                    for curr in range(self.L):
                        P_curr[curr] = alphas[t][curr] * betas[t][curr]

                    # Normalize the probabilities.
                    norm = sum(P_curr)
                    for curr in range(len(P_curr)):
                        P_curr[curr] /= norm

                    for curr in range(self.L):
                        #if t != M:
                        A_den[curr] += P_curr[curr]
                        O_den[curr] += P_curr[curr]
                        O_num[curr][observations[x[t - 1]]] += P_curr[curr]

                # E: Update the expectedP(y^j = a, y^j+1 = b, x) for given (x, y)
                for t in range(1, M):
                    P_curr_nxt = [[0. for _ in range(self.L)] for _ in range(self.L)]

                    for curr in range(self.L):
                        for nxt in range(self.L):
                            P_curr_nxt[curr][nxt] = alphas[t][curr] \
                                                    * self.A[curr][nxt] \
                                                    * self.O[nxt][observations[x[t]]] \
                                                    * betas[t + 1][nxt]

                    # Normalize:
                    norm = 0
                    for lst in P_curr_nxt:
                        norm += sum(lst)
                    for curr in range(self.L):
                        for nxt in range(self.L):
                            P_curr_nxt[curr][nxt] /= norm

                    # Update A_num
                    for curr in range(self.L):
                        for nxt in range(self.L):
                            A_num[curr][nxt] += P_curr_nxt[curr][nxt]

            for curr in range(self.L):
                for nxt in range(self.L):
                    self.A[curr][nxt] = A_num[curr][nxt] / A_den[curr]

            for curr in range(self.L):
                for xt in range(self.D):
                    self.O[curr][xt] = O_num[curr][xt] / O_den[curr]

    def generate_best_emission(self, indexes, syllabCount, endWord):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random. 

        Arguments:
            indexes: Dictionary mapping each index to its word
            syllabCount: number of syllables each line should have

        Returns:
            emission:   The randomly generated emission as a string.
        '''

        emission = ''
        # use cmu dictionary from nltk to calculate syllable counts
        d = cmudict.dict()
        numSyllables = 0
        defaultSyllabCount = 2
        numWord = 0
        state = -1

        # if we have an end word for line
        if(endWord != ''):
            # start with state that most likely generated that word
            maxProb = 0
            # get index of word
            wordIndex = self.observations[endWord]
            # find state with highest probability of generating word
            for i in range(0, self.L):
                prob = self.O[i][wordIndex]
                if(prob > maxProb):
                    state = i
                    maxProb = prob
            # use that state as the start state
            # add endWord to emission seq
            emission = endWord
            if endWord not in d:
                numSyllables  += defaultSyllabCount
            else:
                numSyllables += nsyl(endWord, d)

            # Find highest probability previous state 
            # to have reverse HMM generation
            maxProb = 0
            prev_state = -1
            for i in range(0, self.L):
                prob = self.O[i][state]
                if(prob > maxProb):
                    prev_state = i
                    maxProb = prob

            state = prev_state

        # no end word specified, choose random start state
        else:
            state = random.choice(range(self.L))

        # keep adding words to line until reach required syllable count
        while numSyllables < syllabCount:
            # Find best observation from this state
            maxProb = 0
            bestWordId = -1
            for wordId in range(0, self.D):
                prob = self.O[state][wordId]
                if(prob > maxProb):
                    bestWordId = wordId
                    maxProb = prob

            # found next observation
            nextWord = indexes[bestWordId]
            
            wordSyllab = 0
            # word isn't in dictionary
            if nextWord not in d:
                #print('****', nextWord, 'not in dictionary ****')
                wordSyllab = defaultSyllabCount
            # get syllable count for most common pronunciation
            else:
                wordSyllab = nsyl(nextWord, d)

            # adding word to line would exceed syllable limit per line
            # skip word, don't add it to line
            if numSyllables + wordSyllab > syllabCount:
                continue
            else:
                numSyllables += wordSyllab

            emission = nextWord + ' ' + emission

            # Find highest probability previous state 
            # to have reverse HMM generation
            maxProb = 0
            prev_state = -1
            for i in range(0, self.L):
                prob = self.O[i][state]
                if(prob > maxProb):
                    prev_state = i
                    maxProb = prob

            state = prev_state

        return emission.strip()

    def generate_emission(self, indexes, syllabCount, endWord):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random. 

        Arguments:
            indexes: Dictionary mapping each index to its word
            syllabCount: number of syllables each line should have

        Returns:
            emission:   The randomly generated emission as a string.
        '''

        emission = ''
        # use cmu dictionary from nltk to calculate syllable counts
        d = cmudict.dict()
        numSyllables = 0
        defaultSyllabCount = 2
        numWord = 0
        state = -1

        # if we have an end word for line
        if(endWord != ''):
            # start with state that most likely generated that word
            maxProb = 0
            # get index of word
            wordIndex = self.observations[endWord]
            # find state with highest probability of generating word
            for i in range(0, self.L):
                prob = self.O[i][wordIndex]
                if(prob > maxProb):
                    state = i
                    maxProb = prob
            # use that state as the start state
            # add endWord to emission seq
            emission  = endWord
            if endWord not in d:
                numSyllables  += defaultSyllabCount
            else:
                numSyllables += nsyl(endWord, d)

            # Sample prev state.
            # to have reverse HMM generation
            rand_var = random.uniform(0, 1)
            prev_state = 0

            while rand_var > 0 and prev_state < self.L:
                rand_var -= self.A[prev_state][state]
                prev_state += 1

            prev_state -= 1
            state = prev_state

        # no end word specified, choose random start state
        else:
            state = random.choice(range(self.L))

        # keep adding words to line until reach required syllable count
        while numSyllables < syllabCount:
            # Sample next observation.
            rand_var = random.uniform(0, 1)
            next_obs = 0

            while rand_var > 0:
                rand_var -= self.O[state][next_obs]
                next_obs += 1
            next_obs -= 1

            # found next observation
            nextWord = indexes[next_obs]
            
            wordSyllab = 0
            # word isn't in dictionary
            if nextWord not in d:
                #print('****', nextWord, 'not in dictionary ****')
                wordSyllab = defaultSyllabCount
            # get syllable count for most common pronunciation
            else:
                wordSyllab = nsyl(nextWord, d)

            # adding word to line would exceed syllable limit per line
            # skip word, don't add it to line
            # if numSyllables + wordSyllab > syllabCount:
            #     continue
            # else:
            # I commented this out, so the resulting line may exceed 10 syllables 
            numSyllables += wordSyllab

            emission = nextWord + ' ' + emission

            # Sample prev state.
            # to have reverse HMM generation
            rand_var = random.uniform(0, 1)
            prev_state = 0

            while rand_var > 0 and prev_state < self.L:
                rand_var -= self.A[prev_state][state]
                prev_state += 1

            prev_state -= 1
            state = prev_state

        return emission.strip()

# helper method to calculate the syllable count
# word is the word to calculate the syllable count for, and d is the dictionary
def nsyl(word, d):
    # get syllable counts for all pronunciations of the word
    pronunciationSyllabs = [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]] 
    # return syllable count for first pronunciation, assuming that is the most common one
    return pronunciationSyllabs[0]

def unsupervised_HMM(X, n_states, n_iters, rhymingDict):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        n_states:   Number of hidden states to use in training.
    '''

    # Make a set of observations.
    observations = {}
    indexes = {}
    index = 0
    for x in X:
        wordList = x
        for word in wordList:
            if word == ' ':
                print('FOUND SPACE')
            observations[word] = index
            indexes[index] = word
            index += 1

    #print(observations)
    
    # Compute L and D.
    L = n_states
    D = index + 1

    print(D)

    # Randomly initialize and normalize matrices A and O.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O, indexes, observations, rhymingDict)
    HMM.unsupervised_learning(X, n_iters, observations)

    return HMM

def supervised_HMM(X, Y, n_states):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learing.

    Arguments:
        X:          A list of variable length emission sequences 
        Y:          A corresponding list of variable length state sequences
                    Note that the elements in X line up with those in Y
    '''
    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)
    
    # Make a set of observations.
    observations = {}
    indexes = {}
    index = 0
    for x in X:
        wordList = x
        for word in wordList:
            if word == ' ':
                print('FOUND SPACE')
            observations[word] = index
            indexes[index] = word
            index += 1

    # Compute L and D.
    L = len(states)
    D = index + 1

    # Randomly initialize and normalize matrices A and O.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O, indexes)
    HMM.supervised_learning(X, Y, observations)

    return HMM
