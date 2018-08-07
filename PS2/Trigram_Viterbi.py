import sys
import re
import math
import itertools
from pprint import pprint
from collections import *

START_SYMBOL = 'init'
STOP_SYMBOL = 'final' 
OOV_WORD= 'OOV'

HMM_FILE = sys.argv[1]

tags = set() # i.e. K in the slides, a set of unique POS tags
trans = {} # transisions
emit = {} # emissions
voc = {} # encountered words
trans2 = {} # transision of bigram 

##########################################
#This part parses the my.hmm file you have g3enerated and obtain the transition and emission values.
# Taken from TA bigram Viterbi algorithm 

with open(HMM_FILE) as hmmfile:
    for line in hmmfile.read().splitlines():
        trans_reg = 'trans\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)' #Trigram
        emit_reg = 'emit\s+(\S+)\s+(\S+)\s+(\S+)'           #emission values
        trans2_reg = 'trans2\s+(\S+)\s+(\S+)\s+(\S+)'       #Bigram 
        trans_match = re.match(trans_reg, line)
        trans2_match = re.match(trans2_reg, line)
        emit_match = re.match(emit_reg, line)
        if trans_match:
            prev, qq, q, p = trans_match.groups() 
            # creating an entry in trans with the POS tag pair
            # e.g. (init, NNP) = log(probability for seeing that transition)
            trans[(prev, qq, q)] = math.log(float(p))
            # add the encountered POS tags to set
            tags.update([prev, qq, q])
        elif trans2_match:
            aa, b, c = trans2_match.groups()
            # creating an entry in trans with the POS tag pair
            # e.g. (init, NNP) = log(probability for seeing that transition)
            trans2[(aa, b)] = math.log(float(c))
            # add the encountered POS tags to set
        elif emit_match:
            q, w, p = emit_match.groups() #(tag, token, float(emissions[tag][token]) / emissionsTotal[tag])
            # creating an entry in emit with the tag and word pair
            # e.g. (NNP, "python") = log(probability for seeing that word with that tag)
            emit[(q, w)] = math.log(float(p))
            # adding the word to encountered words
            voc[w] = 1
            # add the encountered POS tags to set
            tags.update([q])
        else:
            #print 'no'
            pass

###################################################

    #Trigram Viterbi model with backoff using Bigram viterbi
    pi = defaultdict(float)
    bp = {}
    score = 0

    pi[(0, START_SYMBOL, START_SYMBOL)] = 0.0  #Initialization 

    #Define tagset S(K)
    def S(k):
        if k in (-1, 0):
            return {START_SYMBOL}
        else:
            return tags

    
    for word_sent in sys.stdin:  #Read sentence from stdin 
        words = word_sent.split(); 
        
        for i in range(len(words)): #assign unseen words OOV_WORD 
            if words[i] not in voc:
                words[i] = OOV_WORD

        n = len(words)
        
        #for loop for Trigram Viterbi Algoritm 
        for k in range(1, n+1):    
            for u in S(k-1):
                for v in S(k):
                    max_score = float('-Inf')
                    max_tag = None
                    for w in S(k - 2):
                        if (w, u, v) in trans and (v, words[k-1]) in emit:

                            if (k-1, w , u) not in pi:
                                score = trans[(w, u, v)]+ emit[(v, words[k-1])]
                            else:
                                score = pi[(k-1, w, u)] + trans[(w, u, v)]+ emit[(v, words[k-1])]
                        else:
                            score = float("-Inf")

                        if score > max_score:
                            max_score = score 
                            max_tag = w 
                    pi[(k, u, v)] = max_score 
                    bp[(k, u, v)] = max_tag 


        max_score = float('-Inf')
        u_tri, v_tri = None, None
        foundgoal = False

        #for loop for STOP_SYMBOL 
        for u in S(n-1):           
            for v in S(n):
                if (u,v, STOP_SYMBOL) in trans:
                    score = pi[(n, u, v)] + trans[(u,v, STOP_SYMBOL)]

                if score > max_score: 
                    max_score = score 
                    u_tri = u 
                    v_tri = v
                    foundgoal = True 

        #Deque the back pointers in the reverse order in 
        if foundgoal: 
            tagsSet = deque()
            tagsSet.append(v_tri)
            tagsSet.append(u_tri)

            
            for i, k in enumerate(range(n-2, 0, -1)):
                tagsSet.append(bp[(k+2, tagsSet[i+1], tagsSet[i])])

            tagsSet.reverse()
            print ' '.join(tagsSet)


        else:
            #If the trigram viterbi fails then run viterbi bigram model 
            #BIGRAM BACKOFFF
            ##############################################################
            pi2 = defaultdict(float)
            pi2[(0, START_SYMBOL)] = 0.0 #Initalize pi 
            bp2 = {} # backpointers
            score2 = 0

            #for loop for Bigram Viterbi Algorithm 
            for k in range(1, n+1): 
                for u in tags:
    
                    max_score = float('-Inf')
                    max_tag = None
                    for v in tags:
                        if (v, u) in trans2 and (u, words[k-1]) in emit:

                            if (k-1, v) not in pi2:
                                score2 = trans2[(v, u)]+ emit[(u, words[k-1])]
                            else:
                                score2 = pi2[(k-1, v)] + trans2[(v,u)]+ emit[(u, words[k-1])]
                        else:
                            score2 = float("-Inf")

                        if score2 > max_score:
                            max_score = score2 
                            max_tag = v
                            
                    pi2[(k, u)] = max_score 
                    bp2[(k, u)] = max_tag 


            max_score = float('-Inf')
            u_BI =  START_SYMBOL
            foundgoal2 = False
            scoreB = float('Inf')
        
            #for loop for STOP_SYMBOL 
            for v in tags:
                if (v, STOP_SYMBOL) in trans2 and (n, v) in pi2:
                    scoreB = pi2[(n, v)] + trans2[( v, STOP_SYMBOL)]

                    if not foundgoal2 or scoreB > max_score: #or not foundgoal:
                        max_score = scoreB 
                        u_BI = v
                        foundgoal2 = True 
            
            noun = 'NN'
            #Deque the backpointers 
            if foundgoal2:
            # y is the sequence of final chosen tags
                y = []
                y.append(u_BI)
                for i in xrange(n , 1, -1): #counting from the last word
                # bp[(i, tag)] gives you the tag for word[i - 1].
                # we use that and traces through the tags in the sentence.
                    if (i, u_BI) not in bp2:
                        y.append(noun)
                        u_BI = noun
                    else:
                        y.append(bp2[(i, u_BI)])
                        u_BI = bp2[(i, u_BI)]


                y.reverse()
                for i , k in enumerate(y):
                    if k == None:
                        y[i] = 'NN'
                    
                print ' '.join(y)
            else:
            #print 'NN' if something fails
                print("NN")            

