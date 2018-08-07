#!/usr/bin/python

# David Bamman
# 2/14/14
#
# Python port of train_hmm.pl:

# Noah A. Smith
# 2/21/08
# Code for maximum likelihood estimation of a bigram HMM from 
# column-formatted training data.

# Usage:  train_hmm.py tags text > hmm-file

# The training data should consist of one line per sequence, with
# states or symbols separated by whitespace and no trailing whitespace.
# The initial and final states should not be mentioned; they are 
# implied.  
# The output format is the HMM file format as described in viterbi.pl.

import sys,re
from itertools import izip
from collections import defaultdict
import matplotlib.pyplot as plt


TAG_FILE=sys.argv[1]
TOKEN_FILE=sys.argv[2]

vocab={}
OOV_WORD="OOV"
INIT_STATE="init"
FINAL_STATE="final"

emissions={}
transitions={}
transitionsTotal=defaultdict(int)
emissionsTotal=defaultdict(int)

#for bigram 
transitions2={}
transitionsTotal2=defaultdict(int)

with open(TAG_FILE) as tagFile, open(TOKEN_FILE) as tokenFile:
	for tagString, tokenString in izip(tagFile, tokenFile):

		tags=re.split("\s+", tagString.rstrip())
		tokens=re.split("\s+", tokenString.rstrip())
		pairs=zip(tags, tokens)

		prevtag=INIT_STATE
		prevprevtag = INIT_STATE

		for (tag, token) in pairs:

			# this block is a little trick to help with out-of-vocabulary (OOV)
			# words.  the first time we see *any* word token, we pretend it
			# is an OOV.  this lets our model decide the rate at which new
			# words of each POS-type should be expected (e.g., high for nouns,
			# low for determiners).
			if tag not in emissions:
				emissions[tag]=defaultdict(int)

			if token not in vocab:
				vocab[token]=1
				emissions[tag][token] = 1
				token=OOV_WORD

			

			#for trigram
			if prevprevtag not in transitions:
				transitions[prevprevtag] = defaultdict(int)
			if prevtag not in transitions[prevprevtag]:
				transitions[prevprevtag][prevtag]=defaultdict(int)

			#for bigram
			if prevtag not in transitions2:
				transitions2[prevtag]=defaultdict(int)

			# increment the emission/transition observation
			emissions[tag][token]+=1
			emissionsTotal[tag]+=1
			
			transitions[prevprevtag][prevtag][tag]+=1
			transitionsTotal[(prevprevtag,prevtag)]+=1

			transitions2[prevtag][tag]+=1
			transitionsTotal2[prevtag]+=1

			prevprevtag = prevtag
			prevtag=tag

		# don't forget the stop probability for each sentence
		if prevprevtag not in transitions:
			transitions[prevprevtag] = defaultdict(int)
		if prevtag not in transitions[prevprevtag]:
			transitions[prevprevtag][prevtag]=defaultdict(int)

		transitions[prevprevtag][prevtag][FINAL_STATE]+=1
		transitionsTotal[(prevprevtag, prevtag)]+=1

		if prevtag not in transitions2:
			transitions2[prevtag]=defaultdict(int)

		transitions2[prevtag][FINAL_STATE]+=1
		transitionsTotal2[prevtag]+=1

for prevprevtag in transitions:
	for prevtag in transitions[prevprevtag]:
		for tag in transitions[prevprevtag][prevtag]:
			print "trans %s %s %s %s" % (prevprevtag, prevtag, tag, float(transitions[prevprevtag][prevtag][tag]) + 1 / transitionsTotal[(prevprevtag, prevtag)])


for prevtag in transitions2:
	for tag in transitions2[prevtag]:
		print "trans2 %s %s %s" % (prevtag, tag, float(transitions2[prevtag][tag]) + 1 / transitionsTotal2[prevtag])

for tag in emissions:
	for token in emissions[tag]:
		print "emit %s %s %s " % (tag, token, float(emissions[tag][token]) / (emissionsTotal[tag])) #+len(vocab)*0.01))



