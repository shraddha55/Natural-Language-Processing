import math, collections

class SmoothBigramModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.SmoothBigramCountBoth = collections.defaultdict(lambda: 0)
    self.SmoothBigramCounts = collections.defaultdict(lambda: 0)
    self.total = 0
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """  
    previousWord = None
    for sentence in corpus.corpus:
      for datum in sentence.data:  
        token = datum.word
        self.SmoothBigramCounts[token] = self.SmoothBigramCounts[token] + 1
        if previousWord != None: 
            self.SmoothBigramCountBoth[ (previousWord,token)] = self.SmoothBigramCountBoth[ (previousWord,token)] +1
        self.total += 1
        previousWord = token


  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    score = 0.0 
    previousWord = None
    count_both = 0 

    for token in sentence:
      #count_both = self.SmoothBigramCounts[token] + self.SmoothBigramCounts[token-1]
      count = self.SmoothBigramCounts[token]
      if previousWord != None:
        count_both = self.SmoothBigramCountBoth[(previousWord, token)]
        
      if count_both > 0:
        score += math.log(count_both +1)
        score -= math.log(count + self.SmoothBigramCounts[token])
      else:
        score -= math.log(self.total+ self.SmoothBigramCounts[token]) #check here with count 
      previousWord = token

    return score
