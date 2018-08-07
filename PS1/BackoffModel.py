import math, collections

class BackoffModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.BigramCounts = collections.defaultdict(lambda: 0)
    self.SmoothUnigramCounts = collections.defaultdict(lambda: 0)
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
        self.SmoothUnigramCounts[token] = self.SmoothUnigramCounts[token] + 1
        if previousWord != None: 
            self.BigramCounts[(previousWord,token)] = self.BigramCounts[(previousWord,token)] +1
        self.total += 1
        previousWord = token

  #self.SmoothUnigramCounts["UNK"] = 0 

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    score = 0.0 
    previousWord = None
    count_both = 0 

    for token in sentence:
      count = self.SmoothUnigramCounts[token]
      
      if previousWord != None:
        count_both = self.BigramCounts[(previousWord, token)]
        count_prev = self.SmoothUnigramCounts[previousWord]
        
      if count_both > 0:
        score += math.log(count_both)
        score -= math.log(count_prev)
      else:
        #if self.SmoothUnigramCounts['UNK']
        score += math.log(count+1)
        score -= math.log(self.total+ self.SmoothUnigramCounts[token])
        score += math.log(0.8)
          
      previousWord = token

    return score
