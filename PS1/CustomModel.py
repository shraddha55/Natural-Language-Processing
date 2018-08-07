import math, collections
class CustomModel:

  def __init__(self, corpus):
    """Initial custom language model and structures needed by this mode"""
    self.TrigramCounts = collections.defaultdict(lambda: 0)
    self.BigramCounts = collections.defaultdict(lambda: 0)
    self.SmoothUnigramCounts = collections.defaultdict(lambda: 0)
    self.total = 0
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model.
    """  
    previousWord = None
    previous2Word = None
    

    for sentence in corpus.corpus:
      for datum in sentence.data:  
        token = datum.word
        self.SmoothUnigramCounts[token] = self.SmoothUnigramCounts[token] + 1
        if previousWord != None: 
            self.BigramCounts[(previousWord,token)] = self.BigramCounts[(previousWord,token)] +1
        if previous2Word != None:
            self.TrigramCounts[(previous2Word, previousWord, token)] = self.TrigramCounts[(previous2Word, previousWord, token)] + 1
        self.total += 1
        previous2Word = previousWord
        previousWord = token
      
    self.SmoothUnigramCounts["UNK"] = 0 
    #for token in self.BigramCounts:
     # self.BigramCounts[token] += 1
    for token in self.SmoothUnigramCounts:
      self.SmoothUnigramCounts[token] += 1
      self.total += 1

  def score(self, sentence):
    """ With list of strings, return the log-probability of the sentence with language model. Use
        information generated from train.
    """
    score = 0.0 
    previousWord = None
    previous2Word = None 
    count_1 = 0 
    count_2 = 0 
    count_2_1 = 0 
    count_3_2 =0 
    count_3_2_1 = 0 
   
  
    for token in sentence:
      count_1 = self.SmoothUnigramCounts[token] | 1
      
      if previousWord != None:
        count_2_1 = self.BigramCounts[(previousWord, token)] 
        count_2 = self.SmoothUnigramCounts[previousWord] | 1
        
      if previous2Word!= None:
        count_3_2_1 = self.TrigramCounts[(previous2Word, previousWord, token)] 
        count_3_2 = self.BigramCounts[(previous2Word, previousWord)] 

      if count_3_2_1 > 0:
        score += math.log(count_3_2_1)
        score -= math.log(count_3_2)
      
      elif count_2_1 > 0:
        score += math.log(count_2_1)
        score -= math.log(count_2)
        score += math.log(0.8)

      else:

        score += math.log(count_1)
        score -= math.log(self.total)
        score += math.log(0.8)
         
      previous2Word = previousWord 
      previousWord = token
             

    return score



