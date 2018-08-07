import sys
import getopt
import os
import math
import operator
import collections


# Accuracy:
#    Task 1: 0.816500
#    Navie-Bayes: 83.1
#    Best Model: 86.35

#Sources: ECS 189G Discussion slides
#         Piazza post
#         Discussed with Kavitha D.

class NaiveBayes:
    class TrainSplit:
        """
        Set of training and testing data
        """
        def __init__(self):
            self.train = []
            self.test = []

    class Document:
        """
        This class represents a document with a label. classifier is 'pos' or 'neg' while words is a list of strings.
        """
        def __init__(self):
            self.classifier = ''
            self.words = []

    def __init__(self):
        """
        Initialization of naive bayes
        """
        self.stopList = set(self.readFile('data/english.stop'))
        self.bestModel = False
        self.stopWordsFilter = False
        self.naiveBayesBool = False
        self.numFolds = 10

        self.TotalDoc = 0.0
        self.positiveDoc = 0.0
        self.negativeDoc = 0.0
        self.posDocLog = 0.0
        self.negDocLog = 0.0
    
        # store count of positive and neg words
        self.positiveVocab = collections.defaultdict(lambda : 0.0)
        self.negativeVocab = collections.defaultdict(lambda : 0.0)
    
        #count of all positive and negative words
        self.posCount = 0.0
        self.negCount = 0.0
    
        #predict pos and neg sentiment
        self.predictPos = 0.0
        self.predictNeg = 0.0
    
        #for vocab
        self.vocab = {}
        self.totalVocab = set()
    
        #n_k in the Doc for Binary
        self.posBinDoc = collections.defaultdict(lambda : 0.0)
        self.negBinDoc = collections.defaultdict(lambda : 0.0)
    
        #n in the for Binary
        self.posBinCount = 0.0
        self.negBinCount = 0.0
    
        #count line
        #self.countline = 0
    
        # TODO
        # Implement a multinomial naive bayes classifier and a naive bayes classifier with boolean features. The flag
        # naiveBayesBool is used to signal to your methods that boolean naive bayes should be used instead of the usual
        # algorithm that is driven on feature counts. Remember the boolean naive bayes relies on the presence and
        # absence of features instead of feature counts.

        # When the best model flag is true, use your new features and or heuristics that are best performing on the
        # training and test set.

        # If any one of the flags filter stop words, boolean naive bayes and best model flags are high, the other two
        # should be off. If you want to include stop word removal or binarization in your best performing model, you
        # will need to write the code accordingly.

    def classify(self, words):
        """
        Classify a list of words and return a positive or negative sentiment
        """
        # classify a list of words and return the 'pos' or 'neg' classification
        
        self.posDocLog = math.log(self.positiveDoc / self.TotalDoc)
        self.negDocLog = math.log(self.negativeDoc / self.TotalDoc )
        
        if self.stopWordsFilter:
            words = self.filterStopWords(words)
        
        #Task 4
        if self.bestModel:
            
            self.predictPos = self.posDocLog
            self.predictNeg = self.negDocLog
            
            negation = ['not', 'but', 'however' ,'despitethat']
            words = self.filterPunctuation(words)
            
            positiveReview = ['first-rate', 'insightful', 'clever', 'charming' , 'comical', 'charismatic', 'enjoyable', 'uproarious', 'original', 'tender', 'hilarious', 'absorbing', 'sensitive' , 'riveting', 'intriguing',
                              'powerful', 'fascinating', 'pleasant', 'surprising', 'dazzling', 'thought provoking','imaginative', 'legendary', 'unpretentious']
            
            negativeReview = [ 'second-rate', 'violent', 'moronic',  'third-rate', 'flawed', 'juvenile', 'boring',
                              'distasteful', 'ordinary', 'disgusting', 'senseless', 'static', 'brutal',   'confused', 'disappointing', 'bloody', 'silly','predictable', 'stupid', 'uninteresting','weak',
                               'tiresome', 'trite', 'outdated', 'dreadful', 'bland']
            
            
            first = ''
            temp = words[:]
            for word in temp:
                words.append(first + word)
                first = word
            
            checked = set()
            temp = []
            for word in words[1:]:
                if word not in checked:
                    temp.append(word)
                checked.add(word)
            words = temp[:]
            
            uniqueWord = set()
            for word in words:
                if word in self.totalVocab and word not in uniqueWord:
                    if word in negation:
                        self.predictPos = 0
                        self.predictNeg = 0
                    else:
                        if word in positiveReview:
                            self.predictPos += (2*math.log(self.posBinDoc[word]+ 5))
                            self.predictPos -= math.log(self.posBinCount + (len(self.vocab)*2))
                        
                            self.predictNeg += math.log(self.negBinDoc[word]+ 5)
                            self.predictNeg -= math.log(self.negBinCount + (len(self.vocab)*2))
                        else:
                            self.predictPos += math.log(self.posBinDoc[word]+ 5)
                            self.predictPos -= math.log(self.posBinCount + (len(self.vocab)*2))
                            
                            self.predictNeg += math.log(self.negBinDoc[word]+ 5)
                            self.predictNeg -= math.log(self.negBinCount + (len(self.vocab)*2))
        
                    uniqueWord.add(word)
                        
            if self.predictPos >= self.predictNeg:
                return 'pos'
            else:
                return 'neg'
        
        
        #Task 3
        if self.naiveBayesBool:
            self.predictPos = self.posDocLog
            self.predictNeg = self.negDocLog
            
            for word in words:
                
                self.predictPos += math.log(self.posBinDoc[word]+20)
                self.predictPos -= math.log(self.posBinCount + (len(self.vocab)*20))
                
                self.predictNeg += math.log(self.negBinDoc[word]+20)
                self.predictNeg -= math.log(self.negBinCount + (len(self.vocab)*20))
            
            if self.predictPos >= self.predictNeg:
                return 'pos'
            else:
                return 'neg'
        
        #Task 1
        else:
            self.predictPos = self.posDocLog
            self.predictNeg = self.negDocLog
            
            for word in words:
                
                self.predictPos += math.log(self.positiveVocab[word]+1)
                self.predictPos -= math.log(self.posCount + len(self.vocab))
                
                self.predictNeg += math.log(self.negativeVocab[word]+1)
                self.predictNeg -= math.log(self.negCount + len(self.vocab))
            
            if self.predictPos >= self.predictNeg:
                return 'pos'
            else:
                return 'neg'
     

    def addDocument(self, classifier, words):
        """
        Train your model on a document with label classifier (pos or neg) and words (list of strings). You should
        store any structures for your classifier in the naive bayes class. This function will return nothing
        """
        # Train model on document with label classifiers and words
        
        self.TotalDoc += 1
        if classifier == 'pos':
            self.positiveDoc += 1
        elif classifier == 'neg':
            self.negativeDoc += 1

        # Task 4
        if self.bestModel:
            
            #words = words[len(words)/3:]
            
            #bigram
            first = ''
            temp = words[:]
            for word in temp:
                words.append(first + word)
                first = word

            uniqueSet = set()
            for word in words:
               
                if word not in uniqueSet:
                    uniqueSet.add(word)
                    self.totalVocab.add(word)
                    if classifier == 'pos':
                        self.posBinCount += 1
                        self.posBinDoc[word] += 1
                
                    if classifier == 'neg':
                        self.negBinCount += 1
                        self.negBinDoc[word] += 1
                        
                        
        #Task 3
        if self.naiveBayesBool:
            
            for word in set(words):
                if classifier == 'pos':
                    self.posBinDoc[word] += 1
                    self.posBinCount += 1
                        
                if classifier == 'neg':
                    self.negBinDoc[word] += 1
                    self.negBinCount += 1

            for word in words:
                if word not in self.vocab:
                    self.vocab[word] = 1
        #Task 1
        else:

            for word in words:
                if classifier == 'pos':
                    self.posCount += 1
                    if word not in self.positiveVocab:
                        self.positiveVocab[word] = 1
                    else:
                        self.positiveVocab[word] += 1
            
                elif classifier == 'neg':
                    self.negCount += 1
                    if word not in self.negativeVocab:
                        self.negativeVocab[word] = 1
                    else:
                        self.negativeVocab[word] += 1

                if word not in self.vocab:
                    self.vocab[word] = 1

        

    def readFile(self, fileName):
        """
        Reads a file and segments.
        """
        contents = []
        f = open(fileName)
        for line in f:
            contents.append(line)
        f.close()
        str = '\n'.join(contents)
        result = str.split()
        return result

    def trainSplit(self, trainDir):
        """Takes in a trainDir, returns one TrainSplit with train set."""
        split = self.TrainSplit()
        posDocTrain = os.listdir('%s/pos/' % trainDir)
        negDocTrain = os.listdir('%s/neg/' % trainDir)
        for fileName in posDocTrain:
            doc = self.Document()
            doc.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
            doc.classifier = 'pos'
            split.train.append(doc)
        for fileName in negDocTrain:
            doc = self.Document()
            doc.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
            doc.classifier = 'neg'
            split.train.append(doc)
        return split

    def train(self, split):
        for doc in split.train:
            words = doc.words
            if self.stopWordsFilter:
                words = self.filterStopWords(words)
            self.addDocument(doc.classifier, words)

    def crossValidationSplits(self, trainDir):
        """Returns a lsit of TrainSplits corresponding to the cross validation splits."""
        splits = []
        posDocTrain = os.listdir('%s/pos/' % trainDir)
        negDocTrain = os.listdir('%s/neg/' % trainDir)
        # for fileName in trainFileNames:
        for fold in range(0, self.numFolds):
            split = self.TrainSplit()
            for fileName in posDocTrain:
                doc = self.Document()
                doc.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
                doc.classifier = 'pos'
                if fileName[2] == str(fold):
                    split.test.append(doc)
                else:
                    split.train.append(doc)
            for fileName in negDocTrain:
                doc = self.Document()
                doc.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
                doc.classifier = 'neg'
                if fileName[2] == str(fold):
                    split.test.append(doc)
                else:
                    split.train.append(doc)
            yield split

    def test(self, split):
        """Returns a list of labels for split.test."""
        labels = []
        for doc in split.test:
            words = doc.words
            if self.stopWordsFilter:
                words = self.filterStopWords(words)
            guess = self.classify(words)
            labels.append(guess)
        return labels

    def buildSplits(self, args):
        """
        Construct the training/test split
        """
        splits = []
        trainDir = args[0]
        if len(args) == 1:
            print '[INFO]\tOn %d-fold of CV with \t%s' % (self.numFolds, trainDir)

            posDocTrain = os.listdir('%s/pos/' % trainDir)
            negDocTrain = os.listdir('%s/neg/' % trainDir)
            for fold in range(0, self.numFolds):
                split = self.TrainSplit()
                for fileName in posDocTrain:
                    doc = self.Document()
                    doc.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
                    doc.classifier = 'pos'
                    if fileName[2] == str(fold):
                        split.test.append(doc)
                    else:
                        split.train.append(doc)
                for fileName in negDocTrain:
                    doc = self.Document()
                    doc.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
                    doc.classifier = 'neg'
                    if fileName[2] == str(fold):
                        split.test.append(doc)
                    else:
                        split.train.append(doc)
                splits.append(split)
        elif len(args) == 2:
            split = self.TrainSplit()
            testDir = args[1]
            print '[INFO]\tTraining on data set:\t%s testing on data set:\t%s' % (trainDir, testDir)
            posDocTrain = os.listdir('%s/pos/' % trainDir)
            negDocTrain = os.listdir('%s/neg/' % trainDir)
            for fileName in posDocTrain:
                doc = self.Document()
                doc.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
                doc.classifier = 'pos'
                split.train.append(doc)
            for fileName in negDocTrain:
                doc = self.Document()
                doc.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
                doc.classifier = 'neg'
                split.train.append(doc)

            posDocTest = os.listdir('%s/pos/' % testDir)
            negDocTest = os.listdir('%s/neg/' % testDir)
            for fileName in posDocTest:
                doc = self.Document()
                doc.words = self.readFile('%s/pos/%s' % (testDir, fileName))
                doc.classifier = 'pos'
                split.test.append(doc)
            for fileName in negDocTest:
                doc = self.Document()
                doc.words = self.readFile('%s/neg/%s' % (testDir, fileName))
                doc.classifier = 'neg'
                split.test.append(doc)
            splits.append(split)
        return splits

    def filterStopWords(self, words):
        """
        Stop word filter
        """
        removed = []
        for word in words:
            if not word in self.stopList and word.strip() != '':
                removed.append(word)
        return removed

    def filterPunctuation(self, words):
        """
        Stop word filter
        """
        removed = []
        punc = ['.',',',':','-']
        for word in words:
            if not word in punc and word.strip() != '':
                removed.append(word)
        return removed


def test10Fold(args, stopWordsFilter, naiveBayesBool, bestModel):
    nb = NaiveBayes()
    splits = nb.buildSplits(args)
    avgAccuracy = 0.0
    fold = 0
    for split in splits:
        classifier = NaiveBayes()
        classifier.stopWordsFilter = stopWordsFilter
        classifier.naiveBayesBool = naiveBayesBool
        classifier.bestModel = bestModel
        accuracy = 0.0
        for doc in split.train:
            words = doc.words
            classifier.addDocument(doc.classifier, words)

        for doc in split.test:
            words = doc.words
            guess = classifier.classify(words)
            if doc.classifier == guess:
                accuracy += 1.0

        accuracy = accuracy / len(split.test)
        avgAccuracy += accuracy
        print '[INFO]\tFold %d Accuracy: %f' % (fold, accuracy)
        fold += 1
    avgAccuracy = avgAccuracy / fold
    print '[INFO]\tAccuracy: %f' % avgAccuracy


def classifyFile(stopWordsFilter, naiveBayesBool, bestModel, trainDir, testFilePath):
    classifier = NaiveBayes()
    classifier.stopWordsFilter = stopWordsFilter
    classifier.naiveBayesBool = naiveBayesBool
    classifier.bestModel = bestModel
    trainSplit = classifier.trainSplit(trainDir)
    classifier.train(trainSplit)
    testFile = classifier.readFile(testFilePath)
    print classifier.classify(testFile)


def main():
    stopWordsFilter = False
    naiveBayesBool = False
    bestModel = False
    (options, args) = getopt.getopt(sys.argv[1:], 'fbm')
    if ('-f', '') in options:
        stopWordsFilter = True
    elif ('-b', '') in options:
        naiveBayesBool = True
    elif ('-m', '') in options:
        bestModel = True

    if len(args) == 2 and os.path.isfile(args[1]):
        classifyFile(stopWordsFilter, naiveBayesBool, bestModel, args[0], args[1])
    else:
        test10Fold(args, stopWordsFilter, naiveBayesBool, bestModel)


if __name__ == "__main__":
    main()
