# The learning curve plots a performance measure evaluated on a fixed test set (y-axis) 
#against the training dataset size (x-axis).

import sys,re
import matplotlib.pyplot as plt
import matplotlib.patches as patch

X_word =[48433, 95498, 144361, 193290, 241875, 288925, 337932, 385354, 432150, 479140, 525870, 574249, 621350, 669176, 716443, 762807, 810556, 859266, 906271, 950028]
Y_Words =[]
Y_Sentence = []



with open("graph.txt") as fp:
	#print fp.readline()

	for line in fp: 

		Inputstring = line

		#error rate by word:      0.0540917815389984 (2170 errors out of 40117)

		Inputlist = Inputstring.split(":")

		if Inputlist[0] == "error rate by word" :
	  		Inputlist[1].strip()
	  		ErrorList = Inputlist[1].split("(");

	  		ErrorRate = float(ErrorList[0].strip())

	  		Y_Words.append(ErrorRate)

	  	elif Inputlist[0] == "error rate by sentence" :
	  		Inputlist[1].strip()
	  		ErrorList = Inputlist[1].split("(");

	  		ErrorRate = float(ErrorList[0].strip())

	  		Y_Sentence.append(ErrorRate)
	  	else:
	  		pass
	
	        	
for i  in (1, 10):
	print("X axis is: ", X_word[i], "y-axis :", Y_Words[i])
	  		



# y axis is the error and x access is the words 

plt.plot( X_word, Y_Words , label = 'Word error' )
plt.plot( X_word, Y_Sentence , label = 'Sentence error')

plt.xlabel('Number of Words')

plt.ylabel('Error Rate')

plt.title('error rate by word!')

plt.show()

