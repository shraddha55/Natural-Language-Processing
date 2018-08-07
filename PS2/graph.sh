#!/bin/bash


var=2000


while [ $var -le 40000 ]
do
	echo "Enter"
	echo "$count"

	head -$var ptb.2-21.txt > my_temp.txt

	./train_hmm.py ptb.2-21.tgs my_temp.txt > my_try.hmm

	./viterbi.pl my_try.hmm < ptb.22.txt > my_try.out

	./tag_acc.pl ptb.22.tgs my_try.out >> graph.txt 

	wc -w my_temp.txt >> graph.txt


	var=$(( $var + 2000 ))
	echo "$var"
	(( count++ ))

done

exit 


