#!/bin/bash

b='03_capstone_'

for f in `ls *.py`
do
	mv $f $b$f
	echo mv $f $b$f
	#echo $f
done
