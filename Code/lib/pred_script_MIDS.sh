#!/bin/bash
for i in {01..31}
do
	        python -W ignore predict_mids.py --dir_out /humbug-data/MozzWearPlot/2021-04-$i/ --to_dash True --batch_size 8 /home/ivank/dbmount/MozzWear/2021-04-$i/ .aac
	done
