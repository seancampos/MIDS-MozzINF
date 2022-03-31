#!/bin/bash
for i in {09..30}
do
	        python -W ignore predict_mids.py --dir_out /humbug-data/midsv2_medfilt_5/MozzWearPlot/2021-04-$i/ --to_dash True --batch_size 16 /home/ivank/dbmount/MozzWear/2021-04-$i/ .aac
	done
