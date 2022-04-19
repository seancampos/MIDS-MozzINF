#!/bin/bash
for i in {08..30}
do
	python -W ignore predict_mids.py --dir_out /humbug-data/mids_pres_draft/MozzWearPlot/2021-04-$i/ --to_dash True /home/ivank/dbmount/MozzWear/2021-04-$i/ .aac
done
