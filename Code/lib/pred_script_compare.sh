#!/bin/bash
for i in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
	        python predict_mids.py --dir_out /home/ivank/ComparE/mids_out/$i/ --threshold $i /home/ivank/ComparE/labelled_audio/ .aac
	done
