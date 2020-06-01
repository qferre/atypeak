MAKEFILE = Makefile
WDIR = ${PWD}
SHELL = /bin/bash

# ------------------------------ Parameters ---------------------------------- #

# The CRM matrices should have this size (for makewindows only; final padding will be done in Python later)
CRM_WINDOW_SIZE = 3200

# File names (relative to `input_raw` directory) after decompression
CRM_FILE = remap2018_crm_macs2_hg38_v1_2_selection.bed
PEAKS_FILE = remap2018_peaks_hg38_v1_2_selection.bed

# ---------------------------------------------------------------------------- #

.ONESHELL: # Run all in one shell


## Preparation

install:
	conda init
	# Create the conda environment we need
	conda env create -f env.yaml


# Turn the raw input_data into something readable by the model
prepare: decompress copybin intersect convert dictionaries split


## Running

# Run the model with the parameters specified in parameters.yaml
run: train process

# Model training
train:
	conda activate atypeak
	python3 train.py

# Reload trained model and process the full data
process:
	conda activate atypeak
	python3 process.py

# Make an individual run (train and process) for all parameter files in parameters_multi
run_all:
	# Save old parameters if present
	[[ ! -f parameters.yaml ]] || mv parameters.yaml parameters.yaml.old	
	# Temporarily replace parameters and Run the FULL analysis for each
	for p in parameters/* ;	do
		cp $$p parameters.yaml 

		echo "############>>------ $${p} IN PROGRESS ------<<############"
		make run	
		
		rm parameters.yaml
	done
	# Restore parameters
	[[ ! -f parameters.yaml.old ]] || mv parameters.yaml.old parameters.yaml	



## Cleaning

clean:
	# Remove all data and results
	rm -rf ./data/input/
	rm -rf ./data/output/
	rm -rf ./data/input_raw/*.bed
	# We however keep the compressed input_raw

clean_env:
	conda remove --name atypeak --all



##############################		RULES		################################

decompress:
	cd data

	# Prepare directories for processed input data and for output
	mkdir input
	mkdir output
	mkdir output/model # For the models
	mkdir output/bed # For the bed files with scores
	mkdir output/diagnostic # For analysis of result

	## Decompress data, keep originals
	cd input_raw
	unxz *.xz -k


copybin:
	# Copy the now-decompressed files
	cp data/input_raw/${CRM_FILE} data/input/
	cp data/input_raw/${PEAKS_FILE} data/input/

	# Divide the CRMs into bins of size n (defined in variables)
	cd data/input
	bedtools makewindows -b ${CRM_FILE} -w ${CRM_WINDOW_SIZE} > crm_split.bed
	rm -rf ${CRM_FILE}


intersect:
	cd data/input

	# Add an ID to the CRMs
	awk 'BEGIN {FS=OFS="\t"} {$$4=($$4"crm-"FNR)}1' crm_split.bed > crm.bed

	# Intersect
	bedtools intersect -wa -wb \
	  -a crm.bed \
	  -b ${PEAKS_FILE} \
	  -filenames -sorted > remap2018_crm_all_intersected_macs2_hg38_v1_2.bed

	# Remove obsolete files
	rm -rf ${PEAKS_FILE}
	rm -rf crm_split.bed

	# We keep a dictionary of CRM coordinates to ID for later use
	cut -f1,2,3,4 crm.bed > crm_id_coord.bed
	rm -rf crm.bed


convert:
	### Reorganize the intersect.bed into a text file with, in order :
	# Cell_line, CRM id, TF, dataset, peak coordinates, score, summit
	# NOTE Strand is discarded here, as all peaks have '.' strand
	cd data/input
	awk 'BEGIN {OFS = "\t"}; {gsub(/\./,"\t",$$8); print}' remap2018_crm_all_intersected_macs2_hg38_v1_2.bed |\
	awk 'BEGIN {OFS = "\t"}; { print $$10,$$4,$$9,$$8,$$5,$$6,$$7,$$11,$$13}' |\
	sort -k1,1 -k2,2V -k3 -k4 -k5,7 --parallel=8 > sorted_intersect.txt

	# Clean up
	rm -f remap2018_crm_all_intersected_macs2_hg38_v1_2.bed


dictionaries:
	cd data/input
	### We also need a dictionary of cell_line/dataset associations, generate it
	# This also contains the number of intersections per dataset
	cut -f1,4 sorted_intersect.txt | sort --parallel=8 | uniq -c | sed "s/^[ \t]*//" | tr ' ' '\t' > cell_line_dataset_couples.txt

	### As well as a dictionary of cell_line/TF associations, so generate it too.
	cut -f1,3 sorted_intersect.txt | sort --parallel=8 | uniq > cell_line_tf_couples.txt

	### And one of cell_line/CRM/TF tuples
	# WARNING : add the line numbers before sorting ! Furthermore, we write FNR-1 so it will be zero-based !
	cut -f1,2,3 sorted_intersect.txt | awk 'BEGIN {OFS = "\t"}; {print $$0,"|" FNR-1}' |\
	sort -t"|" --parallel=8 > cell_line_crm_tf_tuples_untreated.txt
	# Each unique tuple in here will be a unique 2D slice of a matrix so add the line numbers.
	# This way, later, we can quickly know which lines to read in the file to generate the matrices
	# WARNING : The following awk command will produce nonsensical results on non-sorted files !
	awk -F"|" '$$1==i{a=a"|"$$2}$$1!=i{ print i"|"a; a=$$2}{i=$$1}END{print i"|"a}' cell_line_crm_tf_tuples_untreated.txt > cell_line_crm_tf_tuples.txt
	tail -n +2 cell_line_crm_tf_tuples.txt > clcrmtf.tmp && mv clcrmtf.tmp cell_line_crm_tf_tuples.txt # Remove garbage first line from crm_tf_couples.txt

	### We also need, for each of those matrices, the number of datasets which will provide a peak
	cut sorted_intersect.txt -f1,2,3,4 | uniq |\
	rev | uniq -f1 -c | rev | tr ' ' '\t' | cut -f1,2,3,5 > nb_datasets_per_matrix.txt
	# WARNING the line counts are reversed due to the commands used here (they will be reversed again in Python processing)
	# Would be used for weighted losses.

	# Clean up
	rm -f cell_line_crm_tf_tuples_untreated.txt


split:
	cd data/input

	# And finally, split by cell line because we treat each of them independantly
	mkdir cell_line_crm_tf_tuples
	mkdir nb_datasets_per_matrix
	awk 'BEGIN {FS = OFS = "\t"}; { print $$0 >> "cell_line_crm_tf_tuples/"$$1".txt"}' cell_line_crm_tf_tuples.txt
	awk 'BEGIN {FS = OFS = "\t"}; { print $$0 >> "nb_datasets_per_matrix/"$$1".txt"}' nb_datasets_per_matrix.txt

	# Remove the now-useless first column from each of these new files
	for file in ./cell_line_crm_tf_tuples/*
	do
	 	cut -f2- "$$file" > "$$file".tmp && mv "$$file".tmp "$$file"
	done
	for file in ./nb_datasets_per_matrix/*
	do
	 	cut -f2- "$$file" > "$$file".tmp && mv "$$file".tmp "$$file"
	done

	# Now that sorted_intersect.txt is no longer needed, we split it into chunks
	# of 10K lines, ordered, for easy access later
	mkdir sorted_intersect
	mv sorted_intersect.txt ./sorted_intersect/sorted_intersect.txt
	cd sorted_intersect
	split -l 10000 -a 6 -d sorted_intersect.txt "sorted_intersect_" # Left padding with -a
	rm sorted_intersect.txt
	cd ..

	### Clean up
	rm -f cell_line_crm_tf_tuples.txt
	rm -f nb_datasets_per_matrix.txt