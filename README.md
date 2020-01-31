# atyPeak-SCAEN

atyPeak - Stacked Convolutional Autoencoder Neural network

This repository presents a tool for correlating peaks (BED file regions) from multiple datasets using convolutional autoencoders for k-dimensional tensors.

The code is written in Keras.

We use it on ChIP-Seq peaks from ReMap. We present a stacked multiview approach with a comparatively shallow network.

Please see the paper for more details.


## Description

Quickly summarize the philosphy : few layers, stacked multiview, compression to lose anomalies and keep good features


### Results availability

ReMap data has been processed, available here : www.remap/the_tab_that_Benoit_promised_me

### Data source

The raw data included here is a subset of ReMap 2018 data, source is <remap link>



## Usage

TOBE BE MORE DESCRIPTIVE !

- Run `make install`, to create a conda environment.

- Place the input data in *./data/input_raw/* and follow the correct data format, outlined in ./data/input_raw/data_format.md

BE SURE TO CHANGE THE CRM_FILE = remap2018_crm_macs2_hg38_v1_2_selection.bed
PEAKS_FILE = remap2018_peaks_hg38_v1_2_selection.bed PARAMETERS IN THE MAKEFILE !!!

- Run `make prepare` to transform the input_raw into an input readable by the model (mostly involves making an index of peak positions)

- Set the parameters in *parameters.yaml*

Then use `make run`. The network is comparatively shallow and can be run on a CPU, but we still recommend a GPU.

- Once you run the script once, you will have a trained model names trained_{cell_line}_model.h5 ; this way you can now activate use_trained_model in the parameters.

BETTER IDEA : run make train which will give you a trained model with corresponding q-scoresYou can also use pythonwrappers there for grid search
Once satisfied, run make process which will reoad the model and process the data
  I do this manually finally



To sum up :

### Running


Say that you should run some tries, find a model you like, and THEN :
    - flip the load_saved_model switch to True (the model is automatically saved after training). not mandatory, if you don't you'll simply re-train the model with the current parameters.
        TODO MAKE IT HAPPEN AND SAY SO IN README : I believe if load_saved_model is true, we should skip the diagnosis steps like the Q_score
    - flip the process_full_date to True

Set the model parameters in parametrs.yaml. Set load_saved_model and process_full_real_data to False, so you will train a model, evaluate it with Q-score, and save it.
Once satisfied, set those two parameters to True so that you reload the model and process the full data. Also set perform_diagnosis so you don't re-perform the diagnosis.







### Parameters

Parameters such as :
- whether this is a test on artificil data or actual processing of real data
- which cell line is being processed
- whether to train a model or use one that has already been trained

are fixed in the parameters.yaml before a run. main.py will read this file and run its code.


If you run artificial data, it will produce some more diagnostic plots ?



Outputs are named according to the cell line, so you can just re-run with a different cell line in parameters.yaml and not overwrite your data. the models are also saved according to this paradigm






### Data format

If you replace the provided data in ./data/input_raw with your own data, please use the following format :

#### CRM file

a CRM file giving the list

All fields are separated by tabulations
```
chr1	778327	779298	crm	194	.	778690	778691
```
The first three fields are standard BED (chromosome, start, end). The name is irrelevant. The **fifth** field must be a numerical value
WAIT NO I NUMBER THEM MYSELF, THOSE ARE SCORES. SO JUST A STANDARD BED WHERE ONLY THE FIRST THREE COLUMNS ARE CONSIDERED ?


#### peak file

```
chr1	10135	10285	GSE42390.tal1.erythroid	12.67102	.	10261
```
The first three fields are standard bed (chromosome, start, end)

The fourth name field **must** obey this formatting :

dataset_id.transcription_factor_name.cell_line_name


The other fields are discarded.


### Output

Explain the contents

- model contains the saved model and parameters
- diagnostic contains various diagnostic plot
- bed contains the result files, and tsvs of scores

EXPLAIN THE SIGNIFICATION OF THE VARIOUS BEDS

The one you want is "{cell_line}_FINAL_merged_doublons_normalized_corr_group_normalized_by_tf.bed"

The paper as well as comments in main.py explain the rest

## Contributing



Please feel free to post issues on GitHub !





## Credits



Please cite XXXXXXXXXXXXXXXXXXXXX



ALSO EXPLAIN THAT USAGE INSTRUCTIONS IN DETAILS ARE IN THE PAPER


Add at least link to preprint, then link to true paper and links to remap papers. And link back to my remap tab of course




ReMap data :  http://tagc.univ-mrs.fr/remap/



This code is available under the GNU GPL3 license.



## Contact

Quentin Ferré -- quentin.ferre@gmail.com


 Acknowledgements : Jeanne, Cécile Capponi, Benoît Ballester, maybe also Denis
