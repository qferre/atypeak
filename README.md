# atyPeak

This repository presents a tool for correlating peaks (BED file regions) from multiple datasets using convolutional autoencoders for k-dimensional tensors. We present a stacked multiview approach with a comparatively shallow network.

The code is written in Keras ; the data used here are ChIP-Seq peaks from ReMap. 






MAYBE SPLIT THIS INTO SEVERAL READMEs IN A DOC ?




## Description

Quickly summarize the philosphy : few layers, stacked multiview, convolution for combis, autoencoder for compression to lose anomalies and keep good features


Please see the paper for more details. 



### Results

ReMap data has been processed, available here : www.remap/the_tab_that_Benoit_promised_me


## Usage



### Installation

Works only on linux and macos due to pybedtools and makefile

- Run `make install`, to create a conda environment. called 'atypeak'.

- Place the input data in *./data/input_raw/* and follow the correct data format, outlined below

BE SURE TO CHANGE THE CRM_FILE = remap2018_crm_macs2_hg38_v1_2_selection.bed
PEAKS_FILE = remap2018_peaks_hg38_v1_2_selection.bed PARAMETERS IN THE MAKEFILE !!!

- Run `make prepare` to transform the input_raw into an input readable by the model (mostly involves making an index of peak positions)

### Running



- Set the parameters in *parameters.yaml*. The meaning of them is detailed in YAML comments

Then use `make run`. The network is comparatively shallow and can be run on a CPU.

**Set load_saved_model and process_full_real_data to False**, so you will train a model, evaluate it with Q-score, and save it.

- Once you run the script once, you will have a trained model names trained_{cell_line}_model.h5 ; this way you can now activate use_trained_model in the parameters in the yaml.

This will also produce q-score and other diagnostics plots and info in the output directory (presumably if you relaod a model, you have already diagnosed it and can set perform_model_diagnosis to False).

Look at the diagnostic plots : MOST NOTABLY look at some examples of rebuit CRMS  to see if there are enough corr groups, and look at the q-score (see paper)



Once satisfied, **set those two parameters (load_saved_model and process_full_real_data) to True  and then use `make run` again** so that you reload the model and process the full data. Also set perform_diagnosis so you don't re-perform the diagnosis. This is done because processing the real data is the most consuming part




NOTE : I MAY HAVE CHANGED THIS.
    New paradigm is that you can still do the above, or do make train and make denoise separately ??
    Hmm given that many important diagnostic figures are done in train, maybe I do not want to leave that choice to the user





Outputs are named according to the cell line, so you can just re-run with a different cell line in parameters.yaml and not overwrite your data. the models are also saved according to this paradigm





### Data format

If you replace the provided data in ./data/input_raw with your own data, please use the following format :

And remember to replace the parameters :
CRM_FILE and
PEAKS_FILE in the Makefile
and CRM_FILE in the YAML

#### CRM file

a CRM file giving the list

All fields are separated by tabulations
```
chr1	778327	779298	crm	194	.	778690	778691
```
The first three fields are standard BED (chromosome, start, end). The name is irrelevant. The **fifth** field must be a numerical value
WAIT NO I NUMBER THEM MYSELF, THOSE ARE SCORES. SO JUST A STANDARD BED WHERE ONLY THE FIRST THREE COLUMNS ARE CONSIDERED ?


#### Peaks data file

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
- diagnostic contains various diagnostic plot grouped by cell line
- bed contains the result files, and tsvs of scores

EXPLAIN THE SIGNIFICATION OF THE VARIOUS BEDS

EXPLAIN THE SIGNIFICATION OF THE VARIOUS PLOTS (maybe in a separate Readme)

The output bed files are named according to whether they had normalization applied or not. For most use cases, use the BED file wieh "FINAL" in the filename which contains all normalizations.

All output bed FILES have the following format :
chr    start   end    dataset.tf.cell_line    atypeak_score   strand





### BED significance

Each BED has a name depending on the normalizations applied to it for the cell ine "cell_line". THe noteworthy ones are :
- cell_line.bed with raw scores
- cell_line_normalized_corr_group with our custom normalization to correct biases in rebuilding (cf. paper)
- "{cell_line}_FINAL_merged_doublons_normalized_corr_group_normalized_by_tf.bed" with the corr group norlalization and then centered and reduces by TR (transcirpiton regulator) to center scores of each TR around mean for said TR

The ones you want are the last two (elaborate more once decided.)


### Diagnostic figures significance

- `urexamples` contains for each encoded dimension a 2d (sum across X axis, the region size) CRM that would most activate it estimated by gradient ascent
- `crm_examples` contains some example CRMs. Each number is the same crm with a before (its representation), rebuilt (the output of the model) and anomaly (difference between the two, which is 3d). 2d figs are MAX across X axis

- `conv_kernels` contains the weights of all convolutional filters, for datasets and tf (only at first positino x=0 for the latter)

- scores_tf and scores_datasets give the median score for each dimension after corr group normalization

- alone_both give the distribution of scores for each TR in a give TR pair, where the other member of the pair is present or absent. This is done on the final file with corr group AND by tf normalizations

- average_crm is a 2d (averaged across X axis) average of 200*64 (depending on parameters) CRMs



Q-score
- corr_datasets_tf gives the correlation of each dimension with each other (for example, is the presence of peaks from dataset2 correlated with peaks from TF 5).
- posvar: does the presence of a peak from the line dimension increase score of the column dimension
- qscore : individual contributions of each pair to the qscore (with numerical values as a tsv)


STILL MISSING SOME

## Contributing



Please feel free to post issues on GitHub !

The paper as well as comments in main.py explain the rest


ALSO EXPLAIN THAT USAGE INSTRUCTIONS IN DETAILS ARE IN THE PAPER



## Credits



Please cite <atyPeak paper>

The raw data included here is a subset of ReMap 2018 data, source is <remap link>

This code is available under the GNU GPL3 license.



## Contact

Quentin Ferré -- quentin.ferre@gmail.com


 Acknowledgements : Jeanne Chèneby, Cécile Capponi, Benoît Ballester,  Denis Puthier
