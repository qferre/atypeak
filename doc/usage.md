# Detailed usage instructions

## Installation

Only Linux and MacOS systems are supported.

- Run `make install`, which will create a conda environment called "atypeak".

- *If you want to use your own data*, place the input data in *./data/input_raw/* and follow the correct data format, outlined below. 

- Run `make prepare` to transform the input_raw into an input readable by the model (mostly involves making an index of peak positions)

**Remark** : the code supports TensorFlow 2, but uses 1.14 by default (for performance reasons and historical and reproducibility reasons ?? Confirm that before I write it, but it seems so) write that in the article too ! jsut the first part about how I dev on 1.14 but support 2

## Running

- Set the parameters in *parameters.yaml*. Their meaning is detailed in YAML comments.

- **Set `load_saved_model` and `process_full_real_data` to False**, so you will train a model, evaluate it with Q-score, and save it.

- Then use `make run`. The network is comparatively shallow and can be run on a CPU. Once this is done, you will have a trained model named `{cell_line}_trained_model.h5`. This will also produce Q-score and other diagnostics plots and info in the output directory (presumably if you relaod a model, you have already diagnosed it and can set perform_model_diagnosis to False).

Look at the diagnostic plots : MOST NOTABLY look at some examples of rebuit CRMS  to see if there are enough corr groups, and look at the q-score (see paper)

- Once satisfied, **set those two parameters (load_saved_model and process_full_real_data) to True  and then use `make run` again** so that you reload the model and process the full data. Also set perform_diagnosis so you don't re-perform the diagnosis. This is done because processing the real data is the most consuming part

It is also possible to run the training and processing steps separately by running `make train` followed by `make process`.

Outputs are named according to the cell line, so you can just re-run with a different cell line in parameters.yaml and not overwrite your data. the models are also saved according to this paradigm


### Multiple run

atyPeak will always look for its parameters at the root of the directory in the `parameters.yaml` file. However, you can use `make run_all` to run the **full** analysis for all individual parameter files in the `parameters` directory. This is done by copying them at the root, running the model, then deleting them.

Default params are set to artificial, others are in the parameters directory


## Data format

If you replace the provided data in ./data/input_raw with your own data, please use the following format and

Be sure to change the `CRM_FILE` parameter in the Makefile and the parameters YAML, as well as the `PEAKS_FILE` parameter in the Makefile.


#### CRM file

Each region defined here is a candidate CRM region and will be represented as a 3D tensor of peak presence.

a CRM file giving the list

All fields are separated by tabulations
```
chr1	778327	779298	crm	194	.	778690	778691
```
The first three fields are standard BED (chromosome, start, end). The name is irrelevant. The **fifth** field must be a numerical value
WAIT NO I NUMBER THEM MYSELF, THOSE ARE SCORES. SO JUST A STANDARD BED WHERE ONLY THE FIRST THREE COLUMNS ARE CONSIDERED ?


#### Peaks data file

The peaks present in the regions defined above.

```
chr1	10135	10285	GSE42390.tal1.erythroid	12.67102	.	10261
```
The first three fields are standard bed (chromosome, start, end)

The fourth name field **must** obey this formatting :

dataset_id.transcription_factor_name.cell_line_name


The other fields are discarded.