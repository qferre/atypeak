# Detailed usage instructions

## Installation

- Run `make install`, which will create a conda environment called "atypeak".

- *If you want to use your own data*, place the input data in *./data/input_raw/* and follow the correct data format, outlined below.

- Run `make prepare` to transform the input_raw into an input readable by the model (mostly involves making an index of peak positions)

**Remarks** : The code supports TensorFlow 2, but uses 1.15 by default for reproducibility and performance reasons on legacy code. Furthermore, due to dependencies, only Linux and MacOS systems are supported.


## Running

- Set the parameters in *parameters.yaml*. Their meaning is detailed in YAML comments.

- **Set `load_saved_model` and `process_full_real_data` to False**, so you will train a model, evaluate it, and save it, but will not start processing the full data when running the model.

- Then use `make run`. The network is comparatively shallow and can be run on a CPU. Once this is done, you will have a trained model named `{cell_line}_trained_model.h5`. This will also produce Q-score and other diagnostics in the output directory.

- Look at the diagnostic plots, most notably examples of rebuilt CRMs, correlation matrices and Q-score to see if the learned correlation groups are satisfactory.

- Then **set `load_saved_model` and `process_full_real_data` to True** () **and then call `make run` again** to reload the model and process the full data.
  - Presumably if you reload a model, you have already diagnosed it and can set `perform_model_diagnosis` to False.
  - This is done because processing the real data is the most time-consuming part.

It is also possible to run the training and processing steps separately by running `make train` followed by `make process`.

*Remark*: Multithreading and GPUs are disable for training for reproducibility reasons. This should not have much consequence since tensor creation (CPU-boud) is the biggest bottleneck.


### Grid search

At the end of `train.py`, there is an example of code showing how to perform a grid search for parameter choice on our model, which can be adapted and reused.

You can use this, for example in a Jupyter kernel, to systematically test parameters for your data.

### Running for multiple parameters

*atyPeak* will always look for its parameters at the root of the directory in the `parameters.yaml` file. However, you can use `make run_all` to run the **full** analysis for all individual parameter files in the `parameters` directory. This is done by copying them at the root, running the model, then deleting them.

Outputs are named according to the cell line, so results for different cell lines are not overwritten.

Default parameters are set to artificial data, while the parameters for the real data analysis are in the aforementioned `parameters` directory.


## Data format

If you replace the provided data in `data/input_raw` with your own data, please use the following format and be sure to change the `CRM_FILE` parameter in the Makefile and the parameters YAML, as well as the `PEAKS_FILE` parameter in the Makefile.

### CRM file

Each region defined here is a candidate CRM region, and will be represented as a 3D tensor of peak presence.

The lines should be formatted like this, with all fields separated by tabulations:

```
chr1	778327	779298	crm	194	.	778690	778691
```

The first three fields are standard BED (chromosome, start, end) and give the region's coordinates. The other columns (in our example from ReMap, respectively identifier, strand, estimated summit and end of estimated summit) are ignored by the model.


### Peaks data file

The peaks present in the CRM regions defined above. The format should be the following:

```
chr1	10135	10285	GSE42390.tal1.erythroid	12.67102	.	10261
```

The first three fields are standard BED (chromosome, start, end). The fourth field (name) however **must** obey this formatting: **dataset_id.transcription_factor_name.cell_line_name**. The other fields are ignored (in our example, strand and peak center).


## Artificial data

It is possible to use artificial data instead of real data. This is mostly useful for demonstrations of principle.

In broad strokes, the artificial data is made by picking one of several possible predefined correlation groups of sources (TR+dataset pairs) once per artificial region, and then placing a stack of peaks from the sources of this group only.