# Output significance

The contents of the output directory are as follows:

- `model` contains the saved models and parameters,
- `diagnostic` contains various diagnostic figures grouped by cell line,
- `bed` contains the result files, obtained by running our model on all the candidate CRM and performing normalizations.


## Diagnostic figures

By directory:

- `urexamples_encoded_dim` contains, for each variable in the encoded dimension, a 2D (sum across X axis, the region size) CRM that would most activate it. Estimated by gradient ascent.

- `crm_examples` contains some example CRMs. Each number is the same given CRM : for each there is a "before" (its tensor representation), "rebuilt" (the output of the model) and "anomaly" (difference between the two, which is 3D). 2D figs are maximum across X axis (region size).

- `conv_kernels` contains the weights of all convolutional filters, for datasets and TFs (only at the first position X=0 for the latter).

- `tf_pairs_alone_both` gives the distribution of scores for each TF in a given TF pair, where the other member of the pair is present ('both') or absent ('alone'). This is done on the final file with both correlation-group AND TF normalizations. Concerned TFs are defined in parameters.

- Similarily, `estimated_corr_groups` gives estimated correlation groups for sources defined in parameters (a source is a TF + dataset pair). Estimation is done by putting only a peak for this source.

- `data_stats` contains information on the raw data (not treated by the model) calculated on the 3D tensor representations of the CRMs:
  - `average_crm` is a 2d average (across X axis) of many CRMs (that number is *nb of diagnostic batches \* batch size*),
  - Correlation coefficients (R) between dimensions,
  - Jaccard index between TFs,
  - Abundance (number of nucleotides) for peaks in each dimension.

- `distribution_score` contains, for both raw data (produced by the model) and data after final normalization:
  - Mean and max score depending on number of peaks in a CRM ("density"),
  - Score distribution,
  - Median by dataset and tf after normalization,
  - Mean score by TF before (raw data) and after correlation-group normalization (but before the by-TF normalization producing the final file); this is done to evaluate the quality of normalization.

- `median_anomaly_score` gives the median anomaly score per source over many CRMs.

- `q-score` contains several diagnostics.
  - `correlation_matrix_dims` gives the correlation of each dimension with each other (for example, is the presence of peaks from dataset 2 correlated with peaks from TF 5). This is a reference and done on input data.
  - `posvar_when_both_present`: does the presence of a peak from the X dimension increase model score of the Y dimension ? 1 if yes. If the model is properly calibrated, correlations should have an impact on the score and this should look like the correlation matrix above.
  - `Q-score` is the individual contributions of each pair to the Q-score (with numerical values as a tsv, which is roughly the difference between the correlation and posvar plots.

- `normalization_factors` give the correlation-group normalization factors (before the final by-TF normalization).

## BED files

Each BED has a name depending on the normalizations applied to it, for the requested cell line. THe noteworthy ones are :
- `{cell_line}.bed` with raw scores. There is also a version with merged doublons.
- cell lines merged doublons
- `{cell_line}_merged_doublons_normalized_corr_group` with our custom, correlation-group based normalization to correct biases in rebuilding. More details in the paper ; to summarize, we correct for the differences in abundance of the learned correlation groups, as each individual gets a value of 1 only when the group is full.
- `{cell_line}_FINAL_merged_doublons_normalized_corr_group_normalized_by_tf.bed` takes the above BED and centers-reduces the score around 750 (to look more at the less-complete-than-usual cases); this is done separately so the scores of each TR are centered around the mean for said TR.

The directory also contains TSV files giving statistics on scores for this file by dimension.

All output BED files have the following format:

```
chromosome    start   end    dataset.tf.cell_line    atypeak_score   strand
```

In most cases, the one you want is the *FINAL* one containing all normalizations.
