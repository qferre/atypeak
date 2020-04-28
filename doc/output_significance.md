# Output significance

## Diagnostic figures

- `urexamples_encoded_dim` contains for each encoded dimension a 2d (sum across X axis, the region size) CRM that would most activate it estimated by gradient ascent

- `crm_examples` contains some example CRMs. Each number is the same crm with a before (its representation), rebuilt (the output of the model) and anomaly (difference between the two, which is 3D). 2d figs are MAX across X axis

- `conv_kernels` contains the weights of all convolutional filters, for datasets and tf (only at first positino x=0 for the latter)

- scores_tf and scores_datasets give the median score for each dimension after corr group normalization

- tf_pairs_alone_both give the distribution of scores for each TR in a give TR pair, where the other member of the pair is present ('both') or absent ('alone'). This is done on the final file with corr group AND by tf normalizations
concerned trs are defined in parameters

- in the same vein `estimated_corr_groups` for sources defined in parameter (a source is a TF+dataset pair)
    estimation is done by putting only a peak for this source and outputting the rebuilt max (or is it mean ?) across x axis


- in the data_stats directory
  - average_crm is a 2d (averaged across X axis) average of 200*64 (depending on parameters) CRMs
        average_crm_fig.savefig(plot_output_path+'average_crm_2d.pdf')
        tf_corr_fig.savefig(plot_output_path+'tf_correlation_matrix.pdf')
        dataset_corr_fig.savefig(plot_output_path+'dataset_correlation_matrix.pdf')
        tf_abundance_fig.savefig(plot_output_path+'tf_abundance_total_basepairs.pdf')
        dataset_abundance_fig.savefig(plot_output_path+'dataset_abundance_total_basepairs.pdf')
BUG ! HAVE NOT SEEN THEM !


- distribution_score contains, for both raw data (produced by the model) and data after final normalization
  - the average and max score depending on number of peaks in a CRM ("density")
  - score distribution
  - median by dataset and tf after normalization
  - mean score by tf before (raw data) and after corr group normalization (but before tf normalization producing FINAL file) ; this is done to evaluate the normalization









- `median_anomaly_score` gives  median score per source (tf+dataset pair) monte carlo

-


Q-score
- correlation_matrix_dims gives the correlation of each dimension with each other (for example, is the presence of peaks from dataset2 correlated with peaks from TF 5).
- posvar_when_both_present: does the presence of a peak from the line dimension increase score of the column dimension
- qscore : individual contributions of each pair to the qscore (with numerical values as a tsv)
    """Those plots give respectively the Q-score contribution of each pair 
     (lower is better), the true correlation matrix for comparison, and
     the third plot says whether the presence of both results in a change in 
     score and should "look like" the correlation plot."""

STILL MISSING SOME


normalization_factors.tsv give the corr group normalization factors (before final tf normalization)









## BED files


Each BED has a name depending on the normalizations applied to it for the cell ine "cell_line". THe noteworthy ones are :
- cell_line.bed with raw scores
- cell lines merged doublons
- cell_line_merged_doublons_normalized_corr_group with our custom normalization to correct biases in rebuilding (cf. paper)
- "{cell_line}_FINAL_merged_doublons_normalized_corr_group_normalized_by_tf.bed" with the corr group norlalization and then centered and reduces by TR (transcirpiton regulator) to center scores of each TR around mean for said TR

The ones you want are the last two (elaborate more once decided.)



The output bed files are named according to whether they had normalization applied or not. For most use cases, use the BED file wieh "FINAL" in the filename which contains all normalizations.

All output bed FILES have the following format :
chr    start   end    dataset.tf.cell_line    atypeak_score   strand




The directory also contains tsv with the name {file}_scores_datasets or scored_tf, giving stats on scores for this file by dimension