# atyPeak

This repository presents a tool to find anomalous peaks (BED file regions) in a database containing peaks from different sources (Transcription Factors and/or experiments), based on whether or not the peaks respect the usual correlations between sources.

We present a stacked multiview, shallow convolutional autoencoder to do this. CRMs are represented as 3D tensors of peak presence. We use a simple network to compress the data, so as to learn correlation groups present within. Some interpretability is possible to perform combination mining.

Please see *atyPeak, Ferré et al. [TBD]* for more information.

The code is written in Keras. Our result data can be regenerated here, and is also available here: 

- https://github.com/qferre/atypeak-files/
- Mirrored at http://remap2020.univ-amu.fr/download_page 

ChIP-seq can be obscured by data anomalies and biases. *atyPeak* is a deep-learning based method to identify ChIP-seq peaks that have “atypical profiles”, meaning that those peaks are found without their usual collaborators. "Collaborators" are defined as the other Transcriptional Regulators (or corroborating datasets) which are usually found in the same neighborhood, in the Cis Regulatory Elements of this cell line. Peaks get a higher score when more of their usual collaborators are present.


## Usage

Detailed usage instructions can be found in `doc/usage.md`.

To immediately re-process our data with the given parameters, simply run `make install` followed by `make prepare` and `make run_all`.

Here is a summary of the usage steps:

- Prepare the environment using `make install` and `make prepare`,
- Customize the `parameters.yaml` file,
- Call `make run` and check the results using the diagnostics,
- Once satistfied set `process_full_real_data` to True and call `make run` again.

## Output significance

Once the analysis is run, output can be found in `data/output`. See `doc/output_significance.md` for more details.

Generally, anomaly score thresholds and interpretation are at the user’s discretion. Anomalies usually represent noisy peaks. However, a focused study of a single experimental series may rely on low-scoring peaks as they might be caused by certain events of interest. Regions with a high density of high-scoring peaks are the strongest candidate CREs.

## Contributing

Please feel free to post issues on GitHub and fork the code. Details about our approach can be found in the paper and in comments along the Python code.

## Credits

Please cite *atyPeak, Ferré et al. [TBD]*.

The raw data included here is a subset of ReMap 2018 data.

This code is available under the GNU GPL3 license.

**Contact** : Quentin Ferré -- quentin.ferre@gmail.com

**Acknowledgements** : Jeanne Chèneby, Cécile Capponi, Benoît Ballester, Denis Puthier
