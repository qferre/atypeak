# atyPeak

TODO Finish this and the other md it's almost done

This is a tool to find anomalous peaks (BED file regions) in a database containing peaks from different sources (TRs or experiments), based on whether or not the peaks respect the usual correlations between sources.

We present a stacked multiview, shallow convolutional autoencoder to automate this. CRMs are represented as 3D tensors of peak presence.

We use a simple network to compress the data, so as to learn correlation groups present in the data. Some interpretability is possible to perform correlation mining.

Please see *atyPeak, Ferré et al. [TBD]* for more information.

The code is written in Keras. Our result data can be regenerated here, and is also  available here : [www.remap.fr/link_TBD]
TODO Put correct link. In fact put correct lnks everywhere



## Usage

MOVE TO USAGE.MD AND KEEP A SUMMARY HERE

In summary : 
- put files in correct places and change parameters
- prepare using make install and make prepare
- do make run and check results
- once satistfied flip process_real_data and make run

## Output significance

See in output_significance.md

Summary :

we produce diagnostic plots


Explain the contents

- model contains the saved model and parameters
- diagnostic contains various diagnostic plot grouped by cell line, explain a bit
- bed contains the result files, and tsvs of scores And in the bed files we run our model on the 3D representations and perform normalizations






## Contributing

Please feel free to post issues on GitHub !

The paper as well as comments in main.py explain the rest

ALSO EXPLAIN THAT USAGE INSTRUCTIONS IN DETAILS ARE IN THE PAPER



## Credits

Please cite [atyPeak paper].

The raw data included here is a subset of ReMap 2018 data, see [ReMap link].

This code is available under the GNU GPL3 license.



## Contact

Quentin Ferré -- quentin.ferre@gmail.com

**Acknowledgements** : Jeanne Chèneby, Cécile Capponi, Benoît Ballester, Denis Puthier
