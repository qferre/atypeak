import os, io, time, sys, subprocess

import pandas as pd
import numpy as np
import scipy.sparse as sp

# -------------------------- Preparation per cell line ----------------------- #

def get_data_indexes(cell_line, root_path):
    """
    Given a cell line, will collect :
    - the list of datsets containing information for this cell line
    - a dictionary giving, for this cell line, all posible CRM-TF couples
    and the associates lines in sorted_intersect.txt ; each will become a
    2D matrix (ie. a slice of the 3D matrix for each CRM) in our analysis.
    - a dictionary giving, for each tuple of CRM coordinates in format
    (chr<str>, start<int>, end<int>) the corresponding crm id for query
    - a dictionary giving, for each 2D matrix (cf. point 2) the number of datasets
    present (ie. the number of non-empty lines)
    - a dictionary giving the number of peaks (lines of the bed intersect) per
    dataset.
    - the list of transcription factors found in this cell line.

    And return them respectively.
    """

    # ------------------------------ Datasets -------------------------------- #
    # Read the dictionary of datasets for each cell line
    with open(root_path+'/data/input/cell_line_dataset_couples.txt') as f:
        ucld = [tuple(map(str, i.rstrip().split('\t'))) for i in f]

    d = dict()
    peaks_per_dataset = dict() # Also get the number of peaks per dataset
    for n,x,y in ucld:
        if x in d : d[x].append(y)
        else: d[x] = [y]

        peaks_per_dataset[y] = int(n)

    # Fetch the correct list of datasets for the query cell line
    cl_datasets = d[cell_line]


    # ------------------------------ Matrices -------------------------------- #
    # Read the dictionary for the appropriate cell line
    # Each element will become a matrix in our analysis, giving the list of
    # correponsing lines in sorted_intersect.txt
    cl_crm_tf_dict = dict()
    with open(root_path+'/data/input/cell_line_crm_tf_tuples/'+str(cell_line)+'.txt') as cf :
        for l in cf:
            ls = l.rstrip().split('|') # For each line : split by |
            query = tuple(ls[0].rstrip().split('\t')) # First field : split by tab and this will be the query tuple
            cl_crm_tf_dict[query] = ls[1:] # All other fields : the line numbers


    # ------------------------------ CRM number ------------------------------ #
    # Mapping each CRM ID to its coordinates, if we want to query them as such.
    crm_coord_dict = dict()
    with open(root_path+'/data/input/crm_id_coord.bed') as cic :
        for l in cic:
            ls = l.rstrip().split('\t')
            query = tuple([ls[0], int(ls[1]),int(ls[2])])
            crm_coord_dict[query] = ls[3]


    # ------------------------ Nb datasets per matrix ------------------------ #
    # How many datasets contribute to each 2D matrix (CRM/TF couple), in this cell line ?
    cl_nb_datasets_per_matrix = dict()
    with open(root_path+'/data/input/nb_datasets_per_matrix/'+str(cell_line)+'.txt') as cn :
        for l in cn:
            ls = l.rstrip().split('\t')
            query = tuple([ls[0], ls[1]])
            cl_nb_datasets_per_matrix[query] = int(''.join(reversed(ls[2])))
            # I reverse the string because the line counts were reversed in the makefile


    # ------------------------------ TFs ------------------------------------- #
    # Read the dictionary of TFs for each cell line

    with open(root_path+'/data/input/cell_line_tf_couples.txt') as f:
        ucltf = [tuple(map(str, i.rstrip().split('\t'))) for i in f]

    dt = dict()
    for x,y in ucltf:
        if x in dt : dt[x].append(y)
        else: dt[x] = [y]

    # Fetch the correct list of TFs for the query cell line
    cl_tfs = dt[cell_line]


    ### Return everything
    return cl_crm_tf_dict, cl_datasets, crm_coord_dict, cl_nb_datasets_per_matrix, peaks_per_dataset, cl_tfs



# ---------------------------- Matrix processing ----------------------------- #

def process_data_file_into_2d_matrix(subfile, datasets_names, crm_min, crm_max, use_scores = False):
    """
    This takes a datafile as read by extract_matrix and turns it into a matrix
    representing the presence or absence of a peak :
        One column per nucleotide, one line per dataset.

    This function generates a 2D matrix (only one TF), such matrices will be
    stacked later to add the third dimension.
    """

    # -------------------------- Reading the data ---------------------------- #
    # Read the bed 'file' (actually a string) info into a pandas dataframe
    df = pd.read_csv(io.StringIO(''.join(subfile)), sep='\t', header = None)

    # How many different datasets files were there ?
    # This is given as input for consistency so matrices for each cell type have
    # the same number of lines
    #cleaned_names_ori = [dataset_parent_name(n) for n in datasets_names]
    cleaned_names_ori = datasets_names
    cleaned_names = sorted(list(set(cleaned_names_ori)), key=cleaned_names_ori.index) # Make unique and preserve original order
    names_ordered = {cleaned_names[i]:i for i in np.arange(len(cleaned_names))}

    # -------------------------- Processing ---------------------------------- #

    # Initialize an empty matrix for this CRM
    shape = (len(names_ordered),crm_max-crm_min)
    X = sp.lil_matrix(shape)

    # Then, for each line of the bed file, fill in the matrix at the correct places.
    n_lines = len(df)
    for i in np.arange(n_lines):

        line = df.iloc[i,] # Stock the line in a variable for ease of reference

        # Get the name of the current dataset & Strip superfluous TF ID from ENCS* name
        #current_dataset = dataset_parent_name(line[3])
        current_dataset = line[3]
        # Get the line number for the dataset being processed
        line_of_matrix = names_ordered[current_dataset]

        # Get score and peak summit if applicable
        # TODO Use it
        if use_scores:
            score_peak = line[7]
            peak_summit = line[8]
        # Currently all peaks are read as 1
        else:
            score_peak = 1

        # Get endpos and startpos
        startpos_peak = int(line[5])
        endpos_peak = int(line[6])


        # Increase startpos if lower than min, decrease endpos if lower than max
        if startpos_peak < crm_min : startpos_peak = crm_min
        if endpos_peak > crm_max : endpos_peak = crm_max

        # Now write to the matrix
        wvec = np.repeat(score_peak,endpos_peak-startpos_peak) # The vector to be written
        X[line_of_matrix,startpos_peak-crm_min:endpos_peak-crm_min] = wvec

    # Return the result
    X_c = X.tobsr() # Convert to Block Sparse Row
    return X_c


def extract_matrix(crm_id, cell_line, all_tfs, # Parameters
                cl_crm_tf_dict, cl_datasets, crm_coordinates, # Keys to data
                datapath = './data/input/sorted_intersect/',
                return_data_lines = False):
    """
    Given a cell line, a CRM number and a list of transcription factor, will extract the
    correct lines in the data file (in datapath) to generate the corresponding
    3D tensor.

    You must also supply the cl_crm_tf_dict and cl_datasets given by prepare() so
    the function knows respectively which lines to query and how many datasets
    lines should be in the matrix.

    Returns the tensor.
    If return_data_lines is true, returns a tuple (matrix, original lines from sorted_intersect). This is useful to produce the result bed.
    """

    ### To build 3d matrices of shape (position,dataset,tf) we build 2d matrices
    ### (one per TF) and then concatenate them.

    # Get the crm beginning and end, so we may truncate the peaks if needed
    # Although a dictionary is not intended to be used this way, we can still
    # do it because all values (IDs) are unique.
    coord = [item[0] for item in crm_coordinates.items() if item[1]==crm_id][0]
    crm_min = coord[1] ; crm_max = coord[2]

    matrices = list() # List of matrices (one per TF)

    all_lines = list() # Collect all lines.

    # For each transcription factor...
    for tf in all_tfs:

        # --------------------- Build 2D matrix for this TF ------------------ #
        # If the couple (crm_id,tf) does not exist (because this CRM
        # has no peaks for this TF) return an empty matrix !
        if (crm_id, tf) not in cl_crm_tf_dict :
            #cleaned_names_ori = [dataset_parent_name(n) for n in cl_datasets]
            cleaned_names_ori = cl_datasets
            cleaned_names = sorted(list(set(cleaned_names_ori)), key=cleaned_names_ori.index)
            names_ordered = {cleaned_names[i]:i for i in np.arange(len(cleaned_names))}
            zero_matrix = sp.csr.csr_matrix((len(names_ordered),crm_max-crm_min))
            matrices.append(zero_matrix)

        else:

            # First, query the correct lines in the large data file and build the BED
            lines = cl_crm_tf_dict[(crm_id,tf)]
            lines = [int(x) for x in lines]

            # sorted_intersect.txt has been split into chunks of 100k. For each line to fetch,
            # get the correct file and the correct line number in the file.
            subfile = list()
            for l in lines:
                file_id = str(l // 10000).zfill(6) # Left pad
                new_line_number = l - 10000*int(file_id)

                with open(datapath+'/sorted_intersect_'+file_id,"r") as fp:
                    for i, line in enumerate(fp):
                        if i == new_line_number: subfile.append(line)

            # Now get the list of all datasets for the considered cell line and extract the matrix
            matrix = process_data_file_into_2d_matrix(subfile, cl_datasets, crm_min, crm_max)
            matrices.append(matrix)

            all_lines = all_lines + subfile # Add the subfile's lines to all_lines


    # Now, concatenate the list of 2D matrices into a 3D matrix
    # NOTE matrices must be transposed because they are originally in (dataset,length) shape
    matrices_dense = [np.transpose(matrix.toarray()) for matrix in matrices]
    result = np.dstack(matrices_dense)

    # Return result, and data lines, and CRM start point if requested.
    if return_data_lines : return (result, all_lines, crm_min)
    else : return result

    # We will then concatenate all the subfiles as they are generated and return them if asked
    # (ie. if extract matrix is used by itself during denoising, not by the generator)
