import sys
import numpy as np
from scipy.io import mmread
from collections import OrderedDict
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cosine
import argparse
import configparser


def input_parse():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        action='store',
        help="This option allows you to specify a config file.",
        required=False
    )
    
    input_args = parser.parse_args()
    config_filename = input_args.config
    config = configparser.ConfigParser()
    config.read(input_args.config)

    return config


def mtx_read(mtx_file):
    '''
    Reads mtx file and returns numpy array
    '''
    # This takes a few minutes
    count_mtx = mmread(mtx_file)
    count_array = count_mtx.toarray()
    return count_array


def pseudotime_read(pt_file):
    '''
    Opens pseudotime dataframe file and returns a dictionary containing 
    the cell ID's (UMI) (key) and the pseudotime value (value).
    '''
    with open(pt_file, 'r') as ptf:
        header = ptf.readline()
        cell_pt = OrderedDict()
        i = 0
        for cell_line in ptf:
            i += 1
            try:
                idx, umi, pt = cell_line.strip().split(',')
                umi = str(umi.strip('"'))
                cell_pt[umi] = float(pt)
            except:
                print(f"WRONG FORMAT: {cell_line}", file=sys.stderr)
    
    print(f"File {pt_file} length: {i}\n") 
    # {UMI1: pseudotime1, UMI2: pseudotime2, ...}
    return cell_pt


def index_read(id_file):
    '''
    Opens file containing list of cell ID's and returns a dictionary that 
    indexes each of the ID's so that the index can be used to reference 
    the corresponding column in the count matrix.
    '''
    umi_indexes = {}
    with open(id_file, 'r') as idf:
        for i, cell_id_line in enumerate(idf):
            umi_indexes[cell_id_line.strip()] = i

    # {UMI1: col_index1, ...}
    return umi_indexes


def count_extract(count_array, cell_pt, cell_idx):
    '''
    Extracts the columns of mtx that correspond to the UMI's in the pseudo
    time dictionary. Mtx columns are connected to UMI's via UMI index dict. 
    '''
    filter_mtx = OrderedDict()

    for umi, pt in cell_pt.items():
        index = cell_idx[umi] 
        # count array dimensions #gene x #cells (counts[gene,cell])
        cell_counts = count_array[:,index]
        filter_mtx[(pt, umi)] = cell_counts

    return filter_mtx


def inclusion_check(a, b):
    '''
    Checks that all members of <a> are contained in <b> and returns those 
    elements that aren't.
    '''
    absent = [e for e in a if e not in b]
    return absent


def dtw_trajectory(count_dict1, count_dict2, dist=None):
    '''
    Pulls count arrays from the two count dictionaries (assumes dicts are 
    properly sorted) and runs fastdtw on them.
    '''
    x = np.array(list(count_dict1.values()))
    y = np.array(list(count_dict2.values()))
    distance, path = fastdtw(x, y, dist=dist)
    return distance, path


def output_write(
        output_file, 
        distance, 
        path, 
        young_pt_file, 
        aged_pt_file,
        young_dat,
        aged_dat
    ):

    with open(output_file, 'w') as out:
        out.write(f"#Pseudotime files (young, aged): {young_pt_file}, {aged_pt_file}\n")
        out.write(f"#distance={distance}\n")
        out.write("#" + "\t".join(['Y_idx','A_idx','Y_pt','A_pt','Y_umi','A_umi']) + "\n")
        out.write(f"#young\taged\n")

        for i in path:
            y_index = i[0]
            a_index = i[1]
            y_pt, y_umi = list(young_dat.keys())[y_index]
            a_pt, a_umi = list(aged_dat.keys())[a_index]
            out_values = [str(y_index), str(a_index), str(y_pt), str(a_pt), y_umi, a_umi]
            out.write("\t".join(out_values) + "\n")


def main():
    inputs = input_parse()

    files_dir = inputs['IO']['dir']

    young_count_file = files_dir + inputs['COUNTS']['young']
    aged_count_file = files_dir + inputs['COUNTS']['aged']

    young_cell_idx_file = files_dir + inputs['INDEX']['young']
    aged_cell_idx_file = files_dir + inputs['INDEX']['aged']

    young_pt_file = files_dir + inputs['PSEUDOTIME']['young']
    aged_pt_file = files_dir + inputs['PSEUDOTIME']['aged']

    output_file = files_dir + inputs['IO']['output']
    
    dist_metric = inputs['PARAMS']['distance']

    print("Loading young MTX file...")
    mtx_young = mtx_read(young_count_file)
    print("Loading aged MTX file...")
    mtx_aged = mtx_read(aged_count_file)
    print("Loading successful...")

    indx_vals_young = index_read(young_cell_idx_file)
    indx_vals_aged = index_read(aged_cell_idx_file)
    print(f"IDX 1 length: {len(indx_vals_young.keys())}")
    print(f"IDX 2 length: {len(indx_vals_aged.keys())}")
    
    young_pt_vals = pseudotime_read(young_pt_file)
    print(f"Young PT length: {len(young_pt_vals.keys())}")

    aged_pt_vals = pseudotime_read(aged_pt_file)
    print(f"Aged PT length: {len(aged_pt_vals.keys())}")

    missing_young = inclusion_check(young_pt_vals.keys(), indx_vals_young.keys())
    missing_aged = inclusion_check(aged_pt_vals.keys(), indx_vals_aged.keys())
    print(f"WARNING, NOT IN INDEX: {len(missing_young)+len(missing_aged)}")

    # Extract young and aged cell data from MTX object
    young_dat = count_extract(mtx_young, young_pt_vals, indx_vals_young)
    aged_dat = count_extract(mtx_aged, aged_pt_vals, indx_vals_aged)

    # Sort young and aged cell data by pseudotime, key=(umi,pt)
    young_dat = OrderedDict(sorted(young_dat.items(), key=lambda kv: kv[0][0]))
    aged_dat = OrderedDict(sorted(aged_dat.items(), key=lambda kv: kv[0][0]))
    
    if dist_metric == "euclidean":
        dist_metric = euclidean
    elif dist_metric == "cosine":
        dist_metric = cosine
    else:
        raise TypeError("Distance metric must be either 'euclidean' or 'cosine'.")

    #distance, path = dtw_trajectory(young_dat, aged_dat, dist=euclidean)
    distance, path = dtw_trajectory(young_dat, aged_dat, dist=dist_metric)
    
    output_write(
        output_file, 
        distance, 
        path, 
        young_pt_file, 
        aged_pt_file, 
        young_dat, 
        aged_dat
    )


if __name__ == '__main__':
    main()


#    count_file = "./expression_matrix.mtx"
#    cell_idx_file = "./unique_colnames.txt"
##    young_pt_file = "./young_subsampled_umap_pseudotime_df.csv"
##    aged_pt_file = "./aged_subsampled_umap_pseudotime_df.csv"
##    young_pt_file = "./young_branch1_NOTsubsampled_umap_pseudotime_df.csv"
##    aged_pt_file = "./aged_branch1_NOTsubsampled_umap_pseudotime_df.csv"
#    young_pt_file = "./young_branch2_NOTsubsampled_umap_pseudotime_df.csv"
#    aged_pt_file = "./aged_branch2_NOTsubsampled_umap_pseudotime_df.csv"
#    output_file = "./DTW_OUTPUT_NOT_SUBSAMP_BRANCH2.txt"

#    print("Loading MTX file...")
#    mtx = mtx_read(count_file)
#    print("Loading successful...")
#
#    indx_vals = index_read(cell_idx_file)
#    print(f"IDX length: {len(indx_vals.keys())}")
#    
#    young_pt_vals = pseudotime_read(young_pt_file)
#    print(f"Young PT length: {len(young_pt_vals.keys())}")
#
#    aged_pt_vals = pseudotime_read(aged_pt_file)
#    print(f"Aged PT length: {len(aged_pt_vals.keys())}")
#
#    missing_young = inclusion_check(young_pt_vals.keys(), indx_vals.keys())
#    missing_aged = inclusion_check(aged_pt_vals.keys(), indx_vals.keys())
#    print(f"WARNING, NOT IN INDEX: {len(missing_young)+len(missing_aged)}")
#
#    # Extract young and aged cell data from MTX object
#    young_dat = count_extract(mtx, young_pt_vals, indx_vals)
#    aged_dat = count_extract(mtx, aged_pt_vals, indx_vals)
#
#    # Sort young and aged cell data by pseudotime, key=(umi,pt)
#    young_dat = OrderedDict(sorted(young_dat.items(), key=lambda kv: kv[0][0]))
#    aged_dat = OrderedDict(sorted(aged_dat.items(), key=lambda kv: kv[0][0]))

