#!/usr/bin/env python3

import os, math, sys, time, fileinput, re, glob
import subprocess
from subprocess import Popen, PIPE
# from local import *
import argparse
import numpy as np
import operator

import nibabel as nib
from nibabel.processing import resample_to_output as resample
from mo_dots import wrap, Data
import pandas as pd

def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)

def get_nii_hdr_affine(t1w_fname):
    nifti = nib.load(t1w_fname)
    shape = nifti.get_data().shape
    header = nib.load(t1w_fname).header
    affine = header.get_qform()
    return nifti, header, affine, shape

def process_dense_CRF(id, args, root_dir='/host/hamlet/local_raid/data/ravnoor/01_Projects/12_deepMask/src_T1/catanzaro'):

    # filespec="_t1w_*native_brain.nii.gz"
    # t1w_fname = glob.glob(os.path.join(root_dir, id, 'processed', id+filespec))[0]
    filespec = '_t1_final.nii.gz'
    t1w_fname = os.path.join(root_dir, id+filespec)
    print(id)
    start_time = time.time()
    case_id = id

    dst = os.path.join(args.basedir, case_id)
    outfile = os.path.join(dst, case_id+"_denseCrf3dSegmMap.nii.gz")

    _, header, affine, out_shape = get_nii_hdr_affine(t1w_fname) # load original input with header and affine

    print("save {}".format(case_id))
    if not os.path.exists(dst):
        os.makedirs(dst, exist_ok=True)

    config='/host/hamlet/local_raid/data/ravnoor/01_Projects/12_deepMask/src/noel_CIVET_masker/config_densecrf_t1.txt'
    start_time = time.time()
    denseCRF(case_id, t1w_fname, out_shape, config, dst, os.path.join(dst, case_id+"_vnet_maskpred.nii.gz"))
    elapsed_time = time.time() - start_time
    print("=*80")
    print("=> dense 3D-CRF inference time: {} seconds".format(round(elapsed_time,2)))
    print("=*80")


def denseCRF(id, t1, input_shape, config, out_dir, pred_labels):
    X, Y, Z = input_shape
    config_tmp = "/tmp/"+id+"_config_densecrf_t1.txt"
    print(config_tmp)
    subprocess.call(["cp", "-f", config, config_tmp])
    find_str = ["<ID_PLACEHOLDER>", "<T1_FILE_PLACEHOLDER>", "<OUTDIR_PLACEHOLDER>", "<PRED_LABELS_PLACEHOLDER>", "<X_PLACEHOLDER>", "<Y_PLACEHOLDER>", "<Z_PLACEHOLDER>"]
    replace_str = [str(id), str(t1), str(out_dir), str(pred_labels), str(X), str(Y), str(Z)]

    for fs, rs in zip(find_str, replace_str):
        find_replace_re(config_tmp, fs, rs)
    subprocess.call(["/host/hamlet/local_raid/data/ravnoor/01_Projects/12_deepMask/src/densecrf/dense3dCrf/dense3DCrfInferenceOnNiis", "-c", config_tmp])

def find_replace_re(config_tmp, find_str, replace_str):
    with fileinput.FileInput(config_tmp, inplace=True, backup='.bak') as file:
        for line in file:
            print(re.sub(find_str, str(replace_str), line.rstrip(), flags=re.MULTILINE), end='\n')

args=Data()

args.basedir = '/host/hamlet/local_raid/data/ravnoor/01_Projects/12_deepMask/src_T1/predictions/vnet.masker_T1.20180316_2128'

# csv_file = '/host/hamlet/local_raid/data/ravnoorX/MNI.photonAI/data/PAC2019_BrainAge_Training.csv'
# df = pd.read_csv(csv_file)
#
# # exclude: sub-1531 [idx:1358], sub-2835 [1970], sub-2450 [1971], sub-587 [2221]
#
# df['subject_ID'] = df['subject_ID'].str.replace('sub', 'sub-')
# df['subject_ID'] = df['subject_ID'].str.replace(r'(\d+)', lambda m: m.group(1).zfill(4))
#
# # data = data.drop("Ireland", axis=0)
# df = df.set_index('subject_ID')
# df = df.drop(['sub-1531', 'sub-2835', 'sub-2450', 'sub-0587'], axis=0).reset_index()
# df = df[df['subject_ID'] != ]


# df.to_csv('scans.csv')

scan = sys.argv[1]

process_dense_CRF(scan, args)
