#!/usr/bin/env python3

import json
import multiprocessing
import os
import socket
import sys
import time

from config.experiment import options

hostname = socket.getfqdn()

print("-" * 60)
print("hostname : {}".format(hostname))
print("-" * 60)

os.environ["KERAS_BACKEND"] = "theano"

# GPU/CPU options
options['cuda'] = 'cuda2' # cpu, cuda, cuda0, cuda1, or cudaX: flag using gpu 1 or 2
if options['cuda'].startswith('cuda1'):
    os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cuda1,floatX=float32,dnn.enabled=False"
elif options['cuda'].startswith('cpu'):
    os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())
    os.environ['MKL_NUM_THREADS'] = str(multiprocessing.cpu_count())
    os.environ['GOTO_NUM_THREADS'] = str(multiprocessing.cpu_count())
    os.environ['openmp'] = 'True'
    os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,openmp=True,floatX=float32"
else:
    os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cuda0,floatX=float32,dnn.enabled=False"
print(os.environ["THEANO_FLAGS"])

import numpy as np
import pandas as pd
from keras import backend as K
from nibabel import load as load_nii
from tqdm import tqdm

from models.noel_models_keras import *
from utils.base import *
from utils.metrics import *

K.set_image_dim_ordering("th")
K.set_image_data_format("channels_first")  # TH dimension ordering in this code


def function(options, test_val=0):
# set an experiment name to store net weights and segmentation masks
    modalities = ['T1', 'FLAIR']
    x_names = ['_t1_brain-final.nii.gz', '_fl_brain-final.nii.gz']
    options["experiment"] = "noel_deepFCD_dropoutMC_" + str(test_val)
    print("experiment: {}".format(options["experiment"]))
    
    
    options["model_dir"] = os.path.join(options["out_dir"])
    # save the sampled patches as a HDF5 dataset for faster subsequent experiments
    options["save_as_hdf5"] = True 
    options["hdf5_data_dir"] = os.path.join(options["main_dir"], "data", "hdf5")
    
    options["compute_performance"] = True
    sensitivity = 0
    perf = {}
    
    options["train_folder"] = os.path.join(options["main_dir"], "data")
    options["train_lesions"] = os.path.join(options["main_dir"], "lesions")
    
    datafile = "/app/examples/stratified_vol_loc_cv_nG.npy"
    
    split_dict = np.load(datafile, allow_pickle=True)
    train_list = split_dict[test_val].get('train')
    test_list = split_dict[test_val].get('val')
    
    options["load_checkpoint_1"] = False
    options["load_checkpoint_2"] = False
    # options['continue_training_2'] = True
    # options['initial_epoch_2'] = 69

    train_data, test_data = {}, {}

    train_data = {
        f: {
            m: os.path.join(options["train_folder"], 'sub-'+f, 'sub-'+f + n)
            for m, n in zip(modalities, x_names)
        }
        for f in train_list
    }
    train_labels = {
        f: os.path.join(options["train_lesions"], f + '.nii.gz')
        for f in train_list
    }
    subcort_masks_train = {
        f: os.path.join(options["train_folder"], 'sub-'+f, 'sub-'+f +'_exclusive_mask.nii.gz')
        for f in train_list
    }
    
    test_data = {
        f: {
            m: os.path.join(options["train_folder"], 'sub-'+f, 'sub-'+f + n)
            for m, n in zip(modalities, x_names)
        }
        for f in test_list
    }
    test_labels = {
        f: os.path.join(options["train_lesions"], f + '.nii.gz')
        for f in test_list
    }
    subcort_masks_test = {
        f: os.path.join(options["train_folder"], 'sub-'+f, 'sub-'+f + '_exclusive_mask.nii.gz')
        for f in test_list
    }
    print(train_data.keys())
    print(test_data.keys())
    # --------------------------------------------------
    # initialize the CNN
    # --------------------------------------------------
    options["weight_paths"] = os.path.join(options["model_dir"], options["experiment"])
    
    model = None  # clear the CNN
    
    model = off_the_shelf_model(options)
    
    print(model[0].summary())
    
    try:
        os.mkdir(options["weight_paths"])
    except:
        print("{} exists".format(options["weight_paths"]))
    
    start = time.time()
    
    model = train_model(model, train_data, train_labels, options=options, subcort_masks=subcort_masks_train)
    
    if not options["load_checkpoint_2"]:
        print("Saving fold split info")
        f = open((options["weight_paths"] + "/" + "/fold_info.txt"), "w")
        f.write("training list: %s \n \n" % (folds[test_site]["train_pids"]))
        f.write("test list: %s \n \n" % (folds[test_site]["test_pids"]))
        f.close()
    
        print("Saving config")
        opts_json = json.dumps(options, indent=4, sort_keys=True)
        f = open(
            options["weight_paths"] + "/" + options["experiment"] + "_config.json", "w"
        )
        f.write(opts_json)
        f.close()
    
        print("Saving trained models")
        model[0].save(options["weight_paths"] + "/" + options["experiment"] + "_model_1.h5")
        model[1].save(options["weight_paths"] + "/" + options["experiment"] + "_model_2.h5")
    
    end = time.time()
    diff = end - start
    print("training or loading model time elapsed: ~ {} seconds".format(diff))
    
    
    # --------------------------------------------------
    # test the cascaded model
    # --------------------------------------------------
    for _, scan in enumerate(
        tqdm(
            test_list, desc="serving predictions using the trained model", colour="magenta"
        )
    ):
    
        t_data = {}
        t_data[scan] = test_data[scan]
    
        options["pred_folder"] = os.path.join(
            options["test_folder"], "predictions", options["experiment"], scan
        )
    
        pred_mean_fname = os.path.join(
            options["pred_folder"], options["experiment"] + "_prob_mean_1.nii.gz"
        )
        pred_var_fname = os.path.join(
            options["pred_folder"], options["experiment"] + "_prob_var_1.nii.gz"
        )
    
        if np.logical_and(os.path.isfile(pred_mean_fname), os.path.isfile(pred_var_fname)):
            print("prediction for {} already exists".format(scan))
            continue
    
        if not os.path.exists(options["pred_folder"]):
            os.mkdir(options["pred_folder"])
    
        options["test_name"] = scan + "_" + options["experiment"] + ".nii.gz"
        options["test_scan"] = scan
    
        start = time.time()
    
        print("\n")
        print("-" * 80)
        print("testing the model for scan: {} ".format(scan))
        print("-" * 80)
    
        # test0: prediction/stage1
        # test1: prediction/stage2
        # test2: morphological processing + clustered
        # lpred: predicted label only
        # count: number of false positives
        if not options["compute_performance"]:
            test0, test1 = test_model(model, t_data, options, subcort_masks_test)
            print("number of non-zero voxels after CNN#1: {}".format(np.count_nonzero(test0)))
            print("number of non-zero voxels after CNN#2: {}".format(np.count_nonzero(test1)))
        else:
            test0, test1, test2, lpred, count = test_model(model, t_data, options, subcort_masks_test, performance=True)
            label = np.asarray(load_nii(test_labels[scan]).get_data())
            # print("label_shape: {}, label_unique: {}".format(label.shape, np.unique(label)))
            lesion_pred = extract_lesional_clus(label, test1, scan, options)
            print("-" * 80)
            print("computing performance metrics")
            print("-" * 80)
    
            # save metrics in a pandas dataframe
            perf = performancer(perf, scan, test2, label, lesion_pred, count)
        
            print("number of non-zero voxels after CNN#1: {}".format(np.count_nonzero(test0)))
            print("number of non-zero voxels after CNN#2: {}".format(np.count_nonzero(test1)))
            print(
                "number of non-zero voxels after size thresholding (> 20 voxels): {}".format(
                    np.count_nonzero(test2)
                )
            )
    
            if perf[scan]["sensitivity"] != 0:
                sensitivity += 1
            csv_name = (
                os.path.join(options["test_folder"], "predictions")
                + "/" + "results_tbin_" + str(options["t_bin"])
                + "_lmin_" + str(options["l_min"])
                + "_" + str(options["experiment"]) + ".csv"
            )
            # save performance metrics to disk
            df = pd.DataFrame(perf)
            df = df.transpose()
            df.to_csv(csv_name)
    
    
        print("-" * 80)
        end = time.time()
        diff = end - start
        print("=" * 80)
        print("time elapsed: ~ {} seconds".format(diff))
        print("=" * 80)


if __name__ == '__main__':
    intput_dir = os.environ.get('INPUT')
    output_dir = os.environ.get('OUTPUT')
    
    options["parallel_gpu"] = False
   
    options["batch_size"] = 350000
    options["mini_batch_size"] = 2048
    options["main_dir"] = intput_dir
    options["out_dir"] = output_dir
    options["test_folder"] = output_dir

    for i in range(1, 10):
        function(options, test_val=i)
