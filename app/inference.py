import logging
import multiprocessing
import os
import subprocess
import sys
import warnings

from mo_dots import Data

from config.experiment import options

warnings.filterwarnings("ignore")
import time

import numpy as np
import setproctitle as spt
from tqdm import tqdm

from utils.helpers import *

logging.basicConfig(
    level=logging.DEBUG,
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
    format="{asctime} {levelname} {filename}:{lineno}: {message}",
)

os.environ["KERAS_BACKEND"] = "theano"

cores = str(multiprocessing.cpu_count() // 2)
var = os.getenv("OMP_NUM_THREADS", cores)
try:
    logging.info("# of threads initialized: {}".format(int(var)))
except ValueError:
    raise TypeError(
        "The environment variable OMP_NUM_THREADS"
        " should be a number, got '%s'." % var
    )
# os.environ['openmp'] = 'True'
options['cuda'] = 'cuda0' # cpu, cuda, cuda0, cuda1, or cudaX: flag using gpu 1 or 2
if options['cuda'].startswith('cuda1'):
    os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cuda1,floatX=float32,dnn.enabled=False"
elif options['cuda'].startswith('cpu'):
    os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())
    os.environ['MKL_NUM_THREADS'] = str(multiprocessing.cpu_count())
    os.environ['GOTO_NUM_THREADS'] = str(multiprocessing.cpu_count())
    os.environ['openmp'] = 'True'
    os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,openmp=True,floatX=float32"
else:
    os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cuda2,floatX=float32,dnn.enabled=False"
print(os.environ["THEANO_FLAGS"])


from keras import backend as K
from keras.models import load_model

from models.noel_models_keras import *
from utils.base import *
from utils.metrics import *


def inference(args):
    args.t1_fname = f'{args.id}_t1_brain-final.nii.gz'
    args.t2_fname = f'{args.id}_fl_brain-final.nii.gz'

    # our preprocessing:
    args.mask_path = os.path.join(args.dir, args.id, f'{args.id}_exclusive_mask.nii.gz')

    t1 = nib.load(os.path.join(args.dir, args.id, args.t1_fname)) 
    t2 = nib.load(os.path.join(args.dir, args.id, args.t2_fname)) 
    mask = nib.load(args.mask_path).get_fdata() < 0.5
    t1_data = t1.get_fdata()*mask
    t2_data = t2.get_fdata()*mask
    
    args.t1 = os.path.join(args.outdir, args.t1_fname)
    args.t2 = os.path.join(args.outdir, args.t2_fname)

    nib.save(nib.Nifti1Image(t1_data * 100 / t1_data.max(), t1.affine), args.t1)  
    nib.save(nib.Nifti1Image(t2_data * 100 / t2_data.max(), t2.affine), args.t2) 

    # deepFCD configuration
    K.set_image_dim_ordering("th")
    K.set_image_data_format("channels_first")  # TH dimension ordering in this code

    options["parallel_gpu"] = False
    modalities = ["T1", "FLAIR"]
    x_names = options["x_names"]

    options["dropout_mc"] = True
    options["batch_size"] = 350000
    options["mini_batch_size"] = 2048
    options["load_checkpoint_1"] = True
    options["load_checkpoint_2"] = True

    # trained model weights based on 148 histologically-verified FCD subjects
    cwd = os.path.realpath(os.path.dirname(__file__))
    options["weight_paths"] = os.path.join(cwd, "weights")
    options["experiment"] = "noel_deepFCD_dropoutMC"
    logging.info("experiment: {}".format(options["experiment"]))
    spt.setproctitle(options["experiment"])

    # --------------------------------------------------
    # initialize the CNN
    # --------------------------------------------------
    # initialize empty model
    model = None
    # initialize the CNN architecture
    model = off_the_shelf_model(options)

    load_weights = os.path.join(
        options["weight_paths"], "noel_deepFCD_dropoutMC_model_our1.h5"
    )
    logging.info(
        "loading DNN1, model[0]: {} exists".format(load_weights)
    ) if os.path.isfile(load_weights) else sys.exit(
        "model[0]: {} doesn't exist".format(load_weights)
    )
    model[0] = load_model(load_weights)

    load_weights = os.path.join(
        options["weight_paths"], "noel_deepFCD_dropoutMC_model_our2.h5"
    )
    logging.info(
        "loading DNN2, model[1]: {} exists".format(load_weights)
    ) if os.path.isfile(load_weights) else sys.exit(
        "model[1]: {} doesn't exist".format(load_weights)
    )
    model[1] = load_model(load_weights)
    logging.info(model[1].summary())

    # --------------------------------------------------
    # test the cascaded model
    # --------------------------------------------------
    files = [args.t1, args.t2]
    options["test_folder"] = args.outdir

    test_data = {
        args.id: {
            m: os.path.join(options["test_folder"], args.id, n) for m, n in zip(modalities, files)
        }
    }

    options["pred_folder"] = args.outdir

    if not os.path.exists(options["pred_folder"]):
        os.mkdir(options["pred_folder"])

    options["test_scan"] = args.id

    start = time.time()
    logging.info("\n")
    logging.info("-" * 70)
    logging.info("testing the model for scan: {}".format(args.id))
    logging.info("-" * 70)

    test_model(
        model,
        test_data,
        options,
        uncertainty=False
    )

    end = time.time()
    diff = (end - start) // 60
    logging.info("-" * 70)
    logging.info("time elapsed: ~ {} minutes".format(diff))
    logging.info("-" * 70)

    os.remove(args.t1)
    os.remove(args.t2)
    os.remove(os.path.join(options["pred_folder"], args.id + "_" + options["experiment"] + "_prob_mean_0.nii.gz"))


if __name__ == '__main__':
    args = Data()
    args.dir = os.environ.get('INPUT')
    args.outdir = os.environ.get('OUTPUT')

    args.id = os.listdir(args.dir)[0]

    if not os.path.isabs(args.dir):
        args.dir = os.path.abspath(args.dir)

    inference(args)
