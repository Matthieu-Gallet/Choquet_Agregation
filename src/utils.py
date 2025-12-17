from os.path import dirname, abspath, join, basename, exists, normpath
from os import system, remove, makedirs, rename, chdir
import glob, sys, h5py, logging, time
from datetime import datetime
from shutil import copyfile, rmtree
from scipy.stats import spearmanr
from yaml import safe_load
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import pickle

from sklearn.pipeline import Pipeline

# import numpy as np
import autograd.numpy as np

# import matplotlib as mpl

# mpl.use("pgf")
import matplotlib.pyplot as plt

# plt.rcParams.update(
#     {
#         "font.family": "serif",  # use serif/main font for text elements
#         "text.usetex": True,  # use inline math for ticks
#         "pgf.texsystem": "pdflatex",
#         "pgf.preamble": "\n".join(
#             [
#                 r"\usepackage[utf8x]{inputenc}",
#                 r"\usepackage[T1]{fontenc}",
#                 r"\usepackage{cmbright}",
#             ]
#         ),
#     }
# )



def preprocess(x, n):
    x = np.concatenate([x, np.zeros((n, 1))], axis=1)
    x = np.sort(x, axis=1)
    x = np.diff(x, axis=1)
    return x

def transform_groups(gr, corresponding_groups): 
    return np.array([corresponding_groups[gr] for gr in gr])

def extract_windows(image, n, k, max_iter):
    """Extract k windows of size n x n randomly without overlapping inside an image.
    If it fails to extract the  k windows, it will stop after max_iter iterations.

    Parameters
    ----------
    image : np.array
        Image to extract windows from.
    n : int
        Size of the windows.
    k : int
        Number of windows to extract.
    max_iter : int
        Maximum number of iterations to try to extract the windows.

    Returns
    -------
    np.array
        Array of shape (k, n, n) containing the extracted windows.
    """
    nx, ny = image.shape[:2]
    windows = [image[:n, :n].tolist()]
    n_test = 0
    pbar = tqdm(total=k, leave=False)
    while (len(windows) < k) & (n_test < max_iter):
        x = np.random.randint(0, nx - n + 1)
        y = np.random.randint(0, ny - n + 1)
        window = image[x : x + n, y : y + n].tolist()
        if not np.any([np.isin(window, p).any() for p in windows]):
            windows.append(window)
            pbar.update(1)
        else:
            n_test += 1
        if pbar.n >= k:
            break
    pbar.close()
    return np.array(windows)


def scale_image(img):
    img = img - img.min()
    img = img / (img.max() - img.min())
    return img


def load_yaml(file_name):
    with open(file_name, "r") as f:
        opt = safe_load(f)
    return opt


def exist_create_folder(path):
    if not exists(path):
        makedirs(path)
    return 1


def dump_pkl(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return 1


def open_pkl(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def init_logger(path_log):
    now = datetime.now()
    namlog = now.strftime("%d%m%y_%HH%MM%S")
    datestr = "%m/%d/%Y-%I:%M:%S %p "
    s = hex(int(1e6 * now.timestamp()))[-6:]
    logging.basicConfig(
        filename=join(path_log, f"log_{namlog}_{s}.log"),
        level=logging.INFO,
        filemode="w",
        datefmt=datestr,
        format="%(asctime)s: (%(filename)s): %(levelname)s: %(funcName)s Line: %(lineno)d - %(message)s",
    )
    logging.info("Started")
    return logging


def parser_pipeline(opt, ind):
    for imp in opt["import"]:
        exec(imp)
    pipe = opt["pipeline"][ind]
    step = []
    for i in range(len(pipe)):
        name_methode = pipe[i][0]
        estim = locals()[name_methode]()

        if len(pipe[i]) > 1:
            [
                [
                    setattr(estim, param, pipe[i][g][param])
                    for param in pipe[i][g].keys()
                ]
                for g in range(1, len(pipe[i]))
            ]
        step.append((name_methode, estim))
    return Pipeline(step, verbose=True)  # , memory=".cache")


def save_h5(img, label, filename):
    if ".h5" not in filename:
        filename += ".h5"
    with h5py.File(filename, "w") as hf:
        hf.create_dataset(
            "img", np.shape(img), h5py.h5t.IEEE_F32BE, compression="gzip", data=img
        )  # IEEE_F32BE is big endian float32
        hf.create_dataset(
            "label", np.shape(label), compression="gzip", data=label.astype("S")
        )


def norm_one(X_train):
    return (X_train - X_train.min()) / (X_train.max() - X_train.min())


def normalize(X_train, a, b):
    X_train = (a - b) * norm_one(X_train) + b
    return X_train


def load_h5(path, format_date=False):
    """
    Load data from an HDF5 (h5) file.

    Args:
        path (str): The path to the HDF5 file.

    Returns:
        numpy.ndarray: Image data as a NumPy array with data type float32.
        numpy.ndarray: Labels data as a NumPy array with data type str.
        numpy.ndarray: Groups data as a NumPy array with data type int.
    """
    with h5py.File(path, "r") as hf:
        img = hf["img"][:].astype(np.float32)
        labels = hf["labels"][:].astype(str)
        groups = hf["groups"][:].astype(int)
        try:
            org = hf["org"][:].astype(str)
        except:
            org = None
        dates = hf["date"][:].astype(str)
        if format_date:
            if "_" in dates[0]:
                dates = [date.split("_") for date in dates]
                dates = [pd.to_datetime(date, format="%Y-%m-%d") for date in dates]
            else:
                dates = pd.to_datetime(dates, format="%Y-%m-%d")
        dates = np.array(dates)
    if org is not None:
        return img, labels, groups, org, dates
    else:
        return img, labels, groups, dates
