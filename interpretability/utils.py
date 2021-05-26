import os
import torch
import numpy as np

from data.data_handler import Data
from experiments.Imagenet.final.experiment_parameters import exps
from experiments.Imagenet.final.model import get_model
from interpretability.configs import BASE_PATH
from project_utils import to_numpy
from torch.hub import download_url_to_file
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from training.trainer_base import Trainer

matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
params = {'text.usetex': True,
          'font.size': 16,
          'text.latex.preamble': [r"\usepackage{lmodern}"],
          'font.family': 'sans-serif',
          'font.serif': 'Computer Modern Sans serif',
          'text.latex.unicode': True,
          }
plt.rcParams.update(params)

sns.set_style("darkgrid")

_model_urls = {
    "Imagenet": {
        "9L-L-CoDA-SQ-100000": ('https://nextcloud.mpi-klsb.mpg.de/index.php/s/pArrgmncXAAiH5R/download', 60)
    },
    "CIFAR10": {
        "9L-S-CoDA-SQ-10": ('https://nextcloud.mpi-klsb.mpg.de/index.php/s/H8FgwMHJTTTYDJw/download', 200),
        "9L-S-CoDA-SQ-50": ('https://nextcloud.mpi-klsb.mpg.de/index.php/s/SqPBioqmxEWbH9g/download', 200),
        "9L-S-CoDA-SQ-100": ('https://nextcloud.mpi-klsb.mpg.de/index.php/s/yk7LajwLwpXXzf5/download', 200),
        "9L-S-CoDA-SQ-500": ('https://nextcloud.mpi-klsb.mpg.de/index.php/s/zfeYMASLtKNGgsQ/download', 200),
        "9L-S-CoDA-SQ-1000": ('https://nextcloud.mpi-klsb.mpg.de/index.php/s/8ykMc8XLQMtCxHo/download', 200),
        "9L-S-CoDA-SQ-5000": ('https://nextcloud.mpi-klsb.mpg.de/index.php/s/6iY992FYEsfHkeA/download', 200),
    }

}

explainers_color_map = {
    "Grad": (255, 255, 255),
    "Ours": (66, 202, 253),
    "Occlusion": (255, 32, 255),
    "GCam": (255, 255, 22),
    "Occ5": (128, 128, 128),
    "Occ9": (0, 0, 0),
    'DeepLIFT': (255, 165, 46),
    'IntGrad': (99, 38, 84),
    'IxG': (207, 174, 139),
    'GB': np.array([0.63008579, 1., 1.]) * 255,
}  # Adapted from Boyton's colors


def get_pretrained(model="9L-L-CoDA-SQ-100000", dataset="Imagenet"):
    """
    Loading the pretrained models for evaluation.
    Returns:
       trainer object with pretrained model
    """
    assert dataset in _model_urls and model in _model_urls[dataset], "URL for this model is not specified."
    url, epoch = _model_urls[dataset][model]
    # This model_path convention allows to identify the experiment just from the path and simplifies reloading.
    # (E.g., for the interpretability analysis scripts).
    model_path = os.path.join(BASE_PATH, dataset, "final", model)
    model_file = os.path.join(model_path, "model_epoch_{}.pkl".format(epoch))
    if not os.path.exists(model_file):
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        download_url_to_file(url,
                             model_file, progress=True)
    # Specify model parameters defined in Imagenet.final.experiment_parameters
    exp_params = exps[model]
    # Load model according to experiment parameters
    model = get_model(exp_params)
    # Load data according to experiment parameters
    data_handler = Data(dataset, only_test_loader=True, **exp_params)
    trainer = Trainer(model,
                      data_handler,
                      model_path,  # Setting the path in the trainer to be able to use trainer.reload.
                      **exp_params)

    trainer.reload()
    trainer.model.cuda()
    return trainer


@torch.no_grad()
def get_sorted_random_subset(trainer, loader, n=500):
    """
    For the jupyter notebook examples. Evaluates the model in trainer on a random subset of the dataset
    and returns the indices in sorted order (high confidence to low confidence).
    """
    total_length = min(n, len(loader.dataset))  # Loading n random images and to sort them by confidence
    if total_length < n:
        print("More images requested than in data loader. Reset to size of test set.")
    random_idcs = np.random.permutation(range(len(loader.dataset)))[:total_length]
    print("Loading {:5.2f}% of images from the test set.".format(float(total_length) / len(loader.dataset) * 100.))

    correct = []
    confidences = []
    predicted = []
    gt_class = []

    for count, idx in enumerate(random_idcs):
        trainer.model.zero_grad()
        img, tgt = loader.dataset[idx]
        t = tgt.argmax().cuda()
        # In probabilities, many are 1.0, resulting in bad sorting. Hence, stay in logit space.
        pred = trainer.predict(img[None].cuda(), to_probabilities=False)
        conf, pd_class = pred.max(1)
        gt_class.append(t.item())
        predicted.append(pd_class.item())
        confidences.append(conf.item())
        correct.append((pd_class == t).item())
        print("\r{0:6.2f}% done.".format(100 * (count + 1) / total_length), end="")
        cor_mask = (np.array(correct, dtype=bool))

    print("\nSorting indices.")
    idcs = random_idcs[:len(cor_mask)][cor_mask]
    cor_confs = np.array(confidences)[cor_mask]

    srtd_confs, srtd_idcs = np.array(sorted(zip(cor_confs, idcs), key=lambda x: x[0], reverse=True)).T

    srtd_idcs = np.array(srtd_idcs, dtype=int)
    print("Done.")
    return srtd_idcs


def plot_contribution_map(contribution_map, ax=None, vrange=None, vmin=None, vmax=None, hide_ticks=True, cmap="bwr",
                          percentile=100):
    """
    Visualises a contribution map, i.e., a matrix assigning individual weights to each spatial location.
    As default, this shows a contribution map with the "bwr" colormap and chooses vmin and vmax so that the map
    ranges from (-max(abs(contribution_map), max(abs(contribution_map)).
    Args:
        contribution_map: (H, W) matrix to visualise as contributions.
        ax: axis on which to plot. If None, a new figure is created.
        vrange: If None, the colormap ranges from -v to v, with v being the maximum absolute value in the map.
            If provided, it will range from -vrange to vrange, as long as either one of the boundaries is not
            overwritten by vmin or vmax.
        vmin: Manually overwrite the minimum value for the colormap range instead of using -vrange.
        vmax: Manually overwrite the maximum value for the colormap range instead of using vrange.
        hide_ticks: Sets the axis ticks to []
        cmap: colormap to use for the contribution map plot.
        percentile: If percentile is given, this will be used as a cut-off for the attribution maps.

    Returns: The axis on which the contribution map was plotted.

    """
    assert len(contribution_map.shape) == 2, "Contribution map is supposed to only have spatial dimensions.."
    contribution_map = to_numpy(contribution_map)
    cutoff = np.percentile(np.abs(contribution_map), percentile)
    contribution_map = np.clip(contribution_map, -cutoff, cutoff)
    if ax is None:
        fig, ax = plt.subplots(1)
    if vrange is None or vrange == "auto":
        vrange = np.max(np.abs(contribution_map.flatten()))
    im = ax.imshow(contribution_map, cmap=cmap,
                   vmin=-vrange if vmin is None else vmin,
                   vmax=vrange if vmax is None else vmax)

    if hide_ticks:
        ax.set_xticks([])
        ax.set_yticks([])

    return ax, im

