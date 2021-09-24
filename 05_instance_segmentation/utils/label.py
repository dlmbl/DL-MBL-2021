import logging
import mahotas
import numpy as np
import scipy.ndimage
import scipy.special
from skimage import measure
from .mean_shift import MeanShift
import torch

logger = logging.getLogger(__name__)


def watershed(surface, markers, fg):
    # compute watershed
    ws = mahotas.cwatershed(surface, markers)

    # write watershed directly
    logger.debug("watershed output: %s %s %f %f",
                 ws.shape, ws.dtype, ws.max(), ws.min())

    # overlay fg and write
    wsFG = ws * fg
    logger.debug("watershed (foreground only): %s %s %f %f",
                 wsFG.shape, wsFG.dtype, wsFG.max(),
                 wsFG.min())
    wsFGUI = wsFG.astype(np.uint16)

    return wsFGUI

def cluster_nearest_centroid(pred, centroids):
    # make channels last for broadcasted operations
    pred = np.moveaxis(pred,0,-1)
    # get num centroids
    num_cens = centroids.shape[0]
    distances = np.zeros(list(pred.shape[:-1])+[num_cens])
    for i in range(0,num_cens):
        distances[:,:,i] = np.linalg.norm(pred-centroids[i,],ord=2,axis=-1)
    segmentation = np.argmin(distances, axis=2) + 1
    return segmentation.astype(np.float32)

def label(prediction, kind, fg_thresh=0.9, seed_thresh=0.9):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("labelling")

    if kind == "two_class":
        fg = 1.0 * (prediction[0] > fg_thresh)
        ws_surface = 1.0 - prediction[0]

        seeds = (1 * (prediction[0] > seed_thresh)).astype(np.uint8)
    elif kind == "affinities":
        # combine  components of affinities vector
        surface = 0.5 * (prediction[0] + prediction[1])
        # background pixel have affinity zero with everything
        # (including other bg pixel)
        fg = 1.0 * (surface > fg_thresh)
        ws_surface = 1.0 - surface

        seeds = (1 * (prediction > seed_thresh)).astype(np.uint8)
        seeds = (seeds[0] + seeds[1])
        seeds = (seeds > 1).astype(np.uint8)
    elif kind == 'three_class':
        # prediction[0] = bg
        # prediction[1] = inside
        # prediction[2] = boundary
        prediction = scipy.special.softmax(prediction, axis=0)
        fg = 1.0 * ((1.0 - prediction[0, ...]) > fg_thresh)
        ws_surface = 1.0 - prediction[1, ...]
        seeds = (1 * (prediction[1, ...] > seed_thresh)).astype(np.uint8)
    elif kind == 'sdt':
        # distance transform in negative inside an instance
        # so negative values correspond to fg
        if fg_thresh > 0:
            logger.warning("fg threshold should be zero/negative")
        fg = 1.0 * (prediction < fg_thresh)
        fg = fg.astype(np.uint8)

        ws_surface = prediction
        if seed_thresh > 0:
            logger.warning("surface/seed threshold should be negative")
        seeds = (1 * (ws_surface < seed_thresh)).astype(np.uint8)
    elif kind == 'metric_learning':
        fg = 1.0 * (prediction[0] > fg_thresh)
        emb = prediction[1:]
        emb *= fg
        ms = MeanShift(X=torch.Tensor(emb).to(device),bandwidth=2.,chan=3,n_seeds=2000)
        ms = ms
        C_ov = ms.forward()
        C_ov = torch.round(C_ov*100)/100
        C_ov = torch.unique(C_ov, dim=0) # num_centroids x chan
        labelling = cluster_nearest_centroid(emb, C_ov.cpu().numpy()).astype(np.int32)
        labelling = labelling.astype(np.int32) * fg.astype(np.int32)
        labelling = measure.label(labelling, background=0)
        uni, counts = np.unique(labelling, return_counts=True)
        labelling[labelling == uni[counts<50]] = 0
        ws_surface = prediction

    if kind in ["two_class", "affinities",'three_class','sdt']:
        if np.count_nonzero(seeds) == 0:
            logger.warning("no seed points found for watershed")

        markers, cnt = scipy.ndimage.label(seeds)
        logger.info("num markers %s", cnt)
        # compute watershed
        labelling = watershed(ws_surface, markers, fg)

    return labelling, ws_surface
