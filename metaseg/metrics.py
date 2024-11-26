import collections.abc
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np
import skimage.measure
import skimage.segmentation
import torch


class Metrics(collections.abc.Mapping):
    def __init__(
        self,
        probs: np.ndarray,
        target: Optional[np.ndarray] = None,
        ignore_index: Optional[int] = None,
    ) -> None:
        self.probs = probs
        self.ignore_index = ignore_index

        self.labels = probs.argmax(0)
        self.num_classes = probs.shape[0]

        if (target is not None) and (ignore_index is not None):
            self.labels[target == ignore_index] = ignore_index

        self.segments, self.num_segments = skimage.measure.label(
            self.labels, background=-1, return_num=True
        )
        self.segments = self.segments.astype(np.uint16)

        self.boundary_mask = skimage.segmentation.find_boundaries(self.labels + 1, connectivity=2)

        self.metrics: Dict[str, np.ndarray] = {}

        self._compute_base_stats()
        self.add_probabilities(probs)
        if target is not None:
            self.add_adjusted_ious(target)

    def _compute_base_stats(self) -> None:
        labels = self.labels.ravel()
        segments = self.segments.ravel()
        boundary_mask = self.boundary_mask.ravel()

        self.metrics["S"] = n = np.bincount(segments)
        self.metrics["S_bd"] = n_bd = np.bincount(segments * boundary_mask)
        n_bd[0] = n[0] = -1  # no segment has index 0
        self.metrics["S_in"] = n_in = n - n_bd
        self.metrics["S_rel"] = n / n_bd
        self.metrics["S_rel_in"] = n_in / n_bd

        # TODO: Replace this (reasonably fast but) stupid solution. `np.unique` can also be used
        # here but is slower.
        self.metrics["class"] = (np.bincount(segments, weights=labels) / n).astype(int)

        self.metrics["x"], self.metrics["y"] = self._weighted_centroids()

    def _weighted_centroids(
        self, weights: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        segments = self.segments.ravel()

        weights_y, weights_x = np.indices(self.segments.shape).reshape(2, -1)
        weights_sum = self.metrics["S"]
        if weights is not None:
            weights_x *= weights
            weights_y *= weights
            weights_sum = np.bincount(segments, weights=weights)

        x = np.bincount(segments, weights=weights_x) / weights_sum
        y = np.bincount(segments, weights=weights_y) / weights_sum

        return x, y

    def add_heatmap(self, heatmap: np.ndarray, prefix: str, centroids: bool = False) -> None:
        segments = self.segments.ravel()
        boundary_mask = self.boundary_mask.ravel()

        n_tot = self.metrics["S"]
        n_bd = self.metrics["S_bd"]
        n_in = self.metrics["S_in"]
        n_rel = self.metrics["S_rel"]
        n_rel_in = self.metrics["S_rel_in"]

        heatmap = heatmap.ravel()
        heatmap_sq = heatmap**2

        sum_tot = np.bincount(segments, weights=heatmap)
        sum_sq_tot = np.bincount(segments, weights=heatmap_sq)
        sum_bd = np.bincount(segments, weights=heatmap * boundary_mask)
        sum_sq_bd = np.bincount(segments, weights=heatmap_sq * boundary_mask)
        sum_in = sum_tot - sum_bd
        sum_sq_in = sum_sq_tot - sum_sq_bd

        self.metrics[prefix] = mean = sum_tot / n_tot
        self.metrics[f"{prefix}_var"] = var = sum_sq_tot / n_tot - mean**2
        self.metrics[f"{prefix}_in"] = mean_in = _safe_divide(sum_in, n_in)
        self.metrics[f"{prefix}_var_in"] = var_in = _safe_divide(sum_sq_in, n_in) - mean_in**2
        self.metrics[f"{prefix}_bd"] = mean_bd = sum_bd / n_bd
        self.metrics[f"{prefix}_var_bd"] = sum_sq_bd / n_bd - mean_bd**2
        self.metrics[f"{prefix}_rel"] = sum_tot / n_bd  # mean * n_rel
        self.metrics[f"{prefix}_var_rel"] = var * n_rel
        self.metrics[f"{prefix}_rel_in"] = sum_in / n_bd  # mean_in * n_rel_in
        self.metrics[f"{prefix}_var_rel_in"] = var_in * n_rel_in

        if centroids:
            xy = self._weighted_centroids(heatmap)
            self.metrics[f"{prefix}_x"], self.metrics[f"{prefix}_y"] = xy

    def add_probabilities(self, probs: np.ndarray, prefix: Optional[str] = None) -> None:
        prefix = "" if prefix is None else f"{prefix}_"

        segments = self.segments.ravel()
        for c, p in enumerate(probs):
            cprob = np.bincount(segments, weights=p.ravel()) / self.metrics["S"]
            self.metrics[f"{prefix}cprob_{c}"] = cprob

        e = _entropy(probs)
        self.add_heatmap(e, f"{prefix}E")

        v, m = _prob_differences(probs)
        self.add_heatmap(v, f"{prefix}V")
        self.add_heatmap(m, f"{prefix}M")

    def add_adjusted_ious(self, gt_labels: np.ndarray) -> None:
        gt_segments = skimage.measure.label(gt_labels, background=-1).astype(np.uint16)
        iou = _adjusted_ious(
            self.labels,
            self.segments,
            gt_labels,
            gt_segments,
            self.num_classes,
            self.ignore_index,
        )
        self.metrics["iou"] = iou
        self.metrics["iou0"] = iou != 0

    def __getitem__(self, name: str) -> Any:
        return self.metrics[name][1:]

    def __len__(self) -> int:
        return len(self.metrics)

    def __iter__(self) -> Iterator:
        return iter(self.metrics)


def _safe_divide(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.divide(p, q, out=np.zeros_like(p), where=q != 0)


def _split_by_ids(x: np.ndarray, y: np.ndarray, ids: np.ndarray) -> np.ndarray:
    return (y.reshape(1, -1) == ids.reshape(-1, 1)) * x.reshape(1, -1)


def _faster_unique(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(x)
    uniq = np.nonzero(counts)[0]
    counts = counts[uniq]
    return uniq, counts


def _entropy(probs: np.ndarray) -> np.ndarray:
    entropy = (probs * np.log(probs, out=np.zeros_like(probs), where=probs > 0)).sum(0)
    entropy /= -np.log(probs.shape[0])
    return entropy


def _prob_differences(probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    #largest = torch.topk(torch.from_numpy(probs), 2, dim=0).values
    #v = 1 - largest[0]  # variation ratio
    #m = v + largest[1]  # probability margin
    #return v.numpy(), m.numpy()

    high1 = np.max(probs, axis =0)                          #*# this seems to be 3x faster and does not cause a bug where
    high2 = np.max(np.ma.masked_equal(probs, high1),axis=0) #*# you could not compute probs and metadata in a single script

    v2 = 1-high1                                            #*#
    m2 = v2+ high2                                          #*#
    return v2, m2                                           #*#


def _adjusted_ious(
    pr_labels: np.ndarray,
    pr_segments: np.ndarray,
    gt_labels: np.ndarray,
    gt_segments: np.ndarray,
    num_classes: int,
    ignore_index: Optional[int] = None,
) -> np.ndarray:
    ids = np.arange(num_classes)
    if ignore_index is not None:
        ids = ids[ids != ignore_index]

    # The code below can be used to remove ignored areas from the prediction if this has not
    # already been done using the code currently in __init__
    # pr_labels = pr_labels.copy()
    # pr_labels.ravel()[gt_labels.ravel() == ignore_index] = ignore_index

    pr_split = _split_by_ids(pr_segments, pr_labels, ids).ravel()
    gt_split = _split_by_ids(gt_segments, gt_labels, ids).ravel()

    n_pr = pr_segments.max() + 1
    n_gt = gt_segments.max() + 1
    mult = max(n_pr, n_gt)

    uniq, counts = _faster_unique(pr_split + np.multiply(gt_split, mult, dtype=np.uint32))

    div_, mod_ = np.divmod(uniq, mult)  # TODO: faster than (>>, &) ?
    union = np.bincount(mod_, weights=counts, minlength=n_pr)
    mask = div_ != 0
    inter = np.bincount(mod_[mask], weights=counts[mask], minlength=n_pr)

    # TODO: next line is a 'temporary hack' to fix double counting segments where the ground truth
    # segment is entirely covered by a prediction
    counts[mod_ != 0] = 0
    union += np.bincount(
        mod_[mask], weights=counts[np.searchsorted(uniq, div_[mask] * mult)], minlength=n_pr
    )

    union[union == 0] = np.nan

    return inter / union
