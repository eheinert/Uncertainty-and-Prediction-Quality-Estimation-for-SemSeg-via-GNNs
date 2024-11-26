import collections.abc
from collections import namedtuple
from os import PathLike
from pathlib import Path
from typing import Any, Iterator, Tuple

import numpy as np
from PIL import Image


class Template(collections.abc.Mapping):
    """
    Template for a (meta)dataloader for the MetaSeg postprocessing tool. 
    """

    TemplateClass = namedtuple(
        "TemplateClass",
        [
            "name",
            "id",
            "train_id",
            "category",
            "category_id",
            "has_instances",
            "ignore_in_eval",
            "color",
        ],
    )
    """ In labels you can set your classes and how to handle them during training. Below here you see examplary entries which
    you will have to change: """
    labels = [
        TemplateClass("unlabeled", 0, 255, "void", 0, False, True, (0, 0, 0)),
        TemplateClass("background", 1, 255, "void", 0, False, True, (81, 0, 81)),
        TemplateClass("builing", 2, 2, "construction", 2, False, False, (70, 70, 70)),
        TemplateClass("fence", 3, 4, "construction", 2, False, False, (190, 153, 153)),
        TemplateClass("person", 4, 11, "human", 6, True, False, (220, 20, 60)),
        TemplateClass("car", 5, 13, "vehicle", 7, True, False, (0, 0, 142))
    ]

    """ Normalization parameters, need to be set for your custom dataset! E.g. for cityscapes:
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    """
    mean = (None, None, None)
    std = (None, None, None)

    """Here, some useful information from labels will be extracted. Nothing needs to be changed."""
    ignore_in_eval_ids, label_ids, train_ids, train_id2id = ([], [], [], [])
    color_palette_train_ids = [(0, 0, 0) for i in range(256)]
    for i in range(len(labels)):
        if labels[i].ignore_in_eval and labels[i].train_id not in ignore_in_eval_ids:
            ignore_in_eval_ids.append(labels[i].train_id)
    for i in range(len(labels)):
        label_ids.append(labels[i].id)
        if labels[i].train_id not in ignore_in_eval_ids:
            train_ids.append(labels[i].train_id)
            color_palette_train_ids[labels[i].train_id] = labels[i].color
            train_id2id.append(labels[i].id)
    num_label_ids = len(set(label_ids))  # Number of ids
    num_train_ids = len(set(train_ids))  # Number of trainIds
    id2label = {label.id: label for label in labels}
    train_id2label = {label.train_id: label for label in labels}
    color_palette_train_ids = list(sum(color_palette_train_ids, ()))



    """ 
    In init you basically have to define the 4 lists self.targets, self.probabilities, self.all_images, self.all_basenames:
    - self.targets: list of (relative) paths to the semseg labels. In our example:
        [PosixPath('DV3+_WRN38_on_cityscapes/Cityscapes/gtFine/val/frankfurt/frankfurt_000000_000294_gtFine_labelTrainIds.png'), ...]
    - self.probabilities: list of (relative) paths to where the softmax probs shall be saved. In our example:
        [PosixPath('DV3+_WRN38_on_cityscapes/Cityscapes/softmax_probs/val/frankfurt/frankfurt_000000_000294.npy'), ...]
    - self.all_images: list of (relative) paths to where the actual images, i.e. semseg inputs. In our example:
        [PosixPath('DV3+_WRN38_on_cityscapes/Cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png'), ...]
    - self.all_basenames: image 'names', e.g. paths to the images, relative to the 'label' directory. In our example:
        ['val/frankfurt/frankfurt_000000_000294', ...]
    For every Image you want your meta model trained on, you should add an entry to all 4 lists. Order matters!
    """

    def __init__(
        self,
        probs_root: PathLike,
        target_root: PathLike,
        images_root: PathLike = "",
        split: str = "val",
    ):  
        """ Iterator, do not change: """
        self._index = 0

        """ Possible dataset configuration values from dataset cfg, can be handled differently according to your dataset. Set in cfg.
        self.split = split
        self.probs_root = Path(probs_root) / self.split
        self.target_root = Path(target_root) / self.split
        self.images_root = Path(images_root) / self.split
        """
        
        """ All Essential information about the dataset will be stored here: """
        self.targets = []
        self.probabilities = []
        self.all_images = []
        self.all_basenames = []

        """ 
        
                Room to fill the 4 lists. For an example take a look into cityscapes.py!
        
        """


    """ If everything else is set correctly, the functions __getitem__, __len__, __iter__ should be fine as they are:"""

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        probs = np.load(self.probabilities[index])
        target = None
        if self.mode == "meta_train":
            target = np.array(Image.open(self.targets[index]))
        return probs, target

    def __len__(self) -> int:
        return len(self.targets)

    def __iter__(self) -> Iterator:
        for i in range(len(self)):
            yield self[i]
