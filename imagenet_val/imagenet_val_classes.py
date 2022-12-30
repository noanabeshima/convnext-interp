from torchvision import transforms
import torch
from PIL import Image
import json
from IPython.display import display
import os
from interp_utils import get_image_grid, is_iterable
import numpy as np
from functools import cache

dirname = os.path.dirname(__file__)


with open(dirname + "/ordered_class_labels.txt", "r") as f:
    class_labels = np.array(f.read().split("\n"))
simple_class_labels = np.array([label.split(",")[0] for label in class_labels])


center_crop = transforms.CenterCrop(224)

convnext_preprocess = transforms.Compose(
    [
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class ValImage:
    def __init__(self, fname, class_name, class_id, val_index):
        self.fname = fname
        self.class_name = class_name
        self.class_id = class_id
        self._val_index = val_index

    def show(self):
        display(self.image)

    @property
    def image(self):
        return center_crop(Image.open(self.fname).convert("RGB"))

    @property
    def image_array(self):
        return np.array(self.image)

    @property
    def data(self):
        return convnext_preprocess(self.image)

    @property
    def short_name(self):
        return self.class_name.split(",")[0]

    def __repr__(self):
        return f"<ValImage | {self.class_name} | {self.fname.split('/')[-1]}>"


class BlankImage:
    def __init__(self):
        self.class_name = "blank image"
        self.image_array = np.zeros((224, 224, 3), dtype=np.uint8)
        self.image = Image.fromarray(self.image_array)
        self.short_name = "blank image"

    def __repr__(self):
        return f"<ValImage | blank image"

    @property
    def data(self):
        assert False, "You probably don't want to get convnext data from blank image"

    def show(self):
        display(self.image)


blank_image = BlankImage()


class ValData:
    def __init__(self, files=False):
        if files is False:
            self.init_w_entire_validation_set()
        elif isinstance(files[0], ValImage):
            self.val_images = files
        else:
            assert False

    @cache
    def init_w_entire_validation_set(self):
        """Load entire validation dataset"""
        with open(dirname + "/fname_to_info.json", "r") as f:
            fname_to_info = json.load(f)
        self.val_images = []
        for i, fname in enumerate(list(fname_to_info.keys())):
            info = fname_to_info[fname]
            self.val_images.append(
                ValImage(
                    fname=dirname + "/" + fname,
                    class_name=info["class_name"],
                    class_id=info["class_id"],
                    val_index=i,
                )
            )

    def as_batches(self, batch_size):
        if len(self.val_images) % batch_size != 0:
            print(
                f"Warning: given batch size [{batch_size}] doesn't cleanly divide into len(images) = {len(self.val_images)}."
            )
        for i in range(0, len(self.val_images), batch_size):
            yield ValData(self.val_images[i : i + batch_size])

    @cache
    def get_class(self, class_id):
        return ValData(
            [image for image in self.val_images if image.class_id == class_id]
        )

    def __len__(self):
        return len(self.val_images)

    def __getitem__(self, subscript):
        if isinstance(subscript, slice):
            return ValData(
                self.val_images[subscript.start : subscript.stop : subscript.step]
            )
        if is_iterable(subscript):
            return ValData([self.val_images[i] for i in subscript])
        else:
            return self.val_images[subscript]

    def __add__(self, other):
        if isinstance(other, ValData):
            return ValData(self.val_images + other.val_images)
        else:
            assert False, "Can only add ValData to a another ValData"

    def show(self, width=-1, scale=1):
        display(self.get_image_grid(width=width, scale=scale))

    @property
    def data(self):
        return torch.stack([image.data for image in self.val_images])

    @property
    def images(self):
        return [image.image for image in self.val_images]

    @property
    def image_array(self):
        return np.stack([image.image_array for image in self.val_images], axis=0)

    @property
    def class_ids(self):
        return [image.class_id for image in self.val_images]

    @property
    def class_names(self):
        return simple_class_labels[list(set(self.class_ids))]

    @property
    def all_simple_class_labels(self):
        return simple_class_labels

    @property
    def _val_indexes(self):
        return [image._val_index for image in self.val_images]

    def get_image_grid(self, width: int = -1, scale: int = 1):
        return get_image_grid(images=self.images, width=width, scale=scale)
