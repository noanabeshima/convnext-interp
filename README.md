# Convnext Interp

To use `imagenet_val`, run `curl -o images.zip https://nx33059.your-storageshare.de/s/Wr7g489SrK63C38/download?path=/&file_id=5`
and extract `images.zip` in `convnext_interp/imagenet_val/`.

Note that the `convnext` module has a different license than the rest of the repository as it is mostly composed of code from [the ConvNext repository](https://github.com/facebookresearch/ConvNeXt).

`imagenet_val/ordered_class_labels.txt` is slightly edited from the original class labels to make them more clear.


In addition to the packages in requirements.txt, this requires PyTorch installed.