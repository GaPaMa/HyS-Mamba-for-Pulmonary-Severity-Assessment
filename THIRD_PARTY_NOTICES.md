# Third-party components

`src/lung_severity_assesmnt/models/tiny_vim/tvimblock.py` is the adapted TViMBlock/SS2D source supplied with this project. The revised manuscript attributes the implementation to [TinyViM](https://github.com/xwmaxwma/TinyViM/blob/main/model/tvimblock.py), by Xiaowen Ma, Zhenliang Ni and Xinghao Chen. It is preserved byte-for-byte.

TinyViM's README refers to [VMamba](https://github.com/MzeroMiko/VMamba) for the separately compiled selective-scan CUDA operator. The CUDA source and compiled extension are not bundled here. Their terms remain those of their respective authors. The TinyViM repository root inspected for this preparation did not contain a standalone LICENSE file; no MIT relicensing of its adapted block is asserted by this repository.

ConvNeXt is supplied through `timm`, and the ViT implementation and public pretrained weights through Hugging Face `transformers`. These libraries and weights retain their respective terms. Other installed dependencies likewise retain their own licences.

Dataset access rights and manuscript-figure rights are separate from the licence for author-owned software. Dataset images and labels are not distributed in this repository.

