# Unsupervised Deep Video Denoiser for Transmission Electron Microscopy


## Introduction
 This set of code is a fully unsupervised framework, namely **unsupervised deep video denoiser (UDVD)**, to train denoising models using exclusively real noisy data collected from a transmission electron microscope (TEM). The framework enables recovery of atomic-resolution information from TEM data, potentially improving the signal-to-noise ratio (SNR) by more than an order of magnitude.
 
 Assuming the data has minimal correlated noise, the denoiser will take a TEM movie in `.tif` format collected from a direct electron detector and generate the denoised result as a `.npy` file, which can be further converted to other file formats. It is recommended to run this denoiser on high-performance computers (hpc).

## Usage
### Installation
```shell
git clone https://github.com/crozier-del/UDVD-MF-Denoising
cd UDVD-MF-Denoising
conda env create -n denoise-HDR -f environment.yaml
```

### Running
```shell
conda activate denoise-HDR
python denoise_mf.py\
     --data path_to_tiff_file  
     --num-epochs 50
     --batch-size 1
     --image-size 256
```
### Arguments
* `data` **(required)**: Full path to the `.tif` file containing the video to be denoised.
* `num-epochs` Number of training epochs (default: 50).
* `batch-size`: Number of images per batch for training (default: 1). Adjust based on available GPU memory.
* `image-size`: Size of the square image patches used for training (default: 256).

### Example

The provided `PtCeO2_030303.tif` video can be denoised by running the following commands:

```shell
python denoise_mf.py --data "./examples/PtCeO2_030303.tif" 
```
After the denoising process completed, the denoised result `PtCeO2_030303_udvd_mf.npy` can be found in the same folder as the input file.

### Citation

If you use this code, please cite our work: 

*Unsupervised Deep Video Denoising*\
D. Y. Sheth, S. Mohan, J. L. Vincent, R. Manzorro, P. A. Crozier, M. M. Khapra, E. P. Simoncelli, C. Fernandez-Granda; **Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)**, 2021, pp. 1759-1768\
[https://arxiv.org/abs/2011.15045](https://arxiv.org/abs/2011.15045)

*Evaluating Unsupervised Denoising Requires Unsupervised Metrics*\
A. Marcos Morales, M. Leibovich, S. Mohan, J. L. Vincent, P. Haluai, M. Tan, P. A. Crozier, C. Fernandez-Granda; **Proceedings of the 40th International Conference on Machine Learning (ICML)**, PMLR 2023 Vol. 202, pp. 23937-23957.\
[https://arxiv.org/abs/2210.05553](https://arxiv.org/abs/2210.05553)
