# BioAcoustic Inference
Authors: Ilyass Moummad, Romain Serizel, Nicolas Farrugia
---
BioAcoustic Inference provides an easy to use Python code to predict occurences of an animal sound of interest using only few annotations of an audio recording.

The code and the pretrained model is from our ICASSP 2024 work [Regularized Contrastive Learning for Few-shot Bioacoustic Sound Event Detection](https://github.com/ilyassmoummad/RCL_FS_BSED)

## Installation

Todo: requirements.txt with the libraries needed + link to download the model checkpoint file ckpt.pth

## Usage

Put the downloaded checkpoint file 'ckpt.pth' in the 'model/' folder

Have a 'txt' annotation file with few annotations (as little as one) with the same name as the audio file and launch the following code (replace audio_path with the corresponding name):

```python3 inference.py --audiopath audio_path``

This code will create a new 'txt' file (with '_inference' as a suffix) with the inferred annotations

---
### Cite
```
@misc{moummad2023regularized,
      title={Regularized Contrastive Pre-training for Few-shot Bioacoustic Sound Detection}, 
      author={Ilyass Moummad and Romain Serizel and Nicolas Farrugia},
      year={2023},
      eprint={2309.08971},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```