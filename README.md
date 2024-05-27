# BioAcoustic Inference
Authors: Ilyass Moummad, Romain Serizel, Nicolas Farrugia
---
BioAcoustic Inference provides an easy to use Python code to predict occurences of an animal sound of interest using only few annotations of an audio recording.

The code and the pretrained model is from our ICASSP 2024 work "Regularized Contrastive Learning for Few-shot Bioacoustic Sound Event Detection", [Code here](https://github.com/ilyassmoummad/RCL_FS_BSED), and [weights here](https://zenodo.org/records/11353694) (you don't have to download them in advance, they will be automatically downloaded from the code).

## Installation

```pip install -r requirements.txt```

## Usage

Have a 'txt' annotation file with few annotations (as little as one) with the same name as the audio file and launch the following code (replace audio_path with the corresponding name):

```python3 inference.py --audiopath audio_path```

Optionally you can specify where to download the model weights using ```--ckpt```, per default it is ```model/bioacoustics_model.pth```

This code will create a new 'txt' file (with '_inference' as a suffix) with the inferred annotations.

---
### Cite
```
@INPROCEEDINGS{10446409,
  author={Moummad, Ilyass and Farrugia, Nicolas and Serizel, Romain},
  booktitle={ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Regularized Contrastive Pre-Training for Few-Shot Bioacoustic Sound Detection}, 
  year={2024},
  volume={},
  number={},
  pages={1436-1440},
  keywords={Training;Event detection;Animals;Speech coding;Self-supervised learning;Signal processing;Feature extraction;Supervised contrastive learning;total coding rate;transfer learning;few-shot learning;bioacoustics;sound event detection},
  doi={10.1109/ICASSP48485.2024.10446409}}
```