import argparse

parser = argparse.ArgumentParser()

# generic
parser.add_argument("--device", type=str, default='cpu') #device to train on ['cpu', 'cuda', 'cuda:0', 'cuda:1', ...]
parser.add_argument("--workers", type=int, default=1) #number of workers
parser.add_argument("--bs", type=int, default=1) #batch size

# data
parser.add_argument("--audiopath", type=str) #path of audio file
parser.add_argument("--annotpath", type=str) #path to annotation file

# model checkpoint
parser.add_argument("--ckpt", type=str, default='model/bioacoustics_model.pth') #path of (pretrained) model checkpoint 

# few shot
parser.add_argument("--nshot", type=int, default=5) #number of shots

# audio
parser.add_argument("--sr", type=int, default=22050) #sampling rate for audio
parser.add_argument("--len", type=int, default=200) #segment duration for training in ms

# mel spec parameters
parser.add_argument("--nmels", type=int, default=128) #number of mels
parser.add_argument("--nfft", type=int, default=512) #size of FFT
parser.add_argument("--hoplen", type=int, default=128) #hop between STFT windows
parser.add_argument("--fmax", type=int, default=11025) #fmax
parser.add_argument("--fmin", type=int, default=50) #fmin

# data augmentation
parser.add_argument("--tratio", type=float, default=0.9) #time ratio for spectrogram crop
parser.add_argument("--comp", type=float, default=0.9) #compander coefficient to compress signal

# views for support/query at inference
parser.add_argument("--multiview", type=bool, default=True) #create views for support/query
parser.add_argument("--nviews", type=int, default=20) #number of views created

args = parser.parse_args()