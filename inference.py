import torch
import torch.nn as nn
import torchaudio
from torchaudio import transforms as T
from model.model import ResNet
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from augmentations import RandomCrop, Resize, Compander
from args import args

def get_distance(proto_pos, neg_proto, query_set_out):
    prototypes = torch.stack([proto_pos, neg_proto]).squeeze(1)
    dists = torch.cdist(query_set_out, prototypes)
    return dists

if __name__ == "__main__":

    # General params
    CKPT = args.ckpt
    TARGET_SR = args.sr
    N_MELS = args.nmels
    HOP_MEL = args.hoplen
    N_SHOT = args.nshot
    BATCH_SIZE = args.bs
    NUM_WORKERS = args.workers
    fps = TARGET_SR/HOP_MEL
    print(f"Frames per second: {fps} (= target sample rate {args.sr} Hz / hop length {HOP_MEL})")
    emb_dim = 2048 #  dimension of the latent space, check architecture in models.py
    

    # Spectrogram
    mel = T.MelSpectrogram(sample_rate=args.sr, n_fft=args.nfft, hop_length=args.hoplen, f_min=args.fmin, f_max=args.fmax, n_mels=args.nmels)
    power_to_db = T.AmplitudeToDB()
    transform = nn.Sequential(mel, power_to_db)

    inference = {}
    inference['onset'] = []
    inference['offset'] = []
    onset_arr = np.array([])
    offset_arr = np.array([])

    audiopath = args.audiopath
    if args.annotpath is None:
        ext = 'txt'
        annotpath = audiopath.split('.')[-2]+ '.' + ext
    else:
        annotpath = args.annotpath
        ext = annotpath.split('.')[-1]
    csv_path = args.audiopath.split('.')[-2] + '_inference.' + ext
    if ext == 'txt':
        seperator = '\t'
    elif ext == 'csv':
        seperator = ','

    print(f"Processing file {audiopath} with annotation file {annotpath}")
    print(f"Using events annotations from {annotpath}")
    df = pd.read_csv(annotpath, sep=seperator, header=None)
    index_sup = [i for i in range(N_SHOT)]
    start_time = [int(np.floor(start * fps)) for start in df.iloc[:, 0]]
    end_time = [int(np.floor(start * fps)) for start in df.iloc[:, 1]]
    difference = []
    for index in index_sup:
        difference.append(end_time[index] - start_time[index])

    max_len = int(round(np.mean(difference)))
    print(f"Average duration of events: {max_len} frames")

    if max_len <= 17:
        win_len = 17
    elif max_len > 17 and max_len <= 100 :
        win_len = max_len
    elif max_len > 100 and max_len <= 200 :
        win_len = max_len//2
    elif max_len > 200 and max_len <= 400 :
        win_len = max_len//4
    else:
        win_len = max_len//8
    seg_hop = win_len//2
    print(f"Hop length for segmentation: {seg_hop} frames")
    
    print("Loading audio")
    wav, sr = torchaudio.load(audiopath)

    resample = T.Resample(sr, TARGET_SR)

    print(f"Resampling audio from {sr} Hz to {TARGET_SR} Hz")
    wav = resample(wav)
    # if more than one channel, average
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    print("Computing mel spectrogram")
    melspec = transform(wav)

    features_pos = []
    features_neg = []

    print("Segmenting Features from Mel Spectrogram")
    for index in tqdm(range(len(index_sup))):
        # pos features
        start_idx = start_time[index_sup[index]] 
        if end_time[index_sup[index]] - start_idx > win_len:
            while end_time[index_sup[index]] - start_idx > win_len:
                spec = melspec[...,start_idx:start_idx+win_len]
                start_idx += seg_hop
                features_pos.append(spec)  
            if end_time[index_sup[index]] - start_idx > win_len//2:
                spec = melspec[...,start_idx:end_time[index_sup[index]]]
                repeat_num = int(win_len / (spec.shape[-1])) + 1
                spec = spec.repeat(1,1,repeat_num)
                spec = spec[...,:int(win_len)]
                features_pos.append(spec)  
        else:
            if end_time[index_sup[index]] - start_idx > 0:
                spec = melspec[...,start_idx:end_time[index_sup[index]]]
                repeat_num = int(win_len / (spec.shape[-1])) + 1
                spec = spec.repeat(1,1,repeat_num)
                spec = spec[...,:int(win_len)]
                features_pos.append(spec)  

        # neg features
        start_idx = end_time[index_sup[index]]
        if index < len(index_sup)-1:
            if start_time[index_sup[index+1]] - start_idx > win_len:
                while start_time[index_sup[index+1]] - start_idx > win_len:
                    spec = melspec[...,start_idx:start_idx+win_len]
                    start_idx += seg_hop
                    features_neg.append(spec)
                if start_time[index_sup[index+1]] - start_idx > win_len//2:
                    spec = melspec[...,start_idx:start_time[index_sup[index+1]]]
                    repeat_num = int(win_len / (spec.shape[-1])) + 1
                    spec = spec.repeat(1,1,repeat_num)
                    spec = spec[...,:int(win_len)]
                    features_neg.append(spec)
            else:
                if start_time[index_sup[index+1]] - start_idx > 0:
                    spec = melspec[...,start_idx:start_time[index_sup[index+1]]]
                    repeat_num = int(win_len / (spec.shape[-1])) + 1
                    spec = spec.repeat(1,1,repeat_num)
                    spec = spec[...,:int(win_len)]
                    features_neg.append(spec)   
        
        # add event before first pos to neg proto
        end_t0 = start_time[index_sup[0]] 
        start_t0 = 0
        curr_t0 = start_t0
        if end_t0 - curr_t0 > win_len : 
            while end_t0 - curr_t0 > win_len :
                spec = melspec[...,curr_t0:curr_t0+win_len]
                curr_t0 += seg_hop
                features_neg.append(spec)
            if end_t0 - curr_t0 > win_len//2 :
                spec = melspec[...,curr_t0:curr_t0+win_len]
                repeat_num = int(win_len / (spec.shape[-1])) + 1
                spec = spec.repeat(1,1,repeat_num)
                spec = spec[...,:int(win_len)]
                features_neg.append(spec)
        else:
            if end_t0 - curr_t0 > 0 :
                spec = melspec[...,curr_t0:curr_t0+win_len]
                repeat_num = int(win_len / (spec.shape[-1])) + 1
                spec = spec.repeat(1,1,repeat_num)
                spec = spec[...,:int(win_len)]
                features_neg.append(spec)

    features_pos = torch.stack(features_pos)
    features_neg = torch.stack(features_neg)
    nb_positives = features_pos.shape[0]
    nb_negatives = features_neg.shape[0]

    print(f"Number of positive features: {nb_positives}")
    print(f"Number of negative features: {nb_negatives}")

    # query features
    labels_q = []
    features_q = []

    last_frame = melspec.shape[-1]
    curr_frame = end_time[index_sup[-1]]
    if last_frame - curr_frame > win_len:
        while last_frame - curr_frame > win_len:
            spec = melspec[...,curr_frame:curr_frame+win_len]
            features_q.append(spec)
            curr_frame += seg_hop
        if last_frame - curr_frame > win_len//2:
            spec = melspec[...,curr_frame:last_frame]
            repeat_num = int(win_len / (spec.shape[-1])) + 1
            spec = spec.repeat(1,1,repeat_num)
            spec = spec[...,:int(win_len)]
            features_q.append(spec)   
    else:
        if last_frame - curr_frame > 0: #win_len//2
            spec = melspec[...,curr_frame:last_frame]
            repeat_num = int(win_len / (spec.shape[-1])) + 1
            spec = spec.repeat(1,1,repeat_num)
            spec = spec[...,:int(win_len)]
            features_q.append(spec)

    features_q = torch.stack(features_q)
    print(f"Number of query features: {features_q.shape[0]}")
    # Light DA to create different views
    rc = RandomCrop(n_mels=128, time_steps=features_q.shape[-1], tcrop_ratio=0.9)
    resize = Resize(n_mels=128, time_steps=features_q.shape[-1])
    comp = Compander(comp_alpha=0.9)
    makeview = nn.Sequential(rc, resize, comp)

    # Loading model
    print("Loading model..")
    encoder = ResNet(method='scl')
    ckpt = torch.load(CKPT, map_location=torch.device('cpu'))
    encoder.load_state_dict(ckpt['encoder'], strict=False)
    encoder = encoder.to(args.device)

    # Preparing loader
    ds_pos = TensorDataset(features_pos)
    ds_neg = TensorDataset(features_neg)
    ds_q = TensorDataset(features_q)

    loader_pos = DataLoader(ds_pos, batch_size=BATCH_SIZE)
    loader_neg = DataLoader(ds_neg, batch_size=BATCH_SIZE)
    loader_q = DataLoader(ds_q, batch_size=BATCH_SIZE)

    labels_pred = []
    labels_comb = []
    with torch.no_grad():
        encoder.train()

        pos_proto = []
        pos_feat = torch.zeros(0, emb_dim)
        print("Generating positive features...")
        for b_idx, x_p in tqdm(enumerate(loader_pos),total=len(loader_pos)):
            if args.multiview:
                zp_views = []
                for i_v in range(args.nviews):
                    xp = x_p[0].to(args.device)
                    if i_v != 0:
                        xp = makeview(xp)
                    zp, _ = encoder(xp)
                    zp = zp.detach().cpu()
                    zp_views.append(zp)
                zp_views = torch.stack(zp_views)
                zp_mean = zp_views.mean(0).mean(0).unsqueeze(0)
                pos_feat = torch.cat((pos_feat, zp_mean), dim=0)
            else:  
                z_pos, _ = encoder(x_p[0].to(args.device))
                z_pos = z_pos.detach().cpu()
                z_pos_mean = z_pos.mean(dim=0).unsqueeze(0)
                pos_feat = torch.cat((pos_feat, z_pos_mean), dim=0)
        pos_proto = pos_feat.mean(dim=0)

        neg_proto = []
        neg_feat = torch.zeros(0, emb_dim)
        print("Generating negative features...")
        for b_idx, x_n in tqdm(enumerate(loader_neg),total=len(loader_neg)):
            if args.multiview:
                zn_views = []
                z_feat = torch.zeros(0, emb_dim)
                for i_v in range(args.nviews):
                    xn = x_n[0].to(args.device)
                    if i_v != 0:
                        xn = makeview(xn)
                    zn, _ = encoder(xn)
                    zn = zn.detach().cpu()
                    zn_views.append(zn)
                zn_views = torch.stack(zn_views)
                zn_mean = zn_views.mean(0).mean(0).unsqueeze(0)
                neg_feat = torch.cat([neg_feat, zn_mean], dim=0)            
            else:  
                z_neg, _ = encoder(x_n[0].to(args.device))
                z_neg = z_neg.detach().cpu()
                z_neg_mean = z_neg.mean(dim=0).unsqueeze(0)
                neg_feat = torch.cat((neg_feat, z_neg_mean), dim=0)
        neg_proto = neg_feat.mean(dim=0)

        encoder.eval()
        labels_pred = []
        labels_comb = []
        print("Generating Query features...")
        for x_q in tqdm(loader_q, total=len(loader_q)):
            if args.multiview:
                zq_views = []
                for i_v in range(args.nviews):
                    xq = x_q[0].to(args.device)
                    if i_v != 0:
                        xq = makeview(xq)
                    zq, _ = encoder(xq)
                    zq = zq.detach().cpu()
                    zq_views.append(zq)
                zq_views = torch.stack(zq_views)
                z_q = zq_views.mean(0)
            else:
                z_q, _ = encoder(x_q[0].to(args.device))
                z_q = z_q.detach().cpu()

            distances = get_distance(pos_proto, neg_proto, z_q)
            label_pred = torch.argmax(distances, dim=-1)

            labels_pred.extend(label_pred.tolist())

    print("Generating final predictions...")
    labels_comb.append(labels_pred)

    labels_comb = torch.tensor(labels_comb)

    labs_pred, _ = torch.mode(labels_comb, axis=0)

    labs_pred = np.array(labs_pred)

    krn = np.array([1, -1])

    changes = np.convolve(krn, labs_pred)

    onset_frames = np.where(changes == 1)[0]
    offset_frames = np.where(changes == -1)[0]

    str_time_query = end_time[index_sup[-1]] * HOP_MEL / TARGET_SR

    onset = (onset_frames ) * (seg_hop) * HOP_MEL / TARGET_SR
    onset = onset + str_time_query

    offset = (offset_frames ) * (seg_hop) * HOP_MEL / TARGET_SR
    offset = offset + str_time_query

    assert len(onset) == len(offset)
    inference['onset'].append(onset)
    inference['offset'].append(offset)
    onset_arr = np.append(onset_arr, onset)
    offset_arr = np.append(offset_arr, offset)

    df_inference = pd.DataFrame(list(zip(onset_arr, offset_arr)), columns=None)
    print(f"Saving inference to {csv_path}")
    df_inference.to_csv(csv_path)
    #df_inference.to_csv(csv_path, index=False)
    print("Done!")