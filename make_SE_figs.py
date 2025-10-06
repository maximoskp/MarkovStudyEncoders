import torch
import torch.nn as nn
from models import SingleEncoderModel, DualEncoderModel
from data_utils import HM_Dataset
from generate_utils import load_DE, load_SE, nucleus_token_by_token_generate
import matplotlib.pyplot as plt
import seaborn as sns
from plot_utils import save_attention_maps_with_split, save_attention_maps

device_name = 'cpu'
unmasking_order = 'start' # in ['random', 'start', 'end', 'certain', 'uncertain']

for nvis in [None, 0]:
    for subfolder in ['T2_M2', 'T10_M2', 'T2_M10', 'T10_M10']:
        test_dataset = HM_Dataset("data/test_" + subfolder + ".pkl")
        d = test_dataset[0]
        model = load_SE(
            test_dataset.m_vocab_size,
            test_dataset.h_vocab_size,
            test_dataset.seq_len,
            subfolder=subfolder,
            device_name=device_name,
            nvis=nvis,
        )
        h_tokens = nucleus_token_by_token_generate(
            model,
            d['m_seq'],
            test_dataset.mask_token_id,
            temperature=0.5,
            p=0.9,
            unmasking_order='start'
        )
        save_dir='figs/attn_maps/SE/' + subfolder
        if nvis is not None:
            save_dir += '_nvis' + str(nvis)
        self_attns = model.get_attention_maps()
        save_attention_maps_with_split(
            self_attns,
            melody_len=test_dataset.seq_len,
            save_dir=save_dir + '/self/',
            prefix='self',
            title_info=True
        )