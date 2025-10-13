import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from models import SingleEncoderModel, DualEncoderModel
import os

def nucleus_token_by_token_generate(
        model,
        m_seq,            # (1, seq_len, input_dim)
        mask_token_id,          # token ID used for masking
        temperature=1.0,        # optional softmax temperature
        p=0.9,                  # nucleus threshold
        unmasking_order='start', # in ['random', 'start', 'end', 'certain', 'uncertain']
    ):
    device = model.device
    seq_len = m_seq.shape[0]

    # --- 1. Initialize ---
    h_visible = torch.full((1, seq_len), mask_token_id, dtype=torch.long, device=device)

    step = 0
    while (h_visible == mask_token_id).any():
        with torch.no_grad():
            logits = model(
                m_seq.to(device),
                h_visible.to(device),
            )
        # --- Masked position selection ---
        masked_positions = (h_visible == mask_token_id).squeeze(0).nonzero(as_tuple=True)[0]
        if masked_positions.numel() == 0:
            break

        probs = torch.softmax(logits[0, masked_positions] / temperature, dim=-1)
        entropies = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1)

        if unmasking_order == 'random':
            idx = torch.randint(0, masked_positions.numel(), (1,))
            pos = masked_positions[idx].item()
        elif unmasking_order == 'uncertain':
            pos = masked_positions[torch.argmax(entropies)].item()
        elif unmasking_order == 'certain':
            pos = masked_positions[torch.argmin(entropies)].item()
        elif unmasking_order == 'start':
            pos = masked_positions[0].item()
        elif unmasking_order == 'end':
            pos = masked_positions[-1].item()
        else:
            pos = masked_positions[torch.randint(0, masked_positions.numel(), (1,))].item()

        # --- Nucleus sampling step ---
        logits_pos = logits[0, pos] / temperature
        logits_pos[ mask_token_id ] = logits_pos.min().item()/100  # prevent selecting mask token
        probs_pos = torch.softmax(logits_pos, dim=-1)

        # sort probs descending
        sorted_probs, sorted_idx = torch.sort(probs_pos, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # mask out tokens beyond nucleus p
        nucleus_mask = cumulative_probs <= p
        nucleus_mask[0] = True  # keep at least one token
        nucleus_probs = sorted_probs[nucleus_mask]
        nucleus_idx = sorted_idx[nucleus_mask]

        # renormalize
        nucleus_probs = nucleus_probs / nucleus_probs.sum()

        # sample
        sampled_idx = torch.multinomial(nucleus_probs, 1).item()
        token = nucleus_idx[sampled_idx].item()

        # update harmony
        h_visible[0, pos] = token
        step += 1
    
    return h_visible
# end nucleus_token_by_token_generate

def load_SE(
    m_vocab_size, 
    h_vocab_size, 
    seq_len, 
    subfolder=None,
    device_name='cuda:0',
    nvis=None,
):
    if device_name == 'cpu':
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device(device_name)
        else:
            print('Selected device not available: ' + device_name)
            device = torch.device('cpu')
    model = SingleEncoderModel(
        m_vocab_size, 
        h_vocab_size, 
        seq_len, 
        device=device
    )
    model_path = 'saved_models/SE/' + subfolder
    if nvis is not None:
        model_path += '_nvis' + str(nvis)
    model_path += '.pt'
    print('model_path: ',model_path)
    # checkpoint = torch.load(model_path, map_location=device_name, weights_only=True)
    checkpoint = torch.load(model_path, map_location=device_name)
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)
    return model
# end load_SE

def load_DE(
    m_vocab_size, 
    h_vocab_size, 
    seq_len, 
    subfolder=None,
    device_name='cuda:0',
    nvis=None,
):
    if device_name == 'cpu':
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device(device_name)
        else:
            print('Selected device not available: ' + device_name)
            device = torch.device('cpu')
    model = DualEncoderModel(
        m_vocab_size, 
        h_vocab_size, 
        seq_len, 
        device=device
    )
    model_path = 'saved_models/DE/' + subfolder
    if nvis is not None:
        model_path += '_nvis' + str(nvis)
    model_path += '.pt'
    print('model_path: ',model_path)
    # checkpoint = torch.load(model_path, map_location=device_name, weights_only=True)
    checkpoint = torch.load(model_path, map_location=device_name)
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)
    return model
# end load_DE