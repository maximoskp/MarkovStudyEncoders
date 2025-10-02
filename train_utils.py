import torch
from torcheval.metrics.text import Perplexity
import random
from tqdm import tqdm
from data_utils import compute_normalized_token_entropy
import random
import csv
import numpy as np
import os
from transformers import get_cosine_schedule_with_warmup

perplexity_metric = Perplexity(ignore_index=-100)

def full_to_partial_masking(
        h_token_ids,
        mask_token_id,
        num_visible=0
    ):
    """
    Generate visible input and denoising target for diffusion-style training.

    Args:
        harmony_tokens (torch.Tensor): Tensor of shape (B, L) containing target harmony token ids.
        stage (int): Current training stage (0 to total_stages - 1).
        total_stages (int): Total number of diffusion stages.
        mask_token_id (int): The token ID used to mask hidden positions in visible_harmony.
        device (str or torch.device): Target device.

    Returns:
        visible_harmony (torch.Tensor): Tensor of shape (B, L) with visible tokens (others masked).
        denoising_target (torch.Tensor): Tensor of shape (B, L) with tokens to predict (others = -100).
    """
    device = h_token_ids.device
    B, L = h_token_ids.shape

    visible_h = torch.full_like(h_token_ids, fill_value=mask_token_id)
    denoising_target = torch.full_like(h_token_ids, fill_value=-100)  # -100 is ignored by CrossEntropyLoss

    perm = torch.randperm(L, device=device)

    visible_idx = perm[:num_visible]
    predict_idx = perm[num_visible:]  # predict all remaining

    visible_h[:, visible_idx] = h_token_ids[:, visible_idx]
    denoising_target[:, predict_idx] = h_token_ids[:, predict_idx]

    return visible_h, denoising_target
# end full_to_partial_masking

def validation_loop(model, valloader, mask_token_id, bar_token_id, \
                    num_visible, loss_fn, epoch, step, \
                    train_loss, train_accuracy, \
                    train_perplexity, train_token_entropy,
                    best_val_loss, saving_version, results_path=None, transformer_path=None, tqdm_position=0):
    device = model.device
    model.eval()
    with torch.no_grad():
        val_loss = 0
        running_loss = 0
        batch_num = 0
        running_accuracy = 0
        val_accuracy = 0
        running_perplexity = 0
        val_perplexity = 0
        running_token_entropy = 0
        val_token_entropy = 0
        print('validation')
        with tqdm(valloader, unit='batch', position=tqdm_position) as tepoch:
            tepoch.set_description(f'Epoch {epoch}@{step}| val')
            for batch in tepoch:
                perplexity_metric.reset()
                m_seq = batch["m_seq"].to(device)           # (B, 256, 140)
                h_seq = batch["m_seq"].to(device)         # (B, 256)
                
                # Apply masking to h
                h_visible, h_target = full_to_partial_masking(
                    h_seq,
                    mask_token_id,
                    num_visible,
                    bar_token_id=bar_token_id
                )
                
                # Forward pass
                logits = model(
                    m_seq.to(device),
                    h_visible.to(device),
                )

                # Compute loss only on masked tokens
                loss = loss_fn(logits.view(-1, logits.size(-1)), h_target.view(-1))

                # update loss and accuracy
                batch_num += 1
                running_loss += loss.item()
                val_loss = running_loss/batch_num
                # accuracy
                predictions = logits.argmax(dim=-1)
                # mask = harmony_target != harmony_input # harmony_target != -100
                # mask = torch.logical_and(harmony_target != harmony_input, harmony_target != -100)
                mask = h_target != -100
                running_accuracy += (predictions[mask] == h_target[mask]).sum().item()/mask.sum().item()
                val_accuracy = running_accuracy/batch_num
                # perplexity
                running_perplexity += perplexity_metric.update(logits, h_target).compute().item()
                val_perplexity = running_perplexity/batch_num
                # token entropy
                _, entropy_per_batch = compute_normalized_token_entropy(logits, h_target, pad_token_id=-100)
                running_token_entropy += entropy_per_batch
                val_token_entropy = running_token_entropy/batch_num

                tepoch.set_postfix(loss=val_loss, accuracy=val_accuracy)
            # end for batch
    # end with tqdm
    if transformer_path is not None:
        print('saving!')
        saving_version += 1
        best_val_loss = val_loss
        torch.save(model.state_dict(), transformer_path)
    print(f'validation: accuracy={val_accuracy}, loss={val_loss}')
    print('results_path: ', results_path)
    if results_path is not None:
        with open( results_path, 'a' ) as f:
            writer = csv.writer(f)
            writer.writerow( [epoch, step, num_visible, train_loss, train_accuracy, \
                            train_perplexity, train_token_entropy, \
                            val_loss, val_accuracy, \
                            val_perplexity, val_token_entropy, \
                            saving_version] )
    return best_val_loss, saving_version
# end validation_loop

def train_with_curriculum(
    model, optimizer, trainloader, valloader, loss_fn, mask_token_id,
    curriculum_type='fixed',
    epochs=100,
    condition_dim=None,
    exponent=5,
    results_path=None,
    transformer_path=None,
    bar_token_id=None,
    validations_per_epoch=1,
    tqdm_position=0
):
    # device = next(model.parameters()).device
    device = model.device
    perplexity_metric.to(device)
    best_val_loss = np.inf
    saving_version = 0

    # save results and model
    print('results_path:', results_path)
    if results_path is not None:
        result_fields = ['epoch', 'step', 'n_vis', 'train_loss', 'train_acc', \
                        'train_ppl', 'train_te', 'val_loss', \
                        'val_acc', 'val_ppl', 'val_te', 'sav_version']
        with open( results_path, 'w' ) as f:
            writer = csv.writer(f)
            writer.writerow( result_fields )

    # Compute total training steps
    total_steps = len(trainloader) * epochs
    # Define the scheduler
    # warmup_steps = int(0.1 * total_steps)  # 10% of total steps for warmup
    # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    step = 0

    for epoch in range(epochs):
        train_loss = 0
        running_loss = 0
        batch_num = 0
        running_accuracy = 0
        train_accuracy = 0
        running_perplexity = 0
        train_perplexity = 0
        running_token_entropy = 0
        train_token_entropy = 0
        
        with tqdm(trainloader, unit='batch', position=tqdm_position) as tepoch:
            tepoch.set_description(f'Epoch {epoch}@{step} | trn')
            for batch in tepoch:
                perplexity_metric.reset()
                model.train()
                m_seq = batch["m_seq"].to(device)           # (B, 256, 140)
                h_seq = batch["m_seq"].to(device)         # (B, 256)

                # Apply masking to h
                percent_visible = min(1.0, (step+1)/total_steps)**exponent  # 5th power goes around half way near zero
                L = h_seq.shape[1]
                num_visible = min( int(L * percent_visible), L-1 )  # ensure at least one token is predicted
                h_visible, h_target = full_to_partial_masking(
                    h_seq,
                    mask_token_id,
                    num_visible,
                    bar_token_id=bar_token_id
                )
                
                # Forward pass
                logits = model(
                    m_seq.to(device),
                    h_visible.to(device),
                )

                # Compute loss only on masked tokens
                loss = loss_fn(logits.view(-1, logits.size(-1)), h_target.view(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # scheduler.step()

                # update loss and accuracy
                batch_num += 1
                running_loss += loss.item()
                train_loss = running_loss/batch_num
                # accuracy
                predictions = logits.argmax(dim=-1)
                # mask = torch.logical_and(harmony_target != harmony_input, harmony_target != -100)
                mask = h_target != -100
                running_accuracy += (predictions[mask] == h_target[mask]).sum().item()/max(1,mask.sum().item())
                train_accuracy = running_accuracy/batch_num
                # perplexity
                running_perplexity += perplexity_metric.update(logits, h_target).compute().item()
                train_perplexity = running_perplexity/batch_num
                # token entropy
                _, entropy_per_batch = compute_normalized_token_entropy(logits, h_target, pad_token_id=-100)
                running_token_entropy += entropy_per_batch
                train_token_entropy = running_token_entropy/batch_num

                tepoch.set_postfix(loss=train_loss, accuracy=train_accuracy)
                step += 1
                if step%(total_steps//(epochs*validations_per_epoch)) == 0 or step == total_steps:
                    best_val_loss, saving_version = validation_loop(
                        model,
                        valloader,
                        mask_token_id,
                        bar_token_id,
                        num_visible,
                        loss_fn,
                        epoch,
                        step,
                        train_loss,
                        train_accuracy,
                        train_perplexity,
                        train_token_entropy,
                        best_val_loss,
                        saving_version,
                        results_path=results_path,
                        transformer_path=transformer_path,
                        tqdm_position=tqdm_position
                    )
            # end for batch
        # end with tqdm
    # end for epoch
# end train_with_curriculum