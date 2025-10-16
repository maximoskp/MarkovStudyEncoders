import pickle
from data_utils import HM_Dataset
from models import DE_no_pos
import torch
from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import argparse
from train_utils import train_with_curriculum
import os

curriculum_types = ['fixed', 'markov']

def main():

    # Create the argument parser
    parser = argparse.ArgumentParser(description='Script for training a GridMLM model with a specific curriculum type.')

    # Define arguments
    parser.add_argument('-c', '--curriculum', type=str, help='Specify the curriculum type name among: ' + repr(curriculum_types), required=True)
    parser.add_argument('-f', '--subfolder', type=str, help='Specify subfolder to save the model and results. This name also defines tokenizer and token setup.', required=True)
    parser.add_argument('-g', '--gpu', type=int, help='Specify whether and which GPU will be used by used by index. Not using this argument means use CPU.', required=False)
    parser.add_argument('-e', '--epochs', type=int, help='Specify number of epochs. Defaults to 100.', required=False)
    parser.add_argument('-l', '--learningrate', type=float, help='Specify learning rate. Defaults to 5e-5.', required=False)
    parser.add_argument('-b', '--batchsize', type=int, help='Specify batch size. Defaults to 8.', required=False)
    
    # Parse the arguments
    args = parser.parse_args()
    curriculum_type = args.curriculum
    exponent = 5
    subfolder = ''
    if args.subfolder:
        subfolder = args.subfolder
    device_name = 'cpu'
    if args.gpu is not None:
        if args.gpu > -1:
            device_name = 'cuda:' + str(args.gpu)
    epochs = 200
    if args.epochs:
        epochs = args.epochs
    lr = 1e-4
    if args.learningrate:
        lr = args.learningrate
    batchsize = 16
    if args.batchsize:
        batchsize = args.batchsize

    train_dataset = HM_Dataset("data/train_" + subfolder + ".pkl")
    val_dataset = HM_Dataset("data/test_" + subfolder + ".pkl")

    trainloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=batchsize, shuffle=True)

    if device_name == 'cpu':
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device(device_name)
        else:
            print('Selected device not available: ' + device_name)
    # end device selection
    
    loss_fn=CrossEntropyLoss(ignore_index=-100)
    # # Precompute once before training
    # class_weights = compute_class_weights_from_dataset(
    #     train_dataset, tokenizer, scheme="temp", alpha=0.5
    # )

    # # Define loss function with weights
    # loss_fn = torch.nn.CrossEntropyLoss(
    #     weight=class_weights.to(device), ignore_index=-100
    # )
    model = DE_no_pos(
        train_dataset.m_vocab_size, 
        train_dataset.h_vocab_size, 
        train_dataset.seq_len, 
        device=device
    )
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    # save results
    os.makedirs('results/DE_np/', exist_ok=True)
    results_path = 'results/DE_np/' + subfolder + '.csv'

    os.makedirs('saved_models/DE_np/', exist_ok=True)
    save_dir = 'saved_models/DE_np/'
    transformer_path = save_dir + subfolder + '.pt'

    train_with_curriculum(
        model, optimizer, trainloader, valloader, loss_fn, train_dataset.mask_token_id,
        curriculum_type=curriculum_type,
        epochs=epochs,
        exponent=exponent,
        results_path=results_path,
        transformer_path=transformer_path,
    )
    
# end main

if __name__ == '__main__':
    main()