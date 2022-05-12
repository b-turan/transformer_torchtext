''' 
See detailed tutorial:
https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb
'''

import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from dataloader.dataloader import get_data, get_dataloader
from metric.translation import (calculate_bleu, calculate_bleu_alt,
                                translate_sentence_vectorized,
                                translate_single_sentence)
from model.architecture import Decoder, Encoder, build_model
from utils import arg_parser, utils
from utils.vocab import SRC, TRG

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Running on Device:", device)

parser = arg_parser.create_parser()
args = parser.parse_args()
writer = SummaryWriter()

# trainer specific arguments
BATCH_SIZE = args.batch_size
DATASET = args.dataset
LEARNING_RATE = args.learning_rate
N_EPOCHS = args.epochs
CLIP = args.clip
# model specific arguments
HID_DIM = args.hid_dim
ENC_LAYERS = args.enc_layers
DEC_LAYERS = args.dec_layers
ENC_HEADS = args.enc_heads
DEC_HEADS = args.dec_heads
ENC_PF_DIM = args.enc_pf_dim
DEC_PF_DIM = args.dec_pf_dim
ENC_DROPOUT = args.enc_dropout
DEC_DROPOUT = args.dec_dropout
# program specific arguments
SEED = args.seed
CHECKPOINT_PATH = f'checkpoint/{DATASET}_{args.checkpoint}'


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def train_epoch(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        optimizer.zero_grad()
        output, _ = model(src, trg[:,:-1])     
        #output = [batch size, trg len - 1, output dim]
        #trg = [batch size, trg len]   
        output_dim = output.shape[-1]     
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)        
        #output = [batch size * trg len - 1, output dim]
        #trg = [batch size * trg len - 1]  
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output, _ = model(src, trg[:,:-1])
            #output = [batch size, trg len - 1, output dim]
            #trg = [batch size, trg len]
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:,1:].contiguous().view(-1)
            #output = [batch size * trg len - 1, output dim]
            #trg = [batch size * trg len - 1]
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def run_training():
    """
    - Training loop through all epochs. 
    - Prints train and validation loss of each epoch. 
    - Stores parameters of model with lowest validation loss of all epochs to "checkpoint/{DATASET}_checkpoint.pt"
    """
    best_valid_loss = float('inf')
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss = train_epoch(model, train_iterator, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_iterator, criterion)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        pred_trgs, trgs, bleu_score = calculate_bleu(test_iterator, SRC, TRG, model, device)
        # bleu_score = calculate_bleu_alt(test_data, SRC, TRG, model, device)
        print(f'BLEU score = {bleu_score*100:.2f}')
        # logging tensorboard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/valid", valid_loss, epoch)
        writer.add_scalar("BLEU Score", bleu_score, epoch)


def epoch_time(start_time, end_time):
    ''' Returns elapsed time in minutes and seconds. '''
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def load_checkpoint(PATH):
    model.load_state_dict(torch.load(PATH))


if __name__ == "__main__":
    # Data Preprocessing
    train_data, valid_data, test_data = get_data(DATASET)
    SRC.build_vocab(train_data, min_freq = 2)
    TRG.build_vocab(train_data, min_freq = 2)
    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)

    # initialize model
    enc = Encoder(INPUT_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device)
    dec = Decoder(OUTPUT_DIM, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device)
    model = build_model(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device)
    model.apply(initialize_weights)
    print(f'The model has {utils.count_parameters(model):,} trainable parameters')

    # initialize optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

    # initialize dataloaders
    train_iterator, valid_iterator, test_iterator = get_dataloader(train_data, valid_data, test_data, BATCH_SIZE, device)

    if args.train:
        # perform training and evaluation on training and validation set, respectively
        run_training()
        writer.flush()

    # run model on test set
    test_loss = evaluate(model, test_iterator, criterion)
    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

    # check inference on single sentence from validation dataset
    translate_single_sentence(valid_data, SRC, TRG, model, device)

    # calculate BLEU score on test dataset 
    print(f'Final Bleu Score Evaluation on Test Set:')
    pred_trgs, trgs, bleu_score = calculate_bleu(test_iterator, SRC, TRG, model, device)
    print(f'BLEU score = {bleu_score*100:.2f}')
