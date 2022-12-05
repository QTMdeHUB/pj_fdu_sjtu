import argparse
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import transformer.Constants as Constants
import Utils

from preprocess.Dataset import get_dataloader
from transformer.Models import Transformer
from tqdm import tqdm


def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float)


def get_future_mask(seq, time_future):
    """ pad the predict event """
    assert seq.dim() == 2
    gt = ~(seq.gt(time_future)).to(seq.device)
    eq0 = seq.eq(Constants.PAD).to(seq.device)
    return (gt ^ eq0).type(torch.long)


def prepare_dataloader(opt):
    """ Load data and prepare dataloader. """

    def load_data(name, dict_name):
        with open(name, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')
            num_types = data['dim_process']
            data = data[dict_name]
            return data, int(num_types)

    # train_data：list of list of dict，[[{}, {}, ..., {}], ...]
    print('[Info] Loading train data...')
    train_data, num_types = load_data(opt.data + 'train_one_1000.pkl', 'train')
    # print('[Info] Loading dev data...')
    # dev_data, _ = load_data(opt.data + 'dev_one_1000.pkl', 'dev')
    print('[Info] Loading test data...')
    test_data, _ = load_data(opt.data + 'test_one_1000.pkl', 'devtest')

    trainloader = get_dataloader(train_data, opt.batch_size, shuffle=True)
    testloader = get_dataloader(test_data, opt.batch_size, shuffle=False)
    return trainloader, testloader, num_types


def train_epoch(model, training_data, optimizer, pred_loss_func, opt):
    model.train()

    total_event_ll = 0  # cumulative event log-likelihood
    total_time_ae = 0  # cumulative time prediction squared-error
    total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions

    for batch in tqdm(training_data, mininterval=2,
                      desc='  - (Training)   ', leave=False):
        event_time, time_gap, event_type, event_qx = map(lambda x: x.to(opt.device), batch)
        optimizer.zero_grad()

        enc_out, prediction = model(event_type, event_time, event_qx)

        event_ll, non_event_ll = Utils.log_likelihood(model, enc_out, event_time, event_type)
        event_loss = -torch.sum(event_ll - non_event_ll)

        pred_loss, pred_num_event = Utils.type_loss_train(prediction[0], event_type, pred_loss_func)

        ae = Utils.time_loss(prediction[1], event_time)

        loss = event_loss + pred_loss + ae * opt.scaletimeloss
        loss.backward()
        optimizer.step()

        """ note keeping """
        total_event_ll += -event_loss.item()
        total_time_ae += ae.item()
        total_event_rate += pred_num_event.item()
        total_num_event += event_type.ne(Constants.PAD).sum().item()
        # we do not predict the first event
        total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]

    return total_event_ll / total_num_event, total_event_rate / total_num_pred, total_time_ae / total_num_pred


def eval_epoch(model, validation_data, pred_loss_func, opt):
    """ Epoch operation in evaluation phase. """
    model.eval()

    total_event_ll = 0  # cumulative event log-likelihood
    total_time_ae = 0  # cumulative time prediction squared-error
    total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,
                          desc='  - (Validation) ', leave=False):

            """ prepare data """
            event_time, time_gap, event_type, event_qx = map(lambda x: x.to(opt.device), batch)
            future_mask = get_future_mask(event_time, 700)
            future_mask_qx = future_mask.unsqueeze(-1).repeat(1, 1, 6)
            non_pad_mask = get_non_pad_mask(event_type)
            pos = future_mask.sum(dim=1, keepdim=True).long().to(opt.device)

            max_len = event_type.shape[1]
            max_pos = max(future_mask.sum(dim=1))
            input_type = event_type * future_mask
            input_time = event_time * future_mask
            input_qx = event_qx * future_mask_qx

            for i in range(max_len-max_pos-1):
                """ forward """
                enc_out, prediction = model(input_type, input_time, input_qx)

                """ compute loss """
                event_ll, non_event_ll = Utils.log_likelihood(model, enc_out, input_time, input_type)
                event_loss = -torch.sum(event_ll - non_event_ll)
                _, pred_num, pred_type = Utils.type_loss_test(prediction[0], input_type, pred_loss_func)
                ae = Utils.time_loss(prediction[1], input_time)

                """ note keeping """
                total_event_ll += -event_loss.item()
                total_time_ae += ae.item()
                total_event_rate += pred_num.item()
                total_num_event += event_type.ne(Constants.PAD).sum().item()
                total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]

                """ for iteration """
                pred_next_type = pred_type.gather(1, pos - 1)
                pred_next_time = prediction[1].squeeze(-1).gather(1, pos - 1)

                input_type = input_type.clone().scatter_add_(1, pos + 1, pred_next_type)
                input_time = input_time.clone().scatter_add_(1, pos + 1, pred_next_time)

                input_type = (input_type * non_pad_mask).type(torch.long)
                input_time = input_time * non_pad_mask

                pos += 1

                future_mask = future_mask.clone().scatter_add_(1, pos, torch.ones_like(pos).type_as(future_mask))
                future_mask_qx = future_mask.unsqueeze(-1).repeat(1, 1, 6)
                input_qx = input_qx * future_mask_qx * non_pad_mask.unsqueeze(-1).repeat(1, 1, 6)

    return total_event_ll / total_num_event, total_event_rate / total_num_pred, total_time_ae / total_num_pred


def train(model, training_data, validation_data, optimizer, scheduler, pred_loss_func, opt, sw):
    """ Start training. """

    valid_event_losses = []  # validation log-likelihood
    valid_pred_losses = []  # validation event type prediction accuracy
    valid_mae = []  # validation event time prediction RMSE
    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')
        """ train """
        start = time.time()
        train_event, train_type, train_time = train_epoch(model, training_data, optimizer, pred_loss_func, opt)
        print('  - (Training)    loglikelihood: {ll: 8.5f}, '
              'accuracy: {type: 8.5f}, MAE: {mae: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(ll=train_event, type=train_type, mae=train_time, elapse=(time.time() - start) / 60))

        with open(opt.logdir + "/log_train.txt", 'a') as f:
            f.write('{epoch}, {ll: 8.5f}, {acc: 8.5f}, {mae: 8.5f}\n'
                    .format(epoch=epoch, ll=train_event, acc=train_type, mae=train_time))

        """ test """
        start = time.time()
        valid_event, valid_type, valid_time = eval_epoch(model, validation_data, pred_loss_func, opt)
        print('  - (Testing)     loglikelihood: {ll: 8.5f}, '
              'accuracy: {type: 8.5f}, MAE: {mae: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(ll=valid_event, type=valid_type, mae=valid_time, elapse=(time.time() - start) / 60))

        valid_event_losses += [valid_event]
        valid_pred_losses += [valid_type]
        valid_mae += [valid_time]
        print('  - [Info] Maximum ll: {event: 8.5f}, '
              'Maximum accuracy: {pred: 8.5f}, Minimum MAE: {mae: 8.5f}'
              .format(event=max(valid_event_losses), pred=max(valid_pred_losses), mae=min(valid_mae)))

        # logging
        with open(opt.logdir + "/log.txt", 'a') as f:
            f.write('{epoch}, {ll: 8.5f}, {acc: 8.5f}, {mae: 8.5f}\n'
                    .format(epoch=epoch, ll=valid_event, acc=valid_type, mae=valid_time))

        scheduler.step()
        sw.add_scalar("Log-likelihood/train", train_event, global_step=epoch)
        sw.add_scalar("Accuracy/train", train_type, global_step=epoch)
        sw.add_scalar("MAE/train", train_time, global_step=epoch)
        sw.add_scalar("Log-likelihood/test", valid_event, global_step=epoch)
        sw.add_scalar("Accuracy/test", valid_type, global_step=epoch)
        sw.add_scalar("MAE/test", valid_time, global_step=epoch)


def main():
    """ Main function. """

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=False, type=str, default="./data_fx/")
    parser.add_argument('-epoch', type=int, default=30)
    # parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-d_model', type=int, default=64)
    parser.add_argument('-d_rnn', type=int, default=256)
    parser.add_argument('-d_inner_hid', type=int, default=128)
    parser.add_argument('-d_k', type=int, default=16)
    parser.add_argument('-d_v', type=int, default=16)
    parser.add_argument('-n_head', type=int, default=4)
    parser.add_argument('-n_layers', type=int, default=4)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-smooth', type=float, default=0.1)
    parser.add_argument('-logdir', type=str, default='./para1')
    parser.add_argument('-scaletimeloss', type=float, default=1.0)
    parser.add_argument('-comment', type=str, default='none')
    parser.add_argument('-d_qx', type=int, default=64)
    opt = parser.parse_args()

    sw = SummaryWriter(comment=opt.comment)

    opt.device = torch.device('cuda')

    # setup the log file
    with open(opt.logdir + "/log.txt", 'w') as f:
        f.write('Epoch, Log-likelihood, Accuracy, MAE\n')
    with open(opt.logdir + "/log_train.txt", 'w') as f:
        f.write('Epoch, Log-likelihood, Accuracy, MAE\n')

    print('[Info] parameters: {}'.format(opt))

    """ prepare dataloader """
    trainloader, testloader, num_types = prepare_dataloader(opt)

    """ prepare model """
    model = Transformer(
        num_types=num_types,
        d_model=opt.d_model,
        d_rnn=opt.d_rnn,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        d_k=opt.d_k,
        d_v=opt.d_v,
        dropout=opt.dropout,
        d_qx=opt.d_qx
    )
    model.to(opt.device)

    """ optimizer and scheduler """
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                           opt.lr, betas=(0.9, 0.999), eps=1e-05)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

    """ prediction loss function, either cross entropy or label smoothing """
    if opt.smooth > 0:
        pred_loss_func = Utils.LabelSmoothingLoss(opt.smooth, num_types, ignore_index=-1)
    else:
        pred_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

    """ number of parameters """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('[Info] Number of parameters: {}'.format(num_params))

    """ train the model """
    train(model, trainloader, testloader, optimizer, scheduler, pred_loss_func, opt, sw)


if __name__ == '__main__':
    main()
