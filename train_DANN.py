import torch
import argparse
import os
from pathlib import Path
import numpy as np
from data_process import process_small_data, myDataset, myDataset_unlabel
from model import  Bertbaseline, BertDANN
from torch.utils.data import DataLoader, SubsetRandomSampler
from transformers import BertModel, AdamW, get_scheduler, AutoModelForSequenceClassification, AutoTokenizer

import torch.nn.functional as F

from tqdm.auto import tqdm
from datasets import load_metric

seed = 3473497
torch.cuda.manual_seed(seed)
np.random.seed(seed * 13 // 7)

parser = argparse.ArgumentParser(description='PyTorch BERT Text Classification')
parser.add_argument('--output_dir', type=str, default='./results',
                    help='location of the output dir')
parser.add_argument('--ckpt_dir', type=str, default='./checkpoints',
                    help='location of the checkpoint dir')
parser.add_argument('--task_type', type=str, default='in_domain',
                    help='task type, in_domain, single_source, multi_source, DA')
parser.add_argument('--dataset', type=str, default='amazon',
                    help='dataset name')
parser.add_argument('--num_bert', type=int, default=1,
                    help='num of bert')
parser.add_argument('--mask_percentage', type=float, default=0.1,
                    help='mask percentage')
parser.add_argument('--in_domain_loss', type=str, default='nce',
                    help='in domain loss, none, gv, jsd, nce')
parser.add_argument('--cross_domain_loss', type=str, default='nce',
                    help='cross domain loss')
parser.add_argument('--salient_model', type=str, default='gumble',
                    help='salient model, gumble, attn, descriptor, none')
parser.add_argument('--gpu', type=int, default=0,
                    help='gpu cuda visible devices')
parser.add_argument('--lr', type=float, default=1e-5,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=10,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                    help='batch size')
parser.add_argument('--sample_size', type=int, default=16, metavar='N',
                    help='sampling batch size')
parser.add_argument('--lbd', type=float, default=0.1,
                    help='labmda')
parser.add_argument('--lbd1', type=float, default=0.1,
                    help='labmda1')
parser.add_argument('--lbd2', type=float, default=0.1,
                    help='labmda2')
parser.add_argument('--tau', type=float, default=0.12,
                    help='contrastive temperature')
parser.add_argument('--load_from_pretrain', action='store_true',
                    help='if load from a pretrained domain classifier')
parser.add_argument('--max_length', type=int, default=512,
                    help='max length')

args = parser.parse_args()

small_domain_names = ['book', 'electronics', 'beauty', 'music']

torch.cuda.set_device(args.gpu)

def sample_batch(dataset, sample_size = args.sample_size):
    loader = DataLoader(dataset, batch_size=sample_size, shuffle=True)
    for i in loader:
        batch = i
        break
    return batch

def train_single_source(source_domain_name, target_domain_name):
    s_labeled_encodings, s_labeled_labels, s_train_encodings, s_train_labels, s_val_encodings, s_val_labels, s_unlabeled_encodings = process_small_data(source_domain_name, max_length=args.max_length)
    s_train_dataset = myDataset(s_train_encodings, s_train_labels)
    s_val_dataset = myDataset(s_val_encodings, s_val_labels)
    s_unlabeled_dataset = myDataset_unlabel(s_unlabeled_encodings)

    t_labeled_encodings, t_labeled_labels, t_train_encodings, t_train_labels, t_val_encodings, t_val_labels, t_unlabeled_encodings = process_small_data(target_domain_name, max_length=args.max_length)
    t_labeled_dataset = myDataset(t_labeled_encodings, t_labeled_labels)
    t_unlabeled_dataset = myDataset_unlabel(t_unlabeled_encodings)

    source_train_loader = DataLoader(s_train_dataset, batch_size=args.batch_size, shuffle=True)
    source_val_loader = DataLoader(s_val_dataset, batch_size=args.batch_size)
    target_unlabeled_loader = DataLoader(t_unlabeled_dataset, batch_size=args.batch_size, shuffle=True)
    target_test_loader = DataLoader(t_labeled_dataset, batch_size=args.batch_size)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = BertDANN(num_labels=3)

    # ---optimizer---
    #optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5, correct_bias=False)
    #optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=False)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    #optimizer = AdamW(model.parameters(), lr=args.lr)
    # learning rate scheduler, using linear decay
    num_training_steps = args.epochs * len(source_train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0.1 * num_training_steps,
        num_training_steps=num_training_steps
    )

    model.to(device)
    progress_bar = tqdm(range(num_training_steps))
    s_acc = 0
    t_acc = 0
    # ---training---
    model.train()
    for epoch in range(args.epochs):
        i = 0
        for s_l_batch, t_ul_batch in zip(source_train_loader, target_unlabeled_loader):
            #s_ul_batch = sample_batch(s_unlabeled_dataset, sample_size=args.sample_size)
            #t_ul_batch = sample_batch(t_unlabeled_dataset, sample_size=args.sample_size)

            s_l_input_ids = s_l_batch['input_ids'].to(device)
            s_l_attention_mask = s_l_batch['attention_mask'].to(device)  # [8, 128]
            s_l_labels = s_l_batch['labels'].to(device)
            s_l_domain_labels = torch.zeros(s_l_input_ids.shape[0]).long().to(device)

            #s_ul_input_ids = s_ul_batch['input_ids'].to(device)
            #s_ul_attention_mask = s_ul_batch['attention_mask'].to(device)
            #s_ul_domain_labels = torch.zeros(s_ul_input_ids.shape[0]).long().to(device)

            t_ul_input_ids = t_ul_batch['input_ids'].to(device)
            t_ul_attention_mask = t_ul_batch['attention_mask'].to(device)
            t_ul_domain_labels = torch.ones(t_ul_input_ids.shape[0]).long().to(device)


            start_steps = epoch * len(source_train_loader)
            p = float(i+start_steps) / num_training_steps
            alpha = 2. / (1. + np.exp(-args.lbd * p)) - 1

            #outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            #loss = outputs.loss


            #--source labeled--
            s_l_class_loss, s_l_class_logits, s_l_domain_loss, s_l_domain_logits, s_l_hidden_states, s_l_attentions = model(
                input_ids=s_l_input_ids,
                attention_mask=s_l_attention_mask,
                class_labels=s_l_labels,
                domain_labels=s_l_domain_labels,
                alpha=alpha
            )
            loss = s_l_class_loss + s_l_domain_loss
            loss.backward()
            #optimizer.step()
            #optimizer.zero_grad()

            '''
            #--source unlabeled--
            s_ul_class_loss, s_ul_class_logits, s_ul_domain_loss, s_ul_domain_logits, s_ul_hidden_states, s_ul_attentions = model(
                input_ids=s_ul_input_ids,
                attention_mask=s_ul_attention_mask,
                class_labels=None,
                domain_labels=s_ul_domain_labels,
                alpha=alpha
            )
            loss = s_ul_domain_loss
            loss.backward()
            #optimizer.step()
            #optimizer.zero_grad()
            '''

            #--target unlabeled--
            t_ul_class_loss, t_ul_class_logits, t_ul_domain_loss, t_ul_domain_logits, t_ul_hidden_states, t_ul_attentions = model(
                input_ids=t_ul_input_ids,
                attention_mask=t_ul_attention_mask,
                class_labels=None,
                domain_labels=t_ul_domain_labels,
                alpha=alpha
            )
            loss = t_ul_domain_loss
            loss.backward()

            #loss = s_l_class_loss + s_l_domain_loss + s_ul_domain_loss + t_ul_domain_loss
            #loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            i += 1

        # ----------validation----------
        metric_val = load_metric("accuracy")
        metric_val_domain = load_metric("accuracy")
        model.eval()

        for batch in source_val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            class_labels = batch['labels'].to(device)
            domain_labels = torch.zeros(input_ids.shape[0]).long().to(device)
            with torch.no_grad():
                class_loss, class_logits, domain_loss, domain_logits, hidden_states, attentions = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    class_labels=class_labels,
                    domain_labels=domain_labels,
                    alpha=0.5
                )

            predictions = torch.argmax(class_logits, dim=-1)
            metric_val.add_batch(predictions=predictions, references=batch["labels"])
            domain_predictions = torch.argmax(domain_logits, dim=-1)
            metric_val_domain.add_batch(predictions=domain_predictions,
                                        references=torch.zeros(input_ids.shape[0]).long())

        s_score = metric_val.compute()
        s_domain_score = metric_val_domain.compute()

        # ----------testing----------
        metric_test = load_metric("accuracy")
        metric_test_domain = load_metric("accuracy")
        model.eval()

        for batch in target_test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            class_labels = batch['labels'].to(device)
            domain_labels = torch.ones(input_ids.shape[0]).long().to(device)
            with torch.no_grad():
                class_loss, class_logits, domain_loss, domain_logits, hidden_states, attentions = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    class_labels=class_labels,
                    domain_labels=domain_labels,
                    alpha=0.5
                )

            predictions = torch.argmax(class_logits, dim=-1)
            metric_test.add_batch(predictions=predictions, references=batch["labels"])
            domain_predictions = torch.argmax(domain_logits, dim=-1)
            metric_test_domain.add_batch(predictions=domain_predictions,
                                         references=torch.ones(input_ids.shape[0]).long())

        t_score = metric_test.compute()
        t_domain_score = metric_test_domain.compute()

        print(source_domain_name, target_domain_name, s_score, s_domain_score, t_score, t_domain_score)

        if t_score['accuracy'] >= t_acc:
            s_acc = s_score['accuracy']
            t_acc = t_score['accuracy']
            checkpoint_path = args.ckpt_dir + "/" + source_domain_name + "-" + target_domain_name + ".linear.DANN.best.analyze.ckpt"
            torch.save(model.state_dict(), checkpoint_path)

    print (s_acc, t_acc)

    checkpoint_path = args.ckpt_dir + "/" + source_domain_name + "-" + target_domain_name + ".linear.DANN.worst.analyze.ckpt"
    torch.save(model.state_dict(), checkpoint_path)
    return t_acc


if __name__ == "__main__":
    train_single_source('electronics', 'book')

    '''
    for target_domain_name in small_domain_names:
        for source_domain_name in small_domain_names:
            if target_domain_name != source_domain_name:
                acc_list = []
                for i in range(5):
                    acc = train_single_source(source_domain_name, target_domain_name)
                    acc_list.append(acc)

                acc_array = np.array(acc_list)
                print (acc_array, np.average(acc_array), np.std(acc_array))
    '''