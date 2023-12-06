import torch
import argparse
import os
from pathlib import Path
import numpy as np
from data_process import process_small_data, myDataset
from model import  Bertbaseline, BertContrastSequenceClassification
from torch.utils.data import DataLoader, SubsetRandomSampler
from transformers import BertModel, AdamW, get_scheduler, AutoModelForSequenceClassification, AutoTokenizer

import torch.nn.functional as F

from tqdm.auto import tqdm
from datasets import load_metric


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
parser.add_argument('--epochs', type=int, default=8,
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




def train_in_domain(domain_name):
    labeled_encodings, labeled_labels, train_encodings, train_labels, val_encodings, val_labels, unlabeled_encodings = process_small_data(domain_name)
    train_dataset = myDataset(train_encodings, train_labels)
    val_dataset = myDataset(val_encodings, val_labels)


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = Bertbaseline(num_labels=3)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    #---optimizer---
    optimizer = AdamW(model.parameters(), lr=args.lr)
    #learning rate scheduler, using linear decay
    num_training_steps = args.epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer = optimizer,
        num_warmup_steps = 0.1 * num_training_steps,
        num_training_steps = num_training_steps
    )

    model.to(device)
    progress_bar = tqdm(range(num_training_steps))
    acc = 0
    #---training---
    model.train()
    for epoch in range(args.epochs):
        for batch in train_loader:
            #optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device) #[8, 128]
            labels = batch['labels'].to(device)
            #outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss, logits, hidden_states, attentions = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        #---evaluation---
        metric = load_metric("accuracy")
        model.eval()
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            with torch.no_grad():
                loss, logits, hidden_states, attentions = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])

        score = metric.compute()

        print(domain_name, score)

        if score['accuracy'] >= acc:
            acc = score['accuracy']
            checkpoint_path = args.ckpt_dir + "/" + domain_name + "bert-baseline.ckpt"
            torch.save(model.state_dict(), checkpoint_path)

    return

def train_single_source(source_domain_name, target_domain_name):
    s_labeled_encodings, s_labeled_labels, s_train_encodings, s_train_labels, s_val_encodings, s_val_labels, s_unlabeled_encodings = process_small_data(source_domain_name, max_length=args.max_length)
    s_train_dataset = myDataset(s_train_encodings, s_train_labels)
    s_val_dataset = myDataset(s_val_encodings, s_val_labels)


    t_labeled_encodings, t_labeled_labels, t_train_encodings, t_train_labels, t_val_encodings, t_val_labels, t_unlabeled_encodings = process_small_data(target_domain_name, max_length=args.max_length)
    t_labeled_dataset = myDataset(t_labeled_encodings, t_labeled_labels)

    source_train_loader = DataLoader(s_train_dataset, batch_size=args.batch_size, shuffle=True)
    source_val_loader = DataLoader(s_val_dataset, batch_size=args.batch_size)
    target_test_loader = DataLoader(t_labeled_dataset, batch_size=args.batch_size)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = Bertbaseline(num_labels=3)

    # ---optimizer---
    #optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5, correct_bias=False)
    #optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=False)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
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
        for batch in source_train_loader:
            # optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)  # [8, 128]
            labels = batch['labels'].to(device)
            #outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            #loss = outputs.loss
            loss, logits, hidden_states, attentions = model(input_ids,attention_mask=attention_mask, labels=labels)

            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        # ----------validation----------
        metric_val = load_metric("accuracy")
        model.eval()

        for batch in source_val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            with torch.no_grad():
                loss, logits, hidden_states, attentions = model(input_ids,attention_mask=attention_mask, labels=labels)
                #outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                #logits = outputs.logits

            predictions = torch.argmax(logits, dim=-1)
            metric_val.add_batch(predictions=predictions, references=batch["labels"])

        s_score = metric_val.compute()

        # ----------testing----------
        metric_test = load_metric("accuracy")
        model.eval()

        for batch in target_test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            with torch.no_grad():
                loss, logits, hidden_states, attentions = model(input_ids,attention_mask=attention_mask, labels=labels)
                #outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                #logits = outputs.logits

            predictions = torch.argmax(logits, dim=-1)
            metric_test.add_batch(predictions=predictions, references=batch["labels"])

        t_score = metric_test.compute()

        print(source_domain_name, target_domain_name, s_score, t_score)

        if s_score['accuracy'] >= s_acc:
            s_acc = s_score['accuracy']
            t_acc = t_score['accuracy']
            checkpoint_path = args.ckpt_dir + "/" + source_domain_name + "-" + target_domain_name + ".linear.bert-baseline.analyze.ckpt"
            torch.save(model.state_dict(), checkpoint_path)

    print (s_acc, t_acc)
    return t_acc


if __name__ == "__main__":
    '''
    for target_domain_name in small_domain_names:
        for source_domain_name in small_domain_names:
            if target_domain_name != source_domain_name:
                acc_list = []
                for i in range(5):
                    acc = train_single_source(source_domain_name, target_domain_name)
                    acc_list.append(acc)

                acc_array = np.array(acc_list)
                print(acc_array, np.average(acc_array), np.std(acc_array))
    '''
    train_single_source('electronics', 'book')