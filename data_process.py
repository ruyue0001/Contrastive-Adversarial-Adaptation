import torch
import random
import numpy as np
from transformers import AutoTokenizer
from pathlib import Path

small_domain_names = ['beauty', 'book', 'electronics', 'music']

def split_small(domain_name):
    data_dir = "data/small/" + domain_name
    labeled_text_path = data_dir + "/set1_text.txt"
    labeled_label_path = data_dir + "/set1_label.txt"
    unlabeled_text_path = data_dir + '/set2_text.txt'

    l_t_f = open(labeled_text_path, 'r')
    l_l_f = open(labeled_label_path, 'r')
    ul_t_f = open(unlabeled_text_path, 'r')

    pos_text = []
    neg_text = []
    neu_text = []

    for text, label in zip(l_t_f.readlines(), l_l_f.readlines()):
        text = text.strip('\n')
        label = label.strip('\n')
        if label == '5.0' or label == '4.0':
            pos_text.append(text)
        elif label == '3.0':
            neu_text.append(text)
        else:
            neg_text.append(text)
    #pos_text = list(set(pos_text))
    #neg_text = list(set(neg_text))
    #neu_text = list(set(neu_text))
    #print (len(pos_text))
    #print (len(neg_text))
    #print (len(neu_text))

    process_labeled_path = "data/small/" + domain_name + ".labeled"
    process_unlabeled_path = "data/small/" + domain_name + ".unlabeled"
    process_train_path = "data/small/" + domain_name + ".train"
    process_val_path = "data/small/" + domain_name + ".val"

    labeled_f = open(process_labeled_path, 'w')
    unlabeled_f = open(process_unlabeled_path, 'w')
    train_f = open(process_train_path, 'w')
    val_f = open(process_val_path, 'w')

    for text in pos_text:
        labeled_f.write('2\t' + text + '\n')
    for text in neu_text:
        labeled_f.write('1\t' + text + '\n')
    for text in neg_text:
        labeled_f.write('0\t' + text + '\n')

    pos_text_sampled = np.random.choice(pos_text, 334, replace=False)
    neu_text_sampled = np.random.choice(neu_text, 333, replace=False)
    neg_text_sampled = np.random.choice(neg_text, 333, replace=False)


    for text in pos_text:
        if text in pos_text_sampled:
            val_f.write('2\t' + text + '\n')
        else:
            train_f.write('2\t' + text + '\n')
    for text in neu_text:
        if text in neu_text_sampled:
            val_f.write('1\t' + text + '\n')
        else:
            train_f.write('1\t' + text + '\n')
    for text in neg_text:
        if text in neg_text_sampled:
            val_f.write('0\t' + text + '\n')
        else:
            train_f.write('0\t' + text + '\n')

    for text in ul_t_f.readlines():
        unlabeled_f.write(text)

    l_t_f.close()
    l_l_f.close()
    ul_t_f.close()
    labeled_f.close()
    unlabeled_f.close()
    train_f.close()
    val_f.close()

    return

def read_data(dir, is_unlabel = False):
    texts = []
    labels = []
    f = open(dir, "r")
    for line in f.readlines():
        line = line.strip('\n')
        if not is_unlabel:
            labels.append(int(line.split('\t')[0]))
            texts.append(line.split('\t')[1])
        else:
            texts.append((line))
    #print (len(texts))
    return texts, labels

def process_small_data(domain_name, max_length = 512):
    labeled_texts, labeled_labels = read_data("data/small/" + domain_name + ".labeled")
    unlabeled_texts, unlabeled_labels = read_data("data/small/" + domain_name + ".unlabeled", is_unlabel=True)
    train_texts, train_labels = read_data("data/small/" + domain_name + ".train")
    val_texts, val_labels = read_data("data/small/" + domain_name + ".val")

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    labeled_encodings = tokenizer(labeled_texts, padding='max_length', truncation=True, max_length=max_length)
    unlabeled_encodings = tokenizer(unlabeled_texts, padding='max_length', truncation=True, max_length=max_length)
    train_encodings = tokenizer(train_texts, padding='max_length', truncation=True, max_length=max_length)
    val_encodings = tokenizer(val_texts, padding='max_length', truncation=True, max_length=max_length)

    return labeled_encodings, labeled_labels, train_encodings, train_labels, val_encodings, val_labels, unlabeled_encodings

class myDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class myDataset_unlabel(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

if __name__ == "__main__":
    #split_small('beauty')
    for domain_name in small_domain_names:
        split_small(domain_name)
    #labeled_encodings, labeled_labels, train_encodings, train_labels, val_encodings, val_labels, unlabeled_encodings = process_small_data('book')
    #print (len(train_encodings))
    #print (len(val_encodings))