import torch
import argparse
import os
from pathlib import Path
import numpy as np
from data_process import process_small_data, myDataset, myDataset_unlabel
from model import  Bertbaseline, BertAdvContrastSequenceClassification
from loss import SymKlCriterion, JSCriterion, stable_kl

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
parser.add_argument('--epochs', type=int, default=10,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='batch size')
parser.add_argument('--sample_size', type=int, default=16, metavar='N',
                    help='sampling batch size')
parser.add_argument('--load_from_pretrain', action='store_true',
                    help='if load from a pretrained domain classifier')
parser.add_argument('--max_length', type=int, default=512,
                    help='max length')
parser.add_argument('--wd', type=float, default=1e-2,
                    help='weight decay')
parser.add_argument('--adv_steps', type=int, default=1,
                    help="Number of gradient ascent steps for the adversary, should be at least 1")
parser.add_argument('--adv_init_mag', type=float, default=0,
                    help='Magnitude of initial (adversarial?) perturbation, 0, 1e-1, 2e-1, 3e-1, 5e-2, 8e-2')
parser.add_argument('--adv_noise_var', type=float, default=1e-5,
                    help='noise variance 1e-5')
parser.add_argument('--adv_lr', type=float, default=1e-4,
                    help='Step size of gradient ascent, 1e-1, 2e-1, 3e-2, 4e-2, 5e-2')
parser.add_argument('--adv_max_norm', type=float, default=1e-5,
                    help="Maximum norm of adversarial perturbation, set to 0 to be unlimited, 0, 7e-1")
parser.add_argument('--norm_type', type=str, default='linf',
                    help='linf or l2 or l1')
parser.add_argument('--adv_alpha', type=float, default=1,
                    help='virtual adversarial loss alpha')
parser.add_argument('--contrast_lbd', type=float, default=0.05,
                    help='contrastive labmda')
parser.add_argument('--tau', type=float, default=0.12,
                    help='contrastive temperature')
parser.add_argument('--domain_lbd', type=float, default=0.5,
                    help='domain classification lambda')
parser.add_argument('--consis_belta', type=float, default=0,
                    help='belta for consistency loss')

args = parser.parse_args()

small_domain_names = ['book', 'electronics', 'beauty', 'music']

torch.cuda.set_device(args.gpu)



def sample_batch(dataset, sample_size = args.sample_size):
    loader = DataLoader(dataset, batch_size=sample_size, shuffle=True)
    for i in loader:
        batch = i
        break
    return batch

def info_nce_loss(features, n_views, device, batch_size):
    labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
    #labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / args.tau

    return logits, labels

def norm_grad(grad, eff_grad=None, sentence_level=False):
    if args.norm_type == 'l2':
        if sentence_level:
            direction = grad / (torch.norm(grad, dim=(-2, -1), keepdim=True) + args.adv_max_norm)
        else:
            direction = grad / (torch.norm(grad, dim=-1, keepdim=True) + args.adv_max_norm)
    elif args.norm_type == 'l1':
        direction = grad.sign()
    else:
        if sentence_level:
            direction = grad / (grad.abs().max((-2, -1), keepdim=True)[0] + args.adv_max_norm)
        else:
            direction = grad / (grad.abs().max(-1, keepdim=True)[0] + args.adv_max_norm)
            eff_direction = eff_grad / (grad.abs().max(-1, keepdim=True)[0] + args.adv_max_norm)
    return direction, eff_direction

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
    target_test_loader = DataLoader(t_labeled_dataset, batch_size=args.batch_size)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = BertAdvContrastSequenceClassification(num_labels=3)

    # ---optimizer---
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
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
    adv_lf = SymKlCriterion()
    consist_lf = JSCriterion()
    contrast_lf = torch.nn.CrossEntropyLoss()
    cosine_metric = torch.nn.CosineSimilarity(dim=-1)
    acc = 0
    # ====================training=====================
    model.train()
    for epoch in range(args.epochs):
        i = 0
        for s_l_batch in source_train_loader:
            s_ul_batch = sample_batch(s_unlabeled_dataset, sample_size=args.sample_size)
            t_ul_batch = sample_batch(t_unlabeled_dataset, sample_size=args.sample_size)

            s_l_input_ids = s_l_batch['input_ids'].to(device)
            s_l_attention_mask = s_l_batch['attention_mask'].to(device)  # [8, 128]
            s_l_labels = s_l_batch['labels'].to(device)
            s_l_domain_labels = torch.zeros(s_l_input_ids.shape[0]).long().to(device)

            s_ul_input_ids = s_ul_batch['input_ids'].to(device)
            s_ul_attention_mask = s_ul_batch['attention_mask'].to(device)
            s_ul_domain_labels = torch.zeros(s_ul_input_ids.shape[0]).long().to(device)

            t_ul_input_ids = t_ul_batch['input_ids'].to(device)
            t_ul_attention_mask = t_ul_batch['attention_mask'].to(device)
            t_ul_domain_labels = torch.ones(t_ul_input_ids.shape[0]).long().to(device)

            # ==========source labeled data==========
            s_l_class_loss, s_l_class_logits, s_l_domain_loss, s_l_domain_logits, s_l_hidden_states, s_l_attentions, s_l_z = model(
                input_ids=s_l_input_ids,
                attention_mask=s_l_attention_mask,
                class_labels=s_l_labels,
                domain_labels=s_l_domain_labels
            )

            # ===adversarial===
            # adversarial on the domain classification
            s_l_inputs = {"attention_mask": s_l_attention_mask, "labels": s_l_domain_labels}

            s_l_embeds_init = model.bert.embeddings.word_embeddings(s_l_input_ids)

            s_l_noise = torch.zeros_like(s_l_embeds_init).normal_(0, 1) * args.adv_noise_var
            s_l_noise = s_l_noise.detach()
            s_l_noise.requires_grad_()

            for step in range(args.adv_steps):
                s_l_inputs['inputs_embeds'] = s_l_noise + s_l_embeds_init
                adv_s_l_class_loss, adv_s_l_class_logits, adv_s_l_domain_loss, adv_s_l_domain_logits, adv_s_l_hidden_states, adv_s_l_attentions, adv_s_l_z = model(
                    inputs_embeds=s_l_inputs['inputs_embeds'],
                    attention_mask=s_l_inputs["attention_mask"],
                    class_labels=None,
                    domain_labels=s_l_inputs["labels"]
                )

                # (1) calc the adversarial loss - KL divergence
                # adv_logits = adv_logits.view(-1, 1)
                adv_s_l_loss = stable_kl(adv_s_l_domain_logits, s_l_domain_logits.detach(), reduce=False)

                # (2) calc the gradient for the noise
                s_l_delta_grad, = torch.autograd.grad(adv_s_l_loss, s_l_noise, only_inputs=True, retain_graph=False)
                #s_l_norm = s_l_delta_grad.norm()
                #if (torch.isnan(s_l_norm) or torch.isinf(s_l_norm)):
                #    return 0
                s_l_eff_delta_grad = s_l_delta_grad * args.adv_lr
                s_l_delta_grad = s_l_noise + s_l_delta_grad * args.adv_lr
                s_l_noise, s_l_eff_noise = norm_grad(s_l_delta_grad, eff_grad=s_l_eff_delta_grad, sentence_level=False)
                s_l_noise = s_l_noise.detach()
                s_l_noise.requires_grad_()

            s_l_inputs['inputs_embeds'] = s_l_noise + s_l_embeds_init
            adv_s_l_class_loss, adv_s_l_class_logits, adv_s_l_domain_loss, adv_s_l_domain_logits, adv_s_l_hidden_states, adv_s_l_attentions, adv_s_l_z = model(
                inputs_embeds=s_l_inputs['inputs_embeds'],
                attention_mask=s_l_attention_mask,
                class_labels=s_l_labels,
                domain_labels=s_l_domain_labels
            )
            adv_s_l_loss = adv_lf(s_l_domain_logits, adv_s_l_domain_logits)

            # ===contrastive===
            s_l_contrast_logits, s_l_contrast_labels = info_nce_loss(torch.cat([s_l_z, adv_s_l_z], dim=0), n_views=2, device=device, batch_size=s_l_z.shape[0])
            s_l_contrast_loss = contrast_lf(s_l_contrast_logits, s_l_contrast_labels)
            '''
            pos = (cosine_metric(s_l_z, adv_s_l_z) / args.tau).mean()
            neg_matrix = cosine_metric(adv_s_l_z.unsqueeze(0), adv_s_l_z.unsqueeze(1)) / args.tau
            neg_matrix = torch.exp(neg_matrix) * (1 - torch.eye(adv_s_l_z.shape[0], dtype=torch.float).to(device))
            neg = neg_matrix.sum(dim=-1)
            neg = torch.log(neg).mean()
            s_l_contrast_loss = neg - pos
            '''
            # ===consistency loss===
            if args.consis_belta == 0:
                s_l_consistency_loss = 0
            else:
                s_l_consistency_loss=consist_lf(s_l_class_logits, adv_s_l_class_logits)

            # ===loss===
            loss = s_l_class_loss + \
                   args.domain_lbd*(s_l_domain_loss+args.adv_alpha*adv_s_l_loss) +\
                   args.contrast_lbd*s_l_contrast_loss +\
                   args.consis_belta*s_l_consistency_loss
            loss.backward()



            # ==========target unlabel data==========
            t_ul_class_loss, t_ul_class_logits, t_ul_domain_loss, t_ul_domain_logits, t_ul_hidden_states, t_ul_attentions, t_ul_z = model(
                input_ids=t_ul_input_ids,
                attention_mask=t_ul_attention_mask,
                class_labels=None,
                domain_labels=t_ul_domain_labels
            )

            # ===adversarial===
            # adversarial on the domain classification
            t_ul_inputs = {"attention_mask": t_ul_attention_mask, "labels": t_ul_domain_labels}

            t_ul_embeds_init = model.bert.embeddings.word_embeddings(t_ul_input_ids)

            t_ul_noise = torch.zeros_like(t_ul_embeds_init).normal_(0, 1) * args.adv_noise_var
            t_ul_noise = t_ul_noise.detach()
            t_ul_noise.requires_grad_()

            for step in range(args.adv_steps):
                t_ul_inputs['inputs_embeds'] = t_ul_noise + t_ul_embeds_init
                adv_t_ul_class_loss, adv_t_ul_class_logits, adv_t_ul_domain_loss, adv_t_ul_domain_logits, adv_t_ul_hidden_states, adv_t_ul_attentions, adv_t_ul_z = model(
                    inputs_embeds=t_ul_inputs['inputs_embeds'],
                    attention_mask=t_ul_inputs["attention_mask"],
                    class_labels=None,
                    domain_labels=t_ul_inputs["labels"]
                )

                # (1) calc the adversarial loss - KL divergence
                # adv_logits = adv_logits.view(-1, 1)
                adv_t_ul_loss = stable_kl(adv_t_ul_domain_logits, t_ul_domain_logits.detach(), reduce=False)

                # (2) calc the gradient for the noise
                t_ul_delta_grad, = torch.autograd.grad(adv_t_ul_loss, t_ul_noise, only_inputs=True, retain_graph=False)
                #t_ul_norm = t_ul_delta_grad.norm()
                #if (torch.isnan(t_ul_norm) or torch.isinf(t_ul_norm)):
                #    return 0
                t_ul_eff_delta_grad = t_ul_delta_grad * args.adv_lr
                t_ul_delta_grad = t_ul_noise + t_ul_delta_grad * args.adv_lr
                t_ul_noise, t_ul_eff_noise = norm_grad(t_ul_delta_grad, eff_grad=t_ul_eff_delta_grad, sentence_level=False)
                t_ul_noise = t_ul_noise.detach()
                t_ul_noise.requires_grad_()

            t_ul_inputs['inputs_embeds'] = t_ul_noise + t_ul_embeds_init
            adv_t_ul_class_loss, adv_t_ul_class_logits, adv_t_ul_domain_loss, adv_t_ul_domain_logits, adv_t_ul_hidden_states, adv_t_ul_attentions, adv_t_ul_z = model(
                inputs_embeds=t_ul_inputs['inputs_embeds'],
                attention_mask=t_ul_attention_mask,
                class_labels=None,
                domain_labels=t_ul_domain_labels
            )
            adv_t_ul_loss = adv_lf(t_ul_domain_logits, adv_t_ul_domain_logits)

            # ===contrastive===
            t_ul_contrast_logits, t_ul_contrast_labels = info_nce_loss(torch.cat([t_ul_z, adv_t_ul_z], dim=0), n_views=2, device=device, batch_size=t_ul_z.shape[0])
            t_ul_contrast_loss = contrast_lf(t_ul_contrast_logits, t_ul_contrast_labels)
            '''
            pos = (cosine_metric(t_ul_z, adv_t_ul_z) / args.tau).mean()
            neg_matrix = cosine_metric(adv_t_ul_z.unsqueeze(0), adv_t_ul_z.unsqueeze(1)) / args.tau
            neg_matrix = torch.exp(neg_matrix) * (1 - torch.eye(adv_t_ul_z.shape[0], dtype=torch.float).to(device))
            neg = neg_matrix.sum(dim=-1)
            neg = torch.log(neg).mean()
            t_ul_contrast_loss = neg - pos
            '''

            # ===consistency loss===
            if args.consis_belta == 0:
                t_ul_consistency_loss = 0
            else:
                t_ul_consistency_loss = consist_lf(t_ul_class_logits, adv_t_ul_class_logits)

            # ===loss===
            loss = args.domain_lbd * (t_ul_domain_loss + args.adv_alpha*adv_t_ul_loss) + \
                   args.contrast_lbd * t_ul_contrast_loss + \
                   args.consis_belta * t_ul_consistency_loss
            loss.backward()

            # ==========optimizer step==========
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        # ====================evaluation====================
        metric_val = load_metric("accuracy")
        metric_val_domain = load_metric("accuracy")
        model.eval()

        for batch in source_val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            class_labels = batch['labels'].to(device)
            domain_labels = torch.zeros(input_ids.shape[0]).long().to(device)
            with torch.no_grad():
                class_loss, class_logits, domain_loss, domain_logits, hidden_states, attentions, z = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    class_labels=class_labels,
                    domain_labels=domain_labels
                )

            predictions = torch.argmax(class_logits, dim=-1)
            metric_val.add_batch(predictions=predictions, references=batch["labels"])
            domain_predictions = torch.argmax(domain_logits, dim=-1)
            metric_val_domain.add_batch(predictions=domain_predictions, references=torch.zeros(input_ids.shape[0]).long())

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
                class_loss, class_logits, domain_loss, domain_logits, hidden_states, attentions, z = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    class_labels=class_labels,
                    domain_labels=domain_labels,
                )

            predictions = torch.argmax(class_logits, dim=-1)
            metric_test.add_batch(predictions=predictions, references=batch["labels"])
            domain_predictions = torch.argmax(domain_logits, dim=-1)
            metric_test_domain.add_batch(predictions=domain_predictions,
                                        references=torch.ones(input_ids.shape[0]).long())


        t_score = metric_test.compute()
        t_domain_score = metric_test_domain.compute()

        print(source_domain_name, target_domain_name, s_score, s_domain_score, t_score, t_domain_score)

        if t_score['accuracy'] >= acc:
            acc = t_score['accuracy']
            checkpoint_path = args.ckpt_dir + "/" + source_domain_name + "-" + target_domain_name + ".ckpt"
            torch.save(model.state_dict(), checkpoint_path)

    return

if __name__ == "__main__":
    train_single_source('electronics', 'book')