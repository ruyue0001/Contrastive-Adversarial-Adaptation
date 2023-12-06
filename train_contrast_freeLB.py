import torch
import argparse
import os
from pathlib import Path
import numpy as np
from data_process import process_small_data, myDataset, myDataset_unlabel
from model import  Bertbaseline, BertAdvContrastSequenceClassification
from loss import SymKlCriterion, JSCriterion, stable_kl, JSD

from torch.utils.data import DataLoader, SubsetRandomSampler
from transformers import BertModel, AdamW, get_scheduler, AutoModelForSequenceClassification, AutoTokenizer

import torch.nn.functional as F

from tqdm.auto import tqdm
from datasets import load_metric



small_domain_names = ['book', 'electronics', 'beauty', 'music']



def sample_batch(dataset, args):
    sample_size = args.sample_size
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

def train_single_source(source_domain_name, target_domain_name, args):
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
    consist_lf = JSD()
    contrast_lf = torch.nn.CrossEntropyLoss()
    cosine_metric = torch.nn.CosineSimilarity(dim=-1)
    acc = 0
    # ====================training=====================
    model.train()
    for epoch in range(args.epochs):
        for s_l_batch, t_ul_batch in zip(source_train_loader, target_unlabeled_loader):
            optimizer.zero_grad()

            s_l_input_ids = s_l_batch['input_ids'].to(device)
            s_l_attention_mask = s_l_batch['attention_mask'].to(device)  # [8, 128]
            s_l_labels = s_l_batch['labels'].to(device)
            s_l_domain_labels = torch.zeros(s_l_input_ids.shape[0]).long().to(device)

            t_ul_input_ids = t_ul_batch['input_ids'].to(device)
            t_ul_attention_mask = t_ul_batch['attention_mask'].to(device)
            t_ul_domain_labels = torch.ones(t_ul_input_ids.shape[0]).long().to(device)

            # ==========source labeled data==========

            # ===adversarial===
            # adversarial on the domain classification
            inputs = {"attention_mask": s_l_attention_mask, "labels": s_l_domain_labels}

            embeds_init = model.bert.embeddings.word_embeddings(s_l_input_ids)

            if args.adv_init_mag > 0:
                input_mask = inputs['attention_mask'].to(embeds_init)
                input_lengths = torch.sum(input_mask, 1)

                if args.norm_type == "l2":
                    delta = torch.zeros_like(embeds_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
                    dims = input_lengths * embeds_init.size(-1)
                    mag = args.adv_init_mag / torch.sqrt(dims)
                    delta = (delta * mag.view(-1, 1, 1)).detach()
                elif args.norm_type == "linf":
                    delta = torch.zeros_like(embeds_init).uniform_(-args.adv_init_mag, args.adv_init_mag) * input_mask.unsqueeze(2)

            elif args.adv_noise_var > 0:
                input_mask = inputs['attention_mask'].to(embeds_init)
                delta = torch.zeros_like(embeds_init).normal_(0, 1) * args.adv_noise_var
                delta = delta * input_mask.unsqueeze(2)

            else:
                delta = torch.zeros_like(embeds_init)

            for step in range(args.adv_steps):
                delta.requires_grad_()
                inputs['inputs_embeds'] = delta + embeds_init
                adv_s_l_class_loss, adv_s_l_class_logits, adv_s_l_domain_loss, adv_s_l_domain_logits, adv_s_l_hidden_states, adv_s_l_attentions, adv_s_l_z = model(
                    inputs_embeds=inputs['inputs_embeds'],
                    attention_mask=inputs["attention_mask"],
                    class_labels=None,
                    domain_labels=inputs["labels"]
                )

                # (1) calc the adversarial loss - KL divergence
                # adv_logits = adv_logits.view(-1, 1)
                #adv_s_l_loss = stable_kl(adv_s_l_domain_logits, s_l_domain_logits.detach(), reduce=False)

                # (2) calc the gradient for the noise
                #delta_grad, = torch.autograd.grad(adv_s_l_domain_loss, delta, only_inputs=True, retain_graph=False)
                #s_l_norm = s_l_delta_grad.norm()
                #if (torch.isnan(s_l_norm) or torch.isinf(s_l_norm)):
                #    return 0
                adv_s_l_domain_loss.backward()
                delta_grad = delta.grad.clone().detach()

                # (3) update the perturbation
                #s_l_eff_delta_grad = s_l_delta_grad * args.adv_lr
                #s_l_delta_grad = s_l_noise + s_l_delta_grad * args.adv_lr
                #s_l_noise, s_l_eff_noise = norm_grad(s_l_delta_grad, eff_grad=s_l_eff_delta_grad, sentence_level=False)
                #s_l_noise = s_l_noise.detach()
                #s_l_noise.requires_grad_()
                if args.norm_type == "l2":
                    denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                    denorm = torch.clamp(denorm, min=1e-8)
                    delta = (delta + args.adv_lr * delta_grad / denorm).detach()
                    if args.adv_max_norm > 0:
                        delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                        exceed_mask = (delta_norm > args.adv_max_norm).to(embeds_init)
                        reweights = (args.adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)
                        delta = (delta * reweights).detach()
                elif args.norm_type == "linf":
                    denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1, 1)
                    denorm = torch.clamp(denorm, min=1e-8)
                    delta = (delta + args.adv_lr * delta_grad / denorm).detach()
                    if args.adv_max_norm > 0:
                        delta = torch.clamp(delta, -args.adv_max_norm, args.adv_max_norm).detach()
                else:
                    print("Norm type {} not specified.".format(args.norm_type))
                    exit()

                embeds_init = model.bert.embeddings.word_embeddings(s_l_input_ids)
                delta.requires_grad_()
                optimizer.zero_grad()

            # (4) calc the virtual adversarial training loss
            s_l_class_loss, s_l_class_logits, s_l_domain_loss, s_l_domain_logits, s_l_hidden_states, s_l_attentions, s_l_z = model(
                input_ids=s_l_input_ids,
                attention_mask=s_l_attention_mask,
                class_labels=s_l_labels,
                domain_labels=s_l_domain_labels
            )

            inputs['inputs_embeds'] = delta + embeds_init
            adv_s_l_class_loss, adv_s_l_class_logits, adv_s_l_domain_loss, adv_s_l_domain_logits, adv_s_l_hidden_states, adv_s_l_attentions, adv_s_l_z = model(
                inputs_embeds=inputs['inputs_embeds'],
                attention_mask=s_l_attention_mask,
                class_labels=s_l_labels,
                domain_labels=s_l_domain_labels
            )
            if args.virtual_adv:
                adv_s_l_loss = adv_lf(s_l_domain_logits, adv_s_l_domain_logits)
            else:
                adv_s_l_loss = adv_s_l_domain_loss

            # ===contrastive===
            if args.contrast_update == 'one':
                z_cat = torch.cat([s_l_z, adv_s_l_z.detach()], dim=0)
                s_l_contrast_logits, s_l_contrast_labels = info_nce_loss(z_cat,n_views=2, device=device,batch_size=s_l_z.shape[0])
                s_l_contrast_loss = contrast_lf(s_l_contrast_logits, s_l_contrast_labels)
            elif args.contrast_update == 'mix':
                z_cat_1 = torch.cat([s_l_z, adv_s_l_z.detach()], dim=0)
                s_l_contrast_logits_1, s_l_contrast_labels_1 = info_nce_loss(z_cat_1, n_views=2, device=device,batch_size=s_l_z.shape[0])
                s_l_contrast_loss_1 = contrast_lf(s_l_contrast_logits_1, s_l_contrast_labels_1)
                z_cat_2 = torch.cat([s_l_z.detach(), adv_s_l_z], dim=0)
                s_l_contrast_logits_2, s_l_contrast_labels_2 = info_nce_loss(z_cat_2, n_views=2, device=device,batch_size=s_l_z.shape[0])
                s_l_contrast_loss_2 = contrast_lf(s_l_contrast_logits_2, s_l_contrast_labels_2)
                s_l_contrast_loss = (s_l_contrast_loss_1 + s_l_contrast_loss_2) / 2
            else:
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
            s_l_consistency_loss=consist_lf(s_l_class_logits, adv_s_l_class_logits)

            # ===loss===
            loss = s_l_class_loss + \
                   args.domain_lbd*(s_l_domain_loss+args.adv_alpha*adv_s_l_loss) +\
                   args.contrast_lbd*s_l_contrast_loss +\
                   args.consis_belta*s_l_consistency_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


            # ==========target unlabel data==========
            # ===adversarial===
            # adversarial on the domain classification
            inputs = {"attention_mask": t_ul_attention_mask, "labels": t_ul_domain_labels}

            embeds_init = model.bert.embeddings.word_embeddings(t_ul_input_ids)

            if args.adv_init_mag > 0:
                input_mask = inputs['attention_mask'].to(embeds_init)
                input_lengths = torch.sum(input_mask, 1)

                if args.norm_type == "l2":
                    delta = torch.zeros_like(embeds_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
                    dims = input_lengths * embeds_init.size(-1)
                    mag = args.adv_init_mag / torch.sqrt(dims)
                    delta = (delta * mag.view(-1, 1, 1)).detach()
                elif args.norm_type == "linf":
                    delta = torch.zeros_like(embeds_init).uniform_(-args.adv_init_mag,args.adv_init_mag) * input_mask.unsqueeze(2)

            elif args.adv_noise_var > 0:
                input_mask = inputs['attention_mask'].to(embeds_init)
                delta = torch.zeros_like(embeds_init).normal_(0, 1) * args.adv_noise_var
                delta = delta * input_mask.unsqueeze(2)

            else:
                delta = torch.zeros_like(embeds_init)

            for step in range(args.adv_steps):
                delta.requires_grad_()
                inputs['inputs_embeds'] = delta + embeds_init
                adv_t_ul_class_loss, adv_t_ul_class_logits, adv_t_ul_domain_loss, adv_t_ul_domain_logits, adv_t_ul_hidden_states, adv_t_ul_attentions, adv_t_ul_z = model(
                    inputs_embeds=inputs['inputs_embeds'],
                    attention_mask=inputs["attention_mask"],
                    class_labels=None,
                    domain_labels=inputs["labels"]
                )

                # (1) calc the adversarial loss - KL divergence
                # adv_logits = adv_logits.view(-1, 1)
                # adv_s_l_loss = stable_kl(adv_s_l_domain_logits, s_l_domain_logits.detach(), reduce=False)

                # (2) calc the gradient for the noise
                # delta_grad, = torch.autograd.grad(adv_s_l_domain_loss, delta, only_inputs=True, retain_graph=False)
                # s_l_norm = s_l_delta_grad.norm()
                # if (torch.isnan(s_l_norm) or torch.isinf(s_l_norm)):
                #    return 0
                adv_t_ul_domain_loss.backward()
                delta_grad = delta.grad.clone().detach()

                # (3) update the perturbation
                # s_l_eff_delta_grad = s_l_delta_grad * args.adv_lr
                # s_l_delta_grad = s_l_noise + s_l_delta_grad * args.adv_lr
                # s_l_noise, s_l_eff_noise = norm_grad(s_l_delta_grad, eff_grad=s_l_eff_delta_grad, sentence_level=False)
                # s_l_noise = s_l_noise.detach()
                # s_l_noise.requires_grad_()
                if args.norm_type == "l2":
                    denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                    denorm = torch.clamp(denorm, min=1e-8)
                    delta = (delta + args.adv_lr * delta_grad / denorm).detach()
                    if args.adv_max_norm > 0:
                        delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                        exceed_mask = (delta_norm > args.adv_max_norm).to(embeds_init)
                        reweights = (args.adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)
                        delta = (delta * reweights).detach()
                elif args.norm_type == "linf":
                    denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1, 1)
                    denorm = torch.clamp(denorm, min=1e-8)
                    delta = (delta + args.adv_lr * delta_grad / denorm).detach()
                    if args.adv_max_norm > 0:
                        delta = torch.clamp(delta, -args.adv_max_norm, args.adv_max_norm).detach()
                else:
                    print("Norm type {} not specified.".format(args.norm_type))
                    exit()

                embeds_init = model.bert.embeddings.word_embeddings(t_ul_input_ids)
                delta.requires_grad_()
                optimizer.zero_grad()

            # (4) calc the virtual adversarial training loss
            t_ul_class_loss, t_ul_class_logits, t_ul_domain_loss, t_ul_domain_logits, t_ul_hidden_states, t_ul_attentions, t_ul_z = model(
                input_ids=t_ul_input_ids,
                attention_mask=t_ul_attention_mask,
                class_labels=None,
                domain_labels=t_ul_domain_labels
            )

            inputs['inputs_embeds'] = delta + embeds_init
            adv_t_ul_class_loss, adv_t_ul_class_logits, adv_t_ul_domain_loss, adv_t_ul_domain_logits, adv_t_ul_hidden_states, adv_t_ul_attentions, adv_t_ul_z = model(
                inputs_embeds=inputs['inputs_embeds'],
                attention_mask=t_ul_attention_mask,
                class_labels=None,
                domain_labels=t_ul_domain_labels
            )
            if args.virtual_adv:
                adv_t_ul_loss = adv_lf(t_ul_domain_logits, adv_t_ul_domain_logits)
            else:
                adv_t_ul_loss = adv_t_ul_domain_loss

            # ===contrastive===
            if args.contrast_update == 'one':
                z_cat = torch.cat([t_ul_z, adv_t_ul_z.detach()], dim=0)
                t_ul_contrast_logits, t_ul_contrast_labels = info_nce_loss(z_cat, n_views=2, device=device,batch_size=t_ul_z.shape[0])
                t_ul_contrast_loss = contrast_lf(t_ul_contrast_logits, t_ul_contrast_labels)
            elif args.contrast_update == 'mix':
                z_cat_1 = torch.cat([t_ul_z, adv_t_ul_z.detach()], dim=0)
                t_ul_contrast_logits_1, t_ul_contrast_labels_1 = info_nce_loss(z_cat_1, n_views=2, device=device,batch_size=t_ul_z.shape[0])
                t_ul_contrast_loss_1 = contrast_lf(t_ul_contrast_logits_1, t_ul_contrast_labels_1)
                z_cat_2 = torch.cat([t_ul_z.detach(), adv_t_ul_z], dim=0)
                t_ul_contrast_logits_2, t_ul_contrast_labels_2 = info_nce_loss(z_cat_2, n_views=2, device=device,batch_size=t_ul_z.shape[0])
                t_ul_contrast_loss_2 = contrast_lf(t_ul_contrast_logits_2, t_ul_contrast_labels_2)
                t_ul_contrast_loss = (t_ul_contrast_loss_1 + t_ul_contrast_loss_2) / 2
            else:
                t_ul_contrast_logits, t_ul_contrast_labels = info_nce_loss(torch.cat([t_ul_z, adv_t_ul_z], dim=0), n_views=2, device=device,batch_size=t_ul_z.shape[0])
                t_ul_contrast_loss = contrast_lf(t_ul_contrast_logits, t_ul_contrast_labels)

            # ===consistency loss===
            t_ul_consistency_loss = consist_lf(t_ul_class_logits, adv_t_ul_class_logits)

            # ===loss===
            loss = args.domain_lbd * (t_ul_domain_loss + args.adv_alpha * adv_t_ul_loss) + \
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
            checkpoint_path = args.ckpt_dir + "/" + source_domain_name + "-" + target_domain_name + ".tau_0.5.linear.contrast.analyze.ckpt"
            torch.save(model.state_dict(), checkpoint_path)

    print (acc)

    return acc

if __name__ == "__main__":
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
    parser.add_argument('--virtual_adv', action='store_true',
                        help='if using virtual adversarial training to substitute the standard adversarial training')

    parser.add_argument('--contrast_lbd', type=float, default=0.05,
                        help='contrastive labmda')
    parser.add_argument('--tau', type=float, default=0.12,
                        help='contrastive temperature')
    parser.add_argument('--contrast_update', type=str, default='two',
                        help='one, mix, two')

    parser.add_argument('--domain_lbd', type=float, default=0.001,
                        help='domain classification lambda')

    parser.add_argument('--consis_belta', type=float, default=3,
                        help='belta for consistency loss')

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)


    #train_single_source('electronics', 'book', args)
    '''
    for target_domain_name in small_domain_names:
        for source_domain_name in small_domain_names:
            if target_domain_name != source_domain_name:
                acc_list = []
                for i in range(5):
                    acc = train_single_source(source_domain_name, target_domain_name, args)
                    acc_list.append(acc)

                acc_array = np.array(acc_list)
                print (acc_array, np.average(acc_array), np.std(acc_array))
    '''
    train_single_source('electronics', 'book', args)
    #train_single_source('music', 'beauty', args)