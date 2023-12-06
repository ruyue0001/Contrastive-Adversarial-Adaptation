import torch
import math
from transformers import BertModel, AdamW, get_scheduler, BertForSequenceClassification, AutoTokenizer

import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.autograd import Function

def attention(query, key, value, mask=None, prob_function = 'softmax'):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    if prob_function == 'softmax':
        p_attn = F.softmax(scores, dim=-1) #[8,1,128]
    elif prob_function == 'sparsemax':
        sparsemax = Sparsemax(dim=-1)
        p_attn = sparsemax(scores)
    elif prob_function == 'gumblesoftmax':
        gumbels = (
            -torch.empty_like(scores, memory_format=torch.legacy_contiguous_format).exponential_().log()
        )  # ~Gumbel(0,1)
        tau = 1
        gumbels = (scores + gumbels) / tau  # ~Gumbel(logits
        p_attn = gumbels.softmax(dim=-1)
    return torch.matmul(p_attn, value), p_attn

class ReversalLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class Bertbaseline(torch.nn.Module):
    def __init__(self, num_labels=3):
        super().__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.bert.config.output_hidden_states = True
        self.bert.config.output_attentions = True
        self.dropout = torch.nn.Dropout(0.1)
        self.class_classifier = torch.nn.Linear(768, self.num_labels)
        '''
        self.class_classifier = torch.nn.Sequential()
        self.class_classifier.add_module('c_fc1', torch.nn.Linear(768, 100))
        self.class_classifier.add_module('c_bn1', torch.nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', torch.nn.ReLU(True))
        #self.class_classifier.add_module('c_drop1', torch.nn.Dropout(0.1))
        #self.class_classifier.add_module('c_fc2', torch.nn.Linear(100, 100))
        #self.class_classifier.add_module('c_bn2', torch.nn.BatchNorm1d(100))
        #self.class_classifier.add_module('c_relu2', torch.nn.ReLU(True))
        self.class_classifier.add_module('c_fc2', torch.nn.Linear(100, num_labels))
        '''

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, labels=None):
        if inputs_embeds == None:
            outputs = self.bert(input_ids, attention_mask=attention_mask)
        else:
            outputs = self.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        #logits = self.classifier(pooled_output)
        logits = self.class_classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return loss, logits, outputs.last_hidden_state, outputs.attentions


class BertDANN(torch.nn.Module):
    def __init__(self, num_labels = 3):
        super().__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.bert.config.output_hidden_states = True
        self.bert.config.output_attentions = True
        self.dropout = torch.nn.Dropout(0.1)
        self.class_classifier = torch.nn.Linear(768, self.num_labels)
        self.domain_classifier = torch.nn.Linear(768, 2)
        '''
        self.class_classifier = torch.nn.Sequential()
        self.class_classifier.add_module('c_fc1', torch.nn.Linear(768, 100))
        self.class_classifier.add_module('c_bn1', torch.nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', torch.nn.ReLU(True))
        #self.class_classifier.add_module('c_drop1', torch.nn.Dropout(0.1))
        self.class_classifier.add_module('c_fc2', torch.nn.Linear(100, self.num_labels))

        self.domain_classifier = torch.nn.Sequential()
        self.domain_classifier.add_module('d_fc1', torch.nn.Linear(768, 100))
        self.domain_classifier.add_module('d_bn1', torch.nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', torch.nn.ReLU(True))
        #self.class_classifier.add_module('c_drop1', torch.nn.Dropout(0.1))
        self.domain_classifier.add_module('d_fc2', torch.nn.Linear(100, 2))
        '''
    def forward(self, input_ids=None, attention_mask=None, class_labels=None, domain_labels=None, alpha=0):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        reverse_pooled_output = ReversalLayerF.apply(pooled_output, alpha)

        class_logits=self.class_classifier(pooled_output)
        domain_logits=self.domain_classifier(reverse_pooled_output)

        class_loss = None
        if class_labels is not None:
            class_loss_fct = CrossEntropyLoss()
            class_loss = class_loss_fct(class_logits.view(-1, self.num_labels), class_labels.view(-1))

        domain_loss = None
        if domain_labels is not None:
            domain_loss_fct = CrossEntropyLoss()
            domain_loss = domain_loss_fct(domain_logits.view(-1,2), domain_labels.view(-1))

        return class_loss, class_logits, domain_loss, domain_logits, outputs.last_hidden_state, outputs.attentions


class BertAdvContrastSequenceClassification(torch.nn.Module):
    def __init__(self, num_labels=3):
        super().__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.bert.config.output_hidden_states = True
        self.bert.config.output_attentions = True
        #self.bert.config.type_vocab_size = 2
        #single_emb = self.bert.embeddings.token_type_embeddings
        #self.bert.embeddings.token_type_embeddings = torch.nn.Embedding(2, single_emb.embedding_dim)
        #self.bert.embeddings.token_type_embeddings.weight = torch.nn.Parameter(single_emb.weight.repeat([2, 1]))

        self.dropout = torch.nn.Dropout(0.1)

        self.class_classifier = torch.nn.Linear(768, self.num_labels)
        self.domain_classifier = torch.nn.Linear(768, 2)

        '''
        self.class_classifier = torch.nn.Sequential()
        self.class_classifier.add_module('c_fc1', torch.nn.Linear(768, 100))
        self.class_classifier.add_module('c_bn1', torch.nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', torch.nn.ReLU(True))
        #self.class_classifier.add_module('c_drop1', torch.nn.Dropout(0.1))
        self.class_classifier.add_module('c_fc2', torch.nn.Linear(100, self.num_labels))

        self.domain_classifier = torch.nn.Sequential()
        self.domain_classifier.add_module('d_fc1', torch.nn.Linear(768, 100))
        self.domain_classifier.add_module('d_bn1', torch.nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', torch.nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', torch.nn.Linear(100, 2))
        '''
        self.contrast_MLP = torch.nn.Sequential()
        self.contrast_MLP.add_module('cm_fc1', torch.nn.Linear(768, 768))
        #self.contrast_MLP.add_module('cm_bn1', torch.nn.BatchNorm1d(768))
        #self.contrast_MLP.add_module('cm_relu1', torch.nn.ReLU(True))
        #self.contrast_MLP.add_module('c_drop1', torch.nn.Dropout(0.1))
        #self.contrast_MLP.add_module('cm_fc2', torch.nn.Linear(768, 768))

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, class_labels=None, domain_labels=None):
        if inputs_embeds == None:
            outputs = self.bert(input_ids, attention_mask=attention_mask)
        else:
            outputs = self.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        pooled_output = outputs[1]

        class_logits=self.class_classifier(pooled_output)
        domain_logits=self.domain_classifier(pooled_output)

        class_loss = None
        if class_labels is not None:
            class_loss_fct = CrossEntropyLoss()
            class_loss = class_loss_fct(class_logits.view(-1, self.num_labels), class_labels.view(-1))

        domain_loss = None
        if domain_labels is not None:
            domain_loss_fct = CrossEntropyLoss()
            domain_loss = domain_loss_fct(domain_logits.view(-1,2), domain_labels.view(-1))

        z = self.contrast_MLP(pooled_output)

        return class_loss, class_logits, domain_loss, domain_logits, outputs.last_hidden_state, outputs.attentions, z



class BertContrastSequenceClassification(torch.nn.Module):
    def __init__(self, num_domains=2, num_bert=1, num_labels=2, mask_model="gumble", mask_percentage = 0.1):
        super().__init__()
        self.num_domains = num_domains
        self.num_bert = num_bert
        self.num_labels = num_labels
        self.mask_model = mask_model
        self.mask_percentage = mask_percentage

        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.bert.config.output_hidden_states = True
        self.bert.config.output_attentions = True
        if self.num_bert == 2:
            self.bert2 = BertModel.from_pretrained("bert-base-uncased")
            self.bert2.config.output_hidden_states = True
            self.bert2.config.output_attentions = True

        self.domain_embedding = torch.nn.Embedding(self.num_domains, 768)

        self.classifier = torch.nn.Linear(768, self.num_labels)
        self.domain_classifier = torch.nn.Linear(768, self.num_domains)
        self.MLP1 = torch.nn.Linear(768, self.num_domains)
        self.MLP2 = torch.nn.Linear(768,768)
        self.MLP3 = torch.nn.Linear(768,768)
        self.Q = torch.nn.Linear(768, 768*2)
        self.K = torch.nn.Linear(768,768*2)
        self.V = torch.nn.Linear(768*2, 1)
        self.dropout = torch.nn.Dropout(p=0.1)

        self.Relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, labels=None, train_mask=False, bp = True):
        if train_mask:
            if self.mask_model == "gumble":
                outputs = self.bert(input_ids, attention_mask=attention_mask)
                last_hidden_state = outputs.last_hidden_state

                domain_discripter = self.domain_embedding(labels)
                domain_discripter = domain_discripter.unsqueeze(1)
                domain_discripter = domain_discripter.repeat(1,last_hidden_state.shape[1], 1)

                z = torch.cat((last_hidden_state, domain_discripter), dim=-1)

                z = self.MLP1(last_hidden_state)
                z = self.tanh(z)
                gumble = F.gumbel_softmax(z, tau=0.5, hard=True, eps=1e-10, dim=-1)

                device = gumble.device

                text_mask_tmp = list(range(2))
                text_mask_tmp = torch.LongTensor(text_mask_tmp).to(device).float()
                text_mask = torch.matmul(gumble.float(), text_mask_tmp)
                text_mask = text_mask * attention_mask

                mask_code = torch.LongTensor([103]).to(device)
                source_embeddings = self.bert.embeddings.word_embeddings(input_ids)
                maskcode_embeddings = self.bert.embeddings.word_embeddings(mask_code)
                maskcode_embeddings = maskcode_embeddings.repeat(source_embeddings.shape[0], source_embeddings.shape[1], 1)

                text_mask = text_mask.unsqueeze(-1)
                masked_sent_embeds = maskcode_embeddings * text_mask + source_embeddings * (1 - text_mask)

                domain_token_pooling = last_hidden_state * text_mask
                domain_token_pooling = domain_token_pooling.mean(dim=1)

                logits = self.domain_classifier(domain_token_pooling)

                loss = None
                if labels is not None:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

                return loss, logits, text_mask, masked_sent_embeds

            elif self.mask_model == "attn":
                if self.num_bert == 2:
                    outputs = self.bert2(input_ids, attention_mask=attention_mask)
                else:
                    outputs = self.bert(input_ids, attention_mask=attention_mask)

                last_hidden_state = outputs.last_hidden_state
                cls_emb = last_hidden_state[:, 0, :].unsqueeze(1)
                attention_mask_clone = attention_mask.clone()
                attention_mask_clone[:, 0] = 0
                attention_mask_clone = attention_mask_clone.unsqueeze(1)
                x, attn_weight = attention(cls_emb, last_hidden_state, last_hidden_state, attention_mask_clone, prob_function='softmax')
                x = x.squeeze(1)
                attn_weight = attn_weight.squeeze(1)

                device = attn_weight.device
                # percentage
                percentage = self.mask_percentage
                text_mask = torch.zeros(attn_weight.shape).long().to(device)
                for i in range(text_mask.shape[0]):
                    top_k = attention_mask[i].sum().item() * percentage
                    top_k = math.ceil(top_k)
                    top_k_value, top_k_indices = torch.topk(attn_weight[i], top_k, dim=-1)
                    text_mask[i, top_k_indices] = 1

                text_mask_tmp = text_mask - attn_weight.detach() + attn_weight
                text_mask_tmp = text_mask_tmp.unsqueeze(-1)

                mask_code = torch.LongTensor([103]).to(device)
                source_embeddings = self.bert.embeddings.word_embeddings(input_ids)
                maskcode_embeddings = self.bert.embeddings.word_embeddings(mask_code)
                maskcode_embeddings = maskcode_embeddings.repeat(source_embeddings.shape[0], source_embeddings.shape[1], 1)

                masked_sent_embeds = maskcode_embeddings * text_mask_tmp + source_embeddings * (1 - text_mask_tmp)

                domain_token_pooling = last_hidden_state * text_mask_tmp
                domain_token_pooling = domain_token_pooling.mean(dim=1)

                logits = self.domain_classifier(domain_token_pooling)

                loss = None
                if labels is not None:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

                return loss, logits, text_mask, masked_sent_embeds

            elif self.mask_model == "descriptor":
                if self.num_bert == 2:
                    outputs = self.bert2(input_ids, attention_mask=attention_mask)
                else:
                    outputs = self.bert(input_ids, attention_mask=attention_mask)

                last_hidden_state = outputs.last_hidden_state
                domain_descripter = self.domain_embedding(labels)
                domain_descripter = domain_descripter.unsqueeze(1)

                attention_mask_clone = attention_mask.clone()
                attention_mask_clone[:, 0] = 0

                '''
                # dot-product attention
                attention_mask_clone = attention_mask_clone.unsqueeze(1)
                query = self.Q(domain_descripter)
                key = self.K(last_hidden_state)
                value = self.V(last_hidden_state)
                x, attn_weight = attention(query.unsqueeze(1), key, value, attention_mask_clone, prob_function='softmax')
                x = x.squeeze(1)
                attn_weight = attn_weight.squeeze(1)
                '''

                #additive attention
                query = self.Q(domain_descripter)
                key = self.K(last_hidden_state)
                scores = self.tanh(query + key)
                scores = self.V(scores)
                scores = scores.squeeze(-1)
                scores = scores.masked_fill(attention_mask_clone == 0, -1e9)
                attn_weight = F.softmax(scores, dim=-1)
                x = torch.matmul(attn_weight.unsqueeze(1), last_hidden_state)
                x = x.squeeze(1)

                x = self.dropout(x)
                logits = self.domain_classifier(x)

                loss = None
                if labels is not None:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

                # calculate the mask
                percentage = self.mask_percentage
                device = attn_weight.device
                text_mask = torch.zeros(attn_weight.shape).long().to(device)
                for i in range(text_mask.shape[0]):
                    top_k = attention_mask[i].sum().item() * percentage
                    top_k = math.ceil(top_k)
                    top_k_value, top_k_indices = torch.topk(attn_weight[i], top_k, dim=-1)
                    text_mask[i, top_k_indices] = 1

                mask_code = torch.LongTensor([103]).to(device)
                source_embeddings = self.bert.embeddings.word_embeddings(input_ids)
                maskcode_embeddings = self.bert.embeddings.word_embeddings(mask_code)
                maskcode_embeddings = maskcode_embeddings.repeat(source_embeddings.shape[0], source_embeddings.shape[1], 1)

                if bp:
                    text_mask_tmp = text_mask - attn_weight.detach() + attn_weight
                else:
                    text_mask_tmp = text_mask
                text_mask_tmp = text_mask_tmp.unsqueeze(-1)
                masked_sent_embeds = maskcode_embeddings * text_mask_tmp + source_embeddings * (1 - text_mask_tmp)

                return loss, logits, text_mask, masked_sent_embeds


            elif self.mask_model == "none":
                if self.num_bert == 2:
                    outputs = self.bert2(input_ids, attention_mask=attention_mask)
                else:
                    outputs = self.bert(input_ids, attention_mask=attention_mask)

                pooled_output = outputs[1]

                pooled_output = self.dropout(pooled_output)
                logits = self.classifier(pooled_output)

                loss = None
                if labels is not None:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

                last_hidden_state = outputs.last_hidden_state
                cls_emb = last_hidden_state[:, 0, :].unsqueeze(1)
                attention_mask_clone = attention_mask.clone()
                attention_mask_clone[:, 0] = 0
                attention_mask_clone = attention_mask_clone.unsqueeze(1)
                x, attn_weight = attention(cls_emb, last_hidden_state, last_hidden_state, attention_mask_clone, prob_function='softmax')
                x = x.squeeze(1)
                attn_weight = attn_weight.squeeze(1)

                device = attn_weight.device
                # percentage
                percentage = self.mask_percentage
                text_mask = torch.zeros(attn_weight.shape).long().to(device)
                for i in range(text_mask.shape[0]):
                    top_k = attention_mask[i].sum().item() * percentage
                    top_k = math.ceil(top_k)
                    top_k_value, top_k_indices = torch.topk(attn_weight[i], top_k, dim=-1)
                    text_mask[i, top_k_indices] = 1

                masked_sent_embeds = None

                return loss, logits, text_mask, masked_sent_embeds
        else:
            if inputs_embeds == None:
                outputs = self.bert(input_ids, attention_mask=attention_mask)
            else:
                outputs = self.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state

            cls_emb = last_hidden_state[:, 0, :].unsqueeze(1)

            pooled_output = outputs[1]

            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)

            z = self.MLP2(cls_emb)
            z = self.Relu(z)
            z = self.MLP3(z)
            z = self.Relu(z)

            loss = None
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            return loss, logits, outputs.hidden_states, outputs.attentions, z
