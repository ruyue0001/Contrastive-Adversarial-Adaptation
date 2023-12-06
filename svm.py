import numpy as np
from time import time
#import matplotlib
#matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import torch

from sklearn import datasets
from sklearn.manifold import TSNE
from sklearn.svm import SVC

def get_data(source_domain_name, target_domain_name, save_path):
    #save_path = "fig/DANN.best/" + source_domain_name + "-" + target_domain_name
    features = torch.load(save_path + ".linear.analyze.features.pt").cpu().numpy()
    class_labels = torch.load(save_path + ".linear.analyze.class.pt").cpu().numpy()
    domain_labels = torch.load(save_path + ".linear.analyze.domain.pt").cpu().numpy()

    n_samples, n_features = features.shape
    return features, class_labels, domain_labels, n_samples, n_features

def main(source_domain_name, target_domain_name, model_type):
    if model_type == 'bert-baseline':
        save_path = "fig/bert-baseline/" + source_domain_name + "-" + target_domain_name
        fig_path = "fig/bert-baseline/lieanr.analyze.init_random.metric_cosine.pep_20.exa_20.lr_400.png"
    elif model_type == "DANN.best":
        save_path = "fig/DANN.best/" + source_domain_name + "-" + target_domain_name
        fig_path = "fig/DANN.best/linear.analyze.init_random.metric_cosine.pep_20.exa_20.lr_400.png"
    elif model_type == "DANN.worst":
        save_path = "fig/DANN.worst/" + source_domain_name + "-" + target_domain_name
        fig_path = "fig/DANN.worst/linear.analyze.init_random.metric_cosine.pep_20.exa_20.lr_400.png"
    elif model_type == "contrast":
        save_path = "fig/contrast/" + source_domain_name + "-" + target_domain_name
        fig_path = "fig/contrast/tau_0.1.linear.analyze.init_random.metric_cosine.pep_20.exa_20.lr_800.png"
    features, class_labels, domain_labels, n_samples, n_features = get_data(source_domain_name, target_domain_name, save_path)

    print (features.shape)
    source_features = features[0:2000,:]
    target_features = features[2000:,:]
    source_train = source_features[0:1000]
    source_test = source_features[1000:]
    target_train = target_features[0:1000]
    target_test = target_features[1000:]

    train_set = np.concatenate([source_train, target_train])
    test_set = np.concatenate([source_test, target_test])
    label = np.concatenate([np.zeros([1000]).astype(int), np.ones([1000]).astype(int)])

    SVCClf = SVC(kernel='linear', gamma='scale', shrinking=False)
    SVCClf.fit(train_set, label)

    train_score = SVCClf.score(train_set,label)
    print(train_score)
    test_score = SVCClf.score(test_set,label)
    print (test_score)
    #SVCClf.predict(test_set)

if __name__=="__main__":
    main('electronics', 'book', 'bert-baseline')