import numpy as np
from time import time
#import matplotlib
#matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import torch

from sklearn import datasets
from sklearn.manifold import TSNE


def get_data(source_domain_name, target_domain_name, save_path):
    #save_path = "fig/DANN.best/" + source_domain_name + "-" + target_domain_name
    features = torch.load(save_path + ".tau_0.1.linear.analyze.features.pt").cpu().numpy()
    class_labels = torch.load(save_path + ".tau_0.1.linear.analyze.class.pt").cpu().numpy()
    domain_labels = torch.load(save_path + ".tau_0.1.linear.analyze.domain.pt").cpu().numpy()

    n_samples, n_features = features.shape
    return features, class_labels, domain_labels, n_samples, n_features

def plot_embedding(data, class_label, domain_label, title, save_path):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    label = class_label + 3 * domain_label
    data = (data - x_min) / (x_max - x_min)

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        if label[i] == 0:
            color = 'r'
            s0 = plt.scatter(data[i, 0], data[i, 1], s=7, c=color, linewidths=0.5)
        elif label[i] == 1:
            color = 'g'
            s1 = plt.scatter(data[i, 0], data[i, 1], s=7, c=color, linewidths=0.5)
        elif label[i] == 2:
            color = 'y'
            s2 = plt.scatter(data[i, 0], data[i, 1], s=7, c=color, linewidths=0.5)
        elif label[i] == 3:
            color = 'k'
            s3 = plt.scatter(data[i, 0], data[i, 1], s=7, c=color, linewidths=0.5)
        elif label[i] == 4:
            color = 'b'
            s4 = plt.scatter(data[i, 0], data[i, 1], s=7, c=color, linewidths=0.5)
        else:
            color = 'c'
            s5 = plt.scatter(data[i, 0], data[i, 1], s=7, c=color, linewidths=0.5)

        #plt.scatter(data[i, 0], data[i, 1], s=5, c=color, linewidths=0.5)
        #plt.scatter(data[i, 0], data[i, 1], str(label[i]), color=plt.cm.Set1(label[i] / 10.), fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.legend([s0,s1,s2,s3,s4,s5],['source negative','source neutral','source positive','target negative','target neutral','target positive'], loc='best',prop={'size': 6})
    #plt.tick_params(axis='both', which='major', labelsize=14)
    plt.title(title, fontsize=20)
    plt.savefig(save_path)
    return


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
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='random', metric='cosine', random_state=0, perplexity=20, early_exaggeration=20, learning_rate=800)

    data = tsne.fit_transform(features)
    '''
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    label = class_labels + 3 * domain_labels
    data = (data - x_min) / (x_max - x_min)

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    title = 't-SNE embedding of the digits (time %.2fs)' % (time() - t0)
    plt.title(title)
    plt.savefig('fig/bert-baseline/figures.png')
    '''
    plot_embedding(data, class_labels, domain_labels, model_type, fig_path)


if __name__ == '__main__':
    #main('electronics', 'book', 'bert-baseline')
    main('electronics', 'book', 'contrast')
    #main('electronics', 'book', 'DANN.best')
    #main('electronics', 'book', 'DANN.worst')
