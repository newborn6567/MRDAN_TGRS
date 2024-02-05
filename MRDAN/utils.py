import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend('agg')
import seaborn as sns
import csv
import scipy.misc
from skimage import transform, filters


def plot_confusion_matrix(correct_labels, predict_labels, labels, tensor_name='MyFigure/image', normalize=False):
    from textwrap import wrap
    import re
    import itertools
    import tfplot
    import matplotlib
    import numpy as np
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(correct_labels, predict_labels, labels=labels)
    if normalize:
        cm = cm.astype('float') * 10 / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm, copy=True)
        cm = cm.astype('int')

    np.set_printoptions(precision=2)

    fig = matplotlib.figure.Figure(figsize=(7, 7), dpi=320, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap='Oranges')

    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
    classes = ['\n'.join(wrap(l, 40)) for l in classes]

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predict Label', fontsize=7)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=4, rotation=-90, ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=4, va='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i, j] != 0 else '.', horizontalalignment="center", fontsize=6,
                verticalalignment='center', color="black")
    fig.set_tight_layout(True)
    summary = tfplot.figure.to_summary(fig, tag=tensor_name)
    return summary


def scatter(x, colors, num):
    palette = np.array(sns.color_palette("hls", num))
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-40, 40)
    plt.ylim(-40, 40)
    ax.axis('off')
    ax.axis('tight')


def label_to_name(dataset, label_source_np, predict_source_np):
    idx_to_label = {}
    if dataset == 'office':
        with open('../data/label/office_label2id') as in_file:
            lines = csv.reader(in_file, delimiter=' ')
            for line in lines:
                idx_to_label[int(line[1])] = line[0]
        labels = []
        for k, v in idx_to_label.items():
            print(k, v)
            labels.append(v)

        label_source_names = []
        for item in label_source_np:
            print(item)
            idx = int(item)
            label_source_names.append(idx_to_label[idx])
            print(idx_to_label[idx])

        predict_source_names = []
        for item in predict_source_np:
            print(item)
            idx = int(item)
            predict_source_names.append(idx_to_label[idx])
            print(idx_to_label[idx])

    return label_source_names, predict_source_names, labels


def cal_label_distribution(dataset, data):
    data_list_path = '../data/' + dataset + '/' + data + "_list.txt"
    print(data_list_path)
    label_to_num = {}
    with open(data_list_path) as in_file:
        lines = csv.reader(in_file, delimiter='/')
        for line in lines:
            label = line[-2]
            # print(label)
            if label in label_to_num:
                label_to_num[label] += 1
            else:
                label_to_num[label] = 1
    res = []
    for k, v in label_to_num.items():
        print(k, v)
        res.append([k, v])
    label_distribution_path = '../data/label/' + data + "_num"

    with open(label_distribution_path, 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(['label', 'num'])
        writer.writerows(res)


def split_network(network_load):
    network_load = list(network_load[0].children())
    feature_layer = network_load[-3]
    avgpool = network_load[-4]
    bottlenet = network_load[-2]
    classifier_layer = network_load[-1]
    return feature_layer, avgpool, bottlenet, classifier_layer


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_blend_map(img, att_map, blur=True, overlap=True):
    if blur:
        att_map = filters.gaussian(att_map, 0.02 * max(img.shape[:2]))
        att_map -= att_map.min()
        att_map = att_map / att_map.max()
    cmap = plt.get_cmap('jet')
    att_map_v = cmap(att_map)
    att_map_v = np.delete(att_map_v, 3, 2)
    if overlap:
        att_map = 1 * (1 - att_map ** 0.7).reshape(att_map.shape + (1,)) * img + (att_map ** 0.7).reshape(
            att_map.shape + (1,)) * att_map_v
    return att_map


def visualize_and_save(image_path, att, test_epoch, ckpt_dir, mode):
    num_rows = 9
    num_cols = 9  # multiple of 3
    # fig = plt.figure(figsize=(num_cols * 2, num_rows * 2))
    fig = plt.figure()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0)
    orig_img = scipy.misc.imread(image_path)
    orig = scipy.misc.imresize(orig_img, (256, 256), interp='bicubic')
    start_x = (256 - 224 - 1) // 2
    start_y = (256 - 224 - 1) // 2
    orig = orig[start_x:start_x + 224, start_y:start_y + 224, :]
    atten = scipy.misc.imresize(att, (224, 224), interp='bicubic')
    ax = plt.subplot(1, 3, 1)
    ax.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    ax.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
    ax.imshow(orig)

    ax = plt.subplot(1, 3, 2)
    ax.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    ax.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
    ax.imshow(atten, alpha=1.0, cmap=plt.cm.Reds)
    # ax.set_xlabel("just for test", fontsize=10, labelpad=5, ha='center')

    ax = plt.subplot(1, 3, 3)
    ax.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    ax.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
    ax.imshow(get_blend_map(orig / 255.0, atten))
    # plt.show()
    output_dir = os.path.join(ckpt_dir, "test_epoch_%d" % test_epoch)
    output_dir = os.path.join(output_dir, mode)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    image_name = "_".join(image_path.split('/')[-3:]).strip(".jpg")
    output_path = os.path.join(output_dir, image_name + '.pdf')
    fig.savefig(output_path, bbox_inches='tight')


if __name__ == "__main__":
    ## test for scatter
    '''
    a = np.arange(20).reshape(10, 2)
    print(a)
    b = np.array([0,1]*5)
    print(b)
    scatter(a, b, 2)
    out_path = 'result/digits-tsne-try.png'
    plt.savefig(out_path, dpi=120)
    '''

    ## test for plot_confusion_matrix
    '''
    img_d_summary_dir = os.path.join(checkpoint_dir, "summaries", "img")
    img_d_summary_writer = tf.summary.FileWriter(img_d_summary_dir, sess.graph)
    img_d_summary = plot_confusion_matrix(correct_labels, predict_labels, labels, tensor_name='dev/cm')
    img_d_summary_writer.add_summary(img_d_summary, current_step)
    '''

    ## test for label to name
    '''
    dataset = 'office'
    label_to_name(dataset)
    '''

    ## test for label distribution
    dataset = 'office'
    data = 'webcam'
    cal_label_distribution(dataset, data)
