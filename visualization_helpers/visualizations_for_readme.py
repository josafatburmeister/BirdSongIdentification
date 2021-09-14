import numpy as np

import matplotlib.pyplot as plt


def visualize_matrix():
    plt.rcParams["figure.autolayout"] = True

    plots = []

    # for learning rates           0.1,   0.01, 0.001, 0.0001, 0.00001, 0.000001
    lr_batchsize_mean = np.array([[0.730, 0.735, 0.737, 0.744, 0.737, 0.684],  # batch size 64
                                  [0.723, 0.727, 0.736, 0.743, 0.737, 0.642],  # batch size 128
                                  [0.720, 0.727, 0.730, 0.741, 0.737, 0.560],  # batch size 256
                                  [0.711, 0.716, 0.737, 0.743, 0.733, 0.327],  # batch size 512
                                  [0.700, 0.718, 0.726, 0.739, 0.718, 0.302]])  # batch size 1024

    plots.append({
        'title': "F1-Score Mean",
        'matrix': lr_batchsize_mean,
        'y_labels': [64, 128, 256, 512, 1024],
        'x_labels': [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
    })

    # for learning rates           0.1,   0.01, 0.001, 0.0001, 0.00001, 0.000001
    lr_batchsize_max = np.array([[0.843, 0.836, 0.846, 0.853, 0.852, 0.827],  # batch size 64
                                 [0.843, 0.828, 0.857, 0.866, 0.856, 0.807],  # batch size 128
                                 [0.823, 0.846, 0.847, 0.865, 0.850, 0.767],  # batch size 256
                                 [0.838, 0.832, 0.847, 0.857, 0.846, 0.675],  # batch size 512
                                 [0.822, 0.837, 0.850, 0.858, 0.838, 0.529]])  # batch size 1024

    plots.append({
        'title': "F1-Score Max",
        'matrix': lr_batchsize_max,
        'y_labels': [64, 128, 256, 512, 1024],
        'x_labels': [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
    })

    # for learning rates           0.1,   0.01, 0.001, 0.0001, 0.00001, 0.000001
    lr_batchsize_min = np.array([[0.635, 0.647, 0.649, 0.658, 0.647, 0.522],  # batch size 64
                                 [0.628, 0.610, 0.647, 0.649, 0.630, 0.424],  # batch size 128
                                 [0.631, 0.629, 0.615, 0.641, 0.640, 0.205],  # batch size 256
                                 [0.603, 0.615, 0.651, 0.654, 0.626, 0.003],  # batch size 512
                                 [0.596, 0.600, 0.589, 0.632, 0.586, 0.044]])  # batch size 1024

    plots.append({
        'title': "F1-Score Min",
        'matrix': lr_batchsize_min,
        'y_labels': [64, 128, 256, 512, 1024],
        'x_labels': [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
    })

    # for learning rates           0.1,   0.01, 0.001, 0.0001, 0.00001
    lr_batchsize_mean_cut = np.array([[0.730, 0.735, 0.737, 0.744, 0.737],  # batch size 64
                                      [0.723, 0.727, 0.736, 0.743, 0.737],  # batch size 128
                                      [0.720, 0.727, 0.730, 0.741, 0.737],  # batch size 256
                                      [0.711, 0.716, 0.737, 0.743, 0.733],  # batch size 512
                                      [0.700, 0.718, 0.726, 0.739, 0.718]])  # batch size 1024

    plots.append({
        'title': "F1-Score Mean (cut)",
        'matrix': lr_batchsize_mean_cut,
        'y_labels': [64, 128, 256, 512, 1024],
        'x_labels': [0.1, 0.01, 0.001, 0.0001, 0.00001]
    })

    # for learning rates           0.1,   0.01, 0.001, 0.0001, 0.00001, 0.000001
    lr_batchsize_max_cut = np.array([[0.843, 0.836, 0.846, 0.853, 0.852],  # batch size 64
                                     [0.843, 0.828, 0.857, 0.866, 0.856],  # batch size 128
                                     [0.823, 0.846, 0.847, 0.865, 0.850],  # batch size 256
                                     [0.838, 0.832, 0.847, 0.857, 0.846],  # batch size 512
                                     [0.822, 0.837, 0.850, 0.858, 0.838]])  # batch size 1024

    plots.append({
        'title': "F1-Score Max (cut)",
        'matrix': lr_batchsize_max_cut,
        'y_labels': [64, 128, 256, 512, 1024],
        'x_labels': [0.1, 0.01, 0.001, 0.0001, 0.00001]
    })

    # for learning rates           0.1,   0.01, 0.001, 0.0001, 0.00001, 0.000001
    lr_batchsize_min_cut = np.array([[0.635, 0.647, 0.649, 0.658, 0.647],  # batch size 64
                                     [0.628, 0.610, 0.647, 0.649, 0.630],  # batch size 128
                                     [0.631, 0.629, 0.615, 0.641, 0.640],  # batch size 256
                                     [0.603, 0.615, 0.651, 0.654, 0.626],  # batch size 512
                                     [0.596, 0.600, 0.589, 0.632, 0.586]])  # batch size 1024

    plots.append({
        'title': "F1-Score Min (cut)",
        'matrix': lr_batchsize_min_cut,
        'y_labels': [64, 128, 256, 512, 1024],
        'x_labels': [0.1, 0.01, 0.001, 0.0001, 0.00001]
    })

    for plot in plots:
        im = plt.imshow(plot["matrix"])
        plt.colorbar(im)
        plt.xlabel("learning rate", fontdict={'family': 'serif', 'size': 20})
        plt.ylabel("batch size", fontdict={'family': 'serif', 'size': 20})
        plt.xticks(np.arange(len(plot["x_labels"])), plot["x_labels"], size=10)
        plt.yticks(np.arange(len(plot["y_labels"])), plot["y_labels"], size=10)
        plt.title(plot["title"], fontdict={'family': 'serif', 'size': 24})
        plt.show()
        plt.clf()


def visualize_hyperparameter_bars():
    plots = []

    transfer_learning = np.array([0.755, 0.752, 0.741, 0.703])

    plots.append({
        'title': 'Transfer Learning',
        'array': transfer_learning,
        'labels': ['4', '3', '2', '1'],
        'x-label': 'Layers unfrozen',
        'rotate': 0
    })

    weight_decay = np.array([0.739, 0.743, 0.722, 0.641])

    plots.append({
        'title': 'Weight Decay',
        'array': weight_decay,
        'labels': [0.0001, 0.001, 0.01, 0.1],
        'x-label': '',
        'rotate': 0
    })

    dropout = np.array([0.741, 0.743, 0.742, 0.741, 0.740, 0.741, 0.741])

    plots.append({
        'title': 'Probability of Dropout',
        'array': dropout,
        'labels': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        'x-label': '',
        'rotate': 0
    })

    architechtures = np.array([0.741, 0.698, 0.710, 0.737, 0.728])

    plots.append({
        'title': 'Performance Comparison',
        'array': architechtures,
        'labels': ['Resnet-18', 'Res18 noisy', 'Res18 more data', 'Resnet-50', 'Resnet-121'],
        'x-label': '',
        'rotate': 15
    })

    for plot in plots:
        plt.style.use('ggplot')
        x_pos = [i for i, _ in enumerate(plot['labels'])]

        plt.bar(x_pos, plot['array'], color='green')
        plt.ylabel('F1-score')
        plt.xlabel(plot["x-label"])
        plt.title(plot["title"], fontdict={'family': 'serif', 'size': 24})
        plt.xticks(x_pos, plot['labels'], rotation=plot['rotate'])
        plt.yticks((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1), (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1))
        plt.tight_layout()
        plt.show()
        plt.clf()


def visualize_additionl_data():
    baseline = np.array([0.639, 0.720, 0.740, 0.742, 0.736, 0.726, 0.708, 0.680, 0.627])
    baseline_nips4bplus = np.array([0.258, 0.239, 0.213, 0.175, 0.139, 0.122, 0.098, 0.074, 0.045])
    baseline_nips4bplus_cut = np.array([0.266, 0.247, 0.218, 0.178, 0.139, 0.120, 0.093, 0.072, 0.044])

    noisy = np.array([0.572, 0.663, 0.687, 0.689, 0.683, 0.670, 0.649, 0.616, 0.552])
    noisy_nips4bplus = np.array([0.310, 0.276, 0.239, 0.201, 0.173, 0.148, 0.109, 0.078, 0.045])
    noisy_nips4bplus_cut = np.array([0.310, 0.276, 0.247, 0.198, 0.170, 0.149, 0.113, 0.080, 0.044])

    more_data = np.array([0.613, 0.693, 0.715, 0.719, 0.715, 0.706, 0.690, 0.665, 0.614])
    more_data_nips4bplus = np.array([0.314, 0.308, 0.281, 0.254, 0.219, 0.196, 0.165, 0.129, 0.079])
    more_data_nips4bplus_cut = np.array([0.315, 0.302, 0.279, 0.258, 0.225, 0.193, 0.161, 0.128, 0.081])

    ind = np.arange(9) * 1.5
    width = 0.35
    plt.barh(ind - width, baseline, width, label='Baseline')
    plt.barh(ind, noisy, width, label='Noisy Data')
    plt.barh(ind + width, more_data, width, label='More Data')
    plt.xlabel('F1-Score')
    plt.title('Xeno Canto')
    plt.xticks((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1), (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1))
    plt.yticks(ind + width / 2, (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))
    plt.legend(loc='best')
    plt.show()

    plt.clf()

    # NIPS4BPlus
    plt.barh(ind - width, baseline_nips4bplus, width, label='Baseline')
    plt.barh(ind, noisy_nips4bplus, width, label='Noisy Data')
    plt.barh(ind + width, more_data_nips4bplus, width, label='More Data')
    plt.xlabel('F1-Score')
    plt.title('nips4BPlus')
    plt.xticks((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1), (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1))
    plt.yticks(ind + width / 2, (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))
    plt.legend(loc='best')
    plt.show()

    plt.clf()

    # NIPS4BPlus Cut
    plt.barh(ind - width, baseline_nips4bplus_cut, width, label='Baseline')
    plt.barh(ind, noisy_nips4bplus_cut, width, label='Noisy Data')
    plt.barh(ind + width, more_data_nips4bplus_cut, width, label='More Data')
    plt.xlabel('F1-Score')
    plt.title('nips4BPlus cut')
    plt.xticks((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1), (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1))
    plt.yticks(ind + width / 2, (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))
    plt.legend(loc='best')
    plt.show()


if __name__ == 'main':
    visualize_matrix()
    visualize_hyperparameter_bars()
    visualize_additionl_data()
