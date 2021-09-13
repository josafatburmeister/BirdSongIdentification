import numpy as np
import matplotlib.pyplot as plt


def visualize():
    plt.rcParams["figure.autolayout"] = True
    x_labels = np.array(["Cyanistes_caeruleus_song",
                         "Erithacus_rubecula_song",
                         "Fringilla_coelebs_song",
                         "Luscinia_megarhynchos_song",
                         "Parus_major_song",
                         "Phylloscopus_collybita",
                         "Phylloscopus_collybita_song",
                         "Sylvia_atricapilla_song",
                         "Troglodytes_troglodytes_song",
                         "Turdus_philomelos_song"])

    x_nums = np.arange(10) + 1
    x = np.arange(2)
    width = 0.5

    adam_medium_results3 = np.array(
        [0, 0.5365853309631348, 0.13333334028720856, 0.14814816415309906, 0, 0, 0.5384615063667297, 0,
         0.43478262424468994, 0])
    adam_full_results3 = np.array([0, 0.48888885974884033, 0.11764706671237946, 0, 0, 0, 0.3636363446712494, 0, 0.5, 0])

    adam_medium_results2 = np.array(
        [0, 0.5641025900840759, 0.1428571343421936, 0.2222222238779068, 0, 0, 0.5454545021057129, 0, 0.52173912525177,
         0])
    adam_full_results2 = np.array(
        [0, 0.5, 0.1250000149011612, 0.08000000566244125, 0, 0, 0.3333333134651184, 0, 0.5833333134651184, 0])

    adam_medium_results1 = np.array(
        [0.6710037589073181, 0.591316819190979, 0.6348580121994019, 0.6882014274597168, 0.7657743692398071,
         0.8100394010543823, 0.7477155327796936, 0.5760869383811951, 0.7127272486686707, 0.6175653338432312])
    adam_full_results1 = np.array(
        [0.6182531714439392, 0.5685840845108032, 0.631952702999115, 0.6804906725883484, 0.7560269832611084,
         0.8308433890342712, 0.7155812382698059, 0.5542949438095093, 0.7030170559883118, 0.6251744031906128])

    plt.xticks(np.arange(2), np.array(['Adam med', 'Adam full']))
    plt.bar(x, np.array([adam_medium_results3.mean(), 0]), width=width)
    plt.bar(x, np.array([0, adam_full_results3.mean()]), width=width, color="#4CAF50")
    plt.show()
