import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re

from performance.statistics_utils import get_max_F1
from performance.statistics_utils import pr_auc_score

default_dict_sample = {'ED': 'ED_1h', 'healthy': 'WT', 'energy depleted': 'ED_1h',
                       'ED_6h': 'ED_6h', 'S.pombe': 'S.pombe', 'S.cerevisiae': 'S.cerevisiae'}
default_excluded_labels = ['tomo_name']


def get_tomo_names_dict(tomos_names):
    tomo_names_dict = {}
    for element in tomos_names:
        tomo_names_dict[element] = element
    return tomo_names_dict


def extract_float_parameter(label, param):
    search = re.findall('(?<=' + param + ')\d+\.\d+', label)
    if len(search) == 0:
        value = 0
    else:
        value = float(search[0][1:])
    return value


def extract_parameter_from_label(label, param):
    if param in ['shuffle', 'BN']:
        search = re.search('(?<=' + param + '_)\w+', label)
        if search is None:
            param_value = 'false'
        else:
            rest = search.group(0)
            print("rest =", rest)
            search_bool = re.search(param + '_true', label)
            if search_bool is None:
                print("search_bool is ", search_bool)
                param_value = 'false'
            else:
                print("search_bool is ", search_bool)
                param_value = 'true'
    elif param == 'DA':
        search = re.search('(?<=' + param + '_)\w+', label)
        if search is None:
            DA = 'false'
            G = 0
            E = 0
            R = 0
            SP = 0
            DArounds = 0
        else:
            param = 'G'
            search = re.findall(param + '\d+\.\d+', label)
            if len(search) == 0:
                search = re.findall(param + '\d', label)
                if len(search) == 0:
                    G = 0
                else:
                    G = int(search[0][1:])
            else:
                G = float(search[0][1:])

            param = 'R'
            search = re.findall(param + '\d+', label)
            if len(search) == 0:
                R = 0
            else:
                R = int(search[0][1:])

            param = 'E'
            search = re.findall(param + '\d+.\d+', label)
            if len(search) == 0:
                search = re.findall(param + '\d', label)
                if len(search) == 0:
                    E = 0
                else:
                    E = int(search[0][1:])
            else:
                E = float(search[0][1:])

            param = 'SP'
            search = re.findall(param + '\d+.\d+', label)
            if len(search) == 0:
                search = re.findall(param + '\d', label)
                if len(search) == 0:
                    SP = 0
                else:
                    SP = int(search[0][2:])
            else:
                SP = float(search[0][2:])

            param = 'DArounds'
            search = re.findall(param + '\d+', label)
            if len(search) == 0:
                DArounds = 0
            else:
                DArounds = int(search[0][8:])
            if np.max([G, R, E, SP]) > 0:
                DA = 'true'
            else:
                DA = 'false'
        param_value = DA, G, R, E, SP, DArounds
    elif param in ['encoder_dropout', 'decoder_dropout']:
        param_value = extract_float_parameter(label, param + '_')
    elif param in ['auPRC', 'F1']:
        regex = '(' + param + ')'
        search = re.search(regex, label)
        if search is None:
            param_value = False
        else:
            print("value of search", search.group(0))
            param_value = search.group(0)
            if isinstance(param_value, str):
                param_value = True
            else:
                param_value = False
    else:
        regex = '(?<=' + param + '_)\d+'
        search = re.search(regex, label)
        if search is None:
            param_value = False
        else:
            param_value = search.group(0)

    return param_value


def load_plot_df(statistics_df, dataset_df, test_tomos, tomo_names_dict, excluded_labels=None,
                 radius=8, dict_sample_type=None, dice=False):
    if excluded_labels is None:
        excluded_labels = default_excluded_labels
    if dict_sample_type is None:
        dict_sample_type = default_dict_sample
    stats_df = pd.read_csv(statistics_df)
    dataset_df = pd.read_csv(dataset_df)
    labels = []
    for label in stats_df.keys():
        if label in excluded_labels:
            print(label, "will be excluded.")
        elif not dice:
            if extract_parameter_from_label(label, "radius") == str(radius):
                labels.append(label)
                print(label, "radius ok")
        else:
            if label[-4:] == "dice":
                labels.append(label)
            else:
                print(label, "will be excluded.")
    plot_df = pd.DataFrame({})
    for label in labels:
        print(label)
        frac = extract_parameter_from_label(label, 'frac')
        D = extract_parameter_from_label(label, 'D')
        IF = extract_parameter_from_label(label, 'IF')
        shuffle = extract_parameter_from_label(label, 'shuffle')
        DA, G, R, E, SP, DArounds = extract_parameter_from_label(label, param='DA')
        BN = extract_parameter_from_label(label, 'BN')
        encoder_dropout = extract_parameter_from_label(label, 'encoder_dropout')
        decoder_dropout = extract_parameter_from_label(label, 'decoder_dropout')
        full_dropout = (encoder_dropout, decoder_dropout)
        for tomo in test_tomos:
            tomo_name = tomo + "_" + str(frac) + "_" + str(frac)
            if tomo_name in dataset_df['tomo_name'].values:
                tomo_data = dataset_df[dataset_df['tomo_name'] == tomo_name]
                sample_type = dict_sample_type[tomo_data.iloc[0]['sample_type']]
                species = dict_sample_type[tomo_data.iloc[0]['species']]
                vpp = tomo_data.iloc[0]['vpp']
                stats_tomo_name = extract_tomo_name_from_fraction_name(tomo_name)
                if stats_tomo_name in test_tomos:
                    tmp_stats_df = stats_df[stats_df['tomo_name'] == tomo_name]
                    if tmp_stats_df[label].shape[0] > 0:
                        auPRC = tmp_stats_df[label].values[0]
                        miniplot_df = pd.DataFrame({})
                        miniplot_df['model'] = [label]
                        miniplot_df['tomogram'] = [tomo_names_dict[stats_tomo_name]]
                        miniplot_df['auPRC'] = [auPRC]
                        miniplot_df['D'] = int(D)
                        miniplot_df['IF'] = int(IF)
                        miniplot_df['frac'] = int(frac)
                        miniplot_df['shuffle'] = shuffle
                        miniplot_df['BN'] = BN
                        miniplot_df['DA'] = DA
                        miniplot_df['G'] = G
                        miniplot_df['R'] = R
                        miniplot_df['E'] = E
                        miniplot_df['DArounds'] = DArounds
                        miniplot_df['decoder_dropout'] = decoder_dropout
                        miniplot_df['encoder_dropout'] = encoder_dropout
                        miniplot_df['full_dropout'] = [full_dropout]
                        miniplot_df['sample_type'] = sample_type
                        miniplot_df['species'] = species
                        miniplot_df['vpp'] = vpp
                        plot_df = plot_df.append(miniplot_df, sort=False)
                    else:
                        print('label', label, 'not present for tomo', tomo_name)
                        print(tmp_stats_df[label].shape)
    return stats_df, labels, plot_df


def extract_tomo_name_from_fraction_name(frac_name: str):
    regex = '(\d+)_(\d+)'
    search = re.search(regex, frac_name)
    if search.group(0)[0] == '0':
        tomo_name = search.groups()[0]
    else:
        tomo_name = search.groups()[0] + '_' + search.groups()[1]
    return tomo_name


def sample_type_dictionary(sample_type: str):
    dict_sample_type = dict()
    if sample_type == 'healthy':
        dict_sample_type[sample_type] = 'WT'
    else:
        dict_sample_type[sample_type] = 'ED'
    return dict_sample_type


def load_plot_df_from_full_tomo(statistics_df, dataset_df, test_tomos, tomo_names_dict,
                                excluded_labels=None, radius=8, dict_sample_type=None,
                                dice=False, name_with_frac=True, statistic=None):
    if excluded_labels is None:
        excluded_labels = default_excluded_labels
    if dict_sample_type is None:
        dict_sample_type = default_dict_sample
    stats_df = pd.read_csv(statistics_df)
    dataset_df = pd.read_csv(dataset_df)
    labels = []
    for label in stats_df.keys():
        if label in excluded_labels:
            print(label, "will be excluded.")
        elif not dice:
            if extract_parameter_from_label(label, "radius") == str(radius):
                if statistic is None:
                    labels.append(label)
                else:
                    is_statistic = extract_parameter_from_label(label, statistic)
                    print(is_statistic, "statistic")
                    if is_statistic:
                        labels.append(label)


        else:
            labels.append(label)
    plot_df = pd.DataFrame({})
    for label in labels:
        if statistic is None:
            statistic_name = 'auPRC'
        else:
            statistic_name = statistic
        print(label)
        frac = extract_parameter_from_label(label, 'frac')
        if isinstance(frac, bool):
            frac = extract_parameter_from_label(label, 'set')
        D = extract_parameter_from_label(label, 'D')
        IF = extract_parameter_from_label(label, 'IF')
        shuffle = extract_parameter_from_label(label, 'shuffle')
        DA, G, R, E, SP, DArounds = extract_parameter_from_label(label, param='DA')
        print("DA, G, R, E, SP, DArounds", DA, G, R, E, SP, DArounds)
        BN = extract_parameter_from_label(label, 'BN')

        encoder_dropout = extract_parameter_from_label(label, 'encoder_dropout')
        decoder_dropout = extract_parameter_from_label(label, 'decoder_dropout')
        full_dropout = (encoder_dropout, decoder_dropout)
        for tomo in test_tomos:
            print("Tomo", tomo)
            if name_with_frac:
                tomo_name = tomo + "_" + str(frac)
                stats_tomo_name = extract_tomo_name_from_fraction_name(tomo_name)
            else:
                tomo_name = tomo
                stats_tomo_name = tomo_name
            if tomo_name in dataset_df['tomo_name'].values:
                tomo_data = dataset_df[dataset_df['tomo_name'] == tomo_name]
                sample_type = dict_sample_type[tomo_data.iloc[0]['sample_type']]
                species = dict_sample_type[tomo_data.iloc[0]['species']]
                vpp = tomo_data.iloc[0]['vpp']
                if stats_tomo_name in test_tomos:
                    tmp_stats_df = stats_df[stats_df['tomo_name'] == tomo_name]
                    if tmp_stats_df[label].shape[0] > 0:
                        auPRC = tmp_stats_df[label].values[0]
                        miniplot_df = pd.DataFrame({})
                        miniplot_df['model'] = [label]
                        miniplot_df['frac'] = [frac]
                        miniplot_df['tomogram'] = [tomo_names_dict[stats_tomo_name]]
                        miniplot_df[statistic_name] = [auPRC]
                        miniplot_df['D'] = int(D)
                        miniplot_df['IF'] = int(IF)
                        if name_with_frac:
                            miniplot_df['frac'] = int(frac)
                        miniplot_df['shuffle'] = shuffle
                        miniplot_df['BN'] = BN
                        miniplot_df['DA'] = DA
                        miniplot_df['G'] = G
                        miniplot_df['R'] = R
                        miniplot_df['E'] = E
                        miniplot_df['(G, E, R)'] = [(G, E, R)]
                        miniplot_df['(G, E, R, SP, DArounds)'] = [(G, E, R, SP, DArounds)]
                        miniplot_df['DArounds'] = DArounds
                        miniplot_df['decoder_dropout'] = decoder_dropout
                        miniplot_df['encoder_dropout'] = encoder_dropout
                        miniplot_df['full_dropout'] = [full_dropout]
                        miniplot_df['sample_type'] = sample_type
                        miniplot_df['species'] = species
                        miniplot_df['vpp'] = vpp
                        plot_df = plot_df.append(miniplot_df, sort=False)
                    else:
                        print('label', label, 'not present for tomo', tomo_name)
                        print(tmp_stats_df[label].shape)
    return stats_df, labels, plot_df


def generate_histogram(fig_file: str, x: list, label: str, bins: int, plot_title: str, plot_num: int = 1) -> None:
    matplotlib.use('Agg')
    plt.figure(plot_num)
    if math.isnan(np.max(x)) or math.isnan(np.min(x)):
        print("nan values")
    else:
        plt.hist(x, bins=bins, label=label)
    plt.xlabel("score value")
    plt.ylabel("frequency")
    plt.title(plot_title)
    plt.legend()
    plt.gcf()
    plt.savefig(fname=fig_file, format="png")
    return


def generate_bi_histogram(fig_file: str, tp: list, fp: list, bins: int, plot_title: str, plot_num: int = 1) -> None:
    matplotlib.use('Agg')
    plt.figure(plot_num)
    for color, namex, x in zip([(0, 0, 1, 0.5), (1, 0, 0, 0.5)], ["true positives", "false positives"], [tp, fp]):
        if math.isnan(np.max(x)) or math.isnan(np.min(x)):
            print("nan values")
        else:
            plt.hist(x, bins=bins, label=namex, fc=color)
    plt.xlabel("score value")
    plt.ylabel("frequency")
    plt.title(plot_title)
    plt.legend()
    plt.gcf()
    plt.savefig(fname=fig_file, format="png")
    return


def generate_prc_plot(x: list or np.array, y: list or np.array, legend: str, xlabel: str, ylabel: str,
                      title_str: str, fig_file: str, plot_num: int) -> None:
    plt.figure(plot_num)
    plt.plot(x, y, label=legend)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.title(title_str)
    plt.legend()
    plt.gcf()
    plt.savefig(fname=fig_file, format="png")
    return


def generate_performance_plots(recall, prec, F1_score, predicted_values, tp_pred_scores,
                               fp_pred_scores, figures_dir):
    fig_name = os.path.join(figures_dir, "histogram_predicted.png")
    plot_title = str(len(predicted_values)) + " peaks"
    generate_histogram(fig_file=fig_name, x=predicted_values, label="predicted",
                       bins=45, plot_title=plot_title, plot_num=1)

    fig_name = os.path.join(figures_dir, "histogram_tp_fp.png")
    plot_title = str(len(fp_pred_scores)) + " peaks"
    generate_bi_histogram(fig_file=fig_name, tp=tp_pred_scores, fp=fp_pred_scores,
                          bins=45, plot_title=plot_title, plot_num=2)

    max_F1, optimal_peak = get_max_F1(F1_score)
    f1_legend_str = "(max_F1, best_peaks) = ({}, {})".format(round(max_F1, 4), optimal_peak)
    n = len(F1_score)
    title_str = str(n) + " peaks"
    fig_name = os.path.join(figures_dir, "f1_score_" + title_str + ".png")
    generate_prc_plot(x=F1_score, y=np.linspace(start=0, stop=1, num=n), legend=f1_legend_str,
                      xlabel="number of peaks", ylabel="F1 score", title_str=title_str,
                      fig_file=fig_name, plot_num=3)

    auPRC = pr_auc_score(precision=prec, recall=recall)
    pr_legend_str = "auPRC = {}".format(auPRC)
    fig_name = os.path.join(figures_dir, "pr_" + title_str + ".png")
    generate_prc_plot(x=recall, y=prec, legend=pr_legend_str, xlabel="recall", ylabel="precision",
                      title_str=title_str, fig_file=fig_name, plot_num=4)
    return
