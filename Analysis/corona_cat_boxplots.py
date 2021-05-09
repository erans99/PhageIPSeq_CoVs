import os
import numpy
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
from statannot import add_stat_annotation
import config


if __name__ == '__main__':
    cache_path = config.DATA_DIR
    df_C = pandas.read_csv(os.path.join(cache_path, "df_info.csv"), index_col=0)

    out_path = os.path.join(config.OUTPUT_DIR, "plots")
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    meta = pandas.read_csv(os.path.join(cache_path, "meta.csv"), index_col=0)
    meta['is_recoverred'] = meta.is_exposed
    exist = pandas.read_csv(os.path.join(cache_path, "exist.csv"), index_col=0)
    fold = pandas.read_csv(os.path.join(cache_path, "fold.csv"), index_col=0)
    s = fold[fold < 0].sum()
    bad_olis = s[s < 0].index
    print("Throwing %d bad oligos" % len(bad_olis))
    exist = exist[exist.columns.difference(bad_olis)]

    df_C = df_C[(~df_C.is_control)]

    cat_order = ['Alphacoronavirus', 'Betacoronavirus', 'Gammacoronavirus', 'Deltacoronavirus']

    dist = {}
    col = 'genera' 
    num_inds = {}
    for v in cat_order:
        inds = df_C[df_C[col] == v].index
        num_inds[v] = len(inds)
        dist[(v, 'unexposed')] = exist[exist.columns.intersection(inds)].loc[~meta.is_recoverred].sum(1)
        dist[(v, 'recoverred')] = exist[exist.columns.intersection(inds)].loc[meta.is_recoverred].sum(1)
    dist = pandas.concat(dist).reset_index()
    dist.columns = ['category', 'covid_status', 'SerumName', 'num_passed']

    plt.figure(figsize=(15,5))
    fig = sns.boxplot(data=dist, x='category', y='num_passed', hue='covid_status', order=cat_order)
    box_pairs = [((g, "unexposed"), (g, "recoverred")) for g in cat_order]
    add_stat_annotation(fig, data=dist, x='category', y='num_passed', hue='covid_status', order=cat_order,
                        box_pairs=box_pairs, test='Mann-Whitney', text_format='full', loc='inside', verbose=2)
    fig.set_xticklabels(labels=[str(x._text) + "\n(%d oligos)" % num_inds[str(x._text)] for x in fig.get_xticklabels()])
    fig.set_xlabel(None)
    plt.title("All oligos")
    plt.savefig(os.path.join(out_path, "genera_type_box.png"))
    plt.close('all')

    df_C = df_C[(df_C.SARS_Cov_2_var == False) & (df_C.SARS_Cov_2_ref == False) & df_C.ref_seq]
    
    dist = {}
    col = 'genera' 
    num_inds = {}
    for v in cat_order:
        inds = df_C[df_C[col] == v].index
        num_inds[v] = len(inds)
        dist[(v, 'unexposed')] = exist[exist.columns.intersection(inds)].loc[~meta.is_recoverred].sum(1)
        dist[(v, 'recoverred')] = exist[exist.columns.intersection(inds)].loc[meta.is_recoverred].sum(1)
    dist = pandas.concat(dist).reset_index()
    dist.columns = ['category', 'covid_status', 'SerumName', 'num_passed']

    plt.figure(figsize=(15,5))
    fig = sns.boxplot(data=dist, x='category', y='num_passed', hue='covid_status', order=cat_order)
    box_pairs = [((g, "unexposed"), (g, "recoverred")) for g in cat_order]
    add_stat_annotation(fig, data=dist, x='category', y='num_passed', hue='covid_status', order=cat_order,
                        box_pairs=box_pairs, test='Mann-Whitney', text_format='full', loc='inside', verbose=2)
    fig.set_xticklabels(labels=[str(x._text) + "\n(%d oligos)" % num_inds[str(x._text)] for x in fig.get_xticklabels()])
    fig.set_xlabel(None)
    plt.title("Non SARS-2 oligos")
    plt.savefig(os.path.join(out_path, "genera_type_nonSARS2_box.png"))
    plt.close('all')

    df_C = df_C[df_C.human_host == False]

    dist = {}
    col = 'genera'
    num_inds = {}
    for v in cat_order:
        inds = df_C[df_C[col] == v].index
        num_inds[v] = len(inds)
        dist[(v, 'unexposed')] = exist[exist.columns.intersection(inds)].loc[~meta.is_recoverred].sum(1)
        dist[(v, 'recoverred')] = exist[exist.columns.intersection(inds)].loc[meta.is_recoverred].sum(1)
    dist = pandas.concat(dist).reset_index()
    dist.columns = ['category', 'covid_status', 'SerumName', 'num_passed']

    plt.figure(figsize=(15, 5))
    fig = sns.boxplot(data=dist, x='category', y='num_passed', hue='covid_status', order=cat_order)
    box_pairs = [((g, "unexposed"), (g, "recoverred")) for g in cat_order]
    add_stat_annotation(fig, data=dist, x='category', y='num_passed', hue='covid_status', order=cat_order,
                        box_pairs=box_pairs, test='Mann-Whitney', text_format='full', loc='inside', verbose=2)
    fig.set_xticklabels(labels=[str(x._text) + "\n(%d oligos)" % num_inds[str(x._text)] for x in fig.get_xticklabels()])
    fig.set_xlabel(None)
    plt.title("Non human oligos")
    plt.savefig(os.path.join(out_path, "genera_type_nonhuman_box.png"))
    plt.close('all')

