import os
import numpy
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
from statannot import add_stat_annotation
import config

SORT = True

def get_animal_groups(df_C, col, hs, only_sng=False):
    new_hs = {}
    all_sng = []
    for v in hs:
        if not '&' in v:
            all_sng.append(v)
            if v in new_hs.keys():
                new_hs[v][0] += hs[v]
                new_hs[v][1].append(v)
            else:
                new_hs[v] = [hs[v], [v]]
        else:
            print("%s %d split" % (v, hs[v]))
            for v1 in v.split("&"):
                if v1 in new_hs.keys():
                    new_hs[v1][0] += hs[v]
                    new_hs[v1][1].append(v)
                else:
                    new_hs[v1] = [hs[v], [v]]
    throw = []
    for v in new_hs.keys():
        if sum(df_C[df_C[col] == v].not_human_host) == 0:
            print("Throwing %s" % v)
            throw.append(v)
    for v in throw:
        new_hs.pop(v)
    if only_sng:
        throw = list(set(new_hs.keys()).difference(all_sng))
        for v in throw:
            new_hs.pop(v)
    return new_hs


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

    inds = df_C[(~df_C.is_control) & (df_C.not_human_host)].index

    new_hs = get_animal_groups(df_C, 'host', df_C.loc[inds].host.value_counts().to_dict())
    new_vs = get_animal_groups(df_C, 'virus_name_corrected', df_C.loc[inds].virus_name_corrected.value_counts().to_dict())
    new_hts = get_animal_groups(df_C, 'host type', df_C.loc[inds]['host type'].value_counts().to_dict())


    dist = {}
    col = 'host type' #'virus_name_corrected
    d = new_hts #new_vs
    cat_order = d.keys()
    num_inds = {}
    for v in cat_order:
        inds = df_C[numpy.isin(df_C[col], d[v][1])].index
        num_inds[v] = len(inds)
        dist[(v, 'unexposed')] = exist[exist.columns.intersection(inds)].loc[~meta.is_recoverred].sum(1)
        dist[(v, 'recoverred')] = exist[exist.columns.intersection(inds)].loc[meta.is_recoverred].sum(1)
    dist = pandas.concat(dist).reset_index()
    dist.columns = ['category', 'covid_status', 'SerumName', 'num_passed']

    fig = sns.boxplot(data=dist, x='category', y='num_passed', hue='covid_status', order=cat_order)
    box_pairs = [((virus, "unexposed"), (virus, "recoverred")) for virus in cat_order]
    add_stat_annotation(fig, data=dist, x='category', y='num_passed', hue='covid_status', order=cat_order,
                        box_pairs=box_pairs, test='Mann-Whitney', text_format='star', loc='inside', verbose=2)
    fig.set_xticklabels(labels=[str(x._text) + "\n(%d oligos)" % num_inds[str(x._text)] for x in fig.get_xticklabels()])
    fig.set_xlabel(None)
    plt.savefig(os.path.join(out_path, "animal_host_type_box.png"))
    plt.close('all')

    ylim = [-1, 22]
    all_virus_order = {}
    for h in new_hts.keys():
        inds = df_C[numpy.isin(df_C['host type'], new_hts[h][1])].index
        vs = get_animal_groups(df_C, 'virus_name_corrected',
                               df_C.loc[inds].virus_name_corrected.value_counts().to_dict(), True)
        dist = {}
        col = 'virus_name_corrected'
        d = vs
        virus_order = d.keys()
        num_inds = {}
        for v in virus_order:
            inds = df_C[numpy.isin(df_C[col], d[v][1])].index
            num_inds[v] = len(inds)
            dist[(v, 'unexposed')] = exist[exist.columns.intersection(inds)].loc[~meta.is_recoverred].sum(1)
            dist[(v, 'recoverred')] = exist[exist.columns.intersection(inds)].loc[meta.is_recoverred].sum(1)
        dist = pandas.concat(dist).reset_index()
        dist.columns = ['virus', 'covid_status', 'SerumName', 'num_passed']

        plt.figure(constrained_layout=True, figsize=(len(virus_order), 10))
        fig = sns.boxplot(data=dist, x='virus', y='num_passed', hue='covid_status', order=virus_order)
        box_pairs = [((virus, "unexposed"), (virus, "recoverred")) for virus in virus_order]
        stats = add_stat_annotation(fig, data=dist, x='virus', y='num_passed', hue='covid_status', order=virus_order,
                            box_pairs=box_pairs, test='Mann-Whitney', text_format='star', loc='inside', verbose=2)
        all_virus_order[h] = [[stats[1][i].box1[0], stats[1][i].pval] for i in range(len(stats[1]))]
        fig.set_xticklabels(labels=[str(x._text) + "\n(%d oligos)" % num_inds[str(x._text)]
                                    for x in fig.get_xticklabels()], rotation=75, ha="right")
        fig.set_xlabel(None)
        fig.set_ylim(ylim[0], ylim[1])
        plt.title("%s corona-viruses" % h)

        plt.savefig(os.path.join(out_path, "animal_%s_box.png" % h.replace(" ", "_")))
    plt.close('all')


    dist = {}
    col = 'virus_name_corrected'
    d = new_vs
    if SORT:
        virus_order = []
        for h in ['bat', 'bird', 'Other mammals']:
            virus_order += list(pandas.DataFrame(all_virus_order[h]).sort_values(1)[0].values)
    else:
        virus_order = all_virus_order['bat'][0] + all_virus_order['bird'][0] + all_virus_order['Other mammals'][0]
    num_inds = {}
    for v in virus_order:
        inds = df_C[numpy.isin(df_C[col], d[v][1])].index
        num_inds[v] = len(inds)
        dist[(v, 'unexposed')] = exist[exist.columns.intersection(inds)].loc[~meta.is_recoverred].sum(1)
        dist[(v, 'recoverred')] = exist[exist.columns.intersection(inds)].loc[meta.is_recoverred].sum(1)
    dist = pandas.concat(dist).reset_index()
    dist.columns = ['virus', 'covid_status', 'SerumName', 'num_passed']

    plt.figure(constrained_layout=True, figsize=(20, 10))
    fig = sns.boxplot(data=dist, x='virus', y='num_passed', hue='covid_status', order=virus_order)
    box_pairs = [((virus, "unexposed"), (virus, "recoverred")) for virus in virus_order]
    add_stat_annotation(fig, data=dist, x='virus', y='num_passed', hue='covid_status', order=virus_order,
                        box_pairs=box_pairs, test='Mann-Whitney', text_format='star', loc='inside', verbose=2)
    fig.set_xticklabels(labels=[str(x._text) + "\n(%d oligos)" % num_inds[str(x._text)] for x in fig.get_xticklabels()],
                        rotation=75, ha="right")
    fig.set_xlabel(None)
    plt.title("By corona-virus host")
    if SORT:
        plt.savefig(os.path.join(out_path, "animal_box_sorted.png"))
    else:
        plt.savefig(os.path.join(out_path, "animal_box.png"))
    plt.close('all')
    print()
