import os
import numpy
import pandas
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import config

human_CoVs= {"Severe acute respiratory syndrome coronavirus 2": 'SARS-2',
             "Human coronavirus OC43": 'Common Cold', 
             "Human coronavirus HKU1": 'Common Cold', 
             "Human coronavirus NL63": 'Common Cold',
             "Human coronavirus 229E": 'Common Cold',
             "Severe acute respiratory syndrome-related coronavirus": 'Other Human',
             "Middle East respiratory syndrome-related coronavirus": 'Other Human',
             "Betacoronavirus England 1": 'Other Human',}

def get_groups(df_C):
    new_hs = {}
    all_sng = []
    hs = df_C['host type'].value_counts().to_dict()
    for v in hs:
        if '&' not in v:
            all_sng.append(v)
            if v in new_hs.keys():
                new_hs[v][0] += hs[v]
                new_hs[v][1].append(v)
                new_hs[v][2] += list(df_C[df_C['host type'] == v].index)
            else:
                new_hs[v] = [hs[v], [v], list(df_C[df_C['host type']==v].index), 'host type']
        else:
            v1 = 'overlap'
            if v1 in new_hs.keys():
                new_hs[v1][0] += hs[v]
                new_hs[v1][1].append(v)
                new_hs[v1][2] += list(df_C[df_C['host type'] == v].index)
            else:
                new_hs[v1] = [hs[v], [v], list(df_C[df_C['host type'] == v].index), 'overlap']
    new_hs.pop('human')
    hs = df_C[(df_C.human_host==True)&(df_C.not_human_host==False)]['virus_name_corrected'].value_counts().to_dict()
    for v in hs:
        if '&' not in v:
            if v in human_CoVs.keys():
                if v not in all_sng:
                    all_sng.append(human_CoVs[v])
            else:
                continue
            if human_CoVs[v] in new_hs.keys():
                new_hs[human_CoVs[v]][0] += hs[v]
                new_hs[human_CoVs[v]][1].append(v)
                new_hs[human_CoVs[v]][2] += list(df_C[df_C['virus_name_corrected'] == v].index)
            else:
                new_hs[human_CoVs[v]] = [hs[v], [v], list(df_C[df_C['virus_name_corrected'] == v].index),
                                         'virus_name_corrected']
        else:
            v1 = 'overlap'
            if v1 in new_hs.keys():
                new_hs[v1][0] += hs[v]
                new_hs[v1][1].append(v)
                new_hs[v1][2] += list(df_C[df_C['virus_name_corrected'] == v].index)
            else:
                new_hs[v1] = [hs[v], [v], list(df_C[df_C['virus_name_corrected'] == v].index), 'overlap']
    return pandas.DataFrame(new_hs, index=['num_olis', 'values', 'inds', 'col_name']).T


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

    inds = df_C[(~df_C.is_control)].index
    new_hts = get_groups(df_C.loc[inds])

    df = exist[exist.columns.intersection(inds)]
    dist = {}
    num_inds = {}
    for n in new_hts.index:
        num_inds[n] = len(new_hts.loc[n, 'inds'])
        dist[(n, 'unexposed')] = exist[exist.columns.intersection(new_hts.loc[n, 'inds'])].loc[~meta.is_recoverred].sum(
            0)
        dist[(n, 'recoverred')] = exist[exist.columns.intersection(new_hts.loc[n, 'inds'])].loc[meta.is_recoverred].sum(
            0)
    dist = pandas.concat(dist).reset_index()
    dist.columns = ['category', 'covid_status', 'SerumName', 'num_passed']
    dist['cohort_size'] = numpy.nan
    dist.loc[dist[dist.covid_status == 'unexposed'].index, 'cohort_size'] = (~meta.is_recoverred).sum()
    dist.loc[dist[dist.covid_status == 'recoverred'].index, 'cohort_size'] = meta.is_recoverred.sum()
    dist.to_csv(os.path.join(out_path, "oligos_appear_in_grps.csv"))

    fig = plt.figure(constrained_layout=True, figsize=(18, 9))
    spec = gridspec.GridSpec(ncols=3, nrows=1, figure=fig)
    palette = sns.color_palette(n_colors=len(new_hts))
    pos = 0
    for name, flag in [["recoverred", True], ["unexposed", False]]:
        df = exist.loc[meta.is_recoverred == flag][exist.columns.intersection(inds)]
        res = {}
        cols = []
        for i, n in enumerate(new_hts.index):
            res[n] = [n, new_hts.loc[n, 'num_olis'], (100 * new_hts.loc[n, 'num_olis'] / len(inds))]
            cols = ['grp', 'base_num', 'perc_of_over_0%']
        min_ps = [0.01, 0.03, 0.05, 0.1, 0.2, 0.5]
        for min_p in min_ps:
            # stats = {}
            min_val = min_p * len(df)
            base_nums = []
            num_all = (df.sum() > min_val).sum()
            for i, n in enumerate(new_hts.index):
                base_nums.append((df[df.columns.intersection(new_hts.loc[n, 'inds'])].sum() > min_val).sum())
                res[n] += [base_nums[i], (100 * base_nums[i] / new_hts.loc[n, 'num_olis']),
                           (100 * base_nums[i] / num_all)]
            cols += ['num_over_%g%%' % (100 * min_p), 'perc_over_%g%%' % (100 * min_p),
                     'perc_of_over_%g%%' % (100 * min_p)]
        res = pandas.DataFrame(res, index=cols).T
        res.sort_values('perc_over_5%', ascending=False, inplace=True)
        res.to_csv(os.path.join(out_path, "percent_appear_in_grps_%s.csv" % name))
        res = res.loc[['SARS-2', 'Common Cold', 'Other Human', 'bat', 'Other mammals', 'bird', 'overlap']]
        cols = []
        names = []
        for perc in [0, 3, 5, 10, 20, 50]:
            cols.append('perc_of_over_%d%%' % perc)
            if perc == 0:
                names.append("All oligos")
            else:
                names.append("Percent of passed\nin %d%% of cohort\n(of %d)" % (perc, res['num_over_%g%%'%perc].sum()))
        ax = fig.add_subplot(spec[0, pos])
        pos += 1
        for i, n in enumerate(res.index):
            s = res.iloc[:i][cols].sum()
            ax.bar(cols, res.loc[n][cols].values, bottom=s, label=n, color=palette[i])
        if not flag:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_xticklabels(labels=names, rotation=90)
        ax.set_title("%s cohort" % name)

    plt.savefig(os.path.join(out_path, "box_percent_appear_in_grps.png"))
    plt.close("all")
