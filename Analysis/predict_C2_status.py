import os
import pandas
import numpy
import time
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import LeaveOneOut
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import r2_score, roc_curve, auc
import xgboost
from sklearn.linear_model import SGDClassifier
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.stats import ks_2samp, chisquare
import config

MIN_FILL = 0.01
NUM_TH = 4
MIN_APPEAR = 0.05
MIN_IN_COHORT = 0

params = {'colsample_bytree': 0.1, 'max_depth': 4, 'learning_rate': 0.001,
          'n_estimators': 6000, 'subsample': 0.6, 'min_child_weight': 0.01}

predictor_class = xgboost.XGBClassifier(nthread=NUM_TH, **params)
pred_sgdc = SGDClassifier(loss='log')


def pipeline_cross_val_x(predictor, in_data, is_classifier=True, save_pred=None, plot=None, out_path="", i='all',
                         name=""):
    df_in, df_out = in_data

    if not os.path.exists(save_pred):
        loo = LeaveOneOut()
        results = pandas.DataFrame(index=df_out.index, columns=['y', 'y_hat', 'predicted_status'])
        results['y'] = df_out

        cnt = 0
        for train_index, test_index in loo.split(df_in):
            if (cnt % 10) == 0:
                print("LOO %d of %d" % (cnt, len(df_in)), time.ctime())
            cnt += 1
            X_train, X_test = df_in.iloc[train_index], df_in.iloc[test_index]
            y_train, y_test = df_out.iloc[train_index], df_out.iloc[test_index]
            cv_predictor = sklearn.base.clone(predictor)
            cv_predictor.fit(X_train, y_train)
            if is_classifier:
                prob = cv_predictor.predict_proba(X_test)
                results['y_hat'].iloc[test_index] = prob[0][cv_predictor.classes_ == 1][0]
                if prob[0][cv_predictor.classes_ == 1][0] > 1:
                    print("impossible predicted probability")
                results['predicted_status'].iloc[test_index] = cv_predictor.predict(X_test)[0]
            else:
                pred = cv_predictor.predict(X_test)
                results['y_hat'].iloc[test_index] = pred

        if save_pred is not None:
            results.to_csv(save_pred)
    else:
        results = pandas.read_csv(save_pred, index_col=0)

    if is_classifier:
        fpr, tpr, threshold = roc_curve(results['y'], results['y_hat'])
        res = auc(fpr, tpr)
        if plot is not None:
            plt.plot(fpr, tpr)
            plt.plot([0, 1], [0, 1])
            plt.title("Leave one out predicting corona status\nfrom %s AUC %.2g" % (plot, res))
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.savefig(os.path.join(out_path, "plots_%s_ROC.png" % plot.replace(" ", "_")))
            plt.close("all")

            res_pr = precision_recall_curve(results.y, results.y_hat)
            plt.plot(res_pr[1], res_pr[0])
            plt.hlines(0.5, 0, 1, linestyles="dashed")
            plt.title("PRC curve of leave one out predicting corona status\nfrom %s AUC %.2g" % (plot, res))
            plt.xlabel("Recall (sensitivity)")
            plt.ylabel("Precision (PPV)")
            plt.savefig(os.path.join(out_path, "plots_%s_PRC.png" % plot.replace(" ", "_")))
            plt.close("all")
    else:
        res = r2_score(results['y'], results['y_hat'])
    # results['sample_id'] =

    sc, mean, std = res, results['y_hat'].mean(), results['y_hat'].std()
    pr = ("On %s Corona2 lib%s (no IEDB) oligos (appear >%g, %d oligos) LOO XGB classifier got AUC %g" %
          (i, name.replace("_", " "), MIN_APPEAR, len(df_in.columns), sc))
    print(pr)
    return pr


def get_other_name(x, ser):
    tmp = ser[ser == x]
    if len(tmp) == 1:
        return tmp.index[0]
    else:
        print("WTF %s not found" % x)
        return None


def perform_dimensionality_reduction(out_path, existence_table, cohorts, dimensionality_reduction_class, data_type,
                                     column_prefix, **kwargs):
    transformed_table = dimensionality_reduction_class.fit_transform(existence_table)
    transformed_table = pandas.DataFrame(index=existence_table.index,
                                         columns=list(map(lambda x: f'{column_prefix}{x}',
                                                          range(1, 1 + dimensionality_reduction_class.n_components))),
                                         data=transformed_table)
    transformed_table.to_csv(os.path.join(out_path, f'{data_type}.csv'))

    if 'pca' in data_type:
        pca_info = {}
        for i, c in enumerate(transformed_table.columns):
            pca_info[c] = [dimensionality_reduction_class.explained_variance_ratio_[i]]
            pca_info[c] += list(ks_2samp(transformed_table.loc[cohorts[cohorts].index][c].values,
                                         transformed_table.loc[cohorts[~cohorts].index][c].values, 'less'))
            pca_info[c] += list(ks_2samp(transformed_table.loc[cohorts[cohorts].index][c].values,
                                        transformed_table.loc[cohorts[~cohorts].index][c].values, 'greater'))
        pca_info = pandas.DataFrame(pca_info, index=['exp_var', 'ks_l_stat', 'ks_l_pval', 'ks_g_stat', 'ks_g_pval']).T
        pca_info.to_csv(os.path.join(out_path,  data_type + '_info.csv'))

    # Figure by status
    pca_info['min_ks'] = pca_info[['ks_g_stat', 'ks_g_pval']].min(1)
    pca_info.sort_values('min_ks', inplace=True)
    transformed_table.index = transformed_table.index.get_level_values(0)
    x = pca_info.index[0] #f'{column_prefix}1'
    y = pca_info.index[1] #f'{column_prefix}2'
    fig, ax = plt.subplots(ncols=1, nrows=1)
    sns.scatterplot(x=x, y=y, data=transformed_table, #transformed_table[x].values, y=transformed_table[y].values,
                    hue=cohorts*1, ax=ax)
    if 'pca' in data_type:
        plt.xlabel('%s (explained variance %.2g%%)' % (x, 100 * pca_info.loc[x, 'exp_var']))
        plt.ylabel('%s (explained variance %.2g%%)' % (y, 100 * pca_info.loc[y, 'exp_var']))
    fig.savefig(os.path.join(out_path, f'{data_type}_by_status.png'))
    plt.close(fig)


def perform_pca(out_path, existence_table, cohorts, data_type, n_components=10, **kwargs):
    n_components = min(n_components, existence_table.shape[1])
    pca = PCA(n_components=n_components)
    perform_dimensionality_reduction(out_path, existence_table.fillna(0), cohorts, pca, data_type, 'PC', **kwargs)


def check_singles(cols, fold, meta, out_path, name, df_C=None):
    res = {}
    res_cols = ["ks_l_stat", "ks_l_pval", "ks_g_stat", "ks_g_pval", "num_passed_healthy", "num_passed_recoverred",
                "chisq_nums_passed"]
    cohorts = [meta[~meta.is_recoverred].index, meta[meta.is_recoverred].index]
    num_all = [len(cohorts[0]), len(cohorts[1])]
    for c in cols:
        try:
            vals = [fold.loc[cohorts[0]][c], fold.loc[cohorts[1]][c]]
            res[c] = list(ks_2samp(vals[0], vals[1], 'greater'))
            res[c] += list(ks_2samp(vals[0], vals[1], 'less'))
            res[c] += [(vals[0] > MIN_FILL).sum(), (vals[1] > MIN_FILL).sum()]
            f_obs = [res[c][-2], num_all[0] - res[c][-2], res[c][-1], num_all[1] - res[c][-1]]
            f_exp = []
            for j in range(4):
                f_exp.append((f_obs[j] + f_obs[j ^ 1]) * (f_obs[j] + f_obs[j ^ 2]) / sum(f_obs))
            res[c] += [chisquare(f_obs, f_exp, 1)[1]]

        except:
            print("WTF %s" % c, fold[c].shape)
    res = pandas.DataFrame(res, index=res_cols).T
    res['ks_pval'] = res[['ks_l_pval', 'ks_g_pval']].min(1)
    res.sort_values('ks_pval', inplace=True)
    other = list(res[['ks_l_pval', 'ks_g_pval']].max(1).values)
    res['FDR_qhi'] = multipletests(list(res['chisq_nums_passed'].values), method='fdr_bh')[0]
    res['FDR_ks'] = multipletests(list(res['ks_pval'].values) + other, method='fdr_bh')[0][:len(res)]
    if df_C is not None:
        res['prot name'] = df_C.loc[res.index]['full name'].values
        res['position'] = df_C.loc[res.index]['pos'].values

    res.to_csv(os.path.join(out_path, 'single_on%s_corona.csv' % name))
    pr = ("For lib C2 %s, %d (by chisq %d, overlap %d) oligos differentially expressed" %
          (name, len(res[res.FDR_ks]), len(res[res.FDR_qhi]), len(res[res.FDR_ks & res.FDR_qhi])))
    print(pr)
    return pr


def cntrl_prot(all_df_C, part_df_C, meta, exist, out_path, name):
    part_df_C = part_df_C.loc[part_df_C.index.intersection(exist.columns[exist.sum() >= MIN_IN_COHORT])]
    part_df_C['prot name'] = ['random' if 'random' in x else x for x in part_df_C['full name'].values]
    all_df_C['prot name'] = ['random' if 'random' in x else x for x in all_df_C['full name'].values]
    res = {}
    cohorts = [meta[~meta.is_recoverred].index, meta[meta.is_recoverred].index]
    num_all = [len(cohorts[0]), len(cohorts[1])]
    for p in part_df_C['prot name'].value_counts().index:
        pcols = part_df_C[part_df_C['prot name'] == p].index
        # res[p] = [(exist.loc[cohorts[0]][pcols].sum(1) > MIN_OLIS_IN_PROT).sum(),
        # (exist.loc[cohorts[1]][pcols].sum(1) > MIN_OLIS_IN_PROT).sum()]
        res[p] = [len(all_df_C[all_df_C['prot name'] == p].index), len(pcols),
                  exist.loc[cohorts[0]][pcols].max(1).sum(), exist.loc[cohorts[1]][pcols].max(1).sum()]
        f_obs = [res[p][-2], num_all[0] - res[p][-2], res[p][-1], num_all[1] - res[p][-1]]
        f_exp = []
        for j in range(4):
            f_exp.append((f_obs[j] + f_obs[j ^ 1]) * (f_obs[j] + f_obs[j ^ 2]) / sum(f_obs))
        res[p] += [chisquare(f_obs, f_exp, 1)[1]]
    for p in all_df_C['prot name'].value_counts().index:
        if p in res.keys():
            continue
        res[p] = [len(all_df_C[all_df_C['prot name'] == p].index), 0, None, None, None]
    res = pandas.DataFrame(res, index=['Num_olis', 'Num_olis_in5%%', 'Num_in_healthy', 'Num_in_recovered',
                                       'Chisq_pval']).T
    res.sort_values('Chisq_pval', inplace=True)
    res['FDR_qhi'] = list(multipletests(list(res[~numpy.isnan(res['Chisq_pval'])]['Chisq_pval'].values),
                                        method='fdr_bh')[0]) + [None] * len(res[numpy.isnan(res['Chisq_pval'])])
    if MIN_IN_COHORT > 1:
        res.to_csv(os.path.join(out_path, "prot_control%s_ab%d.csv" % (name, MIN_IN_COHORT)))
    else:
        res.to_csv(os.path.join(out_path, "prot_control%s.csv" % name))


if __name__ == '__main__':
    cache_path = config.DATA_DIR
    df_C = pandas.read_csv(os.path.join(cache_path, "df_info.csv"), index_col=0)

    cntrl_inds = df_C[df_C.is_control].index
    inds = {'all': df_C[~df_C.is_control].index,
            'SARS2_ref': df_C[(~df_C.is_control) & df_C.SARS_Cov_2_ref].index,
            'SARS2_all': df_C[(~df_C.is_control) & df_C.SARS_Cov_2].index,
            'non_SARS2': df_C[(~df_C.is_control) & (df_C.not_SARS_Cov_2)].index,
            'non_SARS2_ref': df_C[(~df_C.is_control) & (df_C.not_SARS_Cov_2) &
                              (df_C.comments != 'not ref genome, added as it is very close to the current Cov2')].index,
            'non_human': df_C[(~df_C.is_control) & (df_C.not_human_host)].index,
            'non_human_ref': df_C[(~df_C.is_control) & (df_C.not_human_host) &
                              (df_C.comments != 'not ref genome, added as it is very close to the current Cov2')].index,
            'non_SARS2_human': df_C[(~df_C.is_control) & (df_C.not_SARS_Cov_2) & (df_C.human_host)].index,
            }

    out_path = os.path.join(config.OUTPUT_DIR, "pred_C2")
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    log = ""
    for name in [""]: #, "_age_matched", "_yob_matched"]:
        meta = pandas.read_csv(os.path.join(cache_path, "meta%s.csv" % name), index_col=0)
        meta['is_recoverred'] = meta.is_exposed
        exist = pandas.read_csv(os.path.join(cache_path, "exist%s.csv" % name), index_col=0)
        fold = pandas.read_csv(os.path.join(cache_path, "fold%s.csv" % name), index_col=0)

        fold.fillna(MIN_FILL, inplace=True)
        s = fold[fold < 0].sum()
        bad_olis = s[s < 0].index
        print("Throwing %d bad oligos" % len(bad_olis))
        fold = fold[fold.columns.difference(bad_olis)]
        exist = exist[exist.columns.difference(bad_olis)]
        # fold[fold == -1] = numpy.nan

        pr = ("For lib C2 %s, %d oligos passed for healthy %d for recovered" %
              (name.replace("_", " "), exist[~meta.is_recoverred].sum(1).mean(),
               exist[meta.is_recoverred].sum(1).mean()))
        print(pr)
        log += pr + "\n"

        if os.path.exists(os.path.join(out_path, 'single_on%s_cntrl_corona.csv' % name)):
            res = pandas.read_csv(os.path.join(out_path, 'single_on%s_cntrl_corona.csv' % name), index_col=0)
            pr = ("For lib C2 cntrl %s, %d (by chisq %d, overlap %d) oligos differentially expressed" %
                  (name.replace("_", " "), len(res[res.FDR_ks]), len(res[res.FDR_qhi]),
                   len(res[res.FDR_ks & res.FDR_qhi])))
            print(pr)
        else:
            cols = exist.columns.intersection(cntrl_inds)
            cols = (fold[cols] > MIN_FILL).sum()
            cols = cols[cols > (MIN_APPEAR * len(fold))].index
            cntrl_prot(df_C.loc[cntrl_inds].copy(), df_C.loc[cols].copy(), meta, exist, out_path, name)
            check_singles(cols, fold[cols], meta, out_path, '%s_cntrl' % name, df_C.loc[cols])

        for i in inds.keys():
            tmp_df = fold[fold.columns.intersection(inds[i])]
            cols = (tmp_df > MIN_FILL).sum()
            cols = cols[cols > (MIN_APPEAR * len(tmp_df))].index

            if i == 'all':
                if os.path.exists(os.path.join(out_path, 'single_on%s_corona.csv' % name)):
                    res = pandas.read_csv(os.path.join(out_path, 'single_on%s_corona.csv' % name), index_col=0)
                    pr = ("For lib C2 %s, %d (by chisq %d, overlap %d) oligos differentially expressed" %
                          (name.replace("_", " "), len(res[res.FDR_ks]), len(res[res.FDR_qhi]),
                           len(res[res.FDR_ks & res.FDR_qhi])))
                    print(pr)
                else:
                    check_singles(cols, fold[cols], meta, out_path, name)
                # log += pr + "\n"

            pr = ("For %s: %d oligos show up, %d in more than %g of cohort" % (i, tmp_df.shape[1], len(cols),
                                                                               MIN_APPEAR))
            print(pr)
            log += pr + "\n"

            perform_pca(out_path, numpy.log(tmp_df.dropna(axis=1)), meta['is_recoverred'], "%s%s_pca_fold" % (i, name))
            perform_pca(out_path, (tmp_df.dropna(axis=1) > MIN_FILL)*1, meta['is_recoverred'],
                        "%s%s_pca_exist" % (i, name))
            pipeline_cross_val_x(predictor_class, [tmp_df[cols], meta['is_recoverred']], True,
                                 os.path.join(out_path, "predictions_on%s_%s_corona.csv" % (name, i)),
                                 "%s oligos " % i, out_path, i, name)
                # log += pr + "\n\n"

    open(os.path.join(out_path, 'log2.txt'), "w").write(log)
    print("Done")

    df = pandas.read_csv(os.path.join(out_path, 'single_on_corona.csv'), index_col=0)
    if os.path.exists(os.path.join(out_path, 'single_on_age_matched_corona.csv')):
        df = df.merge(pandas.read_csv(os.path.join(out_path, 'single_on_age_matched_corona.csv'), index_col=0), 'outer',
                      suffixes=['', '_age_matched'], left_index=True, right_index=True)
    if os.path.exists(os.path.join(out_path, 'single_on_yob_matched_corona.csv')):
        df = df.merge(pandas.read_csv(os.path.join(out_path, 'single_on_yob_matched_corona.csv'), index_col=0), 'outer',
                      suffixes=['', '_yob_matched'], left_index=True, right_index=True)
    df.fillna(False, inplace=True)

    df['virus'] = df_C.loc[df.index].virus_name_corrected
    df['prot'] = df_C.loc[df.index].prot_name_corrected
    df['aa_seq'] = [x.split()[0] for x in df_C.loc[df.index].aa_seq.values]
    df['pos'] = df_C.loc[df.index].pos

    df.to_csv(os.path.join(out_path, 'single_on_corona_combined.csv'))

    if os.path.exists(os.path.join(out_path, 'single_on_age_matched_corona.csv')) and \
        os.path.exists(os.path.join(out_path, 'single_on_yob_matched_corona.csv')):
        print("Separate: %d %d %d passed" % (len(df[df.FDR_ks]), len(df[df.FDR_ks_age_matched]),
                                             len(df[df.FDR_ks_yob_matched])))
        print("Pairs: %d %d %d passed" % (len(df[df.FDR_ks_age_matched & df.FDR_ks]),
                                          len(df[df.FDR_ks_yob_matched & df.FDR_ks]),
                                          len(df[df.FDR_ks_age_matched & df.FDR_ks_yob_matched])))
        print("All 3: %d passed" % len(df[df.FDR_ks_age_matched & df.FDR_ks_yob_matched & df.FDR_ks]))
