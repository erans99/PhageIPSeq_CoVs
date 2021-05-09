import os
import pandas
import matplotlib.pyplot as plt
import config


if __name__ == '__main__':
    cache_path = config.DATA_DIR
    df_C = pandas.read_csv(os.path.join(cache_path, "df_info.csv"), index_col=0)

    out_path = os.path.join(config.OUTPUT_DIR, "plots")
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    dfs = pandas.read_csv(os.path.join(cache_path, "SpecialAll.csv"), index_col=0)
    cols = dfs.columns
    vs = ['A7', 'A9', 'C1', 'C3']
    l = [[], []]
    for i1, v1 in enumerate(vs):
        l[0].append([v1, [2*i1, 2*i1 + 1]])
        for i2, v2 in enumerate(vs[:i1]):
            l[1].append(["%s_vs_%s" % (v2, v1), [2*i2, 2*i1]])
    for name, cmp in l[0] + l[1]:
        plt.scatter(dfs[cols[cmp[0]]].values, dfs[cols[cmp[1]]].values) #, facecolors='none', edgecolors='b')
        plt.title(name)
        plt.xlabel("-log10 of p-value of reaction on %s" % cols[cmp[0]])
        plt.ylabel("-log10 of p-value of reaction on %s" % cols[cmp[1]])
        plt.vlines(6.7, 0, 200, color='r')
        plt.hlines(6.7, 0, 200, color='r')
        plt.savefig(os.path.join(out_path, name + ".png"))
        plt.close('all')
