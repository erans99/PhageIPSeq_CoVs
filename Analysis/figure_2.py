import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from statannot import add_stat_annotation

from config import DATA_DIR, MIN_FILL


class ScatterPlotByVirusAndExposureStatus:
    def __init__(self, local_cache_path=DATA_DIR, **kwargs):
        self.virus_name_dict = {'Human coronavirus HKU1': 'hCoV-HKU1',
                                'Severe acute respiratory syndrome coronavirus 2': 'SARS-CoV-2',
                                'Human coronavirus 229E': 'hCoV-299E',
                                'Middle East respiratory syndrome-related coronavirus': 'MERS-CoV',
                                'Human coronavirus NL63': 'hCoV-NL63',
                                'Severe acute respiratory syndrome-related coronavirus': 'SARS-CoV',
                                'Betacoronavirus England 1': 'MERS-CoV',
                                'Human coronavirus OC43': 'hCoV-OC43'}
        local_exists_path = os.path.join(local_cache_path, 'exist.csv')
        local_meta_path = os.path.join(local_cache_path, 'meta.csv')
        self.exists = pd.read_csv(local_exists_path, index_col=0)
        self.meta = pd.read_csv(local_meta_path, index_col=0)
        self.df_info = pd.read_csv(os.path.join(local_cache_path, 'df_info.csv'), index_col=0,
                                   low_memory=False)
        self.df_info.index.name = 'order'

        # Remove faulty oligos
        fold = pd.read_csv(os.path.join(local_cache_path, 'fold.csv'), index_col=0)
        fold.fillna(MIN_FILL, inplace=True)
        s = fold[fold < 0].sum()
        bad_oligos = s[s < 0].index
        self.exists.drop(columns=bad_oligos, inplace=True, errors='ignore')
        self.exists.drop(index=bad_oligos, inplace=True, errors='ignore')

        self.exists['is_exposed'] = self.meta['is_exposed'].loc[self.exists.index.values]
        self.exists = self.exists.reset_index(drop=True).set_index('is_exposed')
        self.exists_summary = self.exists.reset_index().groupby('is_exposed').sum().transpose()
        self.exists_summary['unexposed_perc'] = self.exists_summary[False] * 100.0 / (
                self.exists.shape[0] - self.exists.index.values.sum())
        self.exists_summary['exposed_perc'] = self.exists_summary[True] * 100.0 / self.exists.index.values.sum()
        self.exists_summary.drop(columns=[True, False], inplace=True)

        # Add virus information

        self.exists_viruses = self.df_info[['virus_name_corrected', 'SARS_Cov_2_ref']].dropna()
        self.exists_viruses['virus_exploded'] = self.exists_viruses['virus_name_corrected'].astype(str).str.split('&')
        self.exists_viruses = self.exists_viruses.explode('virus_exploded')
        self.exists_viruses.drop(columns='virus_name_corrected', inplace=True)
        self.exists_viruses['short_virus_name'] = self.exists_viruses.virus_exploded.apply(
            lambda x: self.virus_name_dict.get(x))
        self.exists_viruses = self.exists_viruses[self.exists_viruses['short_virus_name'].notnull()]
        self.exists_viruses['virus'] = (self.exists_viruses.apply(
            lambda x: x['short_virus_name'] if (x['short_virus_name'] != 'SARS-CoV-2') or (
                x['SARS_Cov_2_ref']) else np.nan, axis=1))

        self.exists_viruses.sort_values(by='short_virus_name', inplace=True)
        self.exists_viruses = pd.merge(self.exists_viruses['virus'], self.exists_summary, how='outer', left_index=True,
                                       right_index=True).dropna()

        self.peptides_per_person_count = self.exists.transpose()
        self.peptides_per_person_count = (self.peptides_per_person_count
                                          .merge(self.exists_viruses['virus'],
                                                 how='outer',
                                                 left_index=True,
                                                 right_index=True))
        self.peptides_per_person_count = self.peptides_per_person_count[
            self.peptides_per_person_count['virus'].notnull()]
        assert not self.peptides_per_person_count.isnull().any().any()
        self.peptides_per_person_count = (self.peptides_per_person_count
                                          .groupby('virus')
                                          .sum()
                                          .transpose()
                                          .stack()
                                          .reset_index()
                                          .rename(columns={0: '# peptides per individual', 'level_0': 'SARS-CoV-2'}))
        self.peptides_per_person_count['SARS-CoV-2'] = self.peptides_per_person_count['SARS-CoV-2'].apply(
            lambda x: 'Exposed' if x else 'Unexposed')

    def generate_figure(self, num_cols=4, num_rows=2, label_font_size=15, ticklabel_font_size=13,
                        title_font_size=20, export_dir=None,
                        **kwargs):
        virus_order = kwargs.get('virus_order',
                                 ['SARS-CoV-2', 'SARS-CoV', 'MERS-CoV', 'hCoV-OC43', 'hCoV-HKU1', 'hCoV-299E',
                                  'hCoV-NL63'])
        insignificant_viruses = kwargs.get('insignificant_viruses', ['hCoV-299E', 'hCoV-NL63'])
        fig = plt.figure(constrained_layout=True, figsize=(20, 10))
        spec = gridspec.GridSpec(ncols=num_cols, nrows=num_rows, figure=fig)
        palette = sns.color_palette(n_colors=len(set(self.virus_name_dict.values())))
        for i, virus_name in enumerate(virus_order):
            ax = fig.add_subplot(spec[(i + 1) // num_cols, (i + 1) % num_cols])
            df = self.exists_viruses[self.exists_viruses['virus'].eq(virus_name)]
            ax.scatter(x=df['unexposed_perc'], y=df['exposed_perc'], color=palette[i], )
            ax.set_xlabel('% unexposed', fontsize=label_font_size)
            ax.set_ylabel('% exposed', fontsize=label_font_size)
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            plt.xticks(fontsize=ticklabel_font_size)
            plt.yticks(fontsize=ticklabel_font_size)
            ax.set_title(virus_name, {'fontsize': title_font_size})

        ax_0 = fig.add_subplot(spec[0, 0])
        sns.boxplot(data=self.peptides_per_person_count, x='virus', y='# peptides per individual', hue='SARS-CoV-2',
                    ax=ax_0,
                    order=virus_order)
        box_pairs = [((virus, "Exposed"), (virus, "Unexposed")) for virus in virus_order if
                     virus not in insignificant_viruses]
        add_stat_annotation(ax_0, data=self.peptides_per_person_count, x='virus', y='# peptides per individual',
                            hue='SARS-CoV-2',
                            order=virus_order,
                            box_pairs=box_pairs,
                            test='Mann-Whitney', text_format='star', loc='inside', verbose=2)
        ax_0.set_xticklabels(labels=ax_0.get_xticklabels(), rotation=45, fontsize=ticklabel_font_size)
        ax_0.set_xlabel(None)
        ax_0.set_ylabel(ax_0.get_ylabel(), fontsize=label_font_size)
        if export_dir is None:
            plt.show()
        else:
            os.makedirs(export_dir, exist_ok=True)
            fig.savefig(os.path.join(export_dir, 'virus_level_scatter_figure.png'))


if __name__ == "__main__":
    drawer = ScatterPlotByVirusAndExposureStatus()
    drawer.generate_figure()
