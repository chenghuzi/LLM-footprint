from utils import load_ctx_from_config
from pdf_util import chunk_string
from collections import defaultdict
import json
from pathlib import Path

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from binoculars.detector import BINOCULARS_ACCURACY_THRESHOLD, BINOCULARS_FPR_THRESHOLD
from scipy import stats
from statannotations.Annotator import Annotator

from textwrap import wrap
sns.set_theme(style="ticks")
categories = [
    "phenomenon description",
    "hypothesis formulation",
    "methodology explanation",
    "data presentation",
    "logical deduction",
    "result interpretation",
    "literature review",
    "comparative analysis",
    "conclusion summarization",
    "future work suggestions",

]
categories2rm = [
    "bibliography",
    "publishing meta data",
]


def p2mark(p):
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    else:
        return ''


def plot_content_types(df: pd.DataFrame, suffix='', text_max_length=30, dpi=400):
    df['afterChatGPT'] = (
        df['date'] >= gpt_release_date).astype(int)
    bino_mean = df['bino_score'].mean()
    print(bino_mean)
    df['bino_score_high'] = df['bino_score'] > bino_mean

    ct_low = df['content_type'].loc[
        df['bino_score_high'] == 0
    ].value_counts(normalize=True).sort_values()

    ct_ord = ct_low.index

    ct_high = df['content_type'].loc[
        df['bino_score_high'] == 1
    ].value_counts(normalize=True).reindex(ct_ord)

    ct_before = df.loc[
        df['afterChatGPT'] == 0
    ].groupby('content_type').agg({'bino_score': 'mean'}).sort_values('bino_score')['bino_score'].reindex(ct_ord)


    ct_after = df.loc[
        df['afterChatGPT'] == 1
    ].groupby('content_type').agg({'bino_score': 'mean'})['bino_score'].reindex(ct_ord)

    ctdiff = (ct_before - ct_after).sort_values(ascending=False)
    ct_ord = ctdiff.index
    print(ct_ord)

    ct_low = ct_low.reindex(ct_ord)
    ct_high = ct_high.reindex(ct_ord)
    ct_before = ct_before.reindex(ct_ord)
    ct_after = ct_after.reindex(ct_ord)

    ctpvals = []
    for ct in ct_ord:
        stat, p = stats.mannwhitneyu(
            df.loc[(df['afterChatGPT'] == 0) & (
                df['content_type'] == ct)]['bino_score'].values,
            df.loc[(df['afterChatGPT'] == 1) & (
                df['content_type'] == ct)]['bino_score'].values,
        )
        ctpvals.append(p2mark(p))
        print(ct, f'p={p}')

    wrapped_label_BA = [
        '\n'.join(wrap(l.capitalize(), text_max_length)) for l in ct_ord]

    wrapped_label_LH = [
        '\n'.join(wrap(l.capitalize(), text_max_length)) for l in ct_ord]

    fig, axes = plt.subplots(1, 2, figsize=[6, 3.5], sharey=True)

    ax = axes[0]
    s = 50
    bms = f"{bino_mean:.2f}"
    ax.scatter(ct_high.values, wrapped_label_LH,
               label='$\it{Binoculars}>$'+bms,
               marker='D', s=s)
    ax.scatter(ct_low.values, wrapped_label_LH,
               label='$\it{Binoculars}\leq$'+bms,
               marker='D', s=s)
    ax.hlines(y=wrapped_label_LH,
              xmin=ct_high.values,
              xmax=ct_low.values, color='#666666')
    ax.legend(frameon=False, loc=(0, 1.05))
    vals = ax.get_xticks()
    ax.set_xticklabels(['{:,.0%}'.format(x) for x in vals])
    ax.set_xlabel('Percentage in Collection')

    ax = axes[1]
    ax.scatter(ct_before.values, wrapped_label_BA,
               label='Before ChatGPT release', s=s)
    ax.scatter(ct_after.values, wrapped_label_BA,
               label='After ChatGPT release', s=s)
    ax.hlines(y=wrapped_label_BA,
              xmin=ct_before.values,
              xmax=ct_after.values, color='#666666')

    bn_diff = (ct_before.values + ct_after.values)/2
    for bnc,  pv, lbl in zip(bn_diff, ctpvals, wrapped_label_BA):
        if pv == '':
            continue
        ax.annotate(pv, xy=(bnc, lbl),
                    ha='center',
                    )

    ax.legend(frameon=False, loc=(0, 1.05))
    ax.set_xlabel('Average Bino Score')

    fig.tight_layout()

    return fig



arxiv_cat_info = {
    "astro-ph": "physics",
    "cond-mat": "physics",
    "gr-qc": "physics",
    "hep-ex": "physics",
    "hep-lat": "physics",
    "hep-ph": "physics",
    "hep-th": "physics",
    "math-ph": "physics",
    "nlin": "physics",
    "nucl-ex": "physics",
    "nucl-th": "physics",
    "physics": "physics",
    "quant-ph": "physics",
    "math": "mathematics",
    "cs": "computer_science",
    "q-bio": "quantitative_biology",
    "q-fin": "quantitative_finance",
    "stat": "statistics",
    "eess": "electrical_engineering_and_systems_science",
    "econ": "economics"
}

gpt_release_date = '2022-11-30'
geo_cache4 = json.load(open('data/geo_cache4.json'))

cr_langs = defaultdict(list)
for langcr in pd.read_csv('data/country_lang.csv', delimiter='	')['Two Letter'].values:
    lang, cr = langcr.split('-')
    cr_langs[cr].append(lang)



def cr_has_en(cr):
    if cr not in cr_langs:
        return 0
    else:
        if 'en' not in cr_langs[cr]:
            return 0
        else:
            if len(cr_langs[cr]) == 1:
                return 1
            else:
                return 1


has_en_mapping = {
    0: 'No English',
    1: 'English+',
    2: 'English Only',
}


def get_countryregion_code(place):
    if place in geo_cache4:
        return geo_cache4[place]
    else:
        return None


def load_chatgpt_trends(fn='data/chatgpt_goog_trend.csv'):
    trend_df = pd.read_csv(fn)
    trend_df.columns = ['trends']
    trend_df.drop('Week', axis=0, inplace=True)
    trend_df['trends'] = pd.to_numeric(trend_df['trends'], errors='coerce')
    trend_df.dropna(inplace=True)
    trend_df.index = pd.to_datetime(trend_df.index)
    trend_df.index.name = 'date'
    return trend_df


@click.group()
@click.option('--config', default='config.yml', type=click.Path())
@click.pass_context
def cli(ctx, config):
    load_ctx_from_config(ctx, config)


def group_analysis(figdir, score_df_bypaper, writing_days, metric, trend_df,
                   ylabel='by_gpt_mean',
                   suffix=''):
    score_df_bypaper_day = score_df_bypaper.groupby(score_df_bypaper['date']).agg(
        {'bino_score': 'mean',
            'by_gpt': 'mean',
         'afterChatGPT': 'min'
         }
    ).reset_index()
    score_df_bypaper_day.sort_values('date', inplace=True)
    score_df_bypaper_day.set_index('date', inplace=True)
    score_df_bypaper_day['by_gpt_mean'] = score_df_bypaper_day['by_gpt'].shift(
        -(writing_days-1)).rolling(writing_days).mean()

    score_df_bypaper_day['bino_score_mean_inv'] = 1. / score_df_bypaper_day['bino_score'].shift(
        -(writing_days-1)).rolling(writing_days).mean()

    score_df_bypaper_day = pd.concat(
        [score_df_bypaper_day, trend_df],
        axis=1, join='inner',
    ).dropna()

    plt.close()
    plt.clf()
    fig, ax = plt.subplots(figsize=[8, 2])
    fig.suptitle(f'(n={len(score_df_bypaper_day)})')
    ax = score_df_bypaper_day.reset_index().plot(
        x='date', y=ylabel, kind='line', ax=ax)
    score_df_bypaper_day['trends'].plot(secondary_y=True, ax=ax, kind='line')
    ax.right_ax.set_ylabel('ChatGPT trends')
    fig.tight_layout()
    fig.savefig(figdir / f'{metric}.gpt_ratio_chatgpt_trends_{suffix}.png')

    try:
        plt.close()
        plt.clf()
        fig, ax = plt.subplots(figsize=[3, 3])
        fig.suptitle(f'(n={len(score_df_bypaper_day)})')
        sns.boxplot(data=score_df_bypaper_day,
                    x='afterChatGPT', y=ylabel,
                    ax=ax)
        pairs = [(False, True)]
        annotator = Annotator(ax, pairs,
                              data=score_df_bypaper_day,
                              x='afterChatGPT',
                              y=ylabel,
                              )
        annotator.configure(
            test='Mann-Whitney',
            text_format='star',
        )
        annotator.apply_and_annotate()
        plt.tight_layout()
        sns.despine(ax=ax)
        plt.savefig(figdir / f'{metric}.by_gpt_mean_before_after_{suffix}.png')
    except Exception as e:
        print(f'visualize boxplot failed due to {e}')

    return score_df_bypaper_day


@cli.command()
@click.pass_context
@click.option('--metric', type=str)
@click.option('--metric_th_min', type=float)
@click.option('--metric_th_mean', type=float)
@click.option('--dir', type=click.Path(), multiple=True)
@click.option('--figdir', type=click.Path())
@click.option('--chunksize', type=int, default=512)
def analyzemedbiorxiv(ctx, metric, metric_th_min, metric_th_mean, dir, figdir, chunksize):
    figdir = Path(figdir)
    figdir.mkdir(exist_ok=True, parents=True)
    topk_cat = 10
    writing_days = 60

    metric_thredholds = {
        'min': metric_th_min,
        'mean': metric_th_mean,
    }
    print(f'Analyzing {dir}')

    trend_df = load_chatgpt_trends()
    begin_date = '2022-01-01'


    scored_data = []
    sample_length = []
    for single_dir in dir:
        platform = single_dir.split('/')[1]
        print(platform)
        for jsf in Path(single_dir).glob('fulltext_info*.json'):
            try:
                print(f'reading {jsf}')
                tmpdata = json.load(open(jsf))
                for sample in tmpdata:
                    sample['platform'] = platform
                    sample_length.append(len(sample['full_text']))
                    if f'bino_scores-{chunksize}' in sample.keys() and len(sample['full_text']) > 5000:
                        scored_data.append(sample)

            except Exception as e:
                print(f'skipped {jsf} due to {e}')

    print(f'{len(scored_data)} paper(s) to be analyzed')
    scored_data_plain = []

    for s_paper in scored_data:
        full_text = s_paper['full_text'].replace('/n', '')
        text_chunks = chunk_string(full_text, chunksize)
        for score_idx, (chunk, score) in enumerate(
            zip(
                text_chunks,
                s_paper[f'bino_scores-{chunksize}']
            )
        ):
            scored_data_plain.append((
                s_paper['_id'], s_paper['platform'],
                s_paper['title'], s_paper['date'],
                s_paper['category'],
                s_paper['authors'],
                s_paper['author_corresponding'],
                s_paper['author_corresponding_institution'],
                s_paper['type'],
                s_paper['version'],
                score_idx /
                len(s_paper[f'bino_scores-{chunksize}']), score, chunk
            ))

    scored_df = pd.DataFrame(scored_data_plain, columns=[
        '_id', 'platform', 'title',  'date',
        'category',
        'authors', 'author_corresponding', 'author_corresponding_institution', 'type', 'version',
        'score_loc',
        'bino_score', 'chunk']).dropna()

    scored_df['date_last'] = pd.to_datetime(scored_df['date'])
    scored_df['date'] = pd.to_datetime(scored_df['_id'].apply(
        lambda x: '-'.join(x.split('/')[1].split('.')[:-1])))
    scored_df = scored_df.loc[scored_df['date'] >= pd.to_datetime(begin_date)]

    max_idx = scored_df['bino_score'].idxmax()
    min_idx = scored_df['bino_score'].idxmin()
    scored_df = scored_df.drop([max_idx, min_idx])

    scored_df['country_region'] = scored_df['author_corresponding_institution'].apply(
        get_countryregion_code)

    scored_df.sort_values('date', inplace=True)
    print(f'medbiorxiv.shape={scored_df.shape}')

    scored_df = scored_df[['_id', 'platform', 'title', 'date', 'category',
                           'country_region', 'authors',
                           'score_loc', 'bino_score', 'chunk'
                           ]].copy()
    for col in ['_id', 'platform', 'title', 'category', 'country_region', 'authors', 'chunk']:
        scored_df.loc[:, col] = scored_df[col].apply(str).values
    for col in ['score_loc', 'bino_score']:
        scored_df.loc[:, col] = scored_df[col].apply(float)

    scored_df[['_id', 'platform', 'score_loc', 'bino_score', 'chunk']].to_pickle('data/medbiorxiv_scored_df.pkl')
    paperinfo_df = scored_df[['_id', 'platform', 'title', 'date', 'category',
                              'country_region', 'authors'
                              ]].drop_duplicates('_id')
    print(paperinfo_df.shape)
    paperinfo_df.to_hdf('data/medbiorxiv_paperinfo_df.hdf5',
                        'score', index=False, format='table')



@cli.command()
@click.pass_context
@click.option('--metric', type=str)
@click.option('--metric_th_min', type=float)
@click.option('--metric_th_mean', type=float)
@click.option('--dir', type=click.Path(), multiple=True)
@click.option('--figdir', type=click.Path())
@click.option('--chunksize', type=int, default=512)
def analyzearxiv(ctx, metric, metric_th_min, metric_th_mean, dir, figdir, chunksize):
    authors = json.load(open('data/arxiv/authors_dict.json'))
    author_country_info = pd.read_csv('data/arxiv/names.csv')
    author_country_info.columns = ['name', 'CR']
    author_country_info = author_country_info.loc[
        author_country_info['CR'] != 'unknown'
    ]
    figdir = Path(figdir)
    figdir.mkdir(exist_ok=True, parents=True)
    topk_cat = 10
    writing_days = 60

    metric_thredholds = {
        'min': metric_th_min,
        'mean': metric_th_mean,
    }
    print(f'Analyzing {dir}')

    trend_df = load_chatgpt_trends()
    begin_date = '2022-01-01'

    scored_data = []
    sample_length = []
    for single_dir in dir:
        platform = single_dir.split('/')[1]
        for jsf in Path(single_dir).glob('fulltext_info*.json'):
            try:
                print(f'reading {jsf}')
                tmpdata = json.load(open(jsf))
                for sample in tmpdata:
                    sample['platform'] = platform
                    sample_length.append(len(sample['full_text']))
                    if f'bino_scores-{chunksize}' in sample.keys() and len(sample['full_text']) > 5000:
                        scored_data.append(sample)

            except Exception as e:
                print(f'skipped {jsf} due to {e}')

    print(f'{len(scored_data)} paper(s) to be analyzed')
    scored_data_plain = []
    has_doi = 0

    category_all = set()
    for s_paper in scored_data:
        authorskey = s_paper['authors'].replace('\n', '')

        if authorskey not in authors.keys():
            continue
        last_author = authors[authorskey][-1]

        plain_authors = ';'.join(
            [''.join(c for c in s if 0 <= ord(c) <= 127) for s in authors[authorskey]])
        dfa = author_country_info.loc[author_country_info['name']
                                      == last_author]
        if len(dfa) == 0:
            continue
        cr = dfa.iloc[0]['CR']

        try:
            category_list = sorted(
                set([c.split('.')[0] for c in s_paper['categories'].split(' ')]))
            cats = set([arxiv_cat_info[cat] for cat in category_list])
            category_all = category_all | cats
            categories = ','.join(cats)
        except Exception:
            continue

        has_doi += 1
        full_text = s_paper['full_text'].replace('/n', '')
        text_chunks = chunk_string(full_text, chunksize)
        for score_idx, (chunk, score) in enumerate(
            zip(
                text_chunks,
                s_paper[f'bino_scores-{chunksize}']
            )
        ):
            scored_data_plain.append((
                s_paper['_id'], s_paper['platform'], s_paper['title'], s_paper['date'],
                categories,
                cr,
                plain_authors,
                score_idx /
                len(s_paper[f'bino_scores-{chunksize}']),
                score,
                chunk
            ))


    print(f'{has_doi} papers have doi')
    scored_df = pd.DataFrame(scored_data_plain, columns=[
        '_id', 'platform', 'title',  'date',
        'category',
        'country_region',
        'authors',
        'score_loc',
        'bino_score', 'chunk']).dropna()
    scored_df['date'] = pd.to_datetime(scored_df['date'])
    scored_df = scored_df.loc[scored_df['date'] >= pd.to_datetime(begin_date)]

    max_idx = scored_df['bino_score'].idxmax()
    min_idx = scored_df['bino_score'].idxmin()
    scored_df = scored_df.drop([max_idx, min_idx])
    scored_df.sort_values('date', inplace=True)

    scored_df = scored_df[['_id', 'platform', 'title', 'date', 'category',
                           'country_region', 'authors',
                           'score_loc', 'bino_score', 'chunk'
                           ]].copy()
    for col in ['_id', 'platform', 'title', 'category', 'country_region', 'authors', 'chunk']:
        scored_df.loc[:, col] = scored_df[col].apply(str).values
    for col in ['score_loc', 'bino_score']:
        scored_df.loc[:, col] = scored_df[col].apply(float)
    scored_df[['_id', 'platform', 'score_loc', 'bino_score', 'chunk']].to_hdf('data/arxiv_scored_df.hdf5',
                                                                              'score', index=False, format='table')

    paperinfo_df = scored_df[[
        '_id',
        'platform',
        'title',
        'date',
        'category',
        'country_region', 'authors'
    ]].drop_duplicates('_id')
    print(paperinfo_df.shape)
    paperinfo_df.to_hdf('data/arxiv_paperinfo_df.hdf5',
                        'score', index=False, format='table')
    print('finished arxiv')



@cli.command()
@click.pass_context
@click.option('--metric', type=str)
@click.option('--metric_th_min', type=float)
@click.option('--metric_th_mean', type=float)
@click.option('--dir', type=click.Path())
@click.option('--figdir', type=click.Path())
@click.option('--chunksize', type=int, default=512)
def analyzeall(ctx, metric, metric_th_min, metric_th_mean, dir, figdir, chunksize):
    dir = Path(dir)
    figdir = Path(figdir)
    figdir.mkdir(exist_ok=True, parents=True)
    topk_cat = 20
    writing_days = 60

    metric_thredholds = {
        'min': metric_th_min,
        'mean': metric_th_mean,
    }
    print(f'Analyzing {dir}')

    trend_df = load_chatgpt_trends()
    begin_date = '2022-01-01'

    dfs = []
    for p in dir.glob('*_paperinfo_df.hdf5'):
        tmpdf = pd.read_hdf(p)
        tmpdf['_id'] = tmpdf['_id'].apply(lambda x: p.stem.split('_')[0]+'_'+x)
        dfs.append(tmpdf.copy())
        print(p, tmpdf.shape)
    paperinfo_df = pd.concat(dfs, axis=0).set_index('_id')
    print(paperinfo_df.shape)
    print(paperinfo_df.head())

    dfs = []
    for p in dir.glob('*_scored_df.hdf5'):
        tmpdf = pd.read_hdf(p)
        tmpdf['_id'] = tmpdf['_id'].apply(lambda x: p.stem.split('_')[0]+'_'+x)
        tmpdf['platform'] = p.stem.split('_')[0]
        dfs.append(tmpdf.copy())
        print(p, tmpdf.shape)
    scored_df = pd.concat(dfs, axis=0)
    print(scored_df.shape)
    print(scored_df.head())

    score_df_bypaper = scored_df.groupby(scored_df['_id']).agg({
        'bino_score': ['min', 'mean', 'var'],
    }).reset_index().dropna().set_index('_id')
    score_df_bypaper.columns = [
        '_'.join(col).strip() for col in score_df_bypaper.columns.values]
    print(score_df_bypaper.shape)
    print(paperinfo_df.shape)
    score_df_bypaper = pd.concat(
        [score_df_bypaper, paperinfo_df], axis=1, join='inner')
    print(score_df_bypaper.shape)
    print(score_df_bypaper.head())
    score_df_bypaper.sort_values('date', inplace=True)

    score_df_bypaper.to_hdf('data/allpaperinfo.hdf5',
                            'paperinfo', index=False, format='table')

if __name__ == '__main__':
    cli()
