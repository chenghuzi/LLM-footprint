import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import click
import pandas as pd
from semanticscholar import SemanticScholar
import time

from utils import load_ctx_from_config

gpt_release_date = pd.to_datetime('2022-11-30')


@click.group()
@click.option('--config', default='config.yml', type=click.Path())
@click.pass_context
def cli(ctx, config):
    load_ctx_from_config(ctx, config)
    sch = SemanticScholar(api_key='xxx')
    ctx.obj['sch'] = sch


def article_fetch_worker(
        sch: SemanticScholar, idx: str, title: str,
        article_info_list: list, lock: threading.Lock, df_f: Path,
        total: int, t0: float, dfp=None):
    try:

        res = sch.search_paper(
            quote(title), limit=1,
            fields=['journal', 'citations'])
        if len(res.items) > 0:
            article_info = res.items[0].raw_data

            try:
                if 'journal' in article_info and article_info['journal'] is not None:
                    journal = article_info['journal']['name']
                else:
                    journal = 'Unknown'
            except Exception:
                journal = 'Unknown'

            with lock:
                article_info_list.append(
                    (idx, len(article_info['citations']), journal)
                )
                df_article_info_list = pd.DataFrame(
                    article_info_list,
                    columns=['_id', 'citations', 'journal']
                ).set_index('_id')
                if dfp is not None:
                    df_article_info_list = pd.concat(
                        [dfp, df_article_info_list],
                        axis=0
                    )
                dt = time.time() - t0
                ratio = len(df_article_info_list) / total
                t4single = dt / len(df_article_info_list)
                reamining_time = total * (1 - ratio) * t4single
                print(
                    f'finished {ratio*100:.2f}%, time remaining: {reamining_time//60} mins')
                df_article_info_list.to_pickle(df_f)
    except Exception as e:
        print(f'encountered error {e}')


@cli.command()
@click.pass_context
def getinfo(ctx):

    df_articlejn_infos_f = Path('data/df_articlejn_infos.pkl')
    sch: SemanticScholar = ctx.obj['sch']
    paperinfo_df = pd.read_hdf('data/allpaperinfo.hdf5')
    paperinfo_df = paperinfo_df.loc[paperinfo_df['country_region'] != 'None'].copy(
    )
    paperinfo_df['afterChatGPT'] = (
        paperinfo_df['date'] >= gpt_release_date).astype(int)
    paperinfo_df.head(3)
    if df_articlejn_infos_f.is_file():
        dfp = pd.read_pickle(df_articlejn_infos_f)
        print(f'read from {df_articlejn_infos_f}\n'
              f'before len(paperinfo_df)={len(paperinfo_df)}'
              )
        paperinfo_df = paperinfo_df.loc[
            ~paperinfo_df.index.isin(dfp.index.to_list())
        ].copy()
        print(f'after: len(paperinfo_df)={len(paperinfo_df)}'
              )
    else:
        dfp = None
    lock = threading.Lock()
    article_infos = []
    total = len(paperinfo_df)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=20) as executor:
        [executor.submit(article_fetch_worker,
                         sch, idx, row.title, article_infos,
                         lock, df_articlejn_infos_f, total, t0, dfp)
         for idx, row in paperinfo_df.iterrows()]

    pass


if __name__ == '__main__':
    cli()
