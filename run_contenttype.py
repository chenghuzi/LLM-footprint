import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
import numpy as np
import pandas as pd
import seaborn as sns
from openai import OpenAI
from tqdm import tqdm
import threading
from run_analysis import plot_content_types
from utils import load_ctx_from_config
import json


@click.group()
@click.option('--config', default='config.yml', type=click.Path())
@click.pass_context
def cli(ctx, config):
    load_ctx_from_config(ctx, config)


final_figdir = Path('/path/to/Downloads')
dir = Path('data')
cls_dir = Path(dir / 'content_type')
sdff = cls_dir/'s_df_wct_cali.pkl'

client = OpenAI(
    base_url='xxx',
    api_key="xxxx"
)


def revise_by_gpt(client, content, model="gpt-3.5-turbo"):
    sys_prompt = """
    You are a helpful assistant.
    The user will send you a message containing an unoptimized piece of academic writing that is excerpted from a paper.
    You will revise the piece and improve it.
    Notice that the piece may be incomplete paragraphs and may have unfnished beginning and ending sentences.
    You need to respect these parts, do not modify them, and only edit parts that will not influence its original content.
    Your response should be pluggable to the original paper seamlessly.
    
    Your response will be nothing but the modified content. DONOT reply anything else.
    """
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": content}
        ]
    )

    return completion.choices[0].message.content


df_pinfo = pd.read_hdf(dir/'allpaperinfo.hdf5')


def read_sdfs_raw():
    dfs = []
    for p in cls_dir.glob('res*.pkl'):
        print(f'reading {p}')
        dfs.append(pd.read_pickle(p))
    s_df_wct = pd.concat(dfs, axis=0)
    return s_df_wct


def read_sdfs(sample_size=1000):
    if not sdff.is_file():
        s_df_wct = read_sdfs_raw()
        print(s_df_wct.shape)

        s_df_wct_cali = s_df_wct.loc[s_df_wct['afterChatGPT'] == 0].copy()
        s_df_wct_cali = s_df_wct_cali.loc[s_df_wct_cali['_id'].isin(
            np.random.choice(
                s_df_wct_cali['_id'].unique(), sample_size, replace=False)
        )].copy()
        s_df_wct_cali['chunk_revised'] = ''
        print(
            f'before the chatgpt release, we have {s_df_wct_cali["_id"].unique().shape[0]} papers')

        ids2modify = np.random.choice(
            s_df_wct_cali["_id"].unique(),
            s_df_wct_cali["_id"].unique().shape[0]//2, replace=False)
        print(len(ids2modify))

        rows2changeall = []
        for idx in tqdm(ids2modify):
            s_df_wct_cali_p = s_df_wct_cali.loc[s_df_wct_cali['_id'] == idx]
            n2change = min(10, len(s_df_wct_cali_p))

            rowids2change = np.random.choice(
                s_df_wct_cali_p.index, n2change, replace=False)
            rows2changeall.append(rowids2change)
            s_df_wct_cali.loc[
                rowids2change, 'chunk_revised'
            ] = s_df_wct_cali_p.loc[rowids2change]['chunk_clean']

        s_df_wct_cali.to_pickle(sdff)
    else:
        s_df_wct_cali = pd.read_pickle(sdff)

    return s_df_wct_cali


@cli.command()
def synthesize():
    read_sdfs()


class ThreadSafeDict:
    def __init__(self, jsf):
        self.lock = threading.Lock()
        self.jsf = jsf
        self.dict = {}

    def set_item(self, key, value):
        sk = str(key)
        with self.lock:
            self.dict[sk] = value
            try:
                with open(self.jsf, 'w') as f:
                    f.write(json.dumps(self.dict, indent=2, ensure_ascii=False))
            except Exception as e:
                print(e)

    def get_item(self, key):
        sk = str(key)
        with self.lock:
            return self.dict.get(sk)


def revise_worker(idx, chunk, shared_dict: ThreadSafeDict, length: int):
    if shared_dict.get_item(idx):
        print(f'found with {idx} ({len(shared_dict.dict)}/{length})')
    else:
        try:
            chunk_revised = revise_by_gpt(client, chunk)
            shared_dict.set_item(idx, chunk_revised)
            print(f'Finish {idx} ({len(shared_dict.dict)}/{length})')
        except Exception as e:
            print(f'({idx}) During edits we got error {e}')


@cli.command()
def plot():
    df = read_sdfs_raw()
    fig = plot_content_types(df)
    fig.savefig(final_figdir / 'content_type.png',
                bbox_inches='tight',
                dpi=400,
                transparent=True
                )


@cli.command()
@click.option('--jobs', default=40)
def edit(jobs):
    revised_d_f = cls_dir / 'chunk_revised_all.json'

    s_df_wct_cali = read_sdfs()
    s_df_wct_cali2edit = s_df_wct_cali.loc[s_df_wct_cali['chunk_revised'].str.len(
    ) > 0]
    print(s_df_wct_cali2edit.shape)
    s_df_wct_cali2edit = s_df_wct_cali2edit
    length = len(s_df_wct_cali2edit)

    shared_dict = ThreadSafeDict(revised_d_f)
    if shared_dict.jsf.is_file():
        shared_dict.dict = json.load(open(shared_dict.jsf))

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        [executor.submit(revise_worker, idx, val, shared_dict, length) for idx, val in zip(s_df_wct_cali2edit['chunk_revised'].index, s_df_wct_cali2edit['chunk_revised'].values)
         ]

    s_df_wct_calip = s_df_wct_cali.copy()
    s_df_wct_calip['chunk_revised'] = s_df_wct_calip['chunk_clean']
    s_df_wct_calip['modified_bygpt'] = 0
    df_revised_bygpt = pd.Series(
        shared_dict.dict).to_frame(name='chunk_revised')
    df_revised_bygpt.index = df_revised_bygpt.index.astype(int)

    revised_scores_f = cls_dir / 'revised_scores.json'
    revised_scores = json.load(open(revised_scores_f))
    df_revised_scores = pd.Series(revised_scores).to_frame(name='bino_score')
    df_revised_scores.index = df_revised_scores.index.astype(int)

    original_scores = s_df_wct_calip.loc[s_df_wct_calip.index.isin(
        df_revised_scores.index)]['bino_score'].copy()
    score_dfiff_revised = original_scores - df_revised_scores['bino_score']
    print(f'diff={score_dfiff_revised.mean()}')

    s_df_wct_calip.update(df_revised_bygpt['chunk_revised'])
    s_df_wct_calip.update(df_revised_scores['bino_score'])

    s_df_wct_calip.loc[s_df_wct_calip.index.isin(
        df_revised_bygpt.index), 'modified_bygpt'] = 1
    dft = s_df_wct_calip.groupby('_id').agg({
        'chunk_revised': list,
        'chunk_clean': list,
        'score_loc': list,
        'bino_score': list,
        'content_type': list,
        '_id': 'first',
        'afterChatGPT': 'first',
        'modified_bygpt': 'max',
    })

    dft['score_loc_order'] = dft['score_loc'].apply(lambda x: np.argsort(x))
    dft['score_loc'] = dft.apply(lambda row: np.array(row['score_loc'])[
                                 row['score_loc_order']], axis=1).apply(lambda x: [float(xc) for xc in x])
    dft['bino_score'] = dft.apply(lambda row: np.array(row['bino_score'])[
                                  row['score_loc_order']], axis=1).apply(lambda x: [float(xc) for xc in x])
    dft['chunk_revised'] = dft.apply(lambda row: np.array(row['chunk_revised'])[
                                     row['score_loc_order']], axis=1).apply(lambda x: [str(xc) for xc in x])
    dft['chunk_clean'] = dft.apply(lambda row: np.array(row['chunk_clean'])[
                                   row['score_loc_order']], axis=1).apply(lambda x: [str(xc) for xc in x])
    dft['content_type'] = dft.apply(lambda row: np.array(row['content_type'])[
                                    row['score_loc_order']], axis=1).apply(lambda x: [str(xc) for xc in x])
    dft.to_pickle(cls_dir/'modified_scored_df.pkl')
    pass


if __name__ == '__main__':
    cli()
