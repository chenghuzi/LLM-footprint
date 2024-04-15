
import json
from pathlib import Path

import click
import joblib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import tiktoken
from binoculars.detector import BINOCULARS_ACCURACY_THRESHOLD, BINOCULARS_FPR_THRESHOLD
from pingouin import ancova
from pySankey.sankey import sankey
from scipy import stats
from scipy.stats import pearsonr
from semanticscholar import SemanticScholar
from statannotations.Annotator import Annotator
from statsmodels.formula.api import glm, ols
from tqdm import tqdm

from utils import load_ctx_from_config

sns.set_theme(style="ticks")


def checkAIdeclariation(client: openai.OpenAI, text: str):
    sys_prompt = """You are an AI assistant whose role is to analyze academic papers submitted by users in plain text format. Your specific task is to determine whether the paper includes any declarations or statements indicating that it has been edited, revised, or written with the assistance of Artificial Intelligence (AI), Language Models (LLM), or specifically ChatGPT. It is crucial to focus solely on the content generation aspect of writing, excluding any involvement of AI in data preparation, data analysis, or other non-writing related activities. After your analysis, you will respond with a single letter: "Y" for Yes if you find evidence indicating that the paper's textual content was AI-generated, or "N" for No if there is no indication of AI-generated writing. Your evaluation should be accurate, honing in on explicit acknowledgments of AI's role in the creation of the paper's written content."""
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": text}
        ]
    )
    return completion.choices[0].message.content


@click.group()
@click.option('--config', default='config.yml', type=click.Path())
@click.pass_context
def cli(ctx, config):
    load_ctx_from_config(ctx, config)
    sch = SemanticScholar(api_key='xxx')
    ctx.obj['sch'] = sch


def check_worker(client, _id, text):
    try:
        res = {'_id': _id, 'has': checkAIdeclariation(client, text[-8000:])}
        print(f'finished {_id}')
    except Exception:
        res = {'_id': _id, 'has': 'unkown'}
    return res


@cli.command()
@click.pass_context
def run(ctx):

    client = openai.OpenAI(
        base_url='xxxx',
        api_key='xxxx'
    )

    gpt_release_date = pd.to_datetime('2022-11-30')

    df_bypaper_scored = pd.read_pickle('data/df_bypaper_scored.pkl')
    df_bypaper_scored['n_tokens_k'] = df_bypaper_scored['n_tokens']//1000
    df_bypaper_scored.describe()

    paperinfo_df = pd.read_hdf('data/allpaperinfo.hdf5')
    paperinfo_df = paperinfo_df.loc[paperinfo_df['country_region'] != 'None'].copy(
    )
    paperinfo_df['afterChatGPT'] = (
        paperinfo_df['date'] >= gpt_release_date).astype(int)

    df_bypaper_scored.merge

    df_bypaper_scored = df_bypaper_scored.merge(
        paperinfo_df[['platform',
                      'title', 'date', 'category', 'country_region', 'authors',
                      'afterChatGPT']],
        right_on='_id',
        left_index=True
    )
    df_bypaper_scored_after = df_bypaper_scored.loc[
        df_bypaper_scored['afterChatGPT'] == 1
    ].copy()

    df_bypaper_scored_short_low_bino = df_bypaper_scored_after.loc[
        df_bypaper_scored_after['bino_score_mean'] < df_bypaper_scored_after['bino_score_mean'].quantile(
            0.10)].copy()

    df_bypaper_scored_short_low_bino = df_bypaper_scored_short_low_bino.sample(
        n=min(1000, len(df_bypaper_scored_short_low_bino)), random_state=42)

    df_bypaper_scored_short_low_bino.shape
    args = []
    for idx, row in df_bypaper_scored_short_low_bino.iterrows():
        args.append((idx, row.text))
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(check_worker, client, idx, text)
                   for idx, text in args]

        res = [future.result() for future in as_completed(futures)]
        json.dump(
            res,
            open('data/merged_declarations.json', 'w'),
            indent=2,
            ensure_ascii=False,
        )


if __name__ == '__main__':
    cli()
