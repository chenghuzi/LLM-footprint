import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
import pandas as pd
import requests
from tqdm import tqdm
import copy
import json
from pdf_util import extract_main_text_from_pdf
from utils import headers, load_ctx_from_config

SIZE_LIMIT = 6 * 1024

pdf_dir = Path('data/arxiv/pdfs')
texts_dir = Path('data/arxiv/texts')


@click.group()
@click.option('--config', default='config.yml', type=click.Path())
@click.pass_context
def cli(ctx, config):
    load_ctx_from_config(ctx, config)


def download_paper(paper_id):
    fn = paper_id
    local_file = pdf_dir / f'{fn}.pdf'
    pdf_url = f'https://arxiv.org/pdf/{fn}.pdf'

    if not local_file.is_file():
        try:
            print(f'downloading {pdf_url}')
            response = requests.get(
                pdf_url, headers=headers, verify=False)
            with open(local_file, 'wb') as f:
                f.write(response.content)

            if os.path.getsize(local_file) < SIZE_LIMIT:
                local_file.unlink()
                print(f"The file is smaller than {SIZE_LIMIT/1024:.1f}KB.")
                return

        except Exception as e:
            print(
                f"error when downloading {pdf_url} to {local_file}. Error: {e}")


@cli.command()
@click.pass_context
@click.option('--limit', type=int)
@click.option('--jobs', type=int)
def download(ctx, limit, jobs):
    df = pd.read_pickle('data/arxiv/sampled_df.pkl')
    paper_ids = df['_id'].values
    print(paper_ids)
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        [executor.submit(download_paper, paper_id) for paper_id in paper_ids
         ]
    os.system('say done')


@cli.command()
@click.pass_context
@click.option('--limit', type=int)
def extract(ctx, limit):
    df = pd.read_pickle('data/arxiv/sampled_df.pkl')
    paper_ids = df['_id'].values

    for paper_id in tqdm(paper_ids):
        fn = paper_id
        local_file = pdf_dir / f'{fn}.pdf'

        target_txtf = texts_dir / f'{fn}.txt'
        if local_file.is_file():
            try:
                full_text = extract_main_text_from_pdf(local_file)
                with open(target_txtf, 'w') as f:
                    f.write(full_text)
            except Exception as e:
                print(f"skipped {local_file} as {e}")
                local_file.unlink()


@cli.command()
@click.option('--page', type=int, default=4000, required=False)
@click.pass_context
def integrate(ctx, page):
    """Integrate all plain texts with their meta data"""

    df = pd.read_pickle('data/arxiv/sampled_df.pkl').sample(frac=1)
    full_text_data = []
    nc = 0
    for row_idx, row in df.iterrows():
        paper = row.to_dict()
        fn = paper['_id']
        paper['date'] = paper['date'].strftime('%Y-%m-%d')
        target_txtf = texts_dir / f'{fn}.txt'

        if target_txtf.is_file():
            paper['full_text'] = open(target_txtf, 'r').read()
            full_text_data.append(copy.deepcopy(paper))

            if len(full_text_data) % page == 0:
                nwf = f'data/arxiv/fulltext_info_{nc}.json'
                print(f'writing to {nwf}')
                with open(nwf, 'w') as f:
                    f.write(json.dumps(full_text_data, indent=2))
                full_text_data = []
                nc += 1


if __name__ == '__main__':
    cli()
