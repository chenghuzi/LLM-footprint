import copy
import hashlib
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests
import json
import urllib3
import tqdm
from db import insert2db_direct

import click
from pdf_util import chunk_string, extract_main_text_from_pdf
from utils import load_ctx_from_config, headers, get_proxy, delete_proxy

SIZE_LIMIT = 6 * 1024

pdf_dir = Path('data/biorxiv/pdfs')
texts_dir = Path('data/biorxiv/texts')


@click.group()
@click.option('--config', default='config.yml', type=click.Path())
@click.pass_context
def cli(ctx, config):
    load_ctx_from_config(ctx, config)


def fetch_by_month(mdb, start_m, end_m, limit=None):
    try:
        collections = []
        start_d = f'{start_m}-01'
        end_d = f'{end_m}-01'
        offset = 0
        finished = False
        while not finished:
            url = f"https://api.biorxiv.org/details/biorxiv/{start_d}/{end_d}/{offset}"

            res = requests.get(url).json()
            msg = res['messages'][0]
            collections += res['collection']
            offset += msg['count']

            print(f'fetching {url}, total={msg["total"]}')

            insert2db_direct(mdb, res['collection'], 'doi')
            if offset >= msg['total']:
                finished = True
            if limit:
                if offset >= limit:
                    finished = True
        print(len(collections))
        pass
    except Exception as e:
        print(e)
        pass


@cli.command()
@click.pass_context
@click.option('--limit', type=int)
def fetchbiorxiv(ctx, limit):
    month_ranges = ctx.obj['month_ranges']
    for month_range in month_ranges:
        start_m, end_m = month_range.split(':')
        fetch_by_month(ctx.obj['mdb_biorxiv']['papers'], start_m, end_m, limit)

    pass


def download_papers(ctx, start_m, pdf_dir, limit=None):
    paper_info_collection = ctx.obj['mdb_biorxiv']['papers']
    papers = paper_info_collection.find({"date": {"$regex": f"^{start_m}"}},
                                        limit=limit)

    for paper in papers:
        url = paper['jatsxml']
        fn = paper['_id'].replace('/', '-')

        pdf_url = url.replace('.source.xml', '.full.pdf')
        local_file = pdf_dir / f'{fn}.pdf'

        if not local_file.is_file():
            try:
                print(f'({start_m}) downloading {pdf_url}')
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
                    f"({paper['domain']}) error when downloading {url} to {local_file}. Error: {e}")
    print(f'finished {start_m}')


@cli.command()
@click.pass_context
@click.option('--limit', type=int)
@click.option('--jobs', type=int)
def download(ctx, limit, jobs):
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        [executor.submit(download_papers, ctx, monthr.split(':')[
            0], pdf_dir, limit) for monthr in ctx.obj['month_ranges']]
    os.system('say done')


@cli.command()
@click.pass_context
@click.option('--limit', type=int)
def extract(ctx, limit):



    paper_info_collection = ctx.obj['mdb_biorxiv']['papers']
    for monthr in ctx.obj['month_ranges']:
        print(monthr)
        start_m = monthr.split(':')[0]
        papers = paper_info_collection.find({"date": {"$regex": f"^{start_m}"}},
                                            limit=limit)
        for paper in papers:
            fn = paper['_id'].replace('/', '-')
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
@click.option('--limit', type=int, required=False)
@click.pass_context
def integrate(ctx, limit):
    """Integrate all plain texts with their meta data"""


    month_suffix = f"{ctx.obj['month_ranges'][0].split(':')[0]}_{ctx.obj['month_ranges'][-1].split(':')[1]}"

    full_text_data = []
    paper_info_collection = ctx.obj['mdb_biorxiv']['papers']
    for monthr in ctx.obj['month_ranges']:
        print(monthr)
        start_m = monthr.split(':')[0]
        papers = paper_info_collection.find({"date": {"$regex": f"^{start_m}"}},
                                            limit=limit)
        for paper in papers:
            fn = paper['_id'].replace('/', '-')
            target_txtf = texts_dir / f'{fn}.txt'
            if target_txtf.is_file():
                paper['full_text'] = open(target_txtf, 'r').read()
                full_text_data.append(copy.deepcopy(paper))

    with open(f'data/biorxiv/fulltext_info_{month_suffix}.json', 'w') as f:
        f.write(json.dumps(full_text_data, indent=2))


if __name__ == '__main__':
    cli()
