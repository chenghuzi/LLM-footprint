import copy
import hashlib
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import urllib3
import tqdm
import click
import requests
from pymongo import errors
from semanticscholar import SemanticScholar

from pdf_util import chunk_string, extract_main_text_from_pdf
from utils import load_ctx_from_config, headers
from db import insert2db, insert_chunks2db




def md5_string(s):
    return hashlib.md5(s.encode()).hexdigest()


pdf_dir = Path('data/ss/pdfs')
texts_dir = Path('data/ss/texts')

fields = ['title', 'citationCount', 'openAccessPdf',
          'publicationDate', 's2FieldsOfStudy', 'journal', 'authors']


open_access_pdf = True


@click.group()
@click.option('--config', default='config.yml', type=click.Path())
@click.pass_context
def cli(ctx, config):
    load_ctx_from_config(ctx, config)


@cli.command()
@click.option('--domains', multiple=True)
@click.option('--ss_apikey')
@click.pass_context
def fetchss(ctx, domains, ss_apikey):
    """Fetch data meta data info of papers
    """
    sch = SemanticScholar(api_key=ss_apikey)
    print(ctx.obj['month_ranges'])
    keyid = 'paperId'
    for domain in domains:
        for monthr in ctx.obj['month_ranges']:
            results = sch.search_paper(
                domain, fields=fields,
                open_access_pdf=open_access_pdf,
                publication_date_or_year=monthr,
            )
            t0 = time.time()
            nd = len(results.items)
            if len(results.items) > 0:
                insert2db(ctx.obj['mdb_ss']['papers'], results.items,
                          keyid, domain=domain)
            else:
                continue
            while results._has_next_page():
                print(
                    f'Fetching {monthr} in {domain} '
                    f'({results.offset+results._limit}/{results._max_results})')
                results._items = []
                results.next_page()
                if len(results.items) > 0:
                    insert2db(ctx.obj['mdb_ss']['papers'],
                              results.items,
                              keyid, domain=domain)
                nd += len(results.items)

                if nd > ctx.obj['num_per_month']:
                    break

            dt = time.time() - t0
            print(f'Fetch {monthr} in {domain} finished in {dt:.2f}s.')


def download_papers(ctx, start_month, pdf_dir, limit=None):
    paper_info_collection = ctx.obj['mdb_ss']['papers']
    papers = paper_info_collection.find({"publicationDate": {"$regex": f"^{start_month}"}},
                                        limit=limit)
    for paper in papers:
        url = paper['openAccessPdf']['url']
        fn = paper['paperId']

        local_file = pdf_dir / f'{fn}.pdf'

        if not local_file.is_file():
            try:
                response = requests.get(url, headers=headers, verify=False)
                with open(local_file, 'wb') as f:
                    f.write(response.content)
            except Exception as e:
                print(
                    f"({paper['domain']}) error when downloading {url} to {local_file}. Error: {e}")


@cli.command()
@click.option('--limit', type=int)
@click.pass_context
def download(ctx, limit):
    with ThreadPoolExecutor(max_workers=20) as executor:
        [executor.submit(download_papers, ctx, monthr.split(':')[
            0], pdf_dir, limit) for monthr in ctx.obj['month_ranges']]
    pass


@cli.command()
def extract():
    for pdff in pdf_dir.glob('*.pdf'):
        print(f'reading {pdff}')
        target_txtf = texts_dir / f'{pdff.stem}.txt'
        if target_txtf.is_file():
            return

        try:
            full_text = extract_main_text_from_pdf(pdff)
            with open(target_txtf, 'w') as f:
                f.write(full_text)
        except Exception as e:
            print(f"skipped {pdff} as {e}")
            pdff.unlink()


def parse_papers(ctx, start_month, texts_dir, limit=None, chunk_size=1024):
    paper_info_collection = ctx.obj['mdb_ss']['papers']
    paper_texts_collection = ctx.obj['mdb_ss']['paper_texts']
    papers = paper_info_collection.find({"publicationDate": {"$regex": f"^{start_month}"}},
                                        limit=limit)

    full_text_data = []
    for paper in papers:
        local_file = texts_dir / f'{paper["paperId"]}.txt'
        if local_file.is_file():
            full_text = open(local_file, 'r').read()
            if len(full_text) < 100:
                continue

            print(
                f'({start_month}) start to parse {local_file}, id={paper["_id"]}')
            paper_info_collection.update_one({'_id': paper['_id']},
                                             {"$set": {"file": str(local_file)}})

            full_text_data.append(
                {
                    '_id': paper['paperId'],
                    'text': full_text,
                }
            )

    if len(full_text_data) > 0:
        insert_chunks2db(paper_texts_collection, full_text_data)


@cli.command()
@click.option('--limit', type=int, required=False)
@click.option('--size', type=int)
@click.pass_context
def parse(ctx, limit, size):
    with ThreadPoolExecutor(max_workers=20) as executor:
        [executor.submit(parse_papers, ctx, monthr.split(':')[
            0], texts_dir, limit, size) for monthr in ctx.obj['month_ranges']]


@cli.command()
@click.option('--limit', type=int, required=False)
@click.option('--size', type=int)
@click.pass_context
def integrate(ctx, limit, size):
    full_text_data = []
    paper_coll = ctx.obj['mdb_ss']['papers']
    for txtf in tqdm.tqdm(texts_dir.glob('*.txt')):
        p_info = paper_coll.find_one({"_id": txtf.stem})
        if p_info:
            p_info['full_text'] = open(txtf, 'r').read()
            full_text_data.append(copy.deepcopy(p_info))

    with open('data/fulltext_info2.json', 'w') as f:
        f.write(json.dumps(full_text_data, indent=2))


def detect_papers(ctx, start_month, limit=None, chunk_size=1024):
    paper_info_collection = ctx.obj['mdb_ss']['papers']
    paper_texts_collection = ctx.obj['mdb_ss']['paper_texts']
    papers = paper_info_collection.find({"publicationDate": {"$regex": f"^{start_month}"}},
                                        limit=limit)
    for paper in papers:
        paper_text = paper_texts_collection.find_one({'_id': paper['paperId']},
                                                     limit=limit)

        text_chunks = chunk_string(paper_text['text'], chunk_size=chunk_size)
        print(len(text_chunks))
        pass


@cli.command()
@click.option('--limit', type=int, required=False)
@click.option('--size', type=int, default=2048)
@click.pass_context
def detect(ctx, limit, size):
    with ThreadPoolExecutor(max_workers=20) as executor:
        [executor.submit(detect_papers, ctx, monthr.split(':')[
            0], limit, size) for monthr in ctx.obj['month_ranges']]


if __name__ == '__main__':
    cli()
