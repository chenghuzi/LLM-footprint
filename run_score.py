from pdf_util import chunk_string
import json
import copy
from pathlib import Path
from collections import defaultdict
from binoculars import Binoculars
from tqdm import tqdm
import click


@click.command
@click.argument('data', type=str)
@click.option('--bs', default=16)
@click.option('--chunksize', default=1024)
@click.option('--limit', type=int, default=-1)
@click.option('--abstract', type=bool, default=False)
@click.option('--force', type=bool, default=False)
@click.option('--page', type=int, default=100)
def run(data, bs, chunksize, limit, abstract, force, page):
    dataf = json.load(open(data, 'r'))
    bino = Binoculars()
    n_p = 0
    if abstract is False:
        key_name = f'bino_scores-{chunksize}'
    else:
        key_name = f'abs_bino_scores-{chunksize}'

    text_chunks_w_ids_all = []
    for doc_idx, text_info in tqdm(enumerate(dataf)):
        if key_name in text_info.keys() and not force:
            continue

        if limit != -1 and n_p > limit:
            break
        text_info['full_text'] = text_info['full_text'].replace('\n', '')
        if abstract is False:
            text_chunks = chunk_string(text_info['full_text'], chunksize)
        else:
            text_chunks = [text_info['abstract']]

        text_chunks_w_ids = [((doc_idx, len(text_chunks), cid), chunk)
                             for cid, chunk in enumerate(text_chunks)]
        text_chunks_w_ids_all += text_chunks_w_ids
    print('start to run score')
    scores_d = defaultdict(list)
    n_changed = 0
    for chunk_idx in tqdm(range(len(text_chunks_w_ids_all)//bs + 1)):
        chunk = text_chunks_w_ids_all[chunk_idx*bs: (chunk_idx+1)*bs]
        chunk_text = [ch[1] for ch in chunk]
        scores = bino.compute_score(chunk_text)
        for ch, score in zip(chunk, scores):
            text_id, loc_count, loc_id = ch[0]
            scores_d[text_id].append((loc_id, score))

            if len(scores_d[text_id]) == loc_count:
                n_changed += 1
                score_of_doc = copy.deepcopy(scores_d[text_id])
                score_of_doc.sort(key=lambda x: x[0])
                dataf[text_id][key_name] = copy.deepcopy(
                    [sinfo[1] for sinfo in score_of_doc])
                if n_changed % page == 0:
                    pass
                    with open(data, 'w') as f:
                        f.write(json.dumps(dataf, indent=2))

    with open(data, 'w') as f:
        f.write(json.dumps(dataf, indent=2))


if __name__ == '__main__':
    run()
