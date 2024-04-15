from datetime import datetime, timedelta
from pathlib import Path
from yaml import load
from scipy import stats
import numpy as np

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

import urllib3
import click

from db import get_database

import requests


def get_proxy():
    return requests.get("http://127.0.0.1:5010/get/").json()


def delete_proxy(proxy):
    requests.get("http://127.0.0.1:5010/delete/?proxy={}".format(proxy))


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'DNT': '1',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}


def load_ctx_from_config(ctx, config):
    if Path(config).is_file():
        with open(config, 'r') as f:
            config = load(f.read(), Loader=Loader)
        ctx.default_map = config

        start_date = datetime.strptime(
            config['month_range']['start'], "%Y-%m-%d")
        end_date = datetime.strptime(config['month_range']['end'], "%Y-%m-%d")

        month_ranges = []
        while start_date < end_date:
            next_month = start_date + timedelta(days=31)
            next_month = next_month.replace(day=1)
            month_ranges.append(
                f"{start_date.strftime('%Y-%m')}:{next_month.strftime('%Y-%m')}")
            start_date = next_month

        ctx.obj = {
            'ss': config['semantic_scholar'],
            'month_ranges': month_ranges,
            'num_per_month': config['month_range']['num_per_month'],
            'mdb_ss': get_database('s2_data',
                                   username=config['mongo']['username'],
                                   password=config['mongo']['password'],
                                   host=config['mongo']['host'],
                                   port=config['mongo']['port'],
                                   ),
            'mdb_biorxiv': get_database('biorxiv_data',
                                        username=config['mongo']['username'],
                                        password=config['mongo']['password'],
                                        host=config['mongo']['host'],
                                        port=config['mongo']['port'],
                                        ),
            'mdb_medrxiv': get_database('medrxiv_data',
                                        username=config['mongo']['username'],
                                        password=config['mongo']['password'],
                                        host=config['mongo']['host'],
                                        port=config['mongo']['port'],
                                        )
        }

    else:
        click.echo('no config, warning')



def compare_correlations(n1, corr1, n2, corr2):
    z1 = np.arctanh(corr1)
    z2 = np.arctanh(corr2)

    se_diff = np.sqrt(1/(n1-3) + 1/(n2-3))

    z = (z1 - z2) / se_diff

    p = stats.norm.sf(abs(z))

    return z, p
