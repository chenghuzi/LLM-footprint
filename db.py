from pymongo import MongoClient
import urllib.parse
import os
import copy
import sys
from pymongo import errors


def get_database(database_name,
                 username,
                 password,
                 host,
                 port=37017,
                 ):

    username = urllib.parse.quote_plus(username)
    password = urllib.parse.quote_plus(password)

    uri = f'mongodb://{username}:{password}@{host}:{port}'
    client = MongoClient(uri)
    return client[database_name]


def blockPrint():
    sys.stdout = open(os.devnull, 'w')


def enablePrint():
    sys.stdout = sys.__stdout__


def insert_chunks2db(mgdb, items):
    try:
        blockPrint()
        mgdb.insert_many(items, ordered=False)
    except errors.BulkWriteError as e:
        print(f'inserting error: {e}')
    enablePrint()


def insert2db(mgdb, items, key_id, domain=None):
    data = []
    for item in items:
        item._data['_id'] = item._data[key_id]

        if domain:
            item._data['domain'] = domain
        exists = mgdb.find_one({"_id": item._data[key_id]}) is not None
        if not exists:
            data.append(copy.deepcopy(item._data))
    try:
        blockPrint()
        if len(data) > 0:
            mgdb.insert_many(data, ordered=False)
    except errors.BulkWriteError as e:
        print(f'inserting error: {e}')
    enablePrint()


def insert2db_direct(mgdb, items, key_id, domain=None):
    data = []
    for item in items:
        item['_id'] = item[key_id]

        if domain:
            item['domain'] = domain
        exists = mgdb.find_one({"_id": item[key_id]}) is not None
        if not exists:
            data.append(copy.deepcopy(item))
    try:
        blockPrint()
        if len(data) > 0:
            mgdb.insert_many(data, ordered=False)
    except errors.BulkWriteError as e:
        print(f'inserting error: {e}')
    enablePrint()


if __name__ == "__main__":
    dbname = get_database('s2_data')
    collection_names = dbname.list_collection_names()
    print(collection_names)