import numpy as np

from jina import Document, DocumentArrayMemmap, DocumentArray
from sklearn.datasets import make_blobs
from pqlite import PQLite
import time
import os
import shutil

def create_data_online(n_examples, D, batch_size):
    np.random.seed(123)
    num = 0
    while True:
        Xtr_batch = make_blobs(n_samples=batch_size, n_features=D)[0].astype(np.float32)
        yield DocumentArray([Document(embedding=x) for x in Xtr_batch])
        num += batch_size

        if num + batch_size >= n_examples:
            break

    if num < n_examples:
        Xtr_batch = make_blobs(n_samples=n_examples - num, n_features=D)[0].astype(
            np.float32
        )
        yield DocumentArray([Document(embedding=x) for x in Xtr_batch])

def clean_workspace():
    if os.path.exists('./pqlite_storage'):
        shutil.rmtree('./pqlite_storage')

    if os.path.exists('./memmap'):
        shutil.rmtree('./memmap')

def insert_docarraymemmmap(n_examples, n_features, batch_size=10_000):

    da_generator = create_data_online(n_examples, n_features, batch_size)
    dam = DocumentArrayMemmap('./memmap')

    total_time = 0
    for da in da_generator:
        t0 = time.time()
        dam.extend(da)
        total_time += time.time() - t0

    total_time = round(total_time, 2)
    print(f'\n\ntotal time docarraymemmap={total_time} seconds\n\n')
    return total_time

def insert_pqlite(n_examples, n_features, batch_size=100_000):

    da_generator = create_data_online(n_examples, n_features, batch_size)
    clean_workspace()
    pqlite = PQLite(dim=n_features, data_path= 'pqlite_storage')

    total_time = 0
    for da in da_generator:
        t0 = time.time()
        #embeddings = da.embeddings
        #cell_ids = [0]*len(embeddings)
        #pqlite.insert(embeddings, cell_ids, da)
        pqlite.doc_store(0).insert(da)
        total_time += time.time() - t0

    total_time = round(total_time, 2)
    print(f'\n\ntotal time pqlite={total_time} seconds\n\n')
    return total_time

n_examples = 1_000_000
n_features = 512

total_time_pqlite = insert_pqlite(n_examples, n_features, batch_size=100_000)
total_time_memmap = insert_docarraymemmmap(n_examples, n_features, batch_size=10_000)
