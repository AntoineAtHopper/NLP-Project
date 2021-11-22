from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from annoy import AnnoyIndex
from beir import util
import pandas as pd
import numpy as np


# Create dataset for the searchable index using SquadV2 and DBPedia.
def get_or_create_contexts():
    if not hasattr(get_or_create_contexts, "contexts"):
        # Fetch SquadV2 dataset
        datasets = load_dataset("squad_v2")
        # Fetch DBpedia Dataset
        dataset = "dbpedia-entity"
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
        data_path = util.download_and_unzip(url, "datasets")
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
        # Merge datasets
        contexts = pd.Series(datasets["validation"]["context"]).unique()
        to_add = []
        for doc in list(corpus.values())[:14000]: # The first 14000 completes the SQuAD v2 validation set for ~10000 samples
            if len(doc["text"].split()) > 50:
                to_add.append(doc["text"])
            contexts = np.concatenate((contexts, np.array(to_add)))
            contexts = np.unique(contexts)
        get_or_create_contexts.contexts = contexts
    # Return contexts
    return get_or_create_contexts.contexts


sentence_transformer = SentenceTransformer('msmarco-distilbert-base-tas-b')
# Get or Create a searchable index.
def get_or_create_searchable_index():
    if not hasattr(get_or_create_searchable_index, "searchable_index"):
        contexts = get_or_create_contexts()
        get_or_create_contexts.searchable_index = sentence_transformer.encode(contexts)
    return get_or_create_contexts.searchable_index


# Query searchable index using Nearest Neighbors.
def get_nn(v, k):
    searchable_index = get_or_create_searchable_index()
    distances = v @ searchable_index.T
    return np.argsort(distances)[::-1][:k]


# Query searchable index using Approximative Nearest Neighbors.
def get_nn_approx(v, k):
    searchable_index = get_or_create_searchable_index()
    if not hasattr(get_nn_approx, "nn"):
        nn = AnnoyIndex(768, "dot")
        for i, embedding in enumerate(searchable_index):
            nn.add_item(i, embedding)
        nn.build(2000)
        get_nn_approx.nn = nn
    topk = get_nn_approx.nn.get_nns_by_vector(v, k)
    return topk


# Search in the index.
def search_contexts(question, *, approximate=False, k=1):
    contexts = get_or_create_contexts()
    q = sentence_transformer.encode(question)
    topk = get_nn_approx(q, k) if approximate else get_nn(q, k)
    return contexts[topk]

