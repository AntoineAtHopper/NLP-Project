import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from annoy import AnnoyIndex
from beir import util
import pandas as pd
import numpy as np
import json

# Create dataset for the searchable index using SquadV2 and DBPedia.
def get_or_create_contexts() -> np.ndarray:
    if not hasattr(get_or_create_contexts, "contexts"):
        contexts = None
        if os.path.isfile("./resources/contexts.txt"):
            with open("./resources/contexts.txt", "r") as f:
                contexts = np.array(json.load(f))
        else:
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
            if not os.path.exists("./resources"):
                os.makedirs("./resources")
            with open("./resources/contexts.txt", "w+") as f:
                json.dump(contexts.tolist(), f)
        get_or_create_contexts.contexts = contexts
    # Return contexts
    return get_or_create_contexts.contexts


sentence_transformer = SentenceTransformer('msmarco-distilbert-base-tas-b')
# Get or Create a searchable index.
def get_or_create_searchable_index() -> np.ndarray:
    if not hasattr(get_or_create_searchable_index, "searchable_index"):
        searchable_index = None
        contexts = get_or_create_contexts()
        if os.path.isfile("./resources/searchable_index.txt"):
            searchable_index = np.loadtxt("./resources/searchable_index.txt").reshape(len(contexts), -1)
        else:
            searchable_index = sentence_transformer.encode(contexts)
            if not os.path.exists("./resources"):
                os.makedirs("./resources")
            np.savetxt("./resources/searchable_index.txt", searchable_index)
        get_or_create_searchable_index.searchable_index = searchable_index
    return get_or_create_searchable_index.searchable_index


# Query searchable index using Nearest Neighbors.
def get_nn(v: int, k: int) -> np.ndarray:
    searchable_index = get_or_create_searchable_index()
    distances = v @ searchable_index.T
    return np.argsort(distances)[::-1][:k]


# Query searchable index using Approximative Nearest Neighbors.
def get_nn_approx(v: int, k: int) -> np.ndarray:
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
def search_contexts(question: str, *, approximate: bool=False, k: int=3) -> np.ndarray:
    contexts = get_or_create_contexts()
    q = sentence_transformer.encode(question)
    topk = get_nn_approx(q, k) if approximate else get_nn(q, k)
    return contexts[topk]

