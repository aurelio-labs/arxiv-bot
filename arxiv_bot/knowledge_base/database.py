import os
from typing import Optional
from uuid import uuid4
from tqdm.auto import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from pinecone_text.sparse.splade_encoder import SpladeEncoder

import pinecone
import openai


dense_models = {
    'text-embedding-ada-002': {
        'source': 'openai',
        'dimension': 1536,
        'api_key': True,
        'metric': 'dotproduct'
    },
    'multilingual-22-12': {
        'source': 'cohere',
        'dimension': 768,
        'api_key': True,
        'metric': 'dotproduct'
    }
}


class Pinecone:
    log = []
    def __init__(
        self,
        index_name: str,
        pinecone_api_key: Optional[str] = None,
        environment: str = 'us-east1-gcp',
        openai_api_key: Optional[str] = None,
        dense_model_name: Optional[str] = 'text-embedding-ada-002',
        sparse_model_name: Optional[str] = 'naver/splade-cocondenser-ensembledistil',
        device: Optional[str] = None
    ):
        # set local device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        # initialize dense model
        openai_api_key = openai_api_key or os.environ['OPENAI_API_KEY']
        self._init_dense(dense_model_name, openai_api_key)
        # initialize sparse model
        self._init_sparse(sparse_model_name)

        # initialize connection to Pinecone
        pinecone_api_key = pinecone_api_key or os.environ['PINECONE_API_KEY']
        pinecone.init(
            api_key=pinecone_api_key,
            environment=environment
        )
        # check if index exists
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                index_name,
                pod_type='s1',
                dimension=self.dense_meta['dimension'],
                metric=self.dense_meta['metric']
            )
        self.index_name = index_name
        self.pinecone_api_key = pinecone_api_key
        self.index = pinecone.Index(self.index_name)
    
    def add(
        self,
        texts: list,
        ids: Optional[list]=None,
        metadata: Optional[list]=None,
        batch_size: int = 64,
        overwrite=False
    ):
        """Encodes texts and upserts to Pinecone index.

        :param texts: List of text strings to encode
        :type texts: list
        :param ids: List of ids to upsert to Pinecone index
        :type ids: list, optional
        :param metadata: List of metadata to upsert to Pinecone index
        :type metadata: list, optional
        :param batch_size: Batch size for encoding, defaults to 64
        :type batch_size: int, optional
        :param overwrite: Whether to overwrite existing ids, defaults to True
        :type overwrite: bool, optional
        """
        if overwrite: raise NotImplementedError("Overwrite argument not implemented yet")
        # check lengths align
        if ids is not None and len(ids) != len(texts):
            raise ValueError("Lengths of texts and ids must match")
        if metadata is not None and len(metadata) != len(texts):
            raise ValueError("Lengths of texts and metadata must match")
        for i in tqdm(range(0, len(texts), batch_size)):
            i_end = min(i + batch_size, len(texts))
            dense_emb, sparse_emb = self._encode(texts[i:i_end])
            if ids is None:
                # if there are no IDs we create them
                ids = [str(uuid4()) for _ in range(i, i_end)]
            if metadata is None:
                # if there is no metadata we create it
                metadata = [{} for _ in range(i, i_end)]
            # add text to metadata
            for j, text in enumerate(texts[i:i_end]):
                metadata[j]['text-field'] = text
            # create upsert list
            upsert = []
            for _id, values, sparse_values, meta in zip(ids, dense_emb, sparse_emb, metadata):
                upsert.append({
                    'id': _id,
                    'values': values,
                    'sparse_values': sparse_values,
                    'metadata': meta
                })
            # upsert to Pinecone
            try:
                self.index.upsert(upsert)
            except ValueError as e:
                # loop through each item and try to upsert individually
                for item in upsert:
                    try:
                        self.index.upsert([item])
                    except ValueError as e:
                        # if there is an error we log error message and values that caused it
                        self.log.append({
                            'error': str(e),
                            'upsert_values': item
                        })
                        print("Error during upsert, see Pinecone.log for details")

        return self.index.describe_index_stats()
    
    def _encode(self, texts: list):
        """Takes texts and creates sparse-dense vectors.
        
        :param text: List of text strings to encode
        :type text: list
        """
        # encode dense
        dense_emb = self.dense_model(texts)
        # encode sparse
        sparse_emb = self.sparse_model(texts)
        # return encoded vectors
        return dense_emb, sparse_emb
        
    def _init_dense(self, model_name: str, api_key: Optional[str] = None):
        if model_name in dense_models.keys():
            self.dense_meta = dense_models[model_name]
        else:
            raise NotImplementedError(
                "We'll use this for sentence transformers"
            )
        
        if self.dense_meta['source'] == 'openai':
            api_key = api_key or os.environ['OPENAI_API_KEY']
            self.dense_model = DenseOpenAI(model_name, api_key)
            
        self.dense_meta['model_name'] = model_name
    
    def _init_sparse(self, model_name: str):
        self.sparse_model = SpladeModel(model_name, self.device)


class SpladeModel:
    def __init__(
        self,
        model_name: Optional[str] = 'naver/splade-cocondenser-ensembledistil',
        device: Optional[str] = None
    ):
        # initialize model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SpladeEncoder(device=device)

    def __call__(self, texts: list):
        if type(texts) == str:
            texts = [texts]
        # encode
        embeds = self.model.encode_documents(texts)
        # return sparse embedding in pinecone format
        return embeds

class DenseOpenAI:
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ['OPENAI_API_KEY']
        self.model_name = model_name
    
    def __call__(self, texts: list):
        res = openai.Embedding.create(
            input=texts,
            engine=self.model_name
        )
        embeddings = [record['embedding'] for record in res['data']]
        return embeddings
