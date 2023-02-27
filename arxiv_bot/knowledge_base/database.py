import os
from typing import Optional
from uuid import uuid4
from tqdm.auto import tqdm

import torch
from transformers import AutoTokenizerFast
from splade.models.transformers_rep import Splade

import pinecone
import langchain


dense_models = {
    'text-embedding-ada-002': {
        'source': 'openai',
        'dimension': 1538,
        'api_key': True
    },
    'multilingual-22-12': {
        'source': 'cohere',
        'dimension': 768,
        'api_key': True
    }
}


class Pinecone:
    def __init__(
        self,
        index_name: str,
        api_key: Optional[str] = None,
        environment: str = 'us-east1-gcp',
        dense_model_name: Optional[str] = 'text-embedding-ada-002',
        sparse_model_name: Optional[str] = 'naver/splade-cocondenser-ensembledistil',
        device: Optional[str] = None
    ):
        # set local device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        # initialize dense model
        self._init_dense(dense_model_name, api_key)
        # initialize sparse model
        self._init_sparse(sparse_model_name)

        # initialize connection to Pinecone
        api_key = api_key or os.environ['PINECONE_API_KEY']
        pinecone.init(
            api_key=api_key,
            environment=environment
        )
        # check if index exists
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                pod_type='s1'
            )
        self.index_name = index_name
        self.api_key = api_key
        self.index = pinecone.Index(name=self.index_name, api_key=self.api_key)
    
    def add(
        self,
        texts: list,
        ids: Optional[list]=None,
        metadata: Optional[list]=None,
        batch_size: int = 64,
        overwrite=True
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
            for j in range(i, i_end):
                upsert.append({
                    'id': ids[j],
                    'values': dense_emb[j],
                    'sparse_values': sparse_emb[j],
                    'metadata': metadata[j]
                })
            # upsert to Pinecone
            self.index.upsert(upsert)
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
    def __init__(self, model_name: str, device: Optional[str] = None):
        # initialize model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Splade(model_name, agg='max')
        # move to cuda if enabled
        self.model.to(self.device)
        # set to inference mode
        self.model.eval()
        # initialize tokenizer
        self.tokenizer = AutoTokenizerFast.from_pretrained(model_name)

    def __call__(self, texts: str):
        if type(texts) == str:
            texts = [texts]
        # tokenize
        inputs = self.tokenizer(
            texts, add_special_tokens=False,
            return_tensors='pt'
        ).to(self.device)
        # create sparse embedding
        with torch.no_grad():
            sparse_emb = self.model(
                d_kwargs=inputs
            )['d_rep'].squeeze()
        # extract indices and their values
        indices = sparse_emb.nonzero().squeeze().cpu().tolist()
        values = sparse_emb[indices].cpu().tolist()
        # return sparse embedding in pinecone format
        return {'indices': indices, 'values': values}

class DenseOpenAI:
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        api_key = api_key or os.environ['OPENAI_API_KEY']
        self.model = langchain.llms.OpenAI(
            model_name=model_name,
            openai_api_key=api_key
        )
    
    def __call__(self, text: str):
        return self.model(text)
