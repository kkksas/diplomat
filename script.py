
from langchain.embeddings import HuggingFaceEmbeddings

import time
import re
import pandas as pd
import numpy as np
import matplotlib

from typing import Any, Coroutine, List
class HuggingFaceE5Embeddings(HuggingFaceEmbeddings):
    def embed_query(self, text: str) -> List[float]:
        text = f"query: {text}"
        return super().embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = [f"passage: {text}" for text in texts]
        return super().embed_documents(texts)

    async def aembed_query(self, text: str) -> Coroutine[Any, Any, List[float]]:
        text = f"query: {text}"
        return await super().aembed_query(text)

    async def aembed_documents(
        self, texts: List[str]
    ) -> Coroutine[Any, Any, List[List[float]]]:
        texts = [f"passage: {text}" for text in texts]
        return await super().aembed_documents(texts)


cds = pd.read_excel('opis.xlsx', usecols=['id','context']).dropna()
cds['len_c'] = cds['context'].apply(lambda x: len(x))
emb = HuggingFaceE5Embeddings(model_name="intfloat/multilingual-e5-large-instruct")

emb_texts = []

for i in cds['context']:
    emb_texts.append(emb.embed_query(i))
df = pd.DataFrame(emb_texts)
df.to_csv('vectorized_opis.csv')
print("text vector", df)