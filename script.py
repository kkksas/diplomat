
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores.faiss import FAISS
from langchain_core.documents import Document

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

k_c = 5

cds = pd.read_excel('opis.xlsx', usecols=['id','context']).dropna()
qds = pd.read_csv('./voprosy.csv', header=None,sep=';', names=['id','question'])



cds['len_c'] = cds['context'].apply(lambda x: len(x))
emb = HuggingFaceE5Embeddings(model_name="intfloat/multilingual-e5-large-instruct")

documents = []
for i in cds.iloc:
    documents.append(Document(page_content=i['context'], metadata = {"id":i['id']}))


faiss_db = FAISS.from_documents(documents, embedding=emb)
retriever = faiss_db.as_retriever(search_kwargs={"k": k_c})


print("Вопрос", qds['question'][9])
anc = retriever.invoke(qds['question'][9])
print(anc)
for res in anc:
    print(f"* {res.page_content} [{res.metadata}]")


faiss_db.save_local('./', index_name="index")


"""
cds = pd.read_excel('opis.xlsx', usecols=['id','context']).dropna()
cds['len_c'] = cds['context'].apply(lambda x: len(x))
emb = HuggingFaceE5Embeddings(model_name="intfloat/multilingual-e5-large-instruct")

emb_texts = []

for i in cds['context']:
    emb_texts.append(emb.embed_query(i))
cds = pd.DataFrame(emb_texts)
cds.to_csv('vectorized_opis.csv', index=False)


cds = pd.read_csv('./voprosy.csv', header=None, sep='\t')
print(cds)

cds['len_c'] = cds[0].apply(lambda x: len(x))
emb = HuggingFaceE5Embeddings(model_name="intfloat/multilingual-e5-large-instruct")

emb_texts = []

for i in cds[0]:
    emb_texts.append(emb.embed_query(i))
cds = pd.DataFrame(emb_texts)
print(cds)
cds.to_csv('vectorized_questions.csv', index=False)

"""
