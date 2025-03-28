
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores.faiss import FAISS
from langchain_core.documents import Document
import string
import time
import re
import pandas as pd
import numpy as np
import matplotlib
K_c = 5
dataset_cap = 5

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

def create_emb(emb):
    cds = pd.read_excel('chunked.xlsx', usecols=['id','context']).dropna()
    cds['len_c'] = cds['context'].apply(lambda x: len(x))
    documents = []
    for i in cds.iloc:
        documents.append(Document(page_content=i['context'], metadata = {"id":i['id']}))
    faiss_db = FAISS.from_documents(documents, embedding=emb)
    faiss_db.save_local("index")
    return faiss_db
qds = pd.DataFrame()
for dataset_num in range(dataset_cap):   
    new_ds = pd.read_csv(f'./ds{dataset_num+1}.csv', dtype=str, sep=';', usecols=['question'])
    qds.reset_index(drop=True, inplace=True)
    new_ds.reset_index(drop=True, inplace=True)
    qds = pd.concat([qds, new_ds])

#qds['relv'] = qds['relv'].apply(lambda x: list(map(int, x.split(','))))
#не оч понял, что подразумевается под контекстом в qds, видимо это чанки подходящих под вопрос описаний туров(в моем случае)
#не проще ли тогда считать точность по id описания к которому чанк принадлежит

# добавляет чанки по id описаний, указанных в файле как релевантные
#qds['context'] = qds['relv'].apply(lambda x: list(cds['context'][cds['id'].isin(x)]))

s_time = time.time() 

#######___faiss___#########################
emb = HuggingFaceE5Embeddings(model_name="intfloat/multilingual-e5-large-instruct")
#загрузить эмбы
faiss_db = FAISS.load_local("index", embeddings=emb, allow_dangerous_deserialization=True)
#переделать эмбы
#faiss_db = create_emb(emb)

end_time =time.time()   
print("faiss_db_spended:", end_time-s_time)


def sim_search_with_thresh(query, k = 5, thresh=-1):
    #res = faiss_db.similarity_search_with_score(query, k = k, score_threshold=thresh)#0.3 is ok
    res = faiss_db.similarity_search_with_relevance_scores(query, k = k, score_threshold = thresh)
    ret_chunks=[]
    score = []
    for elem in res:
        ret_chunks.append(elem[0].metadata['id'])
        score.append(elem[1])
    # ret_chunks = list(elem[0].metadata['id'] for elem in res)
    # score = list(elem[1] for elem in res)
    return [query, ret_chunks, score]

result = qds["question"].apply(sim_search_with_thresh, args=(K_c, 0.8))
result = pd.DataFrame(result.tolist(), columns = ['question', 'ret_chunks', 'score']) 

result.to_excel('res_with_scores.xlsx')
end_time =time.time()   
print("total_spended:", end_time-s_time)

