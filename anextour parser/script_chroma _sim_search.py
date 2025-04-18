
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores.faiss import FAISS
from langchain.vectorstores import Chroma
from langchain_core.documents import Document
from transformers import AutoModel
import time
import re
import pandas as pd
import numpy as np
import matplotlib
K_c = 5
dataset_cap = 3
is_metric_type_mine = True
query_addon = "Дан вопрос, необходимо найти абзац текста с ответом \nвопрос:"
def acc_t(x):
    a = list(pd.Series(x['ret_chunks']).isin(x['relv']).dropna()) 
    return sum(a)/len(a)   

def acc_top(qds) -> float:
    #for each elem in qds: (['ret_chunks']{"25", "23", "105"}, ['relv']{"25", "23", "1","2", "3", "4"} =>{True, True, false} => return mean {0.67})
    a = qds.apply(acc_t, axis = 1)
    s = 1
    return a

def acc_buttom(qds) -> float:
    #for each elem in qds: (['ret_chunks']{"25", "23", "105"}, ['relv']{"25", "23", "1","2", "3", "4"} =>{True, True, false} => True)
    #mean(по всем элементам qds)
    a = qds.apply(lambda x: pd.Series(x['ret_chunks']).isin(x['relv']), axis = 1)
    a = np.sum(a, axis = 1).astype(bool)
    return a.astype(int)

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
        documents.append(Document(page_content=i['context'], metadata = {"id":int(i['id'])}))
    chroma_db = Chroma.from_documents(documents, emb, persist_directory='./chromadb_sb')
    chroma_db.persist()
    return chroma_db
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
# model = AutoModel.from_pretrained("ai-sage/Giga-Embeddings-instruct",trust_remote_code=True)
# model_kwargs = {'device': 'cuda'} 
emb = HuggingFaceE5Embeddings(model_name="intfloat/multilingual-e5-large-instruct")#intfloat/multilingual-e5-large-instruct, ai-sage/Giga-Embeddings-instruct
#загрузить эмбы
chroma_db = Chroma(persist_directory='./chromadb', embedding_function=emb)
#переделать эмбы
#chroma_db = create_emb(emb)

end_time =time.time()   
print("chroma_db_spended:", end_time-s_time)
for dataset_num in range(1,dataset_cap+1):
    qds = pd.read_csv(f'./ds{dataset_num}.csv', dtype=str, sep=';' )
    qds['relv'] = qds['relv'].apply(lambda x: list(map(int, x.split(','))))
    res = qds
    for k_c in range(K_c):
        #retriever = chroma_db.as_retriever(search_kwargs={"k": k_c+1})
        res['ret_chunks'] = res["question"].apply(lambda x: list(elem.metadata['id'] for elem in chroma_db.similarity_search(query_addon+x, k=k_c+1)))   
        if is_metric_type_mine:
            res['acc'+str(k_c+1)] = acc_top(res)
        else:
            res['accB'+str(k_c+1)] = acc_buttom(res)

    if is_metric_type_mine:
        res.to_excel(f'results/faiss_cosine/qds{dataset_num}_res.xlsx', index=False)
    else:
        res.to_excel(f'results/faiss_cosine/qds{dataset_num}_res_alt.xlsx', index=False) 


def sim_search_with_thresh(query, k = 5, thresh=-1):
    #res = chroma_db.similarity_search_with_score(query, k = k, score_threshold=thresh)#0.3 is ok
    res = chroma_db.similarity_search_with_relevance_scores(query, k = k, score_threshold = thresh)
    ret_chunks=[]
    score = []
    for elem in res:
        ret_chunks.append(elem[0].metadata['id'])
        score.append(elem[1])
    # ret_chunks = list(elem[0].metadata['id'] for elem in res)
    # score = list(elem[1] for elem in res)
    return [query, ret_chunks, score]

# result = qds["question"].apply(sim_search_with_thresh, args=(K_c, 0.8))
# result = pd.DataFrame(result.tolist(), columns = ['question', 'ret_chunks', 'score']) 

# result.to_excel('chroma_res.xlsx')
end_time =time.time()   
print("total_spended:", end_time-s_time)

