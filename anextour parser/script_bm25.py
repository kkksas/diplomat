
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
dataset_num = 3
is_metric_type_mine = False #переключает метрику True acc_top False - acc_buttom


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


cds = pd.read_excel('chunked.xlsx', usecols=['id','context']).dropna()
qds = pd.read_csv(f'./ds{dataset_num}.csv', dtype=str, sep=';' )
qds['relv'] = qds['relv'].apply(lambda x: list(map(int, x.split(','))))
#не оч понял, что подразумевается под контекстом в qds, видимо это чанки подходящих под вопрос описаний туров(в моем случае)
#не проще ли тогда считать точность по id описания к которому чанк принадлежит

# добавляет чанки по id описаний, указанных в файле как релевантные
#qds['context'] = qds['relv'].apply(lambda x: list(cds['context'][cds['id'].isin(x)]))


cds['len_c'] = cds['context'].apply(lambda x: len(x))


documents = []
for i in cds.iloc:
    documents.append(Document(page_content=i['context'], metadata = {"id":i['id']}))

s_time = time.time() 


#######___B25___#########################
s_time_1 = time.time()
import string
def tokenize(s: str) -> list[str]:
    """Очень простая функция разбития предложения на слова"""
    return s.lower().translate(str.maketrans("", "", string.punctuation)).split(" ")
#  Получение вектороного представления в тексте
bm25_retriever = BM25Retriever.from_documents(
    documents=documents,
    preprocess_func=tokenize,
    k=1,
)

end_time_1 = time.time()
res_bm25 = qds
for k_c in range(K_c):
    bm25_retriever.k = k_c+1
    res_bm25['ret_chunks'] = res_bm25["question"].apply(lambda x: list(elem.metadata['id'] for elem in bm25_retriever.invoke(x)))   
    if is_metric_type_mine:
        res_bm25['acc'+str(k_c+1)] = acc_top(res_bm25)
    else:
        res_bm25['accB'+str(k_c+1)] = acc_buttom(res_bm25)

if is_metric_type_mine:
    res_bm25.to_excel(f'results/bm25/qds{dataset_num}_res_bm25.xlsx', index=False)
else:
    res_bm25.to_excel(f'results/bm25/qds{dataset_num}_res_alt_bm25.xlsx', index=False) 


end_time =time.time()   
print("total_spended:", end_time-s_time)





































"""
видимо это построение эмбедингов(в лоб)
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
