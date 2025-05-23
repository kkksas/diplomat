
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain_core.documents import Document
import time
import pandas as pd
import numpy as np
import matplotlib
from transformers import AutoModel
K_c = 5
dataset_cap = 3
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
# qds = pd.read_csv(f'./ds{dataset_num}.csv', dtype=str, sep=';' )
# qds['relv'] = qds['relv'].apply(lambda x: list(map(int, x.split(','))))
#не оч понял, что подразумевается под контекстом в qds, видимо это чанки подходящих под вопрос описаний туров(в моем случае)
#не проще ли тогда считать точность по id описания к которому чанк принадлежит

# добавляет чанки по id описаний, указанных в файле как релевантные
#qds['context'] = qds['relv'].apply(lambda x: list(cds['context'][cds['id'].isin(x)]))


cds['len_c'] = cds['context'].apply(lambda x: len(x))


documents = []
for i in cds.iloc:
    documents.append(Document(page_content=i['context'], metadata = {"id":i['id']}))

s_time = time.time() 

#######___faiss___#########################

model = AutoModel.from_pretrained("ai-sage/Giga-Embeddings-instruct",trust_remote_code=True)
model_kwargs = {'device': 'cuda'} 
emb = HuggingFaceE5Embeddings(model_name="ai-sage/Giga-Embeddings-instruct", model_kwargs = {'device': 'cuda'} )
#загрузить эмбы
#faiss_db = FAISS.load_local("index", embeddings=emb, allow_dangerous_deserialization=True)
#переделать эмбы
print('ems')
faiss_db = FAISS.from_documents(documents, embedding=emb)
faiss_db.save_local("index_gigachat")
end_time =time.time()   
print("faiss_db_spended:", end_time-s_time)

#res = qds[qds['relv'].apply(len)>1]
for dataset_num in range(1,dataset_cap+1):
    qds = pd.read_csv(f'./ds{dataset_num}.csv', dtype=str, sep=';' )
    qds['relv'] = qds['relv'].apply(lambda x: list(map(int, x.split(','))))
    res = qds
    for k_c in range(K_c):
        retriever = faiss_db.as_retriever(search_kwargs={"k": k_c+1})
        res['ret_chunks'] = res["question"].apply(lambda x: list(elem.metadata['id'] for elem in retriever.invoke(x)))   
        if is_metric_type_mine:
            res['acc'+str(k_c+1)] = acc_top(res)
        else:
            res['accB'+str(k_c+1)] = acc_buttom(res)

    if is_metric_type_mine:
        res.to_excel(f'res_sbert/faiss/qds{dataset_num}_res.xlsx', index=False)
    else:
        res.to_excel(f'res_sbert/faiss/qds{dataset_num}_res_alt.xlsx', index=False) 


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
