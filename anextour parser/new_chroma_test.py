from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_core.documents import Document
from transformers import AutoModel
import time
import re, string
import pandas as pd
import numpy as np
import matplotlib
K_c = 10

is_metric_type_mine = False
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

def create_emb(emb):
    cds = pd.read_excel('chunked.xlsx', usecols=['id','context']).dropna()
    cds['len_c'] = cds['context'].apply(lambda x: len(x))
    documents = []
    for i in cds.iloc:
        documents.append(Document(page_content=i['context'], metadata = {"id":int(i['id'])}))
    chroma_db = Chroma.from_documents(documents, emb, persist_directory='./chromadb_new', collection_name='v_russ')
    chroma_db.persist()
    return chroma_db

def sim_search_with_thresh(query,  ensemble_retriever, k = 5, thresh=-1):
    #res = chroma_db.similarity_search_with_score(query, k = k, score_threshold=thresh)#0.3 is ok
    res = ensemble_retriever.invoke(query, k = k, score_threshold = thresh)
    ret_chunks=[]
    score = []
    for elem in res:
        ret_chunks.append(elem[0].metadata['id'])
        score.append(elem[1])
    # ret_chunks = list(elem[0].metadata['id'] for elem in res)
    # score = list(elem[1] for elem in res)
    return [query, ret_chunks, score]

questions = []
questions_bad = []
rels = []
for dataset_num in range(1,5):
    qds = pd.read_csv(f'./new_ds{dataset_num}.csv', dtype=str, sep=';' ).dropna()
    questions.extend(qds['question'].to_list())


for dataset_num in range(5,7):  
    qds = pd.read_csv(f'./new_ds{dataset_num}.csv', dtype=str, sep=';' ).dropna()
    questions_bad.extend(qds['question'].to_list())
    
s_time = time.time() 

#######___faiss___#########################
# model = AutoModel.from_pretrained("ai-sage/Giga-Embeddings-instruct",trust_remote_code=True)
model_kwargs = {'device': 'cuda'} 
emb = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct", model_kwargs=model_kwargs)#intfloat/multilingual-e5-large-instruct, ai-sage/Giga-Embeddings-instruct
#загрузить эмбы
chroma_db = Chroma(persist_directory='./chromadb', embedding_function=emb)
def tokenize(s: str) -> list[str]:
    """Очень простая функция разбития предложения на слова"""
    return s.lower().translate(str.maketrans("", "", string.punctuation)).split(" ")
def get_docs():
    cds = pd.read_excel('chunked.xlsx', usecols=['id','context']).dropna()
    cds['len_c'] = cds['context'].apply(lambda x: len(x))
    documents = []
    for i in cds.iloc:
        documents.append(Document(page_content=i['context'], metadata = {"id":i['id']}))
    return documents

#  Получение вектороного представления в тексте
documents = get_docs()

for doc in documents:
    doc.metadata['src']='bm25'
    
bm25_retriever = BM25Retriever.from_documents(
    documents=documents,
    preprocess_func=tokenize
)

threshs = np.linspace(0.7, 0.9, 21)
for k_c in range(1, K_c+1):
    retriever = chroma_db.as_retriever(search_kwargs={'k':k_c})
    bm25_retriever.k=k_c
    ensemble_retriever = EnsembleRetriever(
    retrievers=[retriever, bm25_retriever],
    # weights=[koef[0]*0.01, koef[1]*0.01],
    tags=['e5', 'bm25']
    )
    full_res = []
    for q in questions:
        res = []
        for thresh in threshs:
            _, ret_chunks, score = sim_search_with_thresh(q, ensemble_retriever, k_c, thresh)
            res.append(len(ret_chunks)/k_c)
        full_res.append(res)
    result = pd.DataFrame(full_res, index=questions, columns=threshs)
    result.to_excel(f'results/new/ans_qds_thresh_test_{k_c}_q.xlsx', index=True)
    
    full_res = []  
    for q in questions_bad:
        res = []
        for thresh in threshs:
            _, ret_chunks, score = sim_search_with_thresh(q, k_c, thresh)
            res.append((k_c-len(ret_chunks))/k_c)
        full_res.append(res)
    result = pd.DataFrame(full_res, index=questions_bad, columns=threshs)
    result.to_excel(f'results/new/ans_qds_thresh_test_{k_c}_q_bad.xlsx', index=True)

end_time =time.time()   
print("total_spended:", end_time-s_time)

