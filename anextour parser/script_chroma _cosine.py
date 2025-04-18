import chromadb.utils.embedding_functions as embedding_functions
import time
import pandas as pd
import numpy as np
import chromadb
from langchain.embeddings import HuggingFaceEmbeddings

K_c = 5
dataset_cap = 3
is_metric_type_mine = True


s_time = time.time()   
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

def create_emb(collection, emb):
    collection = client.create_collection(
        name="tours", 
        embedding_function=emb,
        metadata={
            "hnsw:space": "cosine",
            "hnsw:search_ef": 100
        }
    )
    cds = pd.read_excel('chunked.xlsx', usecols=['id','context']).dropna()
    cds['len_c'] = cds['context'].apply(lambda x: len(x))
    cds['metadata'] = cds['id'].apply(lambda x: {'id':x})
    collection.add(
        documents = cds['context'].to_list(),
        metadatas = cds['metadata'].to_list(),
        ids = [str(elem) for elem in range(len(cds['context']))]
    )
    return collection
import chromadb.utils.embedding_functions as embedding_functions
huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
    api_key=,
    #model_name="intfloat/multilingual-e5-large-instruct"
)
#emb = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")

client = chromadb.PersistentClient(path="pers_client_e5")

collection = create_emb(client, huggingface_ef)
#collection = client.get_collection(name="tours", embedding_function=emb)

for dataset_num in range(1,dataset_cap+1):
    qds = pd.read_csv(f'./ds{dataset_num}.csv', dtype=str, sep=';' )
    qds['relv'] = qds['relv'].apply(lambda x: list(map(int, x.split(','))))
    res = qds
    a = collection.query(
        query_texts=qds['question'].iloc[0],
        n_results=10,
    )
    print(a)
    # for k_c in range(K_c):
    #     #retriever = chroma_db.as_retriever(search_kwargs={"k": k_c+1})
    #     res['ret_chunks'] = res["question"].apply(lambda x: list(elem.metadata['id'] for elem in chroma_db.similarity_search(query_addon+x, k=k_c+1)))   
    #     if is_metric_type_mine:
    #         res['acc'+str(k_c+1)] = acc_top(res)
    #     else:
    #         res['accB'+str(k_c+1)] = acc_buttom(res)

    # if is_metric_type_mine:
    #     res.to_excel(f'results/faiss_cosine/qds{dataset_num}_res.xlsx', index=False)
    # else:
    #     res.to_excel(f'results/faiss_cosine/qds{dataset_num}_res_alt.xlsx', index=False) 


end_time =time.time()   
print("total_spended:", end_time-s_time)

