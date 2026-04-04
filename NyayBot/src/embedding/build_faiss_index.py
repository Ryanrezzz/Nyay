import json
import numpy as np
import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__=='__main__':
    load_dotenv()
    print('Load Embeddings')

    embedding_array=np.load('embeddings/section_embeddings.npy')
    print(f'Shape : {embedding_array.shape}')

    #Load Meta data
    with open('embeddings/section_metadata.json','r',encoding='utf-8') as f:
         metadata=json.load(f)
    

    #load full section of page content
    with open("data/structured_json/ipc_structured.json", "r") as f:
        ipc= json.load(f)
    with open("data/structured_json/bns_structured.json", "r") as f:
        bns= json.load(f)
    all_sections= ipc + bns

    text_embedding_pairs =[]
    metadatas=[]

    for i,section in enumerate(all_sections):
        text=f"{section['act']} Section {section['section_number']}: {section['title']}. {section['description']}"
        embedding= embedding_array[i].tolist()
        text_embedding_pairs.append((text,embedding))
        metadatas.append({
            'section_id':section['section_id'],
            'act':section['act'],
            'section_number':section['section_number'],
            'title':section['title'],
        })
    embedding_model=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    print("Building FAISS index")
    vector_store = FAISS.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=embedding_model,
        metadatas=metadatas
    )

    print("Saving FAISS index")
    vector_store.save_local('embeddings/faiss_index')

    print('Test : punishmnet for murder')
    results=vector_store.similarity_search('punishmnet for murder',k=3)
    for i,doc in enumerate(results):
        m=doc.metadata
        print(f"Result {i+1}: {m['act']} {m['section_number']}: {m['title'][:60]}")




        