import json
import numpy as np 
import os 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv



def load_sections(file_path):
    """Load structured sections from a JSON file"""
    
    with open(file_path,'r',encoding='utf-8') as f:
        sections = json.load(f)
    print(f"Loaded{len(sections)} sections from {file_path}")
    return sections

def prepare_text(sections):
    """Combine title+description into one searchable string per section """
    
    texts =[]

    for s in sections:
        text=f'''{s['act']} Section {s['section_number']}: {s['title']}. {s['description']}'''
        if s.get('bailable'):
            text += f"\nBailable: {s['bailable']}. Cognizable: {s['cognizable']}. Triable by: {s.get('triable_by', 'N/A')}."

        texts.append(text)
    return texts


if __name__ == '__main__':
    load_dotenv()

    embedding_model= GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    #load all sections
    ipc_sections = load_sections('data/structured_json/ipc_structured.json')
    bns_sections = load_sections('data/structured_json/bns_structured.json')
    all_sections = ipc_sections+ bns_sections
    print(f"Total sections {len(all_sections)}")

    text = prepare_text(all_sections)
    
    print(f"Generate Embeddings")

    import time

# Process in batches of 90 to stay under 100/min limit
    vectors = []
    batch_size = 90

    for i in range(0, len(text), batch_size):
        batch = text[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(text) + batch_size - 1) // batch_size
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} texts)...")
    
        batch_vectors = embedding_model.embed_documents(batch)
        vectors.extend(batch_vectors)
    
    # Wait 61 seconds between batches to reset the rate limit
        if i + batch_size < len(text):
            print("  Waiting 61 seconds for rate limit reset...")
            time.sleep(61)

    
    # convert to numpy arrays
    embeddings=np.array(vectors,dtype=np.float32)
    print(f" Embeddings shape: {embeddings.shape}")

    #save embeddings
    os.makedirs("data/embeddings",exist_ok=True)
    np.save('embeddings/section_embeddings.npy',embeddings)
    print(f"Saved embeddings to data/embeddings/section_embeddings.npy")


    metadata=[]

    for s in  all_sections:
        metadata.append({
            "section_id": s["section_id"],
            "act": s["act"],
            "section_number": s["section_number"],
            "title": s["title"]
        })

    with open("embeddings/section_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


    print(f"\n Done! {len(embeddings)} embeddings saved")
