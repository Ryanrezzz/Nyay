import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

with open('data/section_classification.json','r') as f:
    CLASSIFICATIONS=json.load(f)

def format_docs(docs):
    """Format retrieved docs with classification data."""
    formatted=[]
    for doc in docs:
        m= doc.metadata
        act=m.get('act',"") 
        section=m.get('section_number',"")
        title=m.get('title',"")
        
        header=f"[{act} Section {section}] {title}"
        content=doc.page_content

        prefix = "BNS" if "Bharatiya" in act else "IPC"
        key = f"{prefix}_{section}"
        if key in CLASSIFICATIONS:
            c=CLASSIFICATIONS[key]
            content+=f"\n\n⚠️ VERIFIED CLASSIFICATION (USE THESE VALUES EXACTLY):"
            content+= f"\n- Cognizable: {'Yes' if c.get('cognizable') else 'No'}"
            content+= f"\n- Bailable: {'Yes' if c.get('bailable') else 'No'}"
            content+= f"\n- Triable by: {c.get('triable_by', 'Not available')}"
            content+= f"\n- Punishment: {c.get('punishment', 'See section text')}"
            if c.get("ipc_equivalent"):
                content+= f"\n- IPC Equivalent: Section {c['ipc_equivalent']}"
            if c.get("bns_equivalent"):
                content+= f"\n- BNS Equivalent: Section {c['bns_equivalent']}"

        formatted.append(f"{header}\n{content}")
    return "\n\n---\n\n".join(formatted)

def build_rag_chain():
    'Build chat based RAG chain with memory'

    load_dotenv()
    embedding_model = GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001')
    vector_store=FAISS.load_local(
        'embeddings/faiss_index',
        embedding_model,
        allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20}
    )
    llm = ChatOpenAI(
        base_url="https://api.cerebras.ai/v1",
        api_key=os.getenv("CEREBRAS_API_KEY"),
        model='qwen-3-235b-a22b-instruct-2507',
        temperature=0.1,
        max_tokens=700 
    )

    prompt=ChatPromptTemplate.from_messages([
        ("system", """You are NyayBot, an expert legal assistant for Indian criminal law.You speak in a friendly, helpful tone — like a knowledgeable lawyer friend explaining the law in simple language.
Start by acknowledging the user's situation, then give legal details.
⚠️ DATABASE SCOPE: This system ONLY contains:
- Indian Penal Code (IPC), 1860 — Valid until 30 June 2024
- Bharatiya Nyaya Sanhita (BNS), 2023 — Valid from 1 July 2024 onwards
IMPORTANT RULES:
1. If the user's query falls under a DIFFERENT ACT (e.g., Child Labour Act, POCSO, IT Act, 
   Motor Vehicles Act, Consumer Protection Act, Property Law, Contract Law, etc.):
   - FIRST display this disclaimer:
     "⚠️ DISCLAIMER: NyayBot's database only covers IPC (1860) and BNS (2023). 
      The issue you described may fall under a different/specific Act not in our database.
      Please consult a qualified lawyer for complete legal advice."
   - THEN provide whatever relevant IPC/BNS sections might partially apply
   - THEN mention which other Act likely applies (from your knowledge)
2. If the query is purely CIVIL (rent disputes, contracts, bills):
   - State clearly: "This is a civil matter, not a criminal offence under IPC/BNS."
   - Do NOT force-fit any criminal sections
3. For EVERY criminal law answer, you MUST include this EXACT format:
📋 **Applicable Sections:**
- BNS Section [X]: [Title]
- IPC Section [X]: [Title] (old law)
⚖️ **Legal Details:**
| Field | Value |
|-------|-------|
| Punishment | [from context] |
| Bailable | Yes/No [from Classification in context] |
| Cognizable | Yes/No [from Classification in context] |
| Triable by | [from Classification in context] |
📅 **Validity:** BNS: from 1 July 2024 | IPC: until 30 June 2024
📌 **Action:** [1-2 lines practical advice]
YOU MUST FILL the Legal Details table. If classification data is in the context, USE IT.
4. BNS is the CURRENT law. Always mention BNS first, IPC second.
5. NEVER hallucinate. If unsure, say so.
6. Keep answers CONCISE — maximum 15-20 lines. No repetition.
7. When mentioning sections NOT found in our database, clearly mark them as:
   "⚠️ [NOT IN DATABASE] Section XYZ — This information is from general legal knowledge, NOT verified from our database."
8. Only cite sections from the RETRIEVED context as verified. Everything else must have the NOT IN DATABASE warning.
9. The context contains "VERIFIED CLASSIFICATION" blocks — these are from our database. 
   You MUST use these exact values for Bailable, Cognizable, Triable by, and IPC/BNS Equivalent. 
   Do NOT override them with your own knowledge.
10. For EACH applicable section, show its OWN legal details separately. 
    Do NOT mix data from different sections.
RELEVANT LEGAL SECTIONS:
{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
    ])

    rag_chain=(
        RunnablePassthrough.assign(
            context=lambda x:format_docs(retriever.invoke(x['question']))
        ) | prompt | llm | StrOutputParser()
    )

    store={}
    def get_session_history(session_id) :
        if session_id not in store:
            store[session_id]=ChatMessageHistory()
        return store[session_id]

    chain_with_history= RunnableWithMessageHistory(
        rag_chain,get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history"
    )
    return chain_with_history

if __name__ == '__main__':
    chain=build_rag_chain()
    session_id='test_session'

    while True:
        question=input("You: ")
        if question.lower() in ['exit','quit']:
            break

        response=chain.invoke(
            {'question':question},config={'configurable':{'session_id':session_id}}
        )
        print("NyayBot:",response)




