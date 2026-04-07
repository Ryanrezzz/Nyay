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
from langchain_groq import ChatGroq

def format_docs(docs):
    formatted = []
    for doc in docs:
        m = doc.metadata
        header = f"[{m.get('act','')} Section {m.get('section_number','')}] {m.get('title','')}"
        formatted.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)





def expand_query(user_query, llm):
    """Convert casual language to legal terminology."""
    prompt = f"""You are a legal language translator. 
Convert the user's everyday language into formal legal terminology 
as used in Indian criminal law (Bharatiya Nyaya Sanhita / BNS).
Rules:
- Give exactly 3 short phrases (5-10 words each)
- Use ONLY legal terminology — no section numbers, no act names
- Do NOT add any explanation or commentary
- Do NOT guess section numbers
- Focus on the CRIME described, not the punishment
User's words: "{user_query}"
"""
    response = llm.invoke(prompt).content
    expanded = [q.strip() for q in response.strip().split("\n") if q.strip()]
    print(f"[DEBUG] Expanded queries: {expanded}") 
    return [user_query] + expanded[:3]


def multi_retrieve(question, retriever, llm):
    """Search FAISS with original + expanded queries, deduplicate."""

    queries = expand_query(question, llm)
    all_docs = []
    for q in queries:
        all_docs.extend(retriever.invoke(q))
    
    seen = set()
    unique = []
    for doc in all_docs:
        key = f"{doc.metadata.get('act')}_{doc.metadata.get('section_number')}"
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    unique.sort(key=lambda d: 0 if "Bharatiya" in d.metadata.get('act','') else 1)
    return unique[:7]


def build_rag_chain():
    'Build chat based RAG chain with memory'

    load_dotenv()
    # Resolve path relative to this file (works on both local and Streamlit Cloud)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    embedding_model = GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001')
    vector_store=FAISS.load_local(
        os.path.join(BASE_DIR, 'embeddings', 'faiss_index'),
        embedding_model,
        allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 5}
    )
    llm = ChatOpenAI(
        base_url="https://api.cerebras.ai/v1",
        api_key=os.getenv("CEREBRAS_API_KEY"),
        model='qwen-3-235b-a22b-instruct-2507',
        temperature=0.1,
        max_tokens=700
    )

    expansion_llm=ChatGroq(
        model='llama-3.3-70b-versatile',
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.1,
        max_tokens=100  
    )

    prompt=ChatPromptTemplate.from_messages([
        ("system", """You are NyayBot, a legal assistant STRICTLY for Indian criminal law under IPC and BNS ONLY.
⚠️ DATABASE SCOPE — You ONLY have access to:
- Bharatiya Nyaya Sanhita (BNS), 2023 — Current criminal law (from 1 July 2024)
- Indian Penal Code (IPC), 1860 — Old criminal law (valid until 30 June 2024)
You do NOT have access to ANY other law — no Motor Vehicles Act, no Consumer Protection Act, no POCSO, no IT Act, no land law, no family law, no civil law, no insurance law.
STRICT RULES:
1. ONLY cite sections that appear in the RELEVANT LEGAL SECTIONS context below. If a section is NOT in the context, DO NOT mention it — not even if you know it exists.
2. If the user's query is about a NON-CRIMINAL matter (civil disputes, insurance claims, land disputes, divorce, consumer complaints, motor vehicle compensation, etc.), respond ONLY with: "⚠️ This query relates to [topic], which is outside our database scope. NyayBot only covers criminal law under IPC and BNS. Please consult a qualified lawyer for this matter."
3. If NO relevant sections are found in the context, say: "I could not find a matching section in our database. Please rephrase or consult a qualified lawyer."
4. If classification data (Bailable/Cognizable/Triable) is NOT explicitly shown in the context for a section, write "Not present in database" — NEVER guess or assume.
5. Always show BNS (current law) FIRST, then IPC (old law) SECOND.
6. Do NOT give advice about insurance, compensation, civil remedies, or anything outside criminal law.
FORMAT (use for EVERY criminal law answer):
📋 **Applicable Sections:**
- [Act] Section [X]: [Title from context]

⚖️ **BNS (Current Law — from 1 July 2024):**
| Field | Value |
|-------|-------|
| Section | [Number]: [Title] |
| Punishment | [ONLY from context] |
| Bailable | [From context, or "Not present in database"] |
| Cognizable | [From context, or "Not present in database"] |
| Triable by | [From context, or "Not present in database"] |

⚖️ **IPC (Old Law — until 30 June 2024):**
| Field | Value |
|-------|-------|
| Section | [Number]: [Title] |
| Punishment | [ONLY from context] |
| Bailable | [From context, or "Not present in database"] |
| Cognizable | [From context, or "Not present in database"] |
| Triable by | [From context, or "Not present in database"] |

If ONLY BNS or ONLY IPC sections are found, show only that one table.
📌 **Action:** [1-2 lines criminal law advice ONLY — no civil/insurance/compensation advice]
Keep answers CONCISE. No repetition. No guessing. No information outside the context.
RELEVANT LEGAL SECTIONS:
{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
    ])

    rag_chain=(
        RunnablePassthrough.assign(
            context=lambda x:format_docs(multi_retrieve(x['question'], retriever, expansion_llm))
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




