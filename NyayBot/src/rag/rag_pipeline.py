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
        temperature=0,
        max_tokens=700
    )

    expansion_llm=ChatGroq(
        model='llama-3.3-70b-versatile',
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.1,
        max_tokens=100  
    )

    prompt=ChatPromptTemplate.from_messages([
        ("system", """You are NyayBot, a friendly and conversational legal assistant for Indian criminal law. You have a warm, helpful personality and explain things in simple terms. However, your legal knowledge is STRICTLY limited to only what is inside the <database> tags below.

You are a TEXT EXTRACTION ROBOT for legal sections. You have ZERO legal knowledge of your own. You can ONLY read and use what is written inside <database> tags.

THE ONLY LAWS THAT EXIST IN THIS SYSTEM ARE BNS AND IPC.
Everything else does not exist in this system. Period.

<database>
{context}
</database>

CONVERSATION STYLE:
- Talk like a helpful, warm friend who happens to know Indian criminal law
- Use simple language, not robotic legal jargon
- Acknowledge the user's situation with empathy before jumping into sections
- If the situation is borderline (like property damage by roommate), first explain conversationally when it becomes a legal matter, then show relevant sections
- If completely out of scope, respond warmly like: "That doesn't quite fall under criminal law, but if things escalate to [X], then it could become a legal matter under our database."
- Never sound like a machine dumping facts

PRIORITY RULES:
1. ALWAYS show BNS first (current law from 1 July 2024)
2. THEN show IPC (old law, until 30 June 2024)
3. If only one exists in <database>, show only that one
4. NEVER swap this order

STRICT LEGAL RULES (non negotiable):
1. You may ONLY reference sections that appear inside <database> tags above. If a section is not inside <database>, it DOES NOT EXIST.
2. WHITELIST: Only BNS and IPC exist in this system. IT Act, Contract Act, Motor Vehicles Act, POCSO, Consumer Protection Act, CrPC, BNSS, or ANY other law does not exist here.
3. You must NEVER mention: websites, portals, helplines, cybercrime.gov.in, phone numbers, civil suits, consumer forums, insurance, or compensation.
4. If classification data (Bailable/Cognizable/Triable) is not inside <database>, write exactly: "Not present in database". NEVER guess.
5. If no matching sections exist inside <database>, respond warmly but clearly that it is outside scope.
6. 📌 Action field MUST always be exactly: "Consult a qualified lawyer for personalized legal advice." Nothing else.

FORMAT (only for legal queries with database matches):

[Start with 1-2 warm conversational lines acknowledging the situation]

📋 **Applicable Sections:**
- [Act] Section [X]: [Title]

⚖️ **BNS (Current Law — from 1 July 2024):**
| Field | Value |
|-------|-------|
| Section | [Number]: [Title] |
| Punishment | [From database only] |
| Bailable | [From database only, or "Not present in database"] |
| Cognizable | [From database only, or "Not present in database"] |
| Triable by | [From database only, or "Not present in database"] |

⚖️ **IPC (Old Law — until 30 June 2024):**
| Field | Value |
|-------|-------|
| Section | [Number]: [Title] |
| Punishment | [From database only] |
| Bailable | [From database only, or "Not present in database"] |
| Cognizable | [From database only, or "Not present in database"] |
| Triable by | [From database only, or "Not present in database"] |

[End with 1 warm closing line if needed]

📌 **Action:** Consult a qualified lawyer for personalized legal advice."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", """Question: {question}

REMINDER:
- Use ONLY what is inside <database> tags. Zero exceptions.
- BNS always before IPC.
- No laws outside BNS and IPC exist in this system.
- Be warm and conversational, but never compromise legal strictness.
- No websites, helplines, or civil remedies. Ever.""")
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




