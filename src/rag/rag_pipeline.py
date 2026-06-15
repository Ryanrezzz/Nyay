import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import re
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

def load_classification_data():
    """Load classification metadata (bailable, cognizable, triable_by, punishment) from JSON."""
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    classification_path = os.path.join(BASE_DIR, 'data', 'section_classification.json')
    try:
        with open(classification_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[WARN] Classification file not found: {classification_path}")
        return {}

# Load once at module level
CLASSIFICATION_DATA = load_classification_data()


def format_docs(docs):
    formatted = []
    for doc in docs:
        m = doc.metadata
        act = m.get('act', '')
        sec_num = m.get('section_number', '')
        header = f"[{act} Section {sec_num}] {m.get('title','')}"
        body = doc.page_content

        # Enrich with classification data from the JSON lookup
        cls_key = f"{act}_{sec_num}"
        cls = CLASSIFICATION_DATA.get(cls_key, {})

        classification_lines = []
        for field in ['punishment', 'bailable', 'cognizable', 'triable_by']:
            val = cls.get(field)
            if val is not None:
                if isinstance(val, bool):
                    val = "Yes" if val else "No"
                classification_lines.append(f"{field.replace('_',' ').title()}: {val}")

        if classification_lines:
            body += "\n📊 Classification: " + " | ".join(classification_lines)

        formatted.append(f"{header}\n{body}")
    return "\n\n---\n\n".join(formatted)





_BULLET_RE = re.compile(r'^[\s\-\*•‣◦\d\.\)\(:"\']+')


def _clean_expansion_line(line):
    """Strip leading bullets, numbering, and surrounding quotes from one expansion line."""
    line = _BULLET_RE.sub('', line).strip()
    return line.strip('"\'').strip()


def expand_query(user_query, llm):
    """Convert casual language to legal terminology.

    Robust to models that wrap phrases in bullets/numbering or add a preamble,
    and never lets an expansion failure break retrieval — it falls back to the
    original query alone.
    """
    prompt = f"""You are a legal language translator.
Convert the user's everyday language into formal legal terminology
as used in Indian criminal law (Bharatiya Nyaya Sanhita / BNS).
Rules:
- Give exactly 3 short phrases (5-10 words each)
- Output ONLY the 3 phrases, one per line — no preamble, no numbering
- Use ONLY legal terminology — no section numbers, no act names
- Do NOT add any explanation or commentary
- Do NOT guess section numbers
- Cover ALL distinct offenses in the scenario (e.g. if someone kills during theft, cover BOTH the killing AND the theft/robbery)
- Focus on the CRIMES described, not the punishment
User's words: "{user_query}"
"""
    try:
        response = llm.invoke(prompt).content
    except Exception as e:
        print(f"[WARN] Query expansion failed, using original query only: {e}")
        return [user_query]

    expanded = []
    for line in response.strip().split("\n"):
        cleaned = _clean_expansion_line(line)
        # Drop blanks, over-long lines, and preamble like "Here are 3 phrases:"
        if not cleaned or len(cleaned.split()) > 12 or ':' in cleaned:
            continue
        expanded.append(cleaned)

    print(f"[DEBUG] Expanded queries: {expanded[:3]}")
    return [user_query] + expanded[:3]


def get_all_docs(vector_store):
    """Return every Document stored in the FAISS index (used for direct section lookup)."""
    try:
        return list(vector_store.docstore._dict.values())
    except Exception as e:
        print(f"[WARN] Could not read FAISS docstore: {e}")
        return []


def build_section_index(corpus_docs):
    """Map (act, normalized_section_number) -> Document for O(1) direct lookups."""
    index = {}
    for doc in corpus_docs:
        act = doc.metadata.get('act')
        sec = str(doc.metadata.get('section_number', '')).upper().replace(' ', '')
        if act and sec:
            index[(act, sec)] = doc
    return index


class HybridRetriever:
    """Reciprocal-rank fusion of a lexical (BM25) and a dense (FAISS) retriever.

    Self-contained so we don't depend on EnsembleRetriever, whose import path
    moved between langchain 0.x and 1.x. Exposes ``.invoke(query)`` so it drops
    into ``multi_retrieve`` like any other retriever.
    """

    def __init__(self, lexical, dense, k=10, c=60):
        self.lexical = lexical
        self.dense = dense
        self.k = k
        self.c = c

    @staticmethod
    def _key(doc):
        return f"{doc.metadata.get('act')}_{doc.metadata.get('section_number')}"

    def invoke(self, query):
        scores, docs = {}, {}
        for retriever in (self.lexical, self.dense):
            try:
                results = retriever.invoke(query)
            except Exception as e:
                print(f"[WARN] {type(retriever).__name__} failed: {e}")
                continue
            for rank, doc in enumerate(results):
                key = self._key(doc)
                scores[key] = scores.get(key, 0.0) + 1.0 / (self.c + rank)
                docs.setdefault(key, doc)
        ranked = sorted(docs.values(), key=lambda d: scores[self._key(d)], reverse=True)
        return ranked[:self.k]


_ACT_ALIASES = [
    (re.compile(r'\bindian penal code\b', re.I), 'ipc'),
    (re.compile(r'\bbharatiya nyaya sanhita\b', re.I), 'bns'),
    (re.compile(r'\bi\.\s?p\.\s?c\.?', re.I), 'ipc'),
    (re.compile(r'\bb\.\s?n\.\s?s\.?', re.I), 'bns'),
]
_SEP = r'(?:[\s.,:#-]|§|sections?|secs?|number|under|of|the|no|s){0,12}'
_ACT_NUM = re.compile(r'\b(ipc|bns)' + _SEP + r'(\d{1,3}[a-z]?)\b', re.I)
_NUM_ACT = re.compile(r'\b(\d{1,3}[a-z]?)' + _SEP + r'(ipc|bns)\b', re.I)
_SECTION_NUM = re.compile(r'(?:\bsections?|\bsecs?|\bs\.|§)\s*(\d{1,3}[a-z]?)\b', re.I)


def find_section_refs(query):
    """Detect explicit section references like 'IPC 302', 'section 420', or 'BNS 103'.

    Returns (act_or_None, section_number) tuples; act is None for a bare section
    number with no act, meaning it should be looked up in both statutes.
    """
    q = query
    for pattern, repl in _ACT_ALIASES:
        q = pattern.sub(repl, q)
    q = q.lower()

    refs = []
    for m in _ACT_NUM.finditer(q):
        refs.append((m.group(1).upper(), m.group(2).upper()))
    for m in _NUM_ACT.finditer(q):
        refs.append((m.group(2).upper(), m.group(1).upper()))
    numbered = {s for _, s in refs}
    for m in _SECTION_NUM.finditer(q):
        sec = m.group(1).upper()
        if sec not in numbered:
            refs.append((None, sec))

    seen, out = set(), []
    for ref in refs:
        if ref not in seen:
            seen.add(ref)
            out.append(ref)
    return out


def lookup_sections(refs, section_index):
    """Resolve detected section references to Documents from the FAISS index."""
    docs = []
    for act, sec in refs:
        for a in ([act] if act else ['BNS', 'IPC']):
            doc = section_index.get((a, sec))
            if doc is not None:
                docs.append(doc)
    return docs


def maybe_rerank(query, docs, top_n=10):
    """Optionally rerank candidates with Cohere when COHERE_API_KEY is set.

    No-op when the key or package is unavailable, so the default deployment needs
    no extra dependency. Install ``langchain-cohere`` and set COHERE_API_KEY to enable.
    """
    if not docs or not os.getenv("COHERE_API_KEY"):
        return docs
    try:
        from langchain_cohere import CohereRerank
        reranker = CohereRerank(model="rerank-english-v3.0", top_n=min(top_n, len(docs)))
        return list(reranker.compress_documents(docs, query))
    except Exception as e:
        print(f"[WARN] Rerank skipped: {e}")
        return docs


def multi_retrieve(question, retriever, llm, section_index=None):
    """Search with expanded queries, pin explicitly-referenced sections, deduplicate."""

    # Pin sections the user named directly (e.g. "IPC 302") so they survive ranking.
    pinned = lookup_sections(find_section_refs(question), section_index) if section_index else []

    queries = expand_query(question, llm)
    retrieved = []
    for q in queries:
        retrieved.extend(retriever.invoke(q))
    
    seen = set()
    unique = []
    for doc in pinned + retrieved:
        key = f"{doc.metadata.get('act')}_{doc.metadata.get('section_number')}"
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    unique = maybe_rerank(question, unique)
    # Keep explicitly-requested sections on top, then BNS (current law) before IPC (old law).
    pinned_keys = {f"{d.metadata.get('act')}_{d.metadata.get('section_number')}" for d in pinned}
    def _order(d):
        key = f"{d.metadata.get('act')}_{d.metadata.get('section_number')}"
        return (0 if key in pinned_keys else 1, 0 if d.metadata.get('act') == 'BNS' else 1)
    unique.sort(key=_order)
    return unique[:10]


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
    dense_retriever = vector_store.as_retriever(
        search_kwargs={"k": 7}
    )
    corpus_docs = get_all_docs(vector_store)
    section_index = build_section_index(corpus_docs)

    # Hybrid retrieval: lexical BM25 (catches exact legal terms) + dense FAISS (semantics).
    if corpus_docs:
        bm25_retriever = BM25Retriever.from_documents(corpus_docs)
        bm25_retriever.k = 7
        retriever = HybridRetriever(bm25_retriever, dense_retriever, k=10)
    else:
        retriever = dense_retriever
    llm = ChatOpenAI(
        base_url="https://api.cerebras.ai/v1",
        api_key=os.getenv("CEREBRAS_API_KEY"),
        model='gpt-oss-120b',
        temperature=0,
        max_tokens=4096
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
5. In Indian law, the definition of an offence and its punishment are in SEPARATE sections (e.g. BNS 101 defines Murder, BNS 103 has the punishment). Show BOTH sections separately if they exist in <database>.

OFFENCE CLASSIFICATION RULES (apply these BEFORE picking sections):
1. THEFT and ROBBERY apply ONLY to MOVABLE property (money, phones, vehicles, jewellery, goods). They can NEVER apply to land, a house, a plot, or any immovable property.
2. If someone FORCIBLY occupies, grabs, or refuses to vacate LAND or a HOUSE, that is NOT theft or robbery. Treat it as CRIMINAL TRESPASS / house-trespass (BNS 329–333) and, if threats or force to scare are involved, CRIMINAL INTIMIDATION (BNS 351). Reference these sections only if they appear inside <database>.
3. A pure OWNERSHIP / TITLE DISPUTE over property (who legally owns it) is NOT a criminal matter on its own. Say so warmly, and explain it only becomes criminal if it escalates to trespass, intimidation, or force — then show those sections.
4. Stay 100% within BNS/IPC criminal-law scope. Do NOT mention the IT Act, civil suits, partition suits, property/title litigation, or any non-criminal remedy.

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

⚖️ **BNS — Definition (Current Law — from 1 July 2024):**
| Field | Value |
|-------|-------|
| Section | [Number]: [Title] |
| Bailable | [From database only, or "Not present in database"] |
| Cognizable | [From database only, or "Not present in database"] |
| Triable by | [From database only, or "Not present in database"] |

⚖️ **BNS — Punishment:**
| Field | Value |
|-------|-------|
| Section | [Punishment section number]: [Title] |
| Punishment | [From database only, or "Not present in database"] |

⚖️ **IPC (Old Law — until 30 June 2024):**
| Field | Value |
|-------|-------|
| Section | [Number]: [Title] |
| Punishment | [From database only, or "Not present in database"] |
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
            context=lambda x:format_docs(multi_retrieve(x['question'], retriever, expansion_llm, section_index))
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




