import streamlit as st
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from src.rag.rag_pipeline import build_rag_chain

st.set_page_config(page_title='NyayBot',page_icon='⚖️',layout='centered')
st.title("⚖️ NyayBot")
st.caption("Your AI Legal Assistant for Indian Criminal Law (IPC & BNS)")

@st.cache_resource
def get_chain():
    return build_rag_chain()

chain=get_chain()


if 'messages' not in st.session_state:
    st.session_state.messages=[]
if 'session_id' not in st.session_state:
    st.session_state.session_id = "streamlit_session"


for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

if question := st.chat_input('Ask about any criminal law situation...'):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching legal database..."):
            response = chain.invoke(
                {"question": question},
                config={"configurable": {"session_id": st.session_state.session_id}}
            )
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

with st.sidebar:
    st.header("📚 About NyayBot")
    st.markdown("""
    **Database Coverage:**
    - 🟢 BNS 2023 (358 sections) — Current law
    - 🔵 IPC 1860 (535 sections) — Old law
    
    **Features:**
    - ✅ Verified bailable/cognizable classification
    - ✅ IPC ↔ BNS section mapping
    - ✅ Multi-turn conversation
    
    **⚠️ Disclaimer:**
    NyayBot provides legal information, NOT legal advice. 
    Always consult a qualified lawyer.
    """)
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()