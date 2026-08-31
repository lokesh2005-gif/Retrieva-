
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import warnings

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import config
from ingest import ingest_pdfs
st.set_page_config(page_title="Retrieva | AI-Powered PDF Q&A Assistant", layout="centered")

# Suppress LangChain deprecation warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()

# Resolve API key: Streamlit Cloud secrets take priority, fallback to .env for local dev
groq_key = st.secrets.get("GROQ_API_KEY", None) or os.getenv("GROQ_API_KEY")
if groq_key:
    os.environ["GROQ_API_KEY"] = groq_key
else:
    st.warning("GROQ_API_KEY not set. Add it to Streamlit Cloud Secrets or your local .env file.")

# The CHROMA_DB_IMPL setting was removed because it was deprecated and causing Chroma to fail.


@st.cache_resource
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)


def get_pdf_files(pdf_directory: str):
    if not os.path.exists(pdf_directory):
        return []
    return sorted(
        filename for filename in os.listdir(pdf_directory)
        if filename.lower().endswith(".pdf")
    )


embeddings = get_embeddings_model()


def get_vector_store(embed_func):
    """Initialize Chroma with a simple, non-deprecated approach."""
    try:
        os.makedirs(config.CHROMA_PERSIST_DIRECTORY, exist_ok=True)
        
        # Simple initialization without PersistentClient wrapper
        vectordb = Chroma(
            persist_directory=config.CHROMA_PERSIST_DIRECTORY,
            embedding_function=embed_func,
            collection_name="pdf_documents",
        )
        return vectordb
    except Exception as exc:
        # Silently continue - the app will work but will show empty results until a PDF is ingested
        pass
    
    # Return None to signal empty state; the query function will handle this gracefully
    return None


vectordb = get_vector_store(embeddings)


@st.cache_resource
def get_chat_model():
    return ChatGroq(
        model="llama3-70b-8192",
        temperature=0.0,
        max_tokens=400,
        groq_api_key=groq_key,
    )


model = get_chat_model()


def call_model(state: MessagesState):
    system_prompt = (
        "You are a PDF-based assistant. Use the retrieved context to answer the question accurately. "
        "If the answer is not in the context, say you do not know. Keep the answer concise and use at most three sentences."
    )
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = model.invoke(messages)
    return {"messages": response}


@st.cache_resource
def get_langgraph_app():
    workflow = StateGraph(state_schema=MessagesState)
    workflow.add_node("model", call_model)
    workflow.add_edge(START, "model")
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


app = get_langgraph_app()


st.title("📄 Retrieva | AI-Powered PDF Q&A Assistant")

with st.sidebar:
    st.header("PDF Library")

    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
    if st.button("Upload & ingest"):
        if uploaded_pdf is not None:
            os.makedirs(config.PDF_SOURCE_DIRECTORY, exist_ok=True)
            save_path = os.path.join(config.PDF_SOURCE_DIRECTORY, uploaded_pdf.name)
            with open(save_path, "wb") as file:
                file.write(uploaded_pdf.getvalue())
            try:
                ingest_pdfs(config.PDF_SOURCE_DIRECTORY, config.CHROMA_PERSIST_DIRECTORY)
                st.success(f"'{uploaded_pdf.name}' uploaded and indexed successfully.")
                st.rerun()
            except Exception as exc:
                st.error(f"Upload failed: {exc}")
        else:
            st.warning("Choose a PDF file before uploading.")

    pdf_files = get_pdf_files(config.PDF_SOURCE_DIRECTORY)
    if pdf_files:
        pdf_options = ["All PDFs"] + pdf_files
        selected_pdf = st.selectbox("Select PDF", pdf_options, index=0)
        st.caption(f"{len(pdf_files)} PDF(s) found in {config.PDF_SOURCE_DIRECTORY}/")
    else:
        selected_pdf = "All PDFs"
        st.warning("No PDFs found in the data folder. Upload a PDF to begin.")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.pop("retrieved_docs", None)


if "messages" not in st.session_state:
    st.session_state.messages = []
if "retrieved_docs" not in st.session_state:
    st.session_state.retrieved_docs = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "streamlit_chat_session"

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.retrieved_docs:
    with st.expander("Relevant passages", expanded=False):
        for item in st.session_state.retrieved_docs:
            st.markdown(f"**{os.path.basename(item['source'])} • Page {item['page']}**")
            st.write(item["text"])
            st.divider()


def get_relevant_docs(question: str, pdf_filter: str):
    if vectordb is None:
        return []
    
    docs = vectordb.similarity_search_with_score(question, k=5)
    if pdf_filter != "All PDFs":
        docs = [
            doc for doc in docs
            if os.path.basename(doc[0].metadata.get("source", "")) == pdf_filter
        ]
    return docs[:3]


if prompt := st.chat_input("Ask a question about your PDF..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching the PDF and generating an answer..."):
            try:
                docs = get_relevant_docs(prompt, selected_pdf)
                if not docs:
                    raise ValueError(
                        "No relevant passages were found in the selected PDF set. Try a different question or choose 'All PDFs'."
                    )

                _docs = pd.DataFrame(
                    [
                        (
                            prompt,
                            doc[0].page_content,
                            doc[0].metadata.get("source"),
                            doc[0].metadata.get("page"),
                            doc[1],
                        )
                        for doc in docs
                    ],
                    columns=["query", "paragraph", "document", "page_number", "relevant_score"],
                )
                current_context = "\n\n".join(_docs["paragraph"].tolist())
                current_turn_message = HumanMessage(
                    content=f"Context: {current_context}\n\nQuestion: {prompt}"
                )

                result = app.invoke(
                    {"messages": [current_turn_message]},
                    config={"configurable": {"thread_id": st.session_state.thread_id}},
                )
                ai_response = result["messages"][-1].content

                source_document = _docs["document"].dropna().iloc[0] if not _docs.empty else "N/A"
                page_numbers = _docs["page_number"].dropna().drop_duplicates().head(3).astype(str).tolist()
                page_numbers_str = ", ".join(page_numbers) if page_numbers else "N/A"

                final_response = (
                    f"{ai_response}\n\n**Source PDF**: {os.path.basename(source_document) if source_document != 'N/A' else 'N/A'}\n"
                    f"**Reference Page Numbers**: {page_numbers_str}"
                )
                st.markdown(final_response)
                st.session_state.messages.append({"role": "assistant", "content": final_response})

                st.session_state.retrieved_docs = [
                    {
                        "source": row["document"],
                        "page": row["page_number"],
                        "text": row["paragraph"],
                    }
                    for _, row in _docs.iterrows()
                ]

                with st.expander("Relevant passages", expanded=True):
                    for item in st.session_state.retrieved_docs:
                        st.markdown(f"**{os.path.basename(item['source'])} • Page {item['page']}**")
                        st.write(item["text"])
                        st.divider()

            except Exception as exc:
                error_message = f"I encountered an issue while processing your request: {exc}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
