import streamlit as st
import torch
from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set page configuration
st.set_page_config(page_title="Ahmad Malik's AI Assistant", page_icon="🤖")

# --- 1. Load and Process the PDF (Runs once on app startup) ---
@st.cache_resource
def load_rag_pipeline():
    # Robust path finding: gets the directory where app.py lives
    script_dir = Path(__file__).parent
    pdf_path = script_dir / "Ahmad-Malik-Portfolio-RAG-Knowledge-Base.pdf"

    # Check if the file exists before trying to read it
    if not pdf_path.exists():
        st.error(f"🚨 CRITICAL ERROR: Could not find the PDF file at {pdf_path}")
        st.error("Please make sure the PDF is uploaded to GitHub and the filename matches exactly.")
        st.stop() # Stops the app if the file is missing

    # Read PDF
    reader = PdfReader(str(pdf_path))
    raw_text = "".join([page.extract_text() + "\n" for page in reader.pages])
    
    # Clean text (Remove "NOT FOUND" lines)
    cleaned_lines = [line for line in raw_text.split('\n') if "NOT FOUND" not in line]
    raw_text = "\n".join(cleaned_lines)
    
    # Chunk text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_text(raw_text)
    
    # Build Hybrid Search (Semantic + Keyword)
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    
    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    keyword_retriever = BM25Retriever.from_texts(chunks)
    keyword_retriever.k = 5
    
    hybrid_retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, keyword_retriever], weights=[0.5, 0.5]
    )
    
    # Load Model (CPU Optimized, 0.5B model for free tier)
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cpu")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return hybrid_retriever, tokenizer, model

# Load everything (this shows a spinner on the first load)
with st.spinner("Loading AI Model and Portfolio Data... This takes a minute on first start."):
    hybrid_retriever, tokenizer, model = load_rag_pipeline()

# --- 2. Chat Function ---
def ask_portfolio_free(question):
    docs = hybrid_retriever.invoke(question)
    unique_docs = list({doc.page_content: doc for doc in docs}.values())
    context = "\n\n".join([doc.page_content for doc in unique_docs[:10]])

    system_prompt = """You are Muhammad Ahmad Malik's professional AI assistant. 
    Answer the user's question accurately using ONLY the facts in the Context provided below.

    STRICT RULES:
    1. WORK EXPERIENCE: If asked about work experience or journey, look for the section '4A. WORK EXPERIENCE'. You MUST pair the exact Job Title with the exact Company. 
       - Example format: 
         * **Full Stack AI Engineer** at Anma Tech (Dates)
         * **Website Development Intern** at Growstep Technologies (Dates)
         * **UI/UX Designer** at Freelance (Dates)
         * **Freelance Full-Stack Developer** at Freelance (Dates)
       DO NOT include Education (Iqra University) or Hackathons in the Work Experience list.
    2. If asked about EDUCATION or HACKATHONS, list them separately under their own heading.
    3. TECH STACK: Look for '3. SKILLS & TECH STACK' and list the categories.
    4. Be descriptive, professional, and use Markdown formatting (bolding, bullet points).
    5. If the answer is truly not in the context, say "I don't have that information." """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

# --- 3. Streamlit UI ---
st.title("🤖 Ahmad Malik's AI Assistant")
st.caption("Ask me anything about Ahmad's experience, projects, or tech stack!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Type your question here..."):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get bot response
    with st.spinner("Bot is thinking..."):
        response = ask_portfolio_free(prompt)
        
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
