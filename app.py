import gradio as gr
import torch
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading PDF and Building Search Engine...")
# 1. Load PDF
reader = PdfReader("Ahmad-Malik-Portfolio-RAG-Knowledge-Base.pdf")
raw_text = "".join([page.extract_text() + "\n" for page in reader.pages])

# Clean NOT FOUND lines
cleaned_lines = [line for line in raw_text.split('\n') if "NOT FOUND" not in line]
raw_text = "\n".join(cleaned_lines)

# 2. Chunk Text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = text_splitter.split_text(raw_text)

# 3. Build Hybrid Search
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vectorstore = FAISS.from_texts(chunks, embeddings)
semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
keyword_retriever = BM25Retriever.from_texts(chunks)
keyword_retriever.k = 5
hybrid_retriever = EnsembleRetriever(retrievers=[semantic_retriever, keyword_retriever], weights=[0.5, 0.5])

# 4. Load Model (CPU Optimized)
print("Loading 1.5B Model...")
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cpu")

# 5. Chat Function
def chat_with_bot(question, history):
    docs = hybrid_retriever.invoke(question)
    unique_docs = list({doc.page_content: doc for doc in docs}.values())
    context = "\n\n".join([doc.page_content for doc in unique_docs[:10]])

    system_prompt = """You are Muhammad Ahmad Malik's professional AI assistant. 
    Answer the user's question accurately using ONLY the facts in the Context provided below.

    STRICT RULES:
    1. If asked about EXPERIENCE, list EVERY role (Job Title, Company, Dates) as bullet points.
    2. If asked about TECH STACK, list skills under categories (Frontend, Backend, etc.).
    3. If asked about PROJECTS, list names, tech stacks, and descriptions.
    4. Be descriptive, professional, and format with Markdown.
    5. If the answer is not in the context, say "I don't have that information." """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]
    
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=350,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = outputs[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(response, skip_special_tokens=True)
    
    history.append((question, answer))
    return "", history

# 6. Gradio UI
with gr.Blocks(title="Ahmad Malik's AI") as demo:
    gr.Markdown("# 🤖 Ahmad Malik's AI Assistant")
    gr.Markdown("Ask me anything about Ahmad's experience, projects, or tech stack!")
    
    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(placeholder="Type your question here...", label="Your Question")
    
    msg.submit(chat_with_bot, [msg, chatbot], [msg, chatbot])
    gr.Button("Clear").click(lambda: None, None, chatbot, queue=False)

demo.launch()