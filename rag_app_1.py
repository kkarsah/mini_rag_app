import os
import warnings
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from rag_loader import load_all_documents
from langchain_anthropic import ChatAnthropic
from transformers import logging as hf_logging
from langchain.text_splitter import TokenTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

# ✅ Check API key
if not os.getenv("ANTHROPIC_API_KEY"):
    raise ValueError("Missing ANTHROPIC_API_KEY in .env")

# 📚 Load docs
documents = load_all_documents(
    data_dir="data",
    text_files=["data/knowledge.txt"],
    web_urls=["https://judicial.gov.gh/index.php/the-constitution"],
    use_ocr=False,  # Set to True if you have Tesseract installed
)

# 🔍 Optional: Debug document metadata
for doc in documents:
    print(doc.metadata)

# ✂️ Split text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", ".", " ", ""]
)
docs = splitter.split_documents(documents)


# 🔍 Embed and store
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma.from_documents(docs, embedding=embedding)
retriever = vectordb.as_retriever()

# 🤖 Setup LLM
llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.3)
qa = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)

# 💬 Ask questions
print("Ask your questions (type 'exit' to quit):\n")
while True:
    query = input(">> ")
    if query.lower() in ["exit", "quit"]:
        break

    result = qa.invoke({"question": query})
    answer = result["answer"]
    sources = result.get("sources", "None")

    print(f"\n🧠 Answer:\n{answer}\n")

    if sources and sources != "None":
        print("📚 Sources:")
        for src in sources.split(", "):
            print("-", src)
    else:
        print("📚 Sources: None")
