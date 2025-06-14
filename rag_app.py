import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from dotenv import load_dotenv
import glob

from langchain_community.document_loaders import UnstructuredPDFLoader, WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQA

# 🔁 Load environment variables
load_dotenv()

# ✅ Make sure ANTHROPIC_API_KEY is set
if not os.getenv("ANTHROPIC_API_KEY"):
    raise ValueError("Missing ANTHROPIC_API_KEY in .env")

# 📄 Load your knowledge base
#loader = TextLoader("data/knowledge.txt")
loaders = [
    TextLoader("data/knowledge.txt"),  # 📄 Plain text
    #UnstructuredPDFLoader("data/sample.pdf"),  # 📚 PDF
    WebBaseLoader(["https://judicial.gov.gh/index.php/the-constitution"])  # 🌐 Web page
]

documents = []

pdf_paths = glob.glob("data/*.pdf")
for pdf in pdf_paths:
    loader = UnstructuredPDFLoader(pdf)
    documents.extend(loader.load())

for loader in loaders:
    documents.extend(loader.load())

# ✂️ Split text into chunks
splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = splitter.split_documents(documents)

# 🧠 Use HuggingFace embeddings (no API key needed)
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 📦 Create vector DB
vectordb = Chroma.from_documents(docs, embedding=embedding)

# 🔍 Setup retriever
retriever = vectordb.as_retriever()

# 🤖 Use Claude (Anthropic) for answers
llm = ChatAnthropic(
    model="claude-3-haiku-20240307",
    temperature=0.3
)

# 🔗 Connect retriever and LLM into a chain
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 🎤 Ask your questions
print("Ask your questions (type 'exit' to quit):\n")
while True:
    query = input(">> ")
    if query.lower() in ["exit", "quit"]:
        break
    response = qa.invoke(query)
    print(f"\n🧠 Answer: {response}\n")
