import os
import glob
import traceback
import logging
from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    WebBaseLoader,
    TextLoader,
)


def load_all_documents(data_dir="data", text_files=None, web_urls=None, use_ocr=True):
    """
    Loads documents from PDFs, text files, and URLs.

    Args:
        data_dir (str): Directory where PDFs are stored.
        text_files (list): List of text file paths.
        web_urls (list): List of webpage URLs.
        use_ocr (bool): Enable OCR (requires Tesseract).

    Returns:
        list: Combined list of LangChain Document objects, each with source metadata.
    """
    documents = []
    strategy = "auto" if use_ocr else "fast"

    # Suppress noisy logs
    logging.getLogger("pdfminer").setLevel(logging.ERROR)
    logging.getLogger("unstructured").setLevel(logging.ERROR)

    # 🔍 Load PDFs
    pdf_paths = glob.glob(os.path.join(data_dir, "*.pdf"))
    for pdf in pdf_paths:
        try:
            loader = UnstructuredPDFLoader(
                pdf, check_extractable=True, strategy=strategy
            )
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = os.path.basename(pdf)
            documents.extend(docs)
            print(f"✅ Loaded PDF: {pdf} ({len(docs)} chunks)")
        except Exception as e:
            print(f"⚠️ Skipped {pdf} due to error:")
            traceback.print_exc()

    # 📄 Load plain text files
    if text_files:
        for path in text_files:
            try:
                loader = TextLoader(path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = os.path.basename(path)
                documents.extend(docs)
                print(f"✅ Loaded text: {path} ({len(docs)} chunks)")
            except Exception as e:
                print(f"⚠️ Skipped {path} due to error:")
                traceback.print_exc()

    # 🌐 Load web pages
    if web_urls:
        try:
            web_loader = WebBaseLoader(web_urls)
            docs = web_loader.load()
            for doc in docs:
                doc.metadata["source"] = doc.metadata.get(
                    "source", doc.metadata.get("url", "web")
                )
            documents.extend(docs)
            print(f"✅ Loaded {len(web_urls)} web page(s)")
        except Exception as e:
            print(f"⚠️ Web loading failed:")
            traceback.print_exc()

    return documents
