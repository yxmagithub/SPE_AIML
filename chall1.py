import warnings
warnings.filterwarnings('ignore')

import os
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_chroma import Chroma
import shutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from textwrap import shorten

HF_TOKEN = None
userdata = None
try:
    from google.colab import userdata as colab_userdata
    userdata = colab_userdata
    HF_TOKEN = userdata.get('HF_TOKEN')
    if HF_TOKEN:
        os.environ['HF_TOKEN'] = HF_TOKEN
        print("✓ Loaded HF_TOKEN from Colab secrets")
except Exception:
    HF_TOKEN = os.getenv('HF_TOKEN')
    if HF_TOKEN:
        print("✓ Loaded HF_TOKEN from environment")
if not HF_TOKEN:
    print("ℹ No HF_TOKEN found (optional for public models)")

MODEL_EMBED = "sentence-transformers/all-MiniLM-L6-v2"
LLM_NAME = "microsoft/Phi-3-mini-4k-instruct"
WORKDIR = Path.cwd()
if 'SpeEuGeoH2025' in str(WORKDIR):
    parts = WORKDIR.parts
    idx = parts.index('SpeEuGeoH2025')
    PROJECT_ROOT = Path(*parts[:idx + 1])
else:
    PROJECT_ROOT = WORKDIR.parent if WORKDIR.name == 'notebooks' else WORKDIR
WELL_DATA_DIR = PROJECT_ROOT / "data" / "Training data-shared with participants" / "Well 1"
DB_DIR = PROJECT_ROOT / "notebooks" / "local_data" / "well1_vector_db"
DB_DIR.parent.mkdir(parents=True, exist_ok=True)
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300
TOP_K = 6
print("Configuration:")
print(f"  Project Root: {PROJECT_ROOT}")
print(f"  Well 1 Data: {WELL_DATA_DIR}")
print(f"  Vector DB: {DB_DIR}")
print(f"  Embedding Model: {MODEL_EMBED}")
print(f"  LLM: {LLM_NAME}")
print(f"  Chunk Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")
if not WELL_DATA_DIR.exists():
    print(f"\n⚠️ WARNING: Well 1 data directory not found at {WELL_DATA_DIR}")
    print("Please ensure you're running from the correct location or update WELL_DATA_DIR")
    raw_documents: List[Document] = []
else:
    print(f"\n✓ Well 1 data directory found")
    raw_documents: List[Document] = []


def load_pdf(file_path: Path) -> List[Document]:
    try:
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()
        for doc in docs:
            doc.metadata['source'] = file_path.name
            doc.metadata['file_type'] = 'pdf'
            doc.metadata['directory'] = file_path.parent.name
        return docs
    except Exception as exc:
        print(f"Error loading PDF {file_path.name}: {exc}")
        return []


def load_excel(file_path: Path) -> List[Document]:
    try:
        excel_file = pd.ExcelFile(file_path)
        docs: List[Document] = []
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            text_content = f"Sheet: {sheet_name}\n\n"
            text_content += f"Columns: {', '.join(df.columns.tolist())}\n\n"
            text_content += df.to_string(index=False)
            doc = Document(
                page_content=text_content,
                metadata={
                    'source': file_path.name,
                    'sheet_name': sheet_name,
                    'file_type': 'excel',
                    'directory': file_path.parent.name,
                    'rows': len(df),
                    'columns': len(df.columns)
                }
            )
            docs.append(doc)
        return docs
    except Exception as exc:
        print(f"Error loading Excel {file_path.name}: {exc}")
        return []


def load_word(file_path: Path) -> List[Document]:
    try:
        loader = UnstructuredWordDocumentLoader(str(file_path))
        docs = loader.load()
        for doc in docs:
            doc.metadata['source'] = file_path.name
            doc.metadata['file_type'] = 'docx'
            doc.metadata['directory'] = file_path.parent.name
        return docs
    except Exception as exc:
        print(f"Error loading Word {file_path.name}: {exc}")
        return []


def load_all_documents(well_dir: Path) -> List[Document]:
    all_docs: List[Document] = []
    print(f"\nScanning {well_dir} for documents...")
    print("=" * 60)
    pdf_files = list(well_dir.rglob("*.pdf"))
    excel_files = list(well_dir.rglob("*.xlsx")) + list(well_dir.rglob("*.xls"))
    word_files = list(well_dir.rglob("*.docx")) + list(well_dir.rglob("*.doc"))
    print(f"Found: {len(pdf_files)} PDFs, {len(excel_files)} Excel files, {len(word_files)} Word docs")
    print()
    for pdf_file in pdf_files:
        print(f"Loading PDF: {pdf_file.name}")
        docs = load_pdf(pdf_file)
        all_docs.extend(docs)
    for excel_file in excel_files:
        print(f"Loading Excel: {excel_file.name}")
        docs = load_excel(excel_file)
        all_docs.extend(docs)
    for word_file in word_files:
        print(f"Loading Word: {word_file.name}")
        docs = load_word(word_file)
        all_docs.extend(docs)
    print()
    print("=" * 60)
    print(f"✓ Total documents loaded: {len(all_docs)}")
    return all_docs


if WELL_DATA_DIR.exists():
    raw_documents = load_all_documents(WELL_DATA_DIR)
    print(f"\nSuccessfully loaded {len(raw_documents)} document sections from Well 1")
else:
    print("\n⚠️ Cannot load documents - Well 1 directory not found")
    raw_documents = []

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = text_splitter.split_documents(raw_documents)
print("Document Chunking Summary:")
print(f"  Original documents: {len(raw_documents)}")
print(f"  Total chunks created: {len(chunks)}")
print(f"  Average chunks per document: {len(chunks) / max(len(raw_documents), 1):.1f}")
if chunks:
    print("\nSample chunk metadata:")
    sample = chunks[0]
    print(f"  Source: {sample.metadata.get('source', 'N/A')}")
    print(f"  Type: {sample.metadata.get('file_type', 'N/A')}")
    print(f"  Directory: {sample.metadata.get('directory', 'N/A')}")
    print(f"  Content preview: {sample.page_content[:200]}...")

print("Initializing embedding model...")
embeddings = HuggingFaceEmbeddings(model_name=MODEL_EMBED)
print(f"✓ Loaded embedding model: {MODEL_EMBED}")
if DB_DIR.exists():
    print(f"Removing existing vector database at {DB_DIR}...")
    shutil.rmtree(DB_DIR)
print(f"\nCreating vector store with {len(chunks)} chunks...")
print("This may take a few minutes...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=str(DB_DIR)
)
print(f"\n✓ Vector store created and persisted at: {DB_DIR}")
print(f"  Total vectors stored: {len(chunks)}")


def build_generation_pipeline(model_id: str):
    use_cuda = torch.cuda.is_available()
    use_mps = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model_kwargs: Dict[str, Any] = {}
    if use_cuda:
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
            print("✓ Using CUDA with 4-bit quantization")
        except Exception:
            model_kwargs["device_map"] = "auto"
            model_kwargs["torch_dtype"] = torch.float16
            print("✓ Using CUDA with float16")
    elif use_mps:
        model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["device_map"] = "auto"
        print("✓ Using Apple MPS (float16)")
    else:
        model_kwargs["torch_dtype"] = torch.float32
        model_kwargs["device_map"] = "auto"
        print("✓ Using CPU (this will be slow)")
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.3,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        return_full_text=False
    )
    return gen_pipe


print("Initializing LLM for summarization...")
gen_pipe = build_generation_pipeline(LLM_NAME)
llm = HuggingFacePipeline(pipeline=gen_pipe)
print("\n✓ LLM initialized and ready")
retriever = vectorstore.as_retriever(
    search_kwargs={"k": TOP_K}
)
template = """You are an expert petroleum engineer specializing in well completion reports.
Your task is to create accurate, comprehensive summaries of well completion documentation.

Guidelines:
1. Focus on technical details: completion design, equipment specs, operational parameters
2. Include key metrics: depths, pressures, flow rates, production data
3. Highlight critical events or decisions during completion
4. Maintain technical accuracy - never invent or assume data
5. Respect the requested word count
6. If information is unavailable, state this clearly
7. Cite specific documents when referencing data

Context from well documentation:
{context}

User request: {question}

Please provide a summary based on the context and user requirements."""
prompt = PromptTemplate.from_template(template)

def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)
print("✓ RAG chain for completion report summarization initialized")
print(f"  Retrieval: Top-{TOP_K} relevant chunks")
print(f"  Generation: Phi-3 with specialized prompting")


def generate_summary(
    prompt_text: str,
    word_count: int = 300,
    focus_areas: List[str] | None = None
) -> Dict[str, Any]:
    enhanced_prompt = f"{prompt_text}\n\nGenerate a summary of approximately {word_count} words."
    if focus_areas:
        enhanced_prompt += f"\nFocus on: {', '.join(focus_areas)}"
    retrieved_docs = retriever.invoke(enhanced_prompt)
    summary = rag_chain.invoke(enhanced_prompt)
    sources = set()
    for doc in retrieved_docs:
        source = doc.metadata.get('source', 'Unknown')
        sources.add(source)
    return {
        'summary': summary,
        'sources': list(sources),
        'retrieved_chunks': len(retrieved_docs),
        'word_count': len(summary.split())
    }


def print_summary_report(result: Dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print("COMPLETION REPORT SUMMARY")
    print("=" * 80)
    print()
    print(result['summary'])
    print()
    print("-" * 80)
    print("Summary Statistics:")
    print(f"  Word count: {result['word_count']}")
    print(f"  Retrieved chunks: {result['retrieved_chunks']}")
    print()
    print("Source Documents:")
    for index, source in enumerate(result['sources'], 1):
        print(f"  [{index}] {source}")
    print("=" * 80)


print("Example 1: General Well Completion Summary\n")
result1 = generate_summary(
    prompt_text="Provide a comprehensive summary of the Well 1 (ANDIJK-GT-01) completion, including completion design, equipment used, and key operational parameters.",
    word_count=250
)
print_summary_report(result1)
print("\n\nExample 2: Equipment and Technical Specifications\n")
result2 = generate_summary(
    prompt_text="Summarize the equipment specifications and technical details of the completion for Well 1, including ESP (Electric Submersible Pump) configuration and wellhead equipment.",
    word_count=200,
    focus_areas=["Equipment specs", "ESP configuration", "Wellhead design"]
)
print_summary_report(result2)
print("\n\nExample 3: Production and Well Test Summary\n")
result3 = generate_summary(
    prompt_text="Summarize the well test results and production performance for Well 1, including flow rates, pressures, and reservoir characteristics.",
    word_count=300,
    focus_areas=["Well test results", "Production rates", "Reservoir properties"]
)
print_summary_report(result3)

def interactive_summarize() -> None:
    print("\n" + "=" * 80)
    print("INTERACTIVE COMPLETION REPORT SUMMARIZATION")
    print("=" * 80)
    print("\nEnter your summarization request:")
    print("(Or use one of the examples above)\n")
    user_prompt = input("Your prompt: ")
    if not user_prompt.strip():
        print("No prompt provided. Using default...")
        user_prompt = "Provide a comprehensive summary of the well completion for Well 1."
    word_count_input = input("Target word count (default 300): ")
    try:
        word_count = int(word_count_input) if word_count_input.strip() else 300
    except ValueError:
        word_count = 300
    print("\nGenerating summary...\n")
    result = generate_summary(user_prompt, word_count)
    print_summary_report(result)


def preview_retrieval(query: str, k: int = 5) -> None:
    print(f"\nQuery: {query}")
    print("=" * 80)
    docs_scores: List[Any] = []
    try:
        docs_scores = vectorstore.similarity_search_with_score(query, k=k)
    except Exception:
        docs = vectorstore.similarity_search(query, k=k)
        docs_scores = [(doc, None) for doc in docs]
    for index, (doc, score) in enumerate(docs_scores, 1):
        metadata = doc.metadata or {}
        source = metadata.get('source', 'Unknown')
        file_type = metadata.get('file_type', 'unknown')
        directory = metadata.get('directory', 'unknown')
        snippet = shorten(doc.page_content, width=200, placeholder=" ...")
        score_str = f" | similarity={score:.4f}" if score is not None else ""
        print(f"\n[{index}] {source} ({file_type}) - {directory}{score_str}")
        print(f"    {snippet}")
    print("\n" + "=" * 80)


preview_retrieval(
    "completion equipment ESP electric submersible pump configuration",
    k=5
)


def evaluate_summary_quality(summary_result: Dict[str, Any], target_word_count: int) -> Dict[str, Any]:
    actual_words = summary_result['word_count']
    word_count_accuracy = 1 - abs(actual_words - target_word_count) / target_word_count
    word_count_accuracy = max(0, min(1, word_count_accuracy))
    source_diversity = len(summary_result['sources'])
    metrics = {
        'word_count_accuracy': word_count_accuracy,
        'target_words': target_word_count,
        'actual_words': actual_words,
        'word_count_deviation': abs(actual_words - target_word_count),
        'source_diversity': source_diversity,
        'chunks_retrieved': summary_result['retrieved_chunks']
    }
    return metrics


def print_evaluation(metrics: Dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print("SUMMARY QUALITY EVALUATION")
    print("=" * 80)
    print(f"Word Count Accuracy: {metrics['word_count_accuracy']:.1%}")
    print(f"  Target: {metrics['target_words']} words")
    print(f"  Actual: {metrics['actual_words']} words")
    print(f"  Deviation: {metrics['word_count_deviation']} words")
    print()
    print("Source Coverage:")
    print(f"  Unique sources: {metrics['source_diversity']}")
    print(f"  Chunks retrieved: {metrics['chunks_retrieved']}")
    print("=" * 80)


if 'result1' in locals():
    metrics = evaluate_summary_quality(result1, target_word_count=250)
    print_evaluation(metrics)

result = generate_summary(
    "Provide a complete overview of the well completion",
    word_count=300
)
result = generate_summary(
    "Describe all completion equipment and their specifications",
    word_count=250
)
result = generate_summary(
    "Summarize key operational events and decisions during completion",
    word_count=200
)
result = generate_summary(
    "Extract and summarize all technical measurements: depths, pressures, rates",
    word_count=250
)
result = generate_summary(
    "Describe reservoir properties and characteristics from well test data",
    word_count=200
)
print("Quick reference queries ready. Uncomment any query above and run.")
