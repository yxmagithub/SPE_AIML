import json
import math
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = (10, 6)
print("✓ Imports ready")

MODEL_EMBED = "sentence-transformers/all-MiniLM-L6-v2"
LLM_NAME = "microsoft/Phi-3-mini-4k-instruct"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300
TOP_K = 6

WORKDIR = Path.cwd()
if 'SpeEuGeoH2025' in str(WORKDIR):
    parts = WORKDIR.parts
    idx = parts.index('SpeEuGeoH2025')
    PROJECT_ROOT = Path(*parts[:idx+1])
else:
    PROJECT_ROOT = WORKDIR.parent if WORKDIR.name == 'notebooks' else WORKDIR

WELL_DATA_DIR = PROJECT_ROOT / "data" / "Training data-shared with participants" / "Well 1"
EXTRA_FILES = [PROJECT_ROOT / "data" / "Training data-shared with participants" / "boreholes.xlsx"]
DB_DIR = PROJECT_ROOT / "notebooks" / "local_data" / "well1_nodal_vector_db"
DB_DIR.parent.mkdir(parents=True, exist_ok=True)

print(f"Project Root: {PROJECT_ROOT}")
print(f"Well 1 Data: {WELL_DATA_DIR}")
print(f"Vector DB: {DB_DIR}")
print(f"Embedding Model: {MODEL_EMBED}")
print(f"LLM: {LLM_NAME}")

if not WELL_DATA_DIR.exists():
    raise FileNotFoundError(f"Well 1 directory not found at {WELL_DATA_DIR}")

missing_extra = [p for p in EXTRA_FILES if not p.exists()]
if missing_extra:
    print(f"⚠️ Missing supplemental files: {missing_extra}")
else:
    print("✓ Supplemental files located")

def load_pdf(file_path: Path) -> List[Document]:
    """Load PDF files into LangChain Documents with metadata."""
    try:
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()
        for doc in docs:
            doc.metadata.update({
                "source": file_path.name,
                "file_type": "pdf",
                "directory": file_path.parent.name,
            })
        return docs
    except Exception as exc:
        print(f"Error loading PDF {file_path.name}: {exc}")
        return []


def load_word(file_path: Path) -> List[Document]:
    """Load Word documents via unstructured loader."""
    try:
        loader = UnstructuredWordDocumentLoader(str(file_path))
        docs = loader.load()
        for doc in docs:
            doc.metadata.update({
                "source": file_path.name,
                "file_type": "docx",
                "directory": file_path.parent.name,
            })
        return docs
    except Exception as exc:
        print(f"Error loading Word {file_path.name}: {exc}")
        return []


def load_excel(file_path: Path) -> List[Document]:
    """Flatten Excel sheets into text-based Documents."""
    docs: List[Document] = []
    try:
        excel_file = pd.ExcelFile(file_path)
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            text_lines = [
                f"Sheet: {sheet_name}",
                f"Columns: {', '.join(map(str, df.columns.tolist()))}",
                df.to_string(index=False),
            ]
            doc = Document(
                page_content="\n\n".join(text_lines),
                metadata={
                    "source": file_path.name,
                    "file_type": "excel",
                    "sheet_name": sheet_name,
                    "rows": int(df.shape[0]),
                    "columns": int(df.shape[1]),
                    "directory": file_path.parent.name,
                },
            )
            docs.append(doc)
    except Exception as exc:
        print(f"Error loading Excel {file_path.name}: {exc}")
    return docs


def load_all_documents(root_dir: Path, extra_files: List[Path]) -> List[Document]:
    """Scan directories and supplemental files into a combined document list."""
    documents: List[Document] = []
    pdf_files = list(root_dir.rglob("*.pdf"))
    word_files = list(root_dir.rglob("*.docx")) + list(root_dir.rglob("*.doc"))
    excel_files = list(root_dir.rglob("*.xlsx")) + list(root_dir.rglob("*.xls"))

    print(f"Loading {len(pdf_files)} PDFs, {len(word_files)} Word docs, {len(excel_files)} spreadsheets from {root_dir}")

    for path in pdf_files:
        documents.extend(load_pdf(path))
    for path in word_files:
        documents.extend(load_word(path))
    for path in excel_files:
        documents.extend(load_excel(path))

    for file_path in extra_files:
        if not file_path.exists():
            continue
        suffix = file_path.suffix.lower()
        if suffix in (".xlsx", ".xls"):
            documents.extend(load_excel(file_path))
        elif suffix in (".pdf",):
            documents.extend(load_pdf(file_path))
        elif suffix in (".docx", ".doc"):
            documents.extend(load_word(file_path))

    print(f"✓ Total documents loaded: {len(documents)}")
    return documents


documents = load_all_documents(WELL_DATA_DIR, EXTRA_FILES)
if not documents:
    raise RuntimeError("No documents loaded. Verify data paths.")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""],
)

chunks = text_splitter.split_documents(documents)
print(f"Original documents: {len(documents)}")
print(f"Total chunks: {len(chunks)}")
if chunks:
    sample = chunks[0]
    preview = sample.page_content[:200].replace("\n", " ")
    print(f"Sample chunk source: {sample.metadata.get('source')}")
    print(f"Preview: {preview}...")

import shutil

if DB_DIR.exists():
    shutil.rmtree(DB_DIR)

embeddings = HuggingFaceEmbeddings(model_name=MODEL_EMBED)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=str(DB_DIR),
)
print(f"✓ Vector store ready at {DB_DIR}")

def build_llm_pipeline(model_id: str) -> HuggingFacePipeline:
    use_cuda = torch.cuda.is_available()
    use_mps = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model_kwargs: Dict[str, Any] = {"device_map": "auto"}

    if use_cuda:
        try:
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(load_in_4bit=True)
            model_kwargs["quantization_config"] = quant_config
            print("✓ Using CUDA with 4-bit quantization")
        except Exception:
            model_kwargs["torch_dtype"] = torch.float16
            print("✓ Using CUDA with float16 precision")
    elif use_mps:
        model_kwargs["torch_dtype"] = torch.float16
        print("✓ Using Apple MPS acceleration")
    else:
        model_kwargs["torch_dtype"] = torch.float32
        print("ℹ Running on CPU; generation will be slower")

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    text_pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.2,
        do_sample=True,
        return_full_text=False,
        pad_token_id=tokenizer.pad_token_id,
    )

    return HuggingFacePipeline(pipeline=text_pipe)


llm = build_llm_pipeline(LLM_NAME)
print("✓ LLM initialised")

retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

prompt_template = """You are a petroleum production engineer extracting nodal-analysis parameters for Well 1 (ANDIJK-GT-01).
Use the supplied context to answer the user's question with factual, structured information.
If specific values are unavailable, state this explicitly.

Context:
{context}

Question: {question}
"""

prompt = PromptTemplate.from_template(prompt_template)

def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

print("✓ RAG chain ready for structured extraction")

def extract_nodal_parameters(question: str) -> Dict[str, Any]:
    """Use the RAG chain to obtain structured nodal-analysis inputs."""
    structured_prompt = (
        question
        + "\n\nReturn ONLY a JSON object with the following keys:"
        + "\nreservoir_pressure_bar"
        + "\nwellhead_pressure_bar"
        + "\nproductivity_index_m3hr_per_bar"
        + "\nfluid_density_kg_m3"
        + "\nfluid_viscosity_pa_s"
        + "\nesp_depth_m"
        + "\nroughness_m"
        + "\npump_curve (list of objects with flow_m3hr and head_m)"
        + "\nwell_segments (list with start_depth_m, end_depth_m, diameter_m)"
        + "\nnotes"
        + "\nEnsure every numeric value is a number, not text."
        + "\nIf a value is unavailable in the context, set it to null and mention it in notes."
    )

    raw_response = rag_chain.invoke(structured_prompt)

    def try_parse_json(text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                snippet = text[start:end + 1]
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError:
                    pass
            raise

    try:
        payload = try_parse_json(raw_response)
    except json.JSONDecodeError as exc:
        print("Raw response could not be parsed as JSON. Returning text instead.")
        return {"raw_response": raw_response, "error": str(exc)}

    return payload


parameter_payload = extract_nodal_parameters(
    "Extract all nodal-analysis inputs for Well 1 / ANDIJK-GT-01 from the available documentation and tables."
)
print(json.dumps(parameter_payload, indent=2))

boreholes_path = EXTRA_FILES[0]
if boreholes_path.exists():
    boreholes_df = pd.read_excel(boreholes_path)
    print("Preview of boreholes.xlsx (first 5 rows):")
    display(boreholes_df.head())
else:
    print("boreholes.xlsx not available for direct inspection")

defaults = {
    "reservoir_pressure_bar": 230.0,
    "wellhead_pressure_bar": 10.0,
    "productivity_index_m3hr_per_bar": 5.0,
    "fluid_density_kg_m3": 1000.0,
    "fluid_viscosity_pa_s": 1e-3,
    "esp_depth_m": 500.0,
    "roughness_m": 1e-5,
}

def coerce_parameters(payload: Dict[str, Any], defaults: Dict[str, float]) -> Dict[str, Any]:
    """Merge retrieved payload with defaults and ensure required structures exist."""
    result: Dict[str, Any] = defaults.copy()

    if payload is None:
        return result

    for key, value in payload.items():
        if key in {"pump_curve", "well_segments", "notes", "raw_response", "error"}:
            result[key] = value
        elif value is None:
            continue
        else:
            try:
                result[key] = float(value)
            except (TypeError, ValueError):
                continue

    pump_curve = payload.get("pump_curve") if isinstance(payload, dict) else None
    if not pump_curve:
        pump_curve = [
            {"flow_m3hr": 0.0, "head_m": 600.0},
            {"flow_m3hr": 100.0, "head_m": 550.0},
            {"flow_m3hr": 200.0, "head_m": 450.0},
            {"flow_m3hr": 300.0, "head_m": 300.0},
            {"flow_m3hr": 400.0, "head_m": 100.0},
        ]
        note_prefix = result.get("notes", "") or ""
        note_suffix = "Fallback pump curve applied."
        result["notes"] = (note_prefix + (" " if note_prefix else "") + note_suffix).strip()

    result["pump_curve"] = pump_curve

    segments = payload.get("well_segments") if isinstance(payload, dict) else None
    if not segments:
        segments = [
            {"start_depth_m": 0.0, "end_depth_m": 500.0, "diameter_m": 0.3397},
            {"start_depth_m": 500.0, "end_depth_m": 1500.0, "diameter_m": 0.2445},
            {"start_depth_m": 1500.0, "end_depth_m": 2500.0, "diameter_m": 0.1778},
        ]
        note_prefix = result.get("notes", "") or ""
        note_suffix = "Fallback well geometry applied."
        result["notes"] = (note_prefix + (" " if note_prefix else "") + note_suffix).strip()

    result["well_segments"] = segments

    return result


nodal_params = coerce_parameters(parameter_payload, defaults)
print(json.dumps(nodal_params, indent=2))
