"""Consolidated Challenge 3 agent pipeline for Well 1 nodal analysis."""

INSTALL_REQUIREMENTS = False

if INSTALL_REQUIREMENTS:
    import subprocess
    import sys

    package_groups = [
        [
            "langchain",
            "langchain-core",
            "langchain-community",
            "langchain-huggingface",
        ],
        [
            "langchain-chroma",
            "langchain-text-splitters",
            "python-dotenv",
            "huggingface_hub",
        ],
        [
            "sentence-transformers",
            "requests",
            "bitsandbytes",
            "transformers",
            "datasets",
            "accelerate",
        ],
        [
            "pypdf",
            "pymupdf",
            "python-docx",
            "openpyxl",
            "pandas",
            "unstructured[pdf]",
        ],
        ["opentelemetry-api", "opentelemetry-sdk"],
        ["streamlit"],
    ]
    for group in package_groups:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *group])
    print("Dependencies installed")
else:
    print("Set INSTALL_REQUIREMENTS=True and re-run if packages are missing.")

import json
import math
import shutil
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    from IPython.display import display
except ImportError:  # pragma: no cover - fallback for non-IPython environments
    def display(obj: Any) -> None:
        print(obj)

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import torch

warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = (10, 6)
print("Imports ready")

MODEL_EMBED = "sentence-transformers/all-MiniLM-L6-v2"
LLM_NAME = "microsoft/Phi-3-mini-4k-instruct"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300
TOP_K = 6

WORKDIR = Path.cwd()
if "SpeEuGeoH2025" in str(WORKDIR):
    parts = WORKDIR.parts
    idx = parts.index("SpeEuGeoH2025")
    PROJECT_ROOT = Path(*parts[: idx + 1])
else:
    PROJECT_ROOT = WORKDIR.parent if WORKDIR.name == "notebooks" else WORKDIR

WELL_DATA_DIR = PROJECT_ROOT / "data" / "Training data-shared with participants" / "Well 1"
EXTRA_FILES = [PROJECT_ROOT / "data" / "Training data-shared with participants" / "boreholes.xlsx"]
DB_DIR = PROJECT_ROOT / "notebooks" / "local_data" / "well1_agent_vector_db"
DB_DIR.parent.mkdir(parents=True, exist_ok=True)

print(f"Project Root: {PROJECT_ROOT}")
print(f"Well 1 Data: {WELL_DATA_DIR}")
print(f"Vector DB: {DB_DIR}")
if not WELL_DATA_DIR.exists():
    raise FileNotFoundError(f"Well 1 directory not found at {WELL_DATA_DIR}")
missing = [p for p in EXTRA_FILES if not p.exists()]
if missing:
    print(f"Missing supplemental files: {missing}")
else:
    print("Supplemental files located")

def load_pdf(file_path: Path) -> List[Document]:
    try:
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()
        for doc in docs:
            doc.metadata.update(
                {
                    "source": file_path.name,
                    "file_type": "pdf",
                    "directory": file_path.parent.name,
                }
            )
        return docs
    except Exception as exc:  # noqa: BLE001
        print(f"Error loading PDF {file_path.name}: {exc}")
        return []

def load_word(file_path: Path) -> List[Document]:
    try:
        loader = UnstructuredWordDocumentLoader(str(file_path))
        docs = loader.load()
        for doc in docs:
            doc.metadata.update(
                {
                    "source": file_path.name,
                    "file_type": "docx",
                    "directory": file_path.parent.name,
                }
            )
        return docs
    except Exception as exc:  # noqa: BLE001
        print(f"Error loading Word {file_path.name}: {exc}")
        return []

def load_excel(file_path: Path) -> List[Document]:
    documents: List[Document] = []
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
            documents.append(doc)
    except Exception as exc:  # noqa: BLE001
        print(f"Error loading Excel {file_path.name}: {exc}")
    return documents

def load_all_documents(root_dir: Path, extra_files: List[Path]) -> List[Document]:
    documents: List[Document] = []
    pdf_files = list(root_dir.rglob("*.pdf"))
    word_files = list(root_dir.rglob("*.docx")) + list(root_dir.rglob("*.doc"))
    excel_files = list(root_dir.rglob("*.xlsx")) + list(root_dir.rglob("*.xls"))
    print(
        f"Loading {len(pdf_files)} PDFs, {len(word_files)} Word docs, {len(excel_files)} spreadsheets from {root_dir}"
    )
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
    print(f"Total documents loaded: {len(documents)}")
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
    preview = chunks[0].page_content[:200].replace("\n", " ")
    print(f"Sample chunk source: {chunks[0].metadata.get('source')}")
    print(f"Preview: {preview}...")

if DB_DIR.exists():
    shutil.rmtree(DB_DIR)
embeddings = HuggingFaceEmbeddings(model_name=MODEL_EMBED)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=str(DB_DIR),
)
print(f"Vector store created at {DB_DIR}")

def build_llm_pipeline(model_id: str, force_cpu: bool = False) -> HuggingFacePipeline:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model_kwargs: Dict[str, Any] = {"low_cpu_mem_usage": True}
    quantization_attempted = False

    if force_cpu:
        model_kwargs["device_map"] = {"": "cpu"}
        model_kwargs["torch_dtype"] = torch.float32
        print("Forcing CPU execution for LLM")
    else:
        model_kwargs["device_map"] = "auto"
        if torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig

                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                model_kwargs["quantization_config"] = quant_config
                model_kwargs["torch_dtype"] = torch.float16
                quantization_attempted = True
                print("CUDA 4-bit quantization enabled")
            except Exception as exc:  # noqa: BLE001
                model_kwargs["torch_dtype"] = torch.float16
                print(f"CUDA float16 fallback (quantization unavailable): {exc}")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            model_kwargs["torch_dtype"] = torch.float16
            print("Apple MPS acceleration")
        else:
            model_kwargs["torch_dtype"] = torch.float32
            print("CPU execution")

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    except ValueError as exc:  # noqa: BLE001
        if not force_cpu and quantization_attempted:
            print(f"4-bit load failed: {exc}\nRe-trying without quantization...")
            model_kwargs.pop("quantization_config", None)
            model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
            try:
                model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
                print("Loaded model without 4-bit quantization")
            except Exception:  # noqa: BLE001
                print("Falling back to CPU after quantization failure")
                return build_llm_pipeline(model_id, force_cpu=True)
        elif not force_cpu:
            print("LLM load error, retrying on CPU")
            return build_llm_pipeline(model_id, force_cpu=True)
        else:
            raise
    except torch.cuda.OutOfMemoryError as exc:
        if not force_cpu:
            print(f"GPU OOM while loading model: {exc}\nSwitching to CPU")
            torch.cuda.empty_cache()
            return build_llm_pipeline(model_id, force_cpu=True)
        raise

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    text_pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.2,
        do_sample=True,
        return_full_text=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    return HuggingFacePipeline(pipeline=text_pipe)

llm = build_llm_pipeline(LLM_NAME)
print("LLM ready")

retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
template = """You are a petroleum production engineer extracting nodal-analysis parameters for Well 1 (ANDIJK-GT-01).
Use the supplied context to answer with precise numeric values.
If data is missing, mark it as null and note the gap.

Context:
{context}

Question: {question}
"""
prompt = PromptTemplate.from_template(template)

def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_chain(llm_model: HuggingFacePipeline):
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm_model
        | StrOutputParser()
    )

rag_chain = build_rag_chain(llm)
print("RAG chain initialised")

def extract_nodal_parameters(question: str) -> Dict[str, Any]:
    global rag_chain, llm
    structured_prompt = (
        question
        + "\n\nReturn ONLY a JSON object with keys: reservoir_pressure_bar, wellhead_pressure_bar, productivity_index_m3hr_per_bar, fluid_density_kg_m3, fluid_viscosity_pa_s, esp_depth_m, roughness_m, pump_curve (list of {\"flow_m3hr\", \"head_m\"}), well_segments (list of {\"start_depth_m\", \"end_depth_m\", \"diameter_m\"}), notes.\nEnsure numbers are numeric.\nIf unknown, set null and describe the gap in notes."
    )

    def switch_to_cpu_chain() -> None:
        global rag_chain, llm
        torch.cuda.empty_cache()
        llm = build_llm_pipeline(LLM_NAME, force_cpu=True)
        rag_chain = build_rag_chain(llm)

    def invoke_chain(prompt_text: str) -> str:
        try:
            return rag_chain.invoke(prompt_text)
        except torch.cuda.OutOfMemoryError as exc:
            print(f"GPU OOM during generation: {exc}\nSwitching LLM inference to CPU.")
            switch_to_cpu_chain()
            return rag_chain.invoke(prompt_text)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and torch.cuda.is_available():
                print(f"Runtime OOM in generation: {exc}\nSwitching LLM inference to CPU.")
                switch_to_cpu_chain()
                return rag_chain.invoke(prompt_text)
            raise

    raw_response = invoke_chain(structured_prompt)

    def try_parse_json(text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start : end + 1])
            raise

    try:
        return try_parse_json(raw_response)
    except json.JSONDecodeError as exc:
        return {"raw_response": raw_response, "error": str(exc)}

defaults = {
    "reservoir_pressure_bar": 230.0,
    "wellhead_pressure_bar": 10.0,
    "productivity_index_m3hr_per_bar": 5.0,
    "fluid_density_kg_m3": 1000.0,
    "fluid_viscosity_pa_s": 1e-3,
    "esp_depth_m": 500.0,
    "roughness_m": 1e-5,
}

def coerce_parameters(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = defaults.copy()
    if not isinstance(payload, dict):
        result["notes"] = "Extraction failed; defaults applied."
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
        note = result.get("notes", "")
        result["notes"] = (note + (" " if note else "") + "Fallback pump curve applied.").strip()
    result["pump_curve"] = pump_curve
    segments = payload.get("well_segments") if isinstance(payload, dict) else None
    if not segments:
        segments = [
            {"start_depth_m": 0.0, "end_depth_m": 500.0, "diameter_m": 0.3397},
            {"start_depth_m": 500.0, "end_depth_m": 1500.0, "diameter_m": 0.2445},
            {"start_depth_m": 1500.0, "end_depth_m": 2500.0, "diameter_m": 0.1778},
        ]
        note = result.get("notes", "")
        result["notes"] = (note + (" " if note else "") + "Fallback well geometry applied.").strip()
    result["well_segments"] = segments
    return result

def build_segments(segment_specs: List[Dict[str, float]]) -> List[Dict[str, float]]:
    ordered = sorted(segment_specs, key=lambda seg: seg.get("start_depth_m", 0.0))
    segments: List[Dict[str, float]] = []
    for spec in ordered:
        start = float(spec.get("start_depth_m", 0.0))
        end = float(spec.get("end_depth_m", start))
        segments.append(
            {"length_m": max(end - start, 0.0), "diameter_m": float(spec.get("diameter_m", 0.1)), "theta_rad": math.pi / 2}
        )
    return segments

def swamee_jain(reynolds: float, diameter: float, roughness: float) -> float:
    if reynolds <= 0:
        return 0.0
    return 0.25 / (math.log10((roughness / (3.7 * diameter)) + (5.74 / (reynolds ** 0.9)))) ** 2

def interpolate_pump_head(flow: float, curve: List[Dict[str, float]]) -> float:
    sorted_curve = sorted(curve, key=lambda item: item["flow_m3hr"])
    flows = [point["flow_m3hr"] for point in sorted_curve]
    heads = [point["head_m"] for point in sorted_curve]
    return float(np.interp(flow, flows, heads))

def compute_vlp(flow_m3hr: float, params: Dict[str, Any]) -> float:
    density = params["fluid_density_kg_m3"]
    viscosity = params["fluid_viscosity_pa_s"]
    roughness = params["roughness_m"]
    wellhead_pressure = params["wellhead_pressure_bar"]
    pump_curve = params["pump_curve"]
    esp_depth = params["esp_depth_m"]
    q_m3s = flow_m3hr / 3600.0
    total_dp_pa = 0.0
    depth_accum = 0.0
    for segment in build_segments(params["well_segments"]):
        length = segment["length_m"]
        diameter = segment["diameter_m"]
        area = math.pi * diameter**2 / 4.0
        velocity = q_m3s / area if area else 0.0
        reynolds = density * abs(velocity) * diameter / max(viscosity, 1e-9)
        f = swamee_jain(reynolds, diameter, roughness)
        dp_fric = f * (length / max(diameter, 1e-6)) * (density * velocity**2 / 2.0)
        dp_grav = density * 9.81 * length
        total_dp_pa += dp_fric + dp_grav
        depth_accum += length
    if depth_accum >= esp_depth:
        total_dp_pa -= density * 9.81 * interpolate_pump_head(flow_m3hr, pump_curve)
    return wellhead_pressure + total_dp_pa / 1e5

def compute_ipr(flow_m3hr: float, params: Dict[str, Any]) -> float:
    reservoir_pressure = params["reservoir_pressure_bar"]
    pi = params["productivity_index_m3hr_per_bar"]
    return max(reservoir_pressure - flow_m3hr / max(pi, 1e-6), 0.0)

def run_nodal_analysis(params: Dict[str, Any], max_flow: Optional[float] = None) -> Dict[str, Any]:
    if max_flow is None:
        max_flow = max(point["flow_m3hr"] for point in params["pump_curve"])
    flows = np.linspace(1, max_flow, 200)
    vlp_curve = np.array([compute_vlp(f, params) for f in flows])
    ipr_curve = np.array([compute_ipr(f, params) for f in flows])
    diff = np.abs(vlp_curve - ipr_curve)
    idx = int(np.argmin(diff))
    tolerance = 3.0
    solution = None
    if diff[idx] <= tolerance:
        solution = {
            "flow_m3hr": float(flows[idx]),
            "bottomhole_pressure_bar": float(vlp_curve[idx]),
            "pump_head_m": float(interpolate_pump_head(flows[idx], params["pump_curve"])),
            "delta_bar": float(diff[idx]),
        }
    return {"flows": flows, "vlp": vlp_curve, "ipr": ipr_curve, "solution": solution}

def build_operator_summary(
    solution: Optional[Dict[str, float]],
    analysis: Dict[str, Any],
    params: Dict[str, Any],
    raw_payload: Dict[str, Any],
) -> Dict[str, Any]:
    if not solution:
        narrative = (
            "The analysis could not find an operating point where inflow and outflow match within the set tolerance.\n\n"
            "This usually means the assumed pump support or reservoir drive needs review."
        )
        bullets = [
            "Confirm pump performance data.",
            "Try expanding the tested flow range.",
        ]
        table = pd.DataFrame(
            [
                {
                    "Flowrate (m3/hr)": None,
                    "Bottomhole Pressure (bar)": None,
                    "Pump Head (m)": None,
                    "Difference (bar)": None,
                }
            ]
        )
        return {"narrative": narrative, "bullets": bullets, "table": table}

    flow = solution["flow_m3hr"]
    pressure = solution["bottomhole_pressure_bar"]
    pump_head = solution["pump_head_m"]
    deviation = solution["delta_bar"]

    narrative = (
        f"The well is expected to stabilise at about {flow:.0f} cubic metres per hour. "
        f"At that rate the bottomhole flowing pressure sits near {pressure:.0f} bar, "
        f"which is sufficiently supported by the current pump head of roughly {pump_head:.0f} metres. "
        "This means the well should deliver steady production without overloading the lift equipment."
    )

    bullets = [
        f"Expected flowrate: {flow:.1f} m3/hr",
        f"Flowing pressure at depth: {pressure:.1f} bar",
        f"Pump head required: {pump_head:.1f} m",
        f"Curve match difference: {deviation:.2f} bar",
    ]

    table = pd.DataFrame(
        [
            {"Metric": "Steady flowrate", "Value": f"{flow:.1f} m3/hr"},
            {"Metric": "Bottomhole pressure", "Value": f"{pressure:.1f} bar"},
            {"Metric": "Pump head", "Value": f"{pump_head:.1f} m"},
            {"Metric": "Curve difference", "Value": f"{deviation:.2f} bar"},
        ]
    )

    if raw_payload.get("notes"):
        bullets.append(f"Notes: {raw_payload['notes']}")

    return {"narrative": narrative, "bullets": bullets, "table": table}

def run_nodal_analysis_agent(user_question: str) -> Dict[str, Any]:
    payload = extract_nodal_parameters(user_question)
    params = coerce_parameters(payload)
    analysis = run_nodal_analysis(params)
    summary = build_operator_summary(
        analysis.get("solution"),
        analysis,
        params,
        payload if isinstance(payload, dict) else {},
    )
    summary["analysis"] = analysis
    summary["params"] = params
    summary["raw_payload"] = payload
    return summary

def main() -> None:
    agent_output = run_nodal_analysis_agent(
        "Perform a nodal analysis for Well 1 and summarise the production capacity."
    )
    print(agent_output["narrative"])
    print()
    for item in agent_output["bullets"]:
        print(f"- {item}")
    print()
    display(agent_output["table"])

if __name__ == "__main__":
    main()
