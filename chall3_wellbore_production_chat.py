import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

warnings.filterwarnings('ignore')

MODEL_EMBED = 'sentence-transformers/all-MiniLM-L6-v2'
LLM_NAME = 'microsoft/Phi-3-mini-4k-instruct'
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300
TOP_K = 6

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

WELL_DATA_DIR = PROJECT_ROOT / 'data' / 'Training data-shared with participants' / 'Well 1'
EXTRA_FILES = [PROJECT_ROOT / 'data' / 'Training data-shared with participants' / 'boreholes.xlsx']
DB_DIR = PROJECT_ROOT / 'notebooks' / 'local_data' / 'well1_agent_chat_db'
DB_DIR.mkdir(parents=True, exist_ok=True)

def load_pdf(file_path: Path) -> List[Document]:
    try:
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()
        for doc in docs:
            doc.metadata.update({'source': file_path.name, 'file_type': 'pdf', 'directory': file_path.parent.name})
        return docs
    except Exception as exc:
        print(f'PDF load error for {file_path.name}: {exc}')
        return []

def load_word(file_path: Path) -> List[Document]:
    try:
        loader = UnstructuredWordDocumentLoader(str(file_path))
        docs = loader.load()
        for doc in docs:
            doc.metadata.update({'source': file_path.name, 'file_type': 'docx', 'directory': file_path.parent.name})
        return docs
    except Exception as exc:
        print(f'Word load error for {file_path.name}: {exc}')
        return []

def load_excel(file_path: Path) -> List[Document]:
    documents: List[Document] = []
    try:
        excel_file = pd.ExcelFile(file_path)
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            text_lines = [
                f'Sheet: {sheet_name}',
                f"Columns: {', '.join(map(str, df.columns.tolist()))}",
                df.to_string(index=False),
            ]
            doc = Document(
                page_content="\n\n".join(text_lines),
                metadata={
                    'source': file_path.name,
                    'file_type': 'excel',
                    'sheet_name': sheet_name,
                    'rows': int(df.shape[0]),
                    'columns': int(df.shape[1]),
                    'directory': file_path.parent.name,
                },
            )
            documents.append(doc)
    except Exception as exc:
        print(f'Excel load error for {file_path.name}: {exc}')
    return documents

def load_all_documents(root_dir: Path, extra_files: List[Path]) -> List[Document]:
    documents: List[Document] = []
    pdf_files = list(root_dir.rglob('*.pdf'))
    word_files = list(root_dir.rglob('*.docx')) + list(root_dir.rglob('*.doc'))
    excel_files = list(root_dir.rglob('*.xlsx')) + list(root_dir.rglob('*.xls'))
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
        if suffix in ('.xlsx', '.xls'):
            documents.extend(load_excel(file_path))
        elif suffix in ('.pdf',):
            documents.extend(load_pdf(file_path))
        elif suffix in ('.docx', '.doc'):
            documents.extend(load_word(file_path))
    return documents

@st.cache_resource(show_spinner=False)
def get_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_EMBED)
    if DB_DIR.exists() and any(DB_DIR.iterdir()):
        return Chroma(persist_directory=str(DB_DIR), embedding_function=embeddings)
    documents = load_all_documents(WELL_DATA_DIR, EXTRA_FILES)
    if not documents:
        raise RuntimeError('No documents loaded for Well 1.')
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=str(DB_DIR))
    vectorstore.persist()
    return vectorstore

@st.cache_resource(show_spinner=False)
def get_llm() -> HuggingFacePipeline:
    tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
    model_kwargs: Dict[str, Any] = {'device_map': 'auto'}
    if torch.cuda.is_available():
        try:
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(load_in_4bit=True)
            model_kwargs['quantization_config'] = quant_config
        except Exception:
            model_kwargs['torch_dtype'] = torch.float16
    elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        model_kwargs['torch_dtype'] = torch.float16
    else:
        model_kwargs['torch_dtype'] = torch.float32
    model = AutoModelForCausalLM.from_pretrained(LLM_NAME, **model_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    text_pipe = pipeline(
        task='text-generation',
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.2,
        do_sample=True,
        return_full_text=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    return HuggingFacePipeline(pipeline=text_pipe)

def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

PROMPT_TEMPLATE = """You are a petroleum production engineer extracting nodal-analysis parameters for Well 1 (ANDIJK-GT-01).
Use the supplied context to answer with precise numeric values.
If data is missing, mark it as null and note the gap.

Context:
{context}

Question: {question}
"""
prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

@st.cache_resource(show_spinner=False)
def get_rag_chain():
    retriever = get_vectorstore().as_retriever(search_kwargs={'k': TOP_K})
    return (
        {'context': retriever | format_docs, 'question': RunnablePassthrough()}
        | prompt
        | get_llm()
        | StrOutputParser()
    )

def extract_nodal_parameters(question: str) -> Dict[str, Any]:
    structured_prompt = (
        question
        + "\n\nReturn ONLY a JSON object with keys: "
        + "reservoir_pressure_bar, wellhead_pressure_bar, productivity_index_m3hr_per_bar, "
        + "fluid_density_kg_m3, fluid_viscosity_pa_s, esp_depth_m, roughness_m, "
        + "pump_curve (list of {\"flow_m3hr\", \"head_m\"}), "
        + "well_segments (list of {\"start_depth_m\", \"end_depth_m\", \"diameter_m\"}), notes."
        + "\nEnsure numbers are numeric."
        + "\nIf unknown, set null and describe the gap in notes."
    )
    rag_chain = get_rag_chain()
    raw_response = rag_chain.invoke(structured_prompt)

    def try_parse_json(text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start:end + 1])
            raise

    try:
        return try_parse_json(raw_response)
    except json.JSONDecodeError as exc:
        return {'raw_response': raw_response, 'error': str(exc)}

defaults = {
    'reservoir_pressure_bar': 230.0,
    'wellhead_pressure_bar': 10.0,
    'productivity_index_m3hr_per_bar': 5.0,
    'fluid_density_kg_m3': 1000.0,
    'fluid_viscosity_pa_s': 1e-3,
    'esp_depth_m': 500.0,
    'roughness_m': 1e-5,
}

def coerce_parameters(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = defaults.copy()
    if not isinstance(payload, dict):
        result['notes'] = 'Extraction failed; defaults applied.'
        return result
    for key, value in payload.items():
        if key in {'pump_curve', 'well_segments', 'notes', 'raw_response', 'error'}:
            result[key] = value
        elif value is None:
            continue
        else:
            try:
                result[key] = float(value)
            except (TypeError, ValueError):
                continue
    pump_curve = payload.get('pump_curve') if isinstance(payload, dict) else None
    if not pump_curve:
        pump_curve = [
            {'flow_m3hr': 0.0, 'head_m': 600.0},
            {'flow_m3hr': 100.0, 'head_m': 550.0},
            {'flow_m3hr': 200.0, 'head_m': 450.0},
            {'flow_m3hr': 300.0, 'head_m': 300.0},
            {'flow_m3hr': 400.0, 'head_m': 100.0},
        ]
        note = result.get('notes', '')
        result['notes'] = (note + (' ' if note else '') + 'Fallback pump curve applied.').strip()
    result['pump_curve'] = pump_curve
    segments = payload.get('well_segments') if isinstance(payload, dict) else None
    if not segments:
        segments = [
            {'start_depth_m': 0.0, 'end_depth_m': 500.0, 'diameter_m': 0.3397},
            {'start_depth_m': 500.0, 'end_depth_m': 1500.0, 'diameter_m': 0.2445},
            {'start_depth_m': 1500.0, 'end_depth_m': 2500.0, 'diameter_m': 0.1778},
        ]
        note = result.get('notes', '')
        result['notes'] = (note + (' ' if note else '') + 'Fallback well geometry applied.').strip()
    result['well_segments'] = segments
    return result

def build_segments(segment_specs: List[Dict[str, float]]) -> List[Dict[str, float]]:
    ordered = sorted(segment_specs, key=lambda seg: seg.get('start_depth_m', 0.0))
    segments: List[Dict[str, float]] = []
    for spec in ordered:
        start = float(spec.get('start_depth_m', 0.0))
        end = float(spec.get('end_depth_m', start))
        segments.append({'length_m': max(end - start, 0.0), 'diameter_m': float(spec.get('diameter_m', 0.1))})
    return segments

def swamee_jain(reynolds: float, diameter: float, roughness: float) -> float:
    if reynolds <= 0:
        return 0.0
    return 0.25 / (math.log10((roughness / (3.7 * diameter)) + (5.74 / (reynolds ** 0.9)))) ** 2

def interpolate_pump_head(flow: float, curve: List[Dict[str, float]]) -> float:
    sorted_curve = sorted(curve, key=lambda item: item['flow_m3hr'])
    flows = [point['flow_m3hr'] for point in sorted_curve]
    heads = [point['head_m'] for point in sorted_curve]
    return float(np.interp(flow, flows, heads))

def compute_vlp(flow_m3hr: float, params: Dict[str, Any]) -> float:
    density = params['fluid_density_kg_m3']
    viscosity = params['fluid_viscosity_pa_s']
    roughness = params['roughness_m']
    wellhead_pressure = params['wellhead_pressure_bar']
    pump_curve = params['pump_curve']
    esp_depth = params['esp_depth_m']
    q_m3s = flow_m3hr / 3600.0
    total_dp_pa = 0.0
    depth_accum = 0.0
    for segment in build_segments(params['well_segments']):
        length = segment['length_m']
        diameter = segment['diameter_m']
        area = math.pi * diameter ** 2 / 4.0
        velocity = q_m3s / area if area else 0.0
        reynolds = density * abs(velocity) * diameter / max(viscosity, 1e-9)
        f = swamee_jain(reynolds, diameter, roughness)
        dp_fric = f * (length / max(diameter, 1e-6)) * (density * velocity ** 2 / 2.0)
        dp_grav = density * 9.81 * length
        total_dp_pa += dp_fric + dp_grav
        depth_accum += length
    if depth_accum >= esp_depth:
        total_dp_pa -= density * 9.81 * interpolate_pump_head(flow_m3hr, pump_curve)
    return wellhead_pressure + total_dp_pa / 1e5

def compute_ipr(flow_m3hr: float, params: Dict[str, Any]) -> float:
    reservoir_pressure = params['reservoir_pressure_bar']
    pi = params['productivity_index_m3hr_per_bar']
    return max(reservoir_pressure - flow_m3hr / max(pi, 1e-6), 0.0)

def run_nodal_analysis(params: Dict[str, Any], max_flow: Optional[float] = None) -> Dict[str, Any]:
    if max_flow is None:
        max_flow = max(point['flow_m3hr'] for point in params['pump_curve'])
    flows = np.linspace(1, max_flow, 200)
    vlp_curve = np.array([compute_vlp(f, params) for f in flows])
    ipr_curve = np.array([compute_ipr(f, params) for f in flows])
    diff = np.abs(vlp_curve - ipr_curve)
    idx = int(np.argmin(diff))
    tolerance = 3.0
    solution = None
    if diff[idx] <= tolerance:
        solution = {
            'flow_m3hr': float(flows[idx]),
            'bottomhole_pressure_bar': float(vlp_curve[idx]),
            'pump_head_m': float(interpolate_pump_head(flows[idx], params['pump_curve'])),
            'delta_bar': float(diff[idx]),
        }
    return {'flows': flows, 'vlp': vlp_curve, 'ipr': ipr_curve, 'solution': solution}

def build_operator_summary(solution: Optional[Dict[str, float]], analysis: Dict[str, Any], params: Dict[str, Any], raw_payload: Dict[str, Any]) -> Dict[str, Any]:
    if not solution:
        narrative = (
            "The analysis could not find an operating point where inflow and outflow match within the set tolerance.\n\n"
            "This usually means the assumed pump support or reservoir drive needs review."
        )
        bullets = [
            'Confirm pump performance data.',
            'Try expanding the tested flow range.',
        ]
        table = pd.DataFrame([{
            'Flowrate (m3/hr)': None,
            'Bottomhole Pressure (bar)': None,
            'Pump Head (m)': None,
            'Difference (bar)': None,
        }])
        return {'narrative': narrative, 'bullets': bullets, 'table': table}
    flow = solution['flow_m3hr']
    pressure = solution['bottomhole_pressure_bar']
    pump_head = solution['pump_head_m']
    deviation = solution['delta_bar']
    narrative = (
        f'The well should stabilise near {flow:.0f} cubic metres per hour. '
        f'At that rate the bottomhole flowing pressure is roughly {pressure:.0f} bar, '
        f'supported by an estimated pump head near {pump_head:.0f} metres. '
        'This operating point balances the reservoir inflow and tubing outflow so the lift system can run smoothly.'
    )
    bullets = [
        f'Expected flowrate: {flow:.1f} m3/hr',
        f'Flowing pressure at depth: {pressure:.1f} bar',
        f'Pump head required: {pump_head:.1f} m',
        f'Curve match difference: {deviation:.2f} bar',
    ]
    table = pd.DataFrame([
        {'Metric': 'Steady flowrate', 'Value': f'{flow:.1f} m3/hr'},
        {'Metric': 'Bottomhole pressure', 'Value': f'{pressure:.1f} bar'},
        {'Metric': 'Pump head', 'Value': f'{pump_head:.1f} m'},
        {'Metric': 'Curve difference', 'Value': f'{deviation:.2f} bar'},
    ])
    if raw_payload.get('notes'):
        bullets.append(f"Notes: {raw_payload['notes']}")
    return {'narrative': narrative, 'bullets': bullets, 'table': table}

def run_nodal_analysis_agent(question: str) -> Dict[str, Any]:
    payload = extract_nodal_parameters(question)
    params = coerce_parameters(payload)
    analysis = run_nodal_analysis(params)
    summary = build_operator_summary(analysis.get('solution'), analysis, params, payload if isinstance(payload, dict) else {})
    summary['analysis'] = analysis
    summary['params'] = params
    summary['raw_payload'] = payload
    return summary

def render_summary(summary: Dict[str, Any]) -> None:
    st.subheader('Operator Guidance')
    st.write(summary['narrative'])
    if summary.get('bullets'):
        st.markdown("\n".join(f"- {item}" for item in summary['bullets']))
    if summary.get('table') is not None:
        st.dataframe(summary['table'], use_container_width=True)
    with st.expander('Diagnostic details', expanded=False):
        st.json({
            'parameters': summary.get('params'),
            'raw_payload': summary.get('raw_payload'),
        })

def main() -> None:
    st.set_page_config(page_title='Well 1 Nodal Analysis Chat', page_icon='🛢️', layout='centered')
    st.title('Well 1 Nodal Analysis Assistant')
    st.caption('Ask one question at a time. The assistant retrieves Well 1 context, runs the nodal analysis, and answers in operator language.')
    sample_questions = [
        'How much can Well 1 produce with the installed pump?',
        'What pump head is required for 200 m3/hr on Well 1?',
        'Summarise the current nodal analysis for Well 1.',
    ]
    with st.sidebar:
        st.markdown('### Tips')
        st.markdown('Use clear production questions. The agent will search the Well 1 files and report the most likely operating point.')
        st.markdown('**Sample prompts:**')
        for question in sample_questions:
            st.markdown(f'- {question}')
        st.markdown('---')
        st.markdown(f'**Data root:** `{WELL_DATA_DIR}`')
        st.markdown(f'**Vector DB:** `{DB_DIR}`')
    question = st.text_area('Question about Well 1', placeholder='Example: Estimate the steady production rate for Well 1.', height=120)
    run_clicked = st.button('Run nodal analysis', use_container_width=True)
    if run_clicked:
        if not question.strip():
            st.warning('Please enter a question before running the analysis.')
            return
        with st.spinner('Consulting Well 1 knowledge base...'):
            summary = run_nodal_analysis_agent(question.strip())
        render_summary(summary)
    st.markdown('---')
    st.markdown('Need another scenario? Adjust the question and rerun the analysis.')

if __name__ == '__main__':
    main()
