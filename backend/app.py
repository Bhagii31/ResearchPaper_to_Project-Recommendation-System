import os
import re
import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Optional deps for "any link" HTML extraction
try:
    import trafilatura  # best HTML-to-text extractor
except Exception:
    trafilatura = None

try:
    from bs4 import BeautifulSoup  # fallback extractor
except Exception:
    BeautifulSoup = None


# -----------------------------
# Config
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
# Support BOTH env var names (your compose uses LLM_MODEL)
OPENAI_MODEL = (os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL") or "gpt-4.1-mini").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "").strip()

MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "140000"))  # guardrail for LLM input
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "25"))
OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "35"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "2"))

# CORS: set in Render env like:
# ALLOWED_ORIGINS=https://your-frontend.vercel.app,http://localhost:3000
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Paper2Project Backend", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Models
# -----------------------------
class ResourceItem(BaseModel):
    title: str
    url: str
    source: str
    extra: Dict[str, Any] = Field(default_factory=dict)


class ExtractedPaper(BaseModel):
    title: Optional[str] = None
    paper_summary: Optional[str] = None  # 3–4 sentences
    technologies: List[str] = Field(default_factory=list)
    abstract: Optional[str] = None
    problem: Optional[str] = None
    methods: List[str] = Field(default_factory=list)
    tasks: List[str] = Field(default_factory=list)
    domains: List[str] = Field(default_factory=list)
    prerequisites: List[str] = Field(default_factory=list)
    key_terms: List[str] = Field(default_factory=list)


class BeginnerProjectIdea(BaseModel):
    name: str
    difficulty: str  # easy/medium
    description: str
    milestones: List[str]
    baseline_model: str
    metrics: List[str]
    suggested_stack: List[str]


class RunOutput(BaseModel):
    run_id: str
    extracted: ExtractedPaper
    projects: List[BeginnerProjectIdea]
    datasets: List[ResourceItem]
    youtube: List[ResourceItem]


# -----------------------------
# Robust JSON parsing for LLM responses
# -----------------------------
def _extract_json_object(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if text.startswith("{") and text.endswith("}"):
        return text
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if m:
        return m.group(1).strip()
    m = re.search(r"(\{.*\})", text, flags=re.S)
    if m:
        return m.group(1).strip()
    return ""


async def openai_json(system: str, user: str) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set inside the backend container.")

    url = f"{OPENAI_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

    payload = {
        "model": OPENAI_MODEL,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
    }

    last_err: Optional[Exception] = None
    for _ in range(OPENAI_MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=OPENAI_TIMEOUT) as client:
                r = await client.post(url, headers=headers, json=payload)

            if r.status_code >= 400:
                raise HTTPException(status_code=500, detail=f"OpenAI error {r.status_code}: {r.text}")

            data = r.json()
            txt = data["choices"][0]["message"]["content"]
            js = _extract_json_object(txt)
            if not js:
                raise HTTPException(status_code=500, detail=f"Model did not return JSON. Raw: {txt[:700]}")
            return json.loads(js)

        except Exception as e:
            last_err = e

    raise HTTPException(status_code=500, detail=f"OpenAI request failed: {last_err}")


# -----------------------------
# PDF extraction
# -----------------------------
def extract_pdf_text(pdf_bytes: bytes, max_chars: int = MAX_INPUT_CHARS) -> Tuple[str, Optional[str]]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    meta_title = None
    try:
        meta = doc.metadata or {}
        meta_title = (meta.get("title") or "").strip() or None
    except Exception:
        meta_title = None

    parts: List[str] = []
    total = 0
    for i in range(len(doc)):
        txt = (doc[i].get_text("text") or "").strip()
        if not txt:
            continue
        chunk = f"\n\n--- Page {i+1} ---\n{txt}"
        parts.append(chunk)
        total += len(chunk)
        if total >= max_chars:
            break

    return "\n".join(parts).strip(), meta_title


def select_relevant_text(full_text: str, max_chars: int = 90000) -> str:
    t = full_text or ""
    if len(t) <= max_chars:
        return t

    anchors = [
        "abstract", "introduction", "background",
        "method", "methods", "approach",
        "experiment", "experiments", "results",
        "discussion", "conclusion"
    ]
    lower = t.lower()
    spans: List[Tuple[int, int]] = []

    for a in anchors:
        idx = lower.find(a)
        if idx != -1:
            start = max(0, idx - 1200)
            end = min(len(t), idx + 14000)
            spans.append((start, end))

    spans.sort()
    merged: List[Tuple[int, int]] = []
    for s, e in spans:
        if not merged or s > merged[-1][1]:
            merged.append((s, e))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))

    selected = ""
    for s, e in merged:
        selected += t[s:e] + "\n\n"

    selected = selected.strip()
    if len(selected) < 12000:
        selected = t[:65000] + "\n\n--- Tail ---\n" + t[-20000:]

    return selected[:max_chars]


# -----------------------------
# ANY LINK ingestion (PDF or HTML)
# -----------------------------
async def fetch_text_or_pdf_from_url(url: str) -> Tuple[Optional[str], Optional[bytes], Optional[str]]:
    """
    Returns (page_text, pdf_bytes, guessed_title).
    - If PDF URL -> pdf_bytes is set
    - If HTML URL -> page_text is set (extracted text)
    """
    u = (url or "").strip()
    if not u:
        raise HTTPException(status_code=400, detail="paper_url is empty.")

    # arXiv abs -> PDF
    if "arxiv.org/abs/" in u:
        arxiv_id = u.split("arxiv.org/abs/")[-1].strip("/")
        u = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    if "arxiv.org/pdf/" in u and not u.endswith(".pdf"):
        u = u + ".pdf"

    headers = {
        "User-Agent": "Mozilla/5.0 (Paper2ProjectBot/1.0)",
        "Accept": "*/*",
    }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True, headers=headers) as client:
            r = await client.get(u)
            r.raise_for_status()
            ctype = (r.headers.get("content-type") or "").lower()
            data = r.content
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch URL: {e}")

    # PDF?
    if "pdf" in ctype or u.lower().endswith(".pdf") or data.startswith(b"%PDF"):
        return None, data, None

    # HTML -> extract readable text
    html = data.decode("utf-8", errors="ignore")

    # Title (best-effort)
    guessed_title = None
    tm = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.I | re.S)
    if tm:
        guessed_title = re.sub(r"\s+", " ", tm.group(1)).strip()[:180] or None

    # Best: trafilatura
    if trafilatura is not None:
        try:
            extracted = trafilatura.extract(html, include_comments=False, include_tables=False)
            if extracted and len(extracted.strip()) > 600:
                return extracted.strip()[:MAX_INPUT_CHARS], None, guessed_title
        except Exception:
            pass

    # Fallback: BeautifulSoup
    if BeautifulSoup is not None:
        try:
            soup = BeautifulSoup(html, "lxml")
            for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
                tag.decompose()
            text = soup.get_text(separator="\n")
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            if len(text) > 600:
                return text[:MAX_INPUT_CHARS], None, guessed_title
        except Exception:
            pass

    # Last resort: strip tags crudely
    text = re.sub(r"<script.*?>.*?</script>", " ", html, flags=re.I | re.S)
    text = re.sub(r"<style.*?>.*?</style>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[:MAX_INPUT_CHARS], None, guessed_title


# -----------------------------
# Fallback extraction if OpenAI fails
# -----------------------------
def _find_abstract(text: str) -> Optional[str]:
    t = text or ""
    m = re.search(
        r"\babstract\b[\s:\-—]*\n(.*?)(\n\s*(?:1\s+)?introduction\b|\n\s*keywords\b|\n\s*index terms\b)",
        t,
        flags=re.I | re.S,
    )
    if m:
        abs_txt = re.sub(r"\s+", " ", m.group(1).strip())
        return abs_txt[:2500]
    return None


def _sentences(s: str) -> List[str]:
    s = re.sub(r"\s+", " ", (s or "").strip())
    if not s:
        return []
    parts = re.split(r"(?<=[.!?])\s+", s)
    return [p.strip() for p in parts if p.strip()]


def fallback_extract(full_text: str, meta_title: Optional[str]) -> ExtractedPaper:
    abstract = _find_abstract(full_text)
    title = meta_title

    if not title:
        for line in (full_text or "").splitlines()[:50]:
            line = line.strip()
            if 10 < len(line) < 180 and re.search(r"[A-Za-z]", line):
                title = line
                break

    base = abstract or (full_text[:2500] if full_text else "")
    sents = _sentences(base)
    paper_summary = " ".join(sents[:4]) if sents else None

    lower = (full_text or "").lower()
    tech_map = [
        ("LLM", ["llm", "large language model"]),
        ("transformer", ["transformer", "attention"]),
        ("RAG", ["retrieval augmented", "rag"]),
        ("BERT", ["bert"]),
        ("CNN", ["cnn", "convolutional"]),
        ("PyTorch", ["pytorch"]),
        ("TensorFlow", ["tensorflow"]),
        ("OpenCV", ["opencv"]),
        ("NLP", ["nlp", "natural language"]),
        ("computer vision", ["computer vision", "image", "vision"]),
        ("time series", ["time series"]),
        ("classification", ["classification"]),
        ("segmentation", ["segmentation"]),
        ("object detection", ["object detection", "yolo", "rcnn"]),
        ("SSIM", ["ssim"]),
        ("PSNR", ["psnr"]),
        ("image fusion", ["image fusion", "fusion"]),
    ]

    technologies: List[str] = []
    for canonical, needles in tech_map:
        if any(n in lower for n in needles):
            technologies.append(canonical)

    # Try to parse keywords line
    km = re.search(r"\bkeywords\b[\s:\-—]*\n?(.*)", full_text, flags=re.I)
    if km:
        kw_line = km.group(1).split("\n")[0]
        for kw in re.split(r"[;,•·]", kw_line):
            kw = kw.strip()
            if 2 < len(kw) < 40:
                technologies.append(kw)

    seen = set()
    tech_clean = []
    for t in technologies:
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        tech_clean.append(t)

    domains = []
    if any(x in lower for x in ["medical", "radiology", "mri", "ct", "x-ray", "ultrasound"]):
        domains.append("medical")
    if any(x in lower for x in ["image", "vision", "segmentation", "detection"]):
        domains.append("computer vision")
    if any(x in lower for x in ["nlp", "language", "text"]):
        domains.append("nlp")

    return ExtractedPaper(
        title=title,
        abstract=abstract,
        paper_summary=paper_summary,
        technologies=tech_clean[:18],
        domains=domains,
        methods=[],
        tasks=[],
        prerequisites=[],
        key_terms=[],
        problem=None,
    )


def fallback_projects(extracted: ExtractedPaper) -> List[BeginnerProjectIdea]:
    tech = " ".join(extracted.technologies or []).lower()
    dom = " ".join(extracted.domains or []).lower()
    title = (extracted.title or "Paper").strip()

    if "nlp" in dom or "nlp" in tech or "language" in tech:
        return [
            BeginnerProjectIdea(
                name="Paper-to-Task Baseline (Text Classification)",
                difficulty="easy",
                description="Build a baseline classifier matching the paper’s task using a small public dataset and evaluate performance.",
                milestones=[
                    "Pick a public dataset matching the task",
                    "Implement preprocessing + baseline model",
                    "Train/evaluate with accuracy/F1",
                    "Add one improvement (better tokenizer/embeddings)",
                    "Write README + demo notebook",
                ],
                baseline_model="LogReg/Linear SVM or DistilBERT baseline",
                metrics=["Accuracy", "F1"],
                suggested_stack=["Python", "scikit-learn", "PyTorch", "Hugging Face"],
            )
        ]

    if "computer vision" in dom or "image" in tech:
        return [
            BeginnerProjectIdea(
                name="Paper-Inspired Vision Baseline",
                difficulty="easy",
                description="Implement a simplified baseline from the paper’s vision pipeline and test on a public dataset.",
                milestones=[
                    "Choose a relevant dataset (classification/segmentation)",
                    "Implement baseline model (ResNet/U-Net)",
                    "Train + evaluate",
                    "Add one ablation or improvement",
                    "Build a small demo page/notebook",
                ],
                baseline_model="ResNet / U-Net baseline",
                metrics=["Accuracy (or IoU/DSC)"],
                suggested_stack=["Python", "PyTorch", "torchvision", "OpenCV"],
            )
        ]

    return [
        BeginnerProjectIdea(
            name=f"{title}: Simplified Reproduction",
            difficulty="easy",
            description="Recreate a simplified version of the paper’s approach and validate on a public dataset.",
            milestones=[
                "Extract task + metrics from paper",
                "Pick a small public dataset",
                "Implement baseline pipeline",
                "Run evaluation + quick ablation",
                "Write README + demo notebook",
            ],
            baseline_model="Simplified baseline from the paper",
            metrics=["Task-appropriate metric"],
            suggested_stack=["Python", "NumPy", "PyTorch"],
        )
    ]


# -----------------------------
# Dataset search (HF + PapersWithCode)
# -----------------------------
async def search_hf_datasets(query: str, limit: int = 6) -> List[ResourceItem]:
    q = (query or "").strip()
    if not q:
        return []
    url = "https://huggingface.co/api/datasets"
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            r = await client.get(url, params={"search": q})
            if r.status_code >= 400:
                return []
            data = r.json()
    except Exception:
        return []

    out: List[ResourceItem] = []
    for d in (data or [])[:limit]:
        ds_id = d.get("id") or d.get("name")
        if not ds_id:
            continue
        out.append(
            ResourceItem(
                title=ds_id,
                url=f"https://huggingface.co/datasets/{ds_id}",
                source="huggingface",
                extra={"likes": d.get("likes"), "downloads": d.get("downloads")},
            )
        )
    return out


async def search_pwc_datasets(query: str, limit: int = 6) -> List[ResourceItem]:
    q = (query or "").strip()
    if not q:
        return []
    url = "https://paperswithcode.com/api/v1/datasets/"
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            r = await client.get(url, params={"search": q})
            if r.status_code >= 400:
                return []
            data = r.json()
    except Exception:
        return []

    out: List[ResourceItem] = []
    for item in (data.get("results") or [])[:limit]:
        name = item.get("name") or "Dataset"
        rel = item.get("url") or ""
        full = ("https://paperswithcode.com" + rel) if rel.startswith("/") else rel
        if not full:
            continue
        out.append(ResourceItem(title=name, url=full, source="paperswithcode", extra={}))
    return out


def dedupe_by_url(items: List[ResourceItem]) -> List[ResourceItem]:
    seen = set()
    out = []
    for it in items:
        if it.url in seen:
            continue
        seen.add(it.url)
        out.append(it)
    return out


# -----------------------------
# YouTube search
# -----------------------------
async def search_youtube_videos(query: str, limit: int = 6) -> List[ResourceItem]:
    if not YOUTUBE_API_KEY:
        return []

    q = (query or "").strip()
    if not q:
        return []

    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "key": YOUTUBE_API_KEY,
        "part": "snippet",
        "q": q,
        "maxResults": limit,
        "type": "video",
        "safeSearch": "strict",
    }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            r = await client.get(url, params=params)
            # Don't crash on key restrictions/quotas
            if r.status_code in (401, 403, 429):
                return []
            r.raise_for_status()
            data = r.json()
    except Exception:
        return []

    out: List[ResourceItem] = []
    for item in data.get("items", []):
        vid = item.get("id", {}).get("videoId")
        snip = item.get("snippet", {}) or {}
        if not vid:
            continue
        out.append(
            ResourceItem(
                title=snip.get("title") or "YouTube Video",
                url=f"https://www.youtube.com/watch?v={vid}",
                source="youtube",
                extra={
                    "channel": snip.get("channelTitle") or "",
                    "description": snip.get("description") or "",
                },
            )
        )
    return out


def build_youtube_queries(extracted: ExtractedPaper, projects: List[BeginnerProjectIdea]) -> List[str]:
    """
    Dynamic queries based on THIS paper (no static CT/MRI).
    Also avoids Autodesk 'Fusion 360' drift by using 'image fusion' when relevant.
    """
    title = (extracted.title or "").strip()
    tech = [t.strip() for t in (extracted.technologies or []) if t and len(t) < 60]
    dom = [d.strip() for d in (extracted.domains or []) if d and len(d) < 60]
    keys = [k.strip() for k in (extracted.key_terms or []) if k and len(k) < 60]

    blob = " ".join([title, " ".join(dom), " ".join(tech), " ".join(keys)]).lower()

    topic_parts: List[str] = []
    if dom:
        topic_parts.append(dom[0])

    if "fusion" in blob:
        # if the paper is about fusion + images -> force "image fusion" to avoid CAD results
        if "image" in blob or "imaging" in blob or "vision" in blob:
            topic_parts.append("image fusion")
        else:
            topic_parts.append("fusion")

    topic = " ".join([p for p in topic_parts if p]).strip()
    if not topic:
        topic = title or (dom[0] if dom else (tech[0] if tech else "research paper"))

    modifiers: List[str] = []
    for t in tech + keys:
        tl = t.lower()
        if len(t) < 3:
            continue
        if tl in ["fusion", "image fusion"] or tl in (topic.lower(),):
            continue
        modifiers.append(t)
        if len(modifiers) >= 3:
            break

    queries = [
        f"{topic} tutorial",
        f"{topic} implementation python",
        f"{topic} explained",
    ]

    if title:
        queries.append(f"{title} paper explained")

    for m in modifiers:
        queries.append(f"{topic} {m} tutorial")

    for p in projects[:2]:
        queries.append(f"{p.name} tutorial")
        queries.append(f"{topic} {p.name} implementation")

    # de-dupe
    seen = set()
    out = []
    for q in queries:
        q = q.strip()
        if not q or q in seen:
            continue
        seen.add(q)
        out.append(q)
    return out[:10]


# -----------------------------
# LLM prompts
# -----------------------------
EXTRACT_SYSTEM = (
    "You are an ML engineer translating a research paper (or technical article) for beginners.\n"
    "Return STRICT JSON only.\n"
    "paper_summary MUST be 3–4 sentences (plain language, no bullets).\n"
    "technologies MUST be concrete technologies/methods/tools actually present in the text.\n"
    "Do not invent datasets or claims not supported by the input."
)

EXTRACT_USER_TEMPLATE = """Extract the following fields from the text.

Return JSON with keys:
- title (string or null)
- paper_summary (string or null)
- technologies (array of strings)
- abstract (string or null)
- problem (string or null)
- methods (array of strings)
- tasks (array of strings)
- domains (array of strings)
- prerequisites (array of strings)
- key_terms (array of strings)

Text:
{paper_text}
"""

PROJECT_SYSTEM = (
    "You are a senior ML engineer and mentor.\n"
    "Return STRICT JSON only.\n"
    "Generate 2–3 beginner-friendly end-to-end project ideas based on the extracted info.\n"
    "Each project should be buildable in 1–2 weeks by a student.\n"
    "Use clear milestones."
)

PROJECT_USER_TEMPLATE = """Given this extracted info, generate 2–3 projects.

Return JSON with key:
- projects: array of objects with fields:
  - name (string)
  - difficulty ("easy" or "medium")
  - description (string)
  - milestones (array of strings)
  - baseline_model (string)
  - metrics (array of strings)
  - suggested_stack (array of strings)

Extracted info:
{extracted_json}
"""


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return {"message": "Paper2Project backend is running. Use /health or /api/run"}


@app.get("/health")
def health():
    return {"ok": True, "model": OPENAI_MODEL}


@app.post("/api/run", response_model=RunOutput)
async def run(
    pdf: Optional[UploadFile] = File(None),
    paper_url: Optional[str] = Form(None),
):
    if pdf is None and (paper_url is None or not paper_url.strip()):
        raise HTTPException(status_code=400, detail="Provide a PDF (pdf) or a URL (paper_url).")

    # 1) Get text
    meta_title: Optional[str] = None
    full_text: str = ""

    if pdf is not None:
        try:
            pdf_bytes = await pdf.read()
        except Exception:
            raise HTTPException(status_code=400, detail="Could not read uploaded PDF.")

        if not pdf_bytes:
            raise HTTPException(status_code=400, detail="Empty PDF.")

        full_text, meta_title = extract_pdf_text(pdf_bytes)

        if len(full_text) < 1200:
            raise HTTPException(
                status_code=400,
                detail="This PDF looks scanned/image-only (not enough extractable text). Try an arXiv link or a text PDF.",
            )

    else:
        # ANY link: PDF or HTML
        page_text, maybe_pdf, guessed_title = await fetch_text_or_pdf_from_url(paper_url or "")
        meta_title = guessed_title

        if maybe_pdf:
            full_text, meta_title2 = extract_pdf_text(maybe_pdf)
            meta_title = meta_title2 or meta_title
            if len(full_text) < 1200:
                raise HTTPException(
                    status_code=400,
                    detail="This PDF link looks scanned/image-only (not enough extractable text). Try a different source.",
                )
        else:
            full_text = page_text or ""
            if len(full_text) < 600:
                raise HTTPException(status_code=400, detail="Could not extract enough readable text from this URL.")

    # 2) Prepare text for LLM
    llm_text = select_relevant_text(full_text)

    # 3) Extract using LLM (fallback if OpenAI fails)
    extracted: ExtractedPaper
    extracted_raw: Dict[str, Any]

    try:
        extract_user = EXTRACT_USER_TEMPLATE.format(paper_text=llm_text[:MAX_INPUT_CHARS])
        extracted_raw = await openai_json(EXTRACT_SYSTEM, extract_user)
        extracted = ExtractedPaper(**extracted_raw)
        if not extracted.title and meta_title:
            extracted.title = meta_title
    except Exception:
        extracted = fallback_extract(full_text, meta_title)
        extracted_raw = extracted.model_dump()

    # 4) Projects using LLM (fallback)
    projects: List[BeginnerProjectIdea] = []
    try:
        proj_user = PROJECT_USER_TEMPLATE.format(extracted_json=json.dumps(extracted_raw, ensure_ascii=False))
        proj_raw = await openai_json(PROJECT_SYSTEM, proj_user)
        for p in (proj_raw.get("projects") or [])[:3]:
            try:
                projects.append(BeginnerProjectIdea(**p))
            except Exception:
                continue
        if not projects:
            projects = fallback_projects(extracted)
    except Exception:
        projects = fallback_projects(extracted)

    # 5) Dataset search
    dataset_queries: List[str] = []
    if extracted.title:
        dataset_queries.append(extracted.title)
    dataset_queries += (extracted.technologies or [])[:3]
    dataset_queries += (extracted.domains or [])[:2]
    dataset_queries = [q for q in dataset_queries if q and len(q) < 90]

    datasets: List[ResourceItem] = []
    for q in dataset_queries[:6]:
        datasets += await search_pwc_datasets(q, limit=4)
        datasets += await search_hf_datasets(q, limit=4)
    datasets = dedupe_by_url(datasets)[:15]

    # 6) YouTube (paper-specific queries)
    youtube: List[ResourceItem] = []
    for q in build_youtube_queries(extracted, projects):
        youtube += await search_youtube_videos(q, limit=6)
    youtube = dedupe_by_url(youtube)[:15]

    return RunOutput(
        run_id=str(uuid.uuid4())[:8],
        extracted=extracted,
        projects=projects,
        datasets=datasets,
        youtube=youtube,
    )