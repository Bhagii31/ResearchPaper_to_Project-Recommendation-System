"use client";

import React, { useMemo, useState } from "react";

type ResourceItem = {
  title: string;
  url: string;
  source: string;
  extra?: Record<string, any>;
};

type Project = {
  name: string;
  difficulty: string;
  description: string;
  milestones: string[];
  baseline_model: string;
  metrics: string[];
  suggested_stack: string[];
};

type Output = {
  run_id: string;
  extracted: {
    title?: string | null;
    paper_summary?: string | null;
    technologies?: string[];
    abstract?: string | null;
    problem?: string | null;
    methods: string[];
    tasks: string[];
    domains: string[];
    prerequisites: string[];
    key_terms: string[];
  };
  projects: Project[];
  datasets: ResourceItem[];
  youtube: ResourceItem[];
};

function getVideoId(urlStr: string): string | null {
  try {
    const url = new URL(urlStr);
    const v = url.searchParams.get("v");
    if (v) return v;
    if (url.hostname.includes("youtu.be")) return url.pathname.replace("/", "").trim() || null;
    const parts = url.pathname.split("/").filter(Boolean);
    const embedIndex = parts.indexOf("embed");
    if (embedIndex !== -1 && parts[embedIndex + 1]) return parts[embedIndex + 1];
    return null;
  } catch {
    return null;
  }
}

function Pill({ children }: { children: React.ReactNode }) {
  return (
    <span
      style={{
        fontSize: 12,
        padding: "7px 10px",
        borderRadius: 999,
        border: "1px solid rgba(255,255,255,0.12)",
        background: "rgba(255,255,255,0.06)",
        color: "rgba(255,255,255,0.92)",
        fontWeight: 700,
      }}
    >
      {children}
    </span>
  );
}

function Button({
  children,
  onClick,
  disabled,
  variant = "primary",
}: {
  children: React.ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  variant?: "primary" | "secondary";
}) {
  const isPrimary = variant === "primary";
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        cursor: disabled ? "not-allowed" : "pointer",
        padding: "11px 14px",
        borderRadius: 14,
        background: disabled
          ? "rgba(255,255,255,0.10)"
          : isPrimary
            ? "linear-gradient(135deg, rgba(140,120,255,0.55), rgba(0,220,255,0.22))"
            : "rgba(255,255,255,0.08)",
        border: "1px solid rgba(255,255,255,0.14)",
        color: "white",
        fontWeight: 950,
        letterSpacing: 0.2,
        whiteSpace: "nowrap",
      }}
    >
      {children}
    </button>
  );
}

function Card({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
}) {
  return (
    <div
      style={{
        borderRadius: 20,
        border: "1px solid rgba(255,255,255,0.10)",
        background: "rgba(255,255,255,0.06)",
        boxShadow: "0 16px 45px rgba(0,0,0,0.20)",
        overflow: "hidden",
      }}
    >
      <div
        style={{
          padding: "16px 18px",
          borderBottom: "1px solid rgba(255,255,255,0.10)",
          display: "flex",
          justifyContent: "space-between",
          gap: 12,
          alignItems: "baseline",
        }}
      >
        <div style={{ fontWeight: 950, fontSize: 14, color: "white" }}>{title}</div>
        {subtitle ? <div style={{ fontSize: 12, opacity: 0.75 }}>{subtitle}</div> : null}
      </div>
      <div style={{ padding: 18 }}>{children}</div>
    </div>
  );
}

function Modal({
  open,
  onClose,
  title,
  children,
}: {
  open: boolean;
  onClose: () => void;
  title: string;
  children: React.ReactNode;
}) {
  if (!open) return null;
  return (
    <div
      onClick={onClose}
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.70)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: 20,
        zIndex: 50,
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          width: "min(980px, 95vw)",
          borderRadius: 20,
          background: "rgba(15,15,18,0.96)",
          border: "1px solid rgba(255,255,255,0.14)",
          boxShadow: "0 22px 70px rgba(0,0,0,0.55)",
          overflow: "hidden",
        }}
      >
        <div
          style={{
            padding: 14,
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            borderBottom: "1px solid rgba(255,255,255,0.10)",
            color: "white",
          }}
        >
          <div style={{ fontWeight: 950 }}>{title}</div>
          <Button variant="secondary" onClick={onClose}>
            Close
          </Button>
        </div>
        <div style={{ padding: 14 }}>{children}</div>
      </div>
    </div>
  );
}

export default function Page() {
  const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

  const [file, setFile] = useState<File | null>(null);
  const [paperUrl, setPaperUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [out, setOut] = useState<Output | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const [activeVideoId, setActiveVideoId] = useState<string | null>(null);
  const [activeVideoTitle, setActiveVideoTitle] = useState<string>("");

  const youtubeCards = useMemo(() => {
    if (!out?.youtube) return [];
    return out.youtube
      .map((v) => {
        const id = getVideoId(v.url);
        if (!id) return null;
        const thumb = `https://i.ytimg.com/vi/${id}/hqdefault.jpg`;
        return { ...v, videoId: id, thumbnail: thumb };
      })
      .filter(Boolean) as Array<ResourceItem & { videoId: string; thumbnail: string }>;
  }, [out]);

  async function callRun(fd: FormData) {
    const res = await fetch(`${API_BASE}/api/run`, { method: "POST", body: fd });
    if (!res.ok) throw new Error(await res.text());
    return (await res.json()) as Output;
  }

  async function onGenerateFromPdf() {
    if (!file) {
      setErr("Please choose a PDF file first.");
      return;
    }
    setLoading(true);
    setErr(null);
    setOut(null);
    try {
      const fd = new FormData();
      fd.append("pdf", file);
      const data = await callRun(fd);
      setOut(data);
    } catch (e: any) {
      setErr(e?.message ?? "Error");
    } finally {
      setLoading(false);
    }
  }

  async function onGenerateFromLink() {
    if (!paperUrl.trim()) {
      setErr("Please paste a paper link first.");
      return;
    }
    setLoading(true);
    setErr(null);
    setOut(null);
    try {
      const fd = new FormData();
      fd.append("paper_url", paperUrl.trim());
      const data = await callRun(fd);
      setOut(data);
    } catch (e: any) {
      setErr(e?.message ?? "Error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main
      style={{
        minHeight: "100vh",
        color: "white",
        fontFamily: "ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial",
        background:
          "radial-gradient(900px 520px at 18% 12%, rgba(140,120,255,0.35), transparent 60%)," +
          "radial-gradient(820px 480px at 88% 20%, rgba(0,220,255,0.22), transparent 60%)," +
          "radial-gradient(700px 520px at 40% 95%, rgba(255,120,180,0.16), transparent 55%)," +
          "linear-gradient(180deg, rgba(255,255,255,0.06), transparent 35%)," +
          "#0b0b10",
      }}
    >
      <div style={{ maxWidth: 1120, margin: "0 auto", padding: "38px 16px 64px" }}>
        {/* Hero */}
        <div style={{ display: "grid", gap: 10, marginBottom: 18 }}>
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
            <Pill>LLM-powered</Pill>
            <Pill>Beginner-friendly</Pill>
            <Pill>Paper → Projects</Pill>
          </div>

          <h1 style={{ fontSize: 38, lineHeight: 1.05, margin: 0, fontWeight: 1000, letterSpacing: -0.8 }}>
            Paper-to-Project Recommendation Engine
          </h1>

          <p style={{ margin: 0, opacity: 0.88, maxWidth: 820, lineHeight: 1.55 }}>
            Upload a research paper PDF or paste any public link. Get a 3–4 sentence summary, extracted technologies,
            beginner-friendly project ideas, dataset links, and YouTube resources.
          </p>
        </div>

        {/* Inputs */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
          <Card title="Upload PDF" subtitle="Text-based PDFs work best">
            <div style={{ display: "grid", gap: 12 }}>
              <div
                style={{
                  borderRadius: 16,
                  padding: 14,
                  border: "1px dashed rgba(255,255,255,0.22)",
                  background: "rgba(0,0,0,0.18)",
                }}
              >
                <div style={{ fontWeight: 900, marginBottom: 6 }}>Choose a PDF</div>
                <input
                  type="file"
                  accept="application/pdf"
                  onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                  style={{ width: "100%", color: "white" }}
                />
                <div style={{ marginTop: 10, fontSize: 13, opacity: 0.85 }}>
                  <b>Selected:</b> {file ? file.name : "None"}
                </div>
              </div>

              <Button onClick={onGenerateFromPdf} disabled={loading || !file}>
                {loading ? "Generating…" : "Generate from PDF"}
              </Button>
            </div>
          </Card>

          <Card title="Paste link" subtitle="Any public URL (paper page or PDF link)">
            <div style={{ display: "grid", gap: 12 }}>
              <input
                value={paperUrl}
                onChange={(e) => setPaperUrl(e.target.value)}
                placeholder="Paste a link (arXiv, publisher page, blog post, direct PDF URL...)"
                style={{
                  width: "100%",
                  padding: "12px 12px",
                  borderRadius: 14,
                  border: "1px solid rgba(255,255,255,0.16)",
                  background: "rgba(255,255,255,0.06)",
                  color: "white",
                  outline: "none",
                }}
              />

              <Button onClick={onGenerateFromLink} disabled={loading || !paperUrl.trim()}>
                {loading ? "Generating…" : "Generate from link"}
              </Button>

              <div style={{ fontSize: 12, opacity: 0.78 }}>
                Note: Paywalled pages may only provide abstract/limited text. Best results from arXiv or open-access pages.
              </div>
            </div>
          </Card>
        </div>

        {/* Error */}
        {err && (
          <div
            style={{
              marginTop: 14,
              padding: 12,
              borderRadius: 16,
              border: "1px solid rgba(255,80,80,0.35)",
              background: "rgba(255,80,80,0.10)",
              whiteSpace: "pre-wrap",
              fontSize: 13,
            }}
          >
            {err}
          </div>
        )}

        {/* Results */}
        {out && (
          <div style={{ marginTop: 16, display: "grid", gridTemplateColumns: "1.1fr 0.9fr", gap: 14 }}>
            <div style={{ display: "grid", gap: 14 }}>
              <Card title={out.extracted.title ?? "Paper"} subtitle={`Run ID: ${out.run_id}`}>
                <div style={{ display: "grid", gap: 10 }}>
                  <div>
                    <div style={{ fontWeight: 950, marginBottom: 6 }}>Summary</div>
                    <div style={{ opacity: 0.92, lineHeight: 1.6 }}>
                      {out.extracted.paper_summary ?? "Summary not available from this paper text."}
                    </div>
                  </div>

                  <div>
                    <div style={{ fontWeight: 950, marginBottom: 6 }}>Technologies</div>
                    <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                      {(out.extracted.technologies ?? []).length ? (
                        out.extracted.technologies!.slice(0, 20).map((t, i) => <Pill key={i}>{t}</Pill>)
                      ) : (
                        <span style={{ opacity: 0.85 }}>No technologies extracted.</span>
                      )}
                    </div>
                  </div>
                </div>
              </Card>

              <Card title="Beginner projects" subtitle="2–3 buildable ideas">
                <div style={{ display: "grid", gap: 12 }}>
                  {out.projects.map((p, i) => (
                    <details
                      key={i}
                      style={{
                        borderRadius: 18,
                        border: "1px solid rgba(255,255,255,0.10)",
                        background: "rgba(0,0,0,0.20)",
                        padding: 14,
                      }}
                      open={i === 0}
                    >
                      <summary style={{ cursor: "pointer", listStyle: "none" as any }}>
                        <div style={{ display: "flex", justifyContent: "space-between", gap: 10, flexWrap: "wrap" }}>
                          <div style={{ fontSize: 16, fontWeight: 1000 }}>{p.name}</div>
                          <Pill>{p.difficulty}</Pill>
                        </div>
                        <div style={{ marginTop: 8, opacity: 0.92, lineHeight: 1.5 }}>{p.description}</div>
                      </summary>

                      <div style={{ marginTop: 12, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                        <div>
                          <div style={{ fontWeight: 950, marginBottom: 6 }}>Milestones</div>
                          <ol style={{ margin: 0, paddingLeft: 18, opacity: 0.92 }}>
                            {p.milestones.map((m, j) => (
                              <li key={j} style={{ marginBottom: 5 }}>
                                {m}
                              </li>
                            ))}
                          </ol>
                        </div>

                        <div style={{ opacity: 0.92, lineHeight: 1.55 }}>
                          <div>
                            <b>Baseline:</b> {p.baseline_model}
                          </div>
                          <div style={{ marginTop: 8 }}>
                            <b>Metrics:</b> {p.metrics.join(", ")}
                          </div>
                          <div style={{ marginTop: 8 }}>
                            <b>Stack:</b> {p.suggested_stack.join(", ")}
                          </div>
                        </div>
                      </div>
                    </details>
                  ))}
                </div>
              </Card>
            </div>

            <div style={{ display: "grid", gap: 14 }}>
              <Card title="Datasets" subtitle="Hugging Face + PapersWithCode">
                {out.datasets.length === 0 ? (
                  <div style={{ opacity: 0.85 }}>No datasets found.</div>
                ) : (
                  <div style={{ display: "grid", gap: 10 }}>
                    {out.datasets.slice(0, 12).map((d, i) => (
                      <a
                        key={i}
                        href={d.url}
                        target="_blank"
                        rel="noreferrer"
                        style={{
                          textDecoration: "none",
                          color: "white",
                          borderRadius: 16,
                          padding: 12,
                          border: "1px solid rgba(255,255,255,0.10)",
                          background: "rgba(0,0,0,0.20)",
                        }}
                      >
                        <div style={{ fontWeight: 950 }}>{d.title}</div>
                        <div style={{ fontSize: 12, opacity: 0.78, marginTop: 4 }}>{d.source}</div>
                      </a>
                    ))}
                  </div>
                )}
              </Card>

              <Card title="YouTube resources" subtitle="Click to watch in-app">
                {youtubeCards.length === 0 ? (
                  <div style={{ opacity: 0.85 }}>No YouTube results.</div>
                ) : (
                  <div style={{ display: "grid", gap: 12 }}>
                    {youtubeCards.slice(0, 10).map((v, i) => (
                      <button
                        key={i}
                        onClick={() => {
                          setActiveVideoId((v as any).videoId);
                          setActiveVideoTitle(v.title);
                        }}
                        style={{
                          cursor: "pointer",
                          textAlign: "left",
                          borderRadius: 18,
                          border: "1px solid rgba(255,255,255,0.10)",
                          background: "rgba(0,0,0,0.20)",
                          padding: 0,
                          overflow: "hidden",
                          color: "white",
                        }}
                      >
                        <div style={{ display: "grid", gridTemplateColumns: "160px 1fr", gap: 12 }}>
                          <img
                            src={(v as any).thumbnail}
                            alt={v.title}
                            style={{ width: 160, height: 96, objectFit: "cover" }}
                          />
                          <div style={{ padding: 12 }}>
                            <div style={{ fontWeight: 1000, lineHeight: 1.25 }}>{v.title}</div>
                            <div style={{ fontSize: 12, opacity: 0.85, marginTop: 6 }}>
                              {v.extra?.channel ? `by ${v.extra.channel}` : "YouTube"}
                            </div>
                            {v.extra?.description && (
                              <div
                                style={{
                                  marginTop: 6,
                                  fontSize: 12,
                                  opacity: 0.72,
                                  lineHeight: 1.35,
                                  overflow: "hidden",
                                  display: "-webkit-box",
                                  WebkitLineClamp: 2 as any,
                                  WebkitBoxOrient: "vertical" as any,
                                }}
                              >
                                {v.extra.description}
                              </div>
                            )}
                          </div>
                        </div>
                      </button>
                    ))}
                  </div>
                )}
              </Card>
            </div>
          </div>
        )}

        <Modal
          open={!!activeVideoId}
          onClose={() => {
            setActiveVideoId(null);
            setActiveVideoTitle("");
          }}
          title={activeVideoTitle || "Video"}
        >
          {activeVideoId && (
            <div style={{ position: "relative", paddingTop: "56.25%" }}>
              <iframe
                src={`https://www.youtube.com/embed/${activeVideoId}`}
                title={activeVideoTitle}
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
                style={{
                  position: "absolute",
                  inset: 0,
                  width: "100%",
                  height: "100%",
                  border: 0,
                  borderRadius: 16,
                }}
              />
            </div>
          )}
        </Modal>
      </div>
    </main>
  );
}