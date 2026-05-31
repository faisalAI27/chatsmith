import { useRef, useState } from "react";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000/api";

const Panel = ({ title, subtitle, children }) => (
  <div className="card">
    <div className="card-head">
      <div>
        <h2>{title}</h2>
        {subtitle ? <p className="card-subtitle">{subtitle}</p> : null}
      </div>
    </div>
    {children}
  </div>
);

const MARKDOWN_LINK_RE = /\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g;

const InlineMarkdown = ({ text }) => {
  const parts = [];
  let lastIndex = 0;
  const value = String(text || "");

  for (const match of value.matchAll(MARKDOWN_LINK_RE)) {
    if (match.index > lastIndex) {
      parts.push(value.slice(lastIndex, match.index));
    }
    parts.push(
      <a key={`${match[2]}-${match.index}`} href={match[2]} target="_blank" rel="noreferrer">
        {match[1]}
      </a>
    );
    lastIndex = match.index + match[0].length;
  }

  if (lastIndex < value.length) {
    parts.push(value.slice(lastIndex));
  }

  return parts.length ? parts : value;
};

const MarkdownMessage = ({ content }) => {
  const lines = String(content || "").split(/\r?\n/);
  const blocks = [];
  let listItems = [];
  let listType = "";

  const flushList = () => {
    if (!listItems.length) return;
    const ListTag = listType === "ol" ? "ol" : "ul";
    blocks.push(
      <ListTag key={`list-${blocks.length}`} className="markdown-list">
        {listItems.map((item, index) => (
          <li key={`${item}-${index}`}>
            <InlineMarkdown text={item} />
          </li>
        ))}
      </ListTag>
    );
    listItems = [];
    listType = "";
  };

  lines.forEach((line, index) => {
    const bulletMatch = line.match(/^\s*[-*]\s+(.+)$/);
    const numberedMatch = line.match(/^\s*\d+\.\s+(.+)$/);

    if (bulletMatch || numberedMatch) {
      const nextType = numberedMatch ? "ol" : "ul";
      if (listType && listType !== nextType) {
        flushList();
      }
      listType = nextType;
      listItems.push((bulletMatch?.[1] || numberedMatch?.[1] || "").trim());
      return;
    }

    flushList();
    const trimmed = line.trim();
    if (!trimmed) {
      return;
    }
    blocks.push(
      <p key={`p-${index}`}>
        <InlineMarkdown text={trimmed} />
      </p>
    );
  });
  flushList();

  return <div className="markdown-message">{blocks}</div>;
};

export default function App() {
  const [status, setStatus] = useState("Ready.");
  const [forceRefresh, setForceRefresh] = useState(false);
  const [urlValue, setUrlValue] = useState("https://example.com");
  const [jobResult, setJobResult] = useState(null);
  const [systemPrompt, setSystemPrompt] = useState("");
  const [websiteId, setWebsiteId] = useState("");
  const [siteName, setSiteName] = useState("Bot");
  const [progressValue, setProgressValue] = useState(0);
  const [progressText, setProgressText] = useState("Idle");
  const [isRunning, setIsRunning] = useState(false);
  const [summaryVisible, setSummaryVisible] = useState(false);
  const [summaryData, setSummaryData] = useState({
    pages: 0,
    searches: 0,
    chunks: 0,
    vectorIndexed: false,
  });
  const [chatMessages, setChatMessages] = useState([]);
  const [chatStatus, setChatStatus] = useState("");

  const urlInputRef = useRef(null);
  const chatInputRef = useRef(null);
  const defaultUrl = "https://example.com";

  const runJob = async () => {
    const targetUrl = (urlInputRef.current?.value || "").trim() || defaultUrl;
    setIsRunning(true);
    setStatus("Submitting job...");
    setJobResult(null);
    setSystemPrompt("");
    setWebsiteId("");
    setChatMessages([]);
    if (chatInputRef.current) chatInputRef.current.value = "";
    setProgressValue(10);
    setProgressText("Starting...");
    setSummaryVisible(false);

    try {
      const resp = await fetch(`${API_BASE_URL}/jobs/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: targetUrl, force_refresh: forceRefresh }),
      });

      if (!resp.ok) {
        const msg = await resp.text();
        throw new Error(msg || `HTTP ${resp.status}`);
      }

      const json = await resp.json();
      const statusText = json?.stats?.status_text || "Completed.";
      setJobResult({ ...json, status_text: statusText });
      setStatus("Job completed.");
      setSystemPrompt(json?.stats?.system_prompt || "");
      setWebsiteId(json?.stats?.website_id || json?.stats?.vector_indexing_summary?.website_id || "");
      setSiteName(json?.stats?.name || "Bot");
      setProgressText(statusText);

      const match = statusText.match(/Progress:\s*([0-9]{1,3})%/i);
      setProgressValue(match ? Math.min(100, Math.max(0, parseInt(match[1], 10))) : 100);
      setSummaryData({
        pages: json?.stats?.pages_scraped ?? 0,
        searches: json?.stats?.searches_run ?? 0,
        chunks: json?.stats?.chunk_count ?? json?.stats?.vector_indexing_summary?.chunks_built ?? 0,
        vectorIndexed: Boolean(json?.stats?.vector_indexed),
      });
      setSummaryVisible(true);
      setTimeout(() => setSummaryVisible(false), 5000);
    } catch (err) {
      console.error("Job failed", err);
      setStatus(`Job failed: ${err.message} | API: ${API_BASE_URL}`);
      setProgressText("Failed");
      setProgressValue(0);
    } finally {
      setIsRunning(false);
    }
  };

  const sendChat = async () => {
    const text = chatInputRef.current?.value || "";
    if (!text.trim()) return;

    const newMessages = [...chatMessages, { role: "user", content: text }];
    setChatMessages(newMessages);
    if (chatInputRef.current) chatInputRef.current.value = "";
    setChatStatus("Thinking...");

    try {
      const resp = await fetch(`${API_BASE_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          website_id: websiteId || undefined,
          question: text,
          system_prompt: systemPrompt,
          messages: newMessages,
        }),
      });

      if (!resp.ok) {
        const msg = await resp.text();
        throw new Error(msg || `HTTP ${resp.status}`);
      }

      const json = await resp.json();
      const assistantMsg = json?.message;
      const enrichedAssistantMsg = assistantMsg
        ? {
            ...assistantMsg,
            sources: json?.sources || [],
            mode: json?.mode || "legacy_prompt",
            warnings: json?.warnings || [],
          }
        : null;
      setChatMessages(enrichedAssistantMsg ? [...newMessages, enrichedAssistantMsg] : newMessages);
      setChatStatus(json?.mode === "rag" ? "Ready - RAG" : "Ready");
    } catch (err) {
      console.error("Chat failed", err);
      setChatStatus(`Chat failed: ${err.message}`);
    }
  };

  const Header = () => (
    <header className="hero">
      <div>
        <h1>ChatSMITH</h1>
        <p className="muted">Website to chatbot generator</p>
        <p className="hero-subtitle">Paste a website URL, generate knowledge, then chat with it.</p>
      </div>
      <div className="status">{status}</div>
    </header>
  );

  return (
    <div className="app-shell">
      <Header />

      <div className="main-page">
        <div className="main-card-wrap">
          <div className="grid single-column">
            <Panel title="Generate Chatbot" subtitle="Paste a URL and generate a chatbot instantly.">
              <label className="label">Website URL</label>
              <input
                placeholder="https://example.com"
                defaultValue={urlValue}
                ref={urlInputRef}
                autoCorrect="off"
                autoCapitalize="none"
                spellCheck={false}
                onChange={(e) => setUrlValue(e.target.value)}
              />
              <label className="checkbox">
                <input
                  type="checkbox"
                  checked={forceRefresh}
                  onChange={(e) => setForceRefresh(e.target.checked)}
                />
                Force refresh
              </label>
              <button className={isRunning ? "loading" : ""} onClick={runJob} disabled={isRunning}>
                {isRunning ? "Running..." : "Run"}
              </button>
              <p className="muted small generate-desc">
                Scrape - gap detection - targeted search - knowledge base.
              </p>

              <div className="progress-container">
                <div className="progress-bar" style={{ width: `${progressValue}%` }} />
              </div>

              {progressText && (
                <pre className="result" style={{ whiteSpace: "pre-wrap" }}>
                  {progressText}
                </pre>
              )}

              {jobResult && (systemPrompt || websiteId) && (
                <>
                  <hr style={{ border: "1px solid rgba(255,255,255,0.06)" }} />
                  {summaryVisible ? (
                    <div className="card summary-card">
                      <h3>Summary</h3>
                      <p className="muted small">Pages scraped: {summaryData.pages}</p>
                      <p className="muted small">Web searches: {summaryData.searches}</p>
                      <p className="muted small">Chunks indexed: {summaryData.chunks}</p>
                      <p className="muted small">
                        Vector index: {summaryData.vectorIndexed ? "ready" : "fallback available"}
                      </p>
                    </div>
                  ) : (
                    <>
                      <div className="muted small">Chatbot: {siteName}</div>
                      <div className="chat-box">
                        {chatMessages.length === 0 && (
                          <div className="muted">Ask anything about the scraped site.</div>
                        )}
                        {chatMessages.map((m, idx) => (
                          <div key={idx} className={`chat-msg ${m.role}`}>
                            <strong>{m.role === "user" ? "You" : siteName}:</strong>
                            {m.role === "assistant" ? (
                              <MarkdownMessage content={m.content} />
                            ) : (
                              <span> {m.content}</span>
                            )}
                            {m.role === "assistant" && m.sources?.length ? (
                              <div className="sources">
                                <div className="sources-title">Sources</div>
                                {m.sources.map((source, sourceIdx) => (
                                  source.source_url ? (
                                    <a
                                      key={`${source.source_url}-${sourceIdx}`}
                                      href={source.source_url}
                                      target="_blank"
                                      rel="noreferrer"
                                    >
                                      {source.page_title || source.source_url} - {source.chunk_type}
                                    </a>
                                  ) : (
                                    <span key={`${source.page_title || "source"}-${sourceIdx}`}>
                                      {source.page_title || "Source"} - {source.chunk_type}
                                    </span>
                                  )
                                ))}
                              </div>
                            ) : null}
                            {m.role === "assistant" && m.warnings?.length ? (
                              <div className="warning">{m.warnings.join(" ")}</div>
                            ) : null}
                          </div>
                        ))}
                      </div>
                      <textarea
                        rows={4}
                        placeholder="Type your question..."
                        ref={chatInputRef}
                        defaultValue=""
                      />
                      <button onClick={sendChat}>Send</button>
                      <div className="status">{chatStatus}</div>
                    </>
                  )}
                </>
              )}
            </Panel>
          </div>
        </div>
      </div>
    </div>
  );
}
