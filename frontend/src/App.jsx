import { useRef, useState } from "react";
import { supabase } from "./supabaseClient";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

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

const ProgressStrip = ({ statusText }) => (
  <pre className="result" style={{ whiteSpace: "pre-wrap" }}>
    {statusText || "Waiting for run..."}
  </pre>
);

export default function App() {
  const [view, setView] = useState("login"); // login | signup | otp | app
  const [emailDisplay, setEmailDisplay] = useState("");
  const [session, setSession] = useState(null);
  const [status, setStatus] = useState("");
  const [forceRefresh, setForceRefresh] = useState(false);
  const [jobResult, setJobResult] = useState(null);
  const [systemPrompt, setSystemPrompt] = useState("");
  const [siteName, setSiteName] = useState("Bot");
  const [progressValue, setProgressValue] = useState(0);
  const [progressText, setProgressText] = useState("Idle");
  const [otpEmail, setOtpEmail] = useState("");
  const [firstNameDisplay, setFirstNameDisplay] = useState("");
  const [resetStatus, setResetStatus] = useState("");
  const [resetEmail, setResetEmail] = useState("");
  const [resetSent, setResetSent] = useState(false);
  const [resetOtpEntered, setResetOtpEntered] = useState(false);
  const [resetOtpValue, setResetOtpValue] = useState("");
  const [isRunning, setIsRunning] = useState(false);
  const resetEmailRef = useRef(null);
  const resetOtpRef = useRef(null);
  const resetNewPassRef = useRef(null);
  const resetNewPassConfirmRef = useRef(null);
  const [signupPassword, setSignupPassword] = useState("");
  const [summaryVisible, setSummaryVisible] = useState(false);
  const [summaryData, setSummaryData] = useState({ pages: 0, searches: 0 });

  // Refs to avoid re-rendering while typing (prevents cursor jump/blur)
  const loginEmailRef = useRef(null);
  const loginPassRef = useRef(null);
  const signupFirstRef = useRef(null);
  const signupLastRef = useRef(null);
  const signupEmailRef = useRef(null);
  const signupPassRef = useRef(null);
  const otpEmailRef = useRef(null);
  const otpPassRef = useRef(null);
  const otpCodeRef = useRef(null);
  const urlInputRef = useRef(null);
  const defaultUrl = "https://example.com";

  const handleSignup = async () => {
    const email = signupEmailRef.current?.value?.trim() || "";
    const password = signupPassRef.current?.value || "";
    const first = signupFirstRef.current?.value?.trim() || "";
    const last = signupLastRef.current?.value?.trim() || "";
    setStatus("Signing up...");
    setSignupPassword(password);
    const { error } = await supabase.auth.signUp({
      email,
      password,
      options: { data: { first_name: first, last_name: last } },
    });
    if (error) {
      setStatus(`Signup failed: ${error.message}`);
    } else {
      setFirstNameDisplay(first || "");
      setStatus("Signup initiated. Check your email for OTP.");
      setOtpEmail(email);
      setView("otp");
    }
  };

  const handleVerifyOtp = async () => {
    const otp = otpCodeRef.current?.value?.trim() || "";
    setStatus("Verifying OTP...");
    const { error } = await supabase.auth.verifyOtp({
      email: otpEmail,
      token: otp,
      type: "signup",
    });
    if (error) {
      setStatus(`OTP failed: ${error.message}`);
    } else {
      const { data: loginData, error: loginError } =
        await supabase.auth.signInWithPassword({
          email: otpEmail,
          password: signupPassRef.current?.value || signupPassword || "",
        });
      if (loginError) {
        setStatus(`Verified; now log in. ${loginError.message}`);
        setView("login");
      } else {
        setSession(loginData.session);
        setEmailDisplay(loginData.session?.user?.email || otpEmail);
        const fn = loginData.session?.user?.user_metadata?.first_name;
        setFirstNameDisplay(fn || firstNameDisplay || "");
        setStatus("Account confirmed and logged in.");
        setView("app");
      }
    }
  };

  const handleLogin = async () => {
    const email = loginEmailRef.current?.value?.trim() || "";
    const password = loginPassRef.current?.value || "";
    setStatus("Logging in...");
    const { data, error } = await supabase.auth.signInWithPassword({
      email,
      password,
    });
    if (error) {
      setStatus(`Login failed: ${error.message}`);
    } else {
      setSession(data.session);
      setEmailDisplay(data.session?.user?.email || email);
      const fn = data.session?.user?.user_metadata?.first_name;
      setFirstNameDisplay(fn || firstNameDisplay || (email ? email.split("@")[0] : ""));
      setStatus("Logged in.");
      setView("app");
    }
  };

  const handleLogout = async () => {
    await supabase.auth.signOut();
    setSession(null);
    setJobResult(null);
    setSystemPrompt("");
    setEmailDisplay("");
    setChatMessages([]);
    if (chatInputRef.current) chatInputRef.current.value = "";
    setProgressValue(0);
    setProgressText("Idle");
    setStatus("Logged out.");
    setView("login");
  };

  const handleSendReset = async () => {
    const email =
      resetEmailRef.current?.value?.trim() ||
      resetEmail ||
      loginEmailRef.current?.value?.trim() ||
      "";
    if (!email) {
      setResetStatus("Enter an email to reset.");
      return;
    }
    setResetEmail(email);
    setResetOtpEntered(false);
    setResetOtpValue("");
    setResetStatus("Sending reset OTP...");
    const { error } = await supabase.auth.resetPasswordForEmail(email);
    if (error) setResetStatus(`Failed: ${error.message}`);
    else {
      setResetStatus("Reset OTP sent. Check your email.");
      setResetSent(true);
    }
  };

  const handleVerifyResetOtp = () => {
    const otp = resetOtpRef.current?.value?.trim() || "";
    if (!otp) {
      setResetStatus("Enter the OTP you received.");
      return;
    }
    setResetOtpValue(otp);
    setResetOtpEntered(true);
    setResetStatus("OTP captured. Enter new password.");
  };

  const handleConfirmReset = async () => {
    const email =
      resetEmailRef.current?.value?.trim() ||
      resetEmail ||
      loginEmailRef.current?.value?.trim() ||
      "";
    const otp = resetOtpValue;
    const newPass = resetNewPassRef.current?.value || "";
    const newPassConfirm = resetNewPassConfirmRef.current?.value || "";
    if (!email || !otp || !newPass) {
      setResetStatus("Enter OTP and new password.");
      return;
    }
    if (newPass !== newPassConfirm) {
      setResetStatus("New password and confirm password do not match.");
      return;
    }
    setResetEmail(email);
    setResetStatus("Resetting password...");
    const { data: verifyData, error: verifyError } = await supabase.auth.verifyOtp({
      email,
      token: otp,
      type: "recovery",
    });
    if (verifyError) {
      setResetStatus(`Failed: ${verifyError.message}`);
      return;
    }

    // After OTP verification, update the password
    const { error: updateError } = await supabase.auth.updateUser({
      password: newPass,
    });
    if (updateError) {
      setResetStatus(`Failed: ${updateError.message}`);
      return;
    }

    setResetStatus("Password reset. You can log in now.");
    setResetSent(false);
    setResetOtpEntered(false);
    setResetOtpValue("");
    if (resetNewPassRef.current) resetNewPassRef.current.value = "";
    if (resetNewPassConfirmRef.current) resetNewPassConfirmRef.current.value = "";
    setView("login");
  };

  const runJob = async () => {
    const targetUrl = urlInputRef.current?.value?.trim() || defaultUrl;
    setIsRunning(true);
    setStatus("Submitting job...");
    setJobResult(null);
    setSystemPrompt("");
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
      setJobResult(json);
      const statusText = json?.stats?.status_text || "Completed.";
      setStatus("Job completed.");
      setSystemPrompt(json?.stats?.system_prompt || "");
      setSiteName(json?.stats?.name || "Bot");
      setJobResult((prev) => ({ ...prev, status_text: statusText }));
      setProgressText(statusText);
      const match = statusText.match(/Progress:\s*([0-9]{1,3})%/i);
      setProgressValue(match ? Math.min(100, Math.max(0, parseInt(match[1], 10))) : 100);
      setSummaryData({
        pages: json?.stats?.pages_scraped ?? 0,
        searches: json?.stats?.searches_run ?? 0,
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

  const Header = () => (
    <header className="hero">
      <div>
        <h1>ChatSMITH</h1>
        <p className="muted">
          {firstNameDisplay
            ? `Welcome, ${firstNameDisplay}`
            : emailDisplay
              ? `Welcome, ${emailDisplay.split("@")[0]}`
              : "Welcome"}
        </p>
        <p className="hero-subtitle">AI-powered chatbot generator for any website.</p>
      </div>
      <div className="status">{status}</div>
    </header>
  );

  const LoginCard = () => (
    <Panel title="Login" subtitle="AI-powered chatbot generator for any website.">
      <input
        placeholder="Email"
        ref={loginEmailRef}
        defaultValue=""
      />
      <input
        placeholder="Password"
        type="password"
        ref={loginPassRef}
        defaultValue=""
      />
      <button onClick={handleLogin}>Log In</button>
      <p className="link" onClick={() => setView("signup")}>
        Don’t have an account? Sign up
      </p>
      <div className="muted small" style={{ marginTop: 8, cursor: "pointer" }} onClick={() => setView("reset")}>
        Forgot password?
      </div>
    </Panel>
  );

  const SignupCard = () => (
    <Panel title="Sign Up" subtitle="Create your account and start building.">
      <input
        placeholder="First name"
        ref={signupFirstRef}
        defaultValue=""
      />
      <input placeholder="Last name" ref={signupLastRef} defaultValue="" />
      <input placeholder="Email" ref={signupEmailRef} defaultValue="" />
      <input placeholder="Password" type="password" ref={signupPassRef} defaultValue="" />
      <button onClick={handleSignup}>Sign Up</button>
      <p className="link" onClick={() => setView("login")}>
        Back to login
      </p>
    </Panel>
  );

  const ResetCard = () => (
    <Panel title="Reset Password" subtitle="Securely recover access with OTP.">
      {!resetSent && (
        <>
          <input
            placeholder="Email for reset"
            ref={resetEmailRef}
            defaultValue=""
          />
          <button onClick={handleSendReset}>Send reset OTP</button>
        </>
      )}
      {resetSent && !resetOtpEntered && (
        <>
          <input
            placeholder="Reset OTP"
            ref={resetOtpRef}
            defaultValue=""
          />
          <button onClick={handleVerifyResetOtp}>Verify OTP</button>
        </>
      )}
      {resetSent && resetOtpEntered && (
        <>
          <div className="muted small">OTP captured. Enter new password.</div>
          <input
            placeholder="New password"
            type="password"
            ref={resetNewPassRef}
            defaultValue=""
          />
          <input
            placeholder="Confirm new password"
            type="password"
            ref={resetNewPassConfirmRef}
            defaultValue=""
          />
          <button onClick={handleConfirmReset}>Confirm reset</button>
        </>
      )}
      <div className="status">{resetStatus}</div>
      <p className="link" onClick={() => setView("login")}>
        Back to login
      </p>
    </Panel>
  );

  const OtpCard = () => (
    <Panel title="Enter OTP" subtitle="Check your inbox for the 6-digit code.">
      <div className="muted small">OTP sent to: {otpEmail || "your email"}</div>
      <input placeholder="OTP code" ref={otpCodeRef} defaultValue="" />
      <button onClick={handleVerifyOtp}>Verify OTP & Login</button>
      <p className="link" onClick={() => setView("login")}>
        Back to login
      </p>
    </Panel>
  );

  const AppCards = () => (
    <div className="grid single-column">
      <Panel title="Generate Chatbot" subtitle="Paste a URL and generate a chatbot instantly.">
        <label className="label">Website URL</label>
        <input
          placeholder="https://example.com"
          defaultValue={defaultUrl}
          ref={urlInputRef}
        />
        <label className="checkbox">
          <input type="checkbox" checked={forceRefresh} onChange={(e) => setForceRefresh(e.target.checked)} />
          Force refresh
        </label>
        <button className={isRunning ? "loading" : ""} onClick={runJob} disabled={isRunning}>
          {isRunning ? "Running..." : "Run"}
        </button>
        <p className="muted small generate-desc">Paste a URL and generate a chatbot instantly. Scrape → gap detection → targeted search → knowledge base.</p>

        <div className="progress-container">
          <div className="progress-bar" style={{ width: `${progressValue}%` }} />
        </div>

        {systemPrompt && (
          <>
            <hr style={{ border: "1px solid rgba(255,255,255,0.06)" }} />
            {summaryVisible ? (
              <div className="card summary-card">
                <h3>Summary</h3>
                <p className="muted small">Pages scraped: {summaryData.pages}</p>
                <p className="muted small">Web searches: {summaryData.searches}</p>
              </div>
            ) : (
              <>
                <div className="muted small">Chatbot: {siteName}</div>
                <div className="chat-box">
                  {chatMessages.length === 0 && <div className="muted">Ask anything about the scraped site.</div>}
                  {chatMessages.map((m, idx) => (
                    <div key={idx} className={`chat-msg ${m.role}`}>
                      <strong>{m.role === "user" ? "You" : siteName}:</strong> {m.content}
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
                <details style={{ marginTop: 8 }}>
                  <summary className="muted small">View system prompt</summary>
                  <pre className="result" style={{ maxHeight: 160 }}>{systemPrompt}</pre>
                </details>
              </>
            )}
          </>
        )}
      </Panel>

      <div className="logout-row">
        <button className="link logout-small" onClick={handleLogout}>Log out</button>
      </div>
    </div>
  );

  const [chatMessages, setChatMessages] = useState([]);
  const chatInputRef = useRef(null);
  const [chatStatus, setChatStatus] = useState("");

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
      setChatMessages([...newMessages, assistantMsg]);
      setChatStatus("Ready");
    } catch (err) {
      console.error("Chat failed", err);
      setChatStatus(`Chat failed: ${err.message}`);
    }
  };

  return (
    <div className="app-shell">
      <Header />

      {view === "login" && (
        <div className="auth-page">
          <div className="auth-stage">
            <div className="auth-card-wrap">
              <LoginCard />
            </div>
          </div>
        </div>
      )}

      {view === "signup" && (
        <div className="auth-page">
          <div className="auth-stage">
            <div className="auth-card-wrap">
              <SignupCard />
            </div>
          </div>
        </div>
      )}

      {view === "reset" && (
        <div className="auth-page">
          <div className="auth-stage">
            <div className="auth-card-wrap">
              <ResetCard />
            </div>
          </div>
        </div>
      )}

      {view === "otp" && (
        <div className="auth-page">
          <div className="auth-stage">
            <div className="auth-card-wrap">
              <OtpCard />
            </div>
          </div>
        </div>
      )}

      {view === "app" && (
        <div className="main-page">
          <div className="main-card-wrap">
            <AppCards />
          </div>
        </div>
      )}
    </div>
  );
}
