
import React, {useEffect, useState} from "react";
import {createRoot} from "react-dom/client";
import {Upload, Download, Share2, History, Image as ImageIcon, X, Loader2, Map} from "lucide-react";
import "./style.css";

const API = "http://127.0.0.1:8000";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);
  const [error, setError] = useState("");

  const loadHistory = async () => {
    try {
      const r = await fetch(`${API}/api/results`);
      const data = await r.json();
      setHistory(data.results || []);
    } catch {
      setError("Backend is not running. Start FastAPI on port 8000.");
    }
  };

  useEffect(() => { loadHistory(); }, []);

  const chooseFile = (f) => {
    if (!f) return;
    if (!f.type.startsWith("image/")) {
      setError("Please choose a satellite image.");
      return;
    }
    setError("");
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
  };

  const extract = async () => {
    if (!file) return;
    setLoading(true);
    setError("");
    const form = new FormData();
    form.append("file", file);

    try {
      const r = await fetch(`${API}/api/extract`, {method: "POST", body: form});
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || "Extraction failed");
      setResult(data);
      await loadHistory();
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const share = async () => {
    if (!result) return;
    const url = `${API}${result.mask_url}`;
    try {
      if (navigator.share) {
        await navigator.share({title: "Extracted Road Network", url});
      } else {
        await navigator.clipboard.writeText(url);
        alert("Result link copied.");
      }
    } catch {}
  };

  const openHistory = (item) => {
    setResult({
      id: item.id,
      filename: item.original_name,
      mask_url: `/api/results/${item.id}/mask`,
      original_url: `/api/results/${item.id}/original`
    });
    setFile(null);
    setPreview(`${API}/api/results/${item.id}/original`);
  };

  const maskUrl = result ? `${API}${result.mask_url}` : null;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand"><div className="logo"><Map size={22}/></div>
          <div><h1>RoadVision AI</h1><span>Satellite Road Network Extraction</span></div>
        </div>
        <button className="historyBtn" onClick={loadHistory}><History size={18}/> Previous Images</button>
      </header>

      <main>
        <section className="hero">
          <div>
            <p className="eyebrow">CNN + U-NET</p>
            <h2>Extract road networks<br/>from satellite imagery.</h2>
            <p className="sub">Upload a satellite image and generate a clean binary road map where roads are white and other objects are black.</p>
          </div>
        </section>

        {error && <div className="error"><X size={18}/>{error}</div>}

        <section className="workspace">
          <div className="panel">
            <div className="panelHead"><h3>1. Upload Satellite Image</h3><span>JPG / PNG / TIFF</span></div>
            <label className="dropzone">
              <input type="file" accept="image/*" onChange={e=>chooseFile(e.target.files[0])}/>
              <Upload size={34}/>
              <strong>Choose image from files</strong>
              <span>or drag and drop it here</span>
            </label>

            {preview && <div className="previewWrap">
              <img src={preview} alt="Satellite preview"/>
              <button className="remove" onClick={()=>{setFile(null);setPreview(null);setResult(null)}}><X/></button>
            </div>}

            <button className="primary" disabled={!file || loading} onClick={extract}>
              {loading ? <><Loader2 className="spin"/> Extracting roads...</> : <><Map size={19}/> Extract Road Network</>}
            </button>
          </div>

          <div className="panel resultPanel">
            <div className="panelHead"><h3>2. Road Extraction Result</h3><span>WHITE = ROAD</span></div>
            <div className="resultBox">
              {maskUrl ? <img src={maskUrl} alt="Road mask"/> :
                <div className="empty"><ImageIcon size={44}/><strong>Your road mask will appear here</strong><span>Upload an image and click Extract Road Network</span></div>}
            </div>
            <div className="actions">
              <a className={!result ? "disabled action" : "action"} href={result ? `${API}/api/results/${result.id}/download` : "#"} download><Download size={18}/> Download</a>
              <button className="action" disabled={!result} onClick={share}><Share2 size={18}/> Share</button>
              {result && <a className="action" href={`${API}${result.original_url}`} target="_blank"}><ImageIcon size={18}/> Original</a>}
            </div>
          </div>
        </section>

        <section className="historySection">
          <div className="sectionTitle"><div><p className="eyebrow">LIBRARY</p><h3>Previously Obtained Images</h3></div><button onClick={loadHistory}><History size={17}/> Refresh</button></div>
          <div className="historyGrid">
            {history.length === 0 && <div className="emptyHistory">No processed images yet.</div>}
            {history.map(item => (
              <button className="historyCard" key={item.id} onClick={()=>openHistory(item)}>
                <img src={`${API}/api/results/${item.id}/mask`} alt="Road result"/>
                <div><strong>{item.original_name}</strong><span>{new Date(item.created_at).toLocaleString()}</span></div>
              </button>
            ))}
          </div>
        </section>
      </main>
      <footer>RoadVision AI • CNN + U-Net Satellite Road Extraction</footer>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
