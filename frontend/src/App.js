// frontend/src/App.js
import React, { useState } from "react";
import { inferImage, checkBackend } from "./api";

function App() {
  const [file, setFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [backendStatus, setBackendStatus] = useState("Checking...");

  React.useEffect(() => {
    checkBackend()
      .then((ok) => setBackendStatus(ok ? "✅ Backend Connected" : "❌ Backend Unreachable"))
      .catch(() => setBackendStatus("❌ Backend Unreachable"));
  }, []);

  const handleFileChange = (e) => {
    const f = e.target.files[0];
    if (!f) return;
    setFile(f);
    setResult(null);
    setImagePreview(URL.createObjectURL(f));
  };

  const handleRun = async () => {
    if (!file) {
      alert("Please choose an image first.");
      return;
    }

    setLoading(true);
    setResult(null);

    try {
      const res = await inferImage(file);
      console.log("Inference result:", res);
      setResult(res);
    } catch (err) {
      console.error(err);
      // user-friendly message
      alert(err.message || "Inference failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        fontFamily: "Arial, sans-serif",
        padding: 20,
        maxWidth: 700,
        margin: "0 auto",
      }}
    >
      <h1>🌾 Agritech Image Detection</h1>
      <p style={{ color: backendStatus.includes("✅") ? "green" : "red" }}>
        {backendStatus}
      </p>

      {/* File Upload */}
      <input
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        style={{ marginTop: 10 }}
      />

      {imagePreview && (
        <div style={{ marginTop: 20 }}>
          <h3>Original Image</h3>
          <img
            src={imagePreview}
            alt="Preview"
            style={{
              width: "100%",
              maxHeight: 400,
              objectFit: "contain",
              border: "1px solid #ccc",
              borderRadius: 8,
            }}
          />
        </div>
      )}

      <button
        onClick={handleRun}
        disabled={loading}
        style={{
          marginTop: 20,
          padding: "10px 20px",
          backgroundColor: loading ? "#888" : "#007bff",
          color: "white",
          border: "none",
          borderRadius: 5,
          cursor: loading ? "not-allowed" : "pointer",
        }}
      >
        {loading ? "Running Inference..." : "Run Inference"}
      </button>

      {/* Results */}
      {result && (
        <div style={{ marginTop: 30 }}>
          <h2>Detection Results</h2>
          <p>
            Objects detected: <strong>{result.boxes_count ?? 0}</strong>
          </p>

          {result.image_url ? (
            <img
              src={result.image_url}
              alt="Predicted"
              style={{
                width: "100%",
                maxHeight: 600,
                objectFit: "contain",
                border: "1px solid #ccc",
                borderRadius: 8,
                marginTop: 10,
              }}
            />
          ) : (
            <p>No annotated image available.</p>
          )}

          {result.boxes && result.boxes.length > 0 && (
            <div style={{ marginTop: 10 }}>
              <h4>Detected Objects:</h4>
              <ul>
                {result.boxes.map((b, i) => (
                  <li key={i}>
                    Label: <strong>{b.label}</strong> — Score:{" "}
                    {( (b.score ?? 0) * 100 ).toFixed(1)}%
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* raw debug info */}
          <details style={{ marginTop: 12 }}>
            <summary>Raw response (debug)</summary>
            <pre style={{ whiteSpace: "pre-wrap" }}>{JSON.stringify(result.raw, null, 2)}</pre>
          </details>
        </div>
      )}
    </div>
  );
}

export default App;