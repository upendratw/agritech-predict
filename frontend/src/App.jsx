import React, { useRef, useState } from "react";

/**
 * Agritech Predictor Frontend
 * - Large annotated image
 * - Clean detections list
 * - Collapsible debug panel
 */

const API_URL = "http://127.0.0.1:8000/predict";

export default function App() {
  const fileRef = useRef(null);
  const imgRef = useRef(null);
  const canvasRef = useRef(null);

  const [loading, setLoading] = useState(false);
  const [annotSrc, setAnnotSrc] = useState(null);
  const [detections, setDetections] = useState([]);
  const [debug, setDebug] = useState(null);
  const [showDebug, setShowDebug] = useState(false);
  const [scoreThresh, setScoreThresh] = useState(0.25);

  async function upload() {
    const f = fileRef.current?.files?.[0];
    if (!f) {
      alert("Please choose an image first.");
      return;
    }

    setLoading(true);
    setAnnotSrc(null);
    setDetections([]);
    setDebug(null);
    setShowDebug(false);

    const form = new FormData();
    form.append("file", f);

    try {
      const resp = await fetch(
        `${API_URL}?score_thresh=${encodeURIComponent(scoreThresh)}`,
        { method: "POST", body: form }
      );

      if (!resp.ok) {
        throw new Error(await resp.text());
      }

      const data = await resp.json();

      if (data.annotated_image_base64) {
        setAnnotSrc("data:image/png;base64," + data.annotated_image_base64);
      }

      setDetections(Array.isArray(data.detections) ? data.detections : []);
      setDebug(data.debug ?? null);
    } catch (err) {
      alert("Prediction failed: " + err.message);
    } finally {
      setLoading(false);
    }
  }

  function drawBoxes() {
    const img = imgRef.current;
    const canvas = canvasRef.current;
    if (!img || !canvas) return;

    const ctx = canvas.getContext("2d");
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;

    canvas.style.width = img.clientWidth + "px";
    canvas.style.height = img.clientHeight + "px";

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 3;
    ctx.font = "18px Arial";

    detections.forEach((d) => {
      const x = d.x1;
      const y = d.y1;
      const w = d.x2 - d.x1;
      const h = d.y2 - d.y1;

      ctx.strokeStyle = "lime";
      ctx.strokeRect(x, y, w, h);

      const label = `${d.label ?? d.class_id} ${(d.score ?? 0).toFixed(2)}`;
      ctx.fillStyle = "rgba(0,0,0,0.7)";
      ctx.fillRect(x, y - 22, ctx.measureText(label).width + 8, 22);
      ctx.fillStyle = "white";
      ctx.fillText(label, x + 4, y - 18);
    });
  }

  return (
    <div style={{ maxWidth: 1200, margin: "0 auto", padding: 20 }}>
      <h1>🌾 Agritech Predictor</h1>

      {/* Controls */}
      <div style={{ display: "flex", gap: 12, marginBottom: 16 }}>
        <input ref={fileRef} type="file" accept="image/*" />
        <label>
          Threshold
          <input
            type="number"
            step="0.01"
            min="0"
            max="1"
            value={scoreThresh}
            onChange={(e) => setScoreThresh(+e.target.value)}
            style={{ width: 80, marginLeft: 6 }}
          />
        </label>
        <button onClick={upload} disabled={loading}>
          {loading ? "Running…" : "Upload & Predict"}
        </button>
      </div>

      {/* Image */}
      <div style={{ position: "relative", width: "100%" }}>
        {annotSrc ? (
          <>
            <img
              ref={imgRef}
              src={annotSrc}
              alt="annotated"
              style={{ width: "100%", borderRadius: 8 }}
              onLoad={drawBoxes}
            />
            <canvas
              ref={canvasRef}
              style={{
                position: "absolute",
                left: 0,
                top: 0,
                pointerEvents: "none",
              }}
            />
          </>
        ) : (
          <div
            style={{
              height: 300,
              border: "2px dashed #aaa",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            Annotated image will appear here
          </div>
        )}
      </div>

      {/* Detections */}
      <h3 style={{ marginTop: 20 }}>
        Detections ({detections.length})
      </h3>
      {detections.length === 0 ? (
        <p>No detections</p>
      ) : (
        detections.map((d, i) => (
          <div key={i}>
            <strong>{d.label}</strong> — conf {(d.score ?? 0).toFixed(3)}
          </div>
        ))
      )}

      {/* Debug */}
      {debug && (
        <>
          <button
            style={{ marginTop: 16 }}
            onClick={() => setShowDebug(!showDebug)}
          >
            {showDebug ? "Hide Debug" : "Show Debug"}
          </button>

          {showDebug && (
            <pre
              style={{
                marginTop: 10,
                maxHeight: 300,
                overflow: "auto",
                background: "#111",
                color: "#0f0",
                padding: 10,
                fontSize: 12,
              }}
            >
              {JSON.stringify(debug, null, 2)}
            </pre>
          )}
        </>
      )}
    </div>
  );
}