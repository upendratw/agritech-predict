// src/components/InferResult.js
import React from "react";

export default function InferResult({ result }) {
  if (!result) return null;

  const { image_base64, detections = [], counts = {}, note } = result;
  const src = `data:image/png;base64,${image_base64}`;

  return (
    <div style={{ marginTop: 16 }}>
      <h3>Inference Result</h3>
      {note && <div style={{ color: "crimson", marginBottom: 8 }}>{note}</div>}
      <div style={{ display: "flex", gap: 12 }}>
        <div style={{ flex: 1 }}>
          <img alt="inferred" src={src} />
        </div>
        <div className="sidebar">
          <div style={{ fontWeight: 700 }}>Detections</div>
          <table className="table">
            <thead>
              <tr><th>Label</th><th>Score</th></tr>
            </thead>
            <tbody>
              {detections.length === 0 && <tr><td colSpan="2" className="small">No detections</td></tr>}
              {detections.map((d, i) => (
                <tr key={i}>
                  <td>{d.label}</td>
                  <td>{d.score.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <div style={{ marginTop: 12, fontWeight: 700 }}>Counts</div>
          <div className="small">
            {Object.keys(counts).length === 0 && <div className="small">No objects</div>}
            {Object.entries(counts).map(([k, v]) => (
              <div key={k}>{k}: {v}</div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}