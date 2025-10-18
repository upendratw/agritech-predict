// frontend/src/api.js
const BASE = process.env.REACT_APP_BACKEND_URL || "http://localhost:8600";

/**
 * Simple health check — ensures backend root returns 200.
 * Returns true/false.
 */
export async function checkBackend(timeoutMs = 2500) {
  try {
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), timeoutMs);

    const res = await fetch(`${BASE}/`, { method: "GET", signal: controller.signal });
    clearTimeout(id);
    return res.ok;
  } catch (err) {
    return false;
  }
}

/**
 * Upload an image file to the backend /infer endpoint.
 *
 * - file: File object from input[type=file]
 * - opts: { image_size, score_threshold, topk }
 *
 * Returns a normalized result object:
 * {
 *   image_url: "/static/results/out_....png" OR fully-qualified,
 *   counts: {label: count, ...},
 *   raw_boxes_count: number,
 *   boxes_count: number,
 *   boxes: [{ label, score }, ...]  // may be empty if backend doesn't return boxes
 * }
 */
export async function inferImage(file, opts = {}) {
  const image_size = opts.image_size ?? 512;
  const score_threshold = opts.score_threshold ?? 0.3;
  const topk = opts.topk ?? 200;

  const query = new URLSearchParams({
    image_size: String(image_size),
    score_threshold: String(score_threshold),
    topk: String(topk),
  }).toString();

  const url = `${BASE}/infer?${query}`;

  const fd = new FormData();
  // IMPORTANT: backend expects multipart field name "image"
  fd.append("image", file, file.name);

  const res = await fetch(url, {
    method: "POST",
    body: fd,
  });

  if (!res.ok) {
    // try to parse returned JSON error message
    let errBody = null;
    try {
      errBody = await res.json();
    } catch (e) {
      // ignore
    }
    const msg = errBody?.detail ? JSON.stringify(errBody.detail) : `${res.status} ${res.statusText}`;
    throw new Error(`Infer failed: ${res.status} ${res.statusText} — ${msg}`);
  }

  const payload = await res.json();

  // Normalize payload into a friendly shape the UI expects
  const result = {
    image_url: null,
    counts: payload.counts ?? {},
    raw_boxes_count: payload.raw_boxes_count ?? 0,
    boxes_count: 0,
    boxes: [],
    raw: payload,
  };

  // image_url may be returned as image_url or out_image_url (some variants)
  const imgPath = payload.image_url ?? payload.out_image_url ?? null;
  if (imgPath) {
    // If backend returned absolute URL (http(s)...) use it; otherwise prefix BASE
    if (imgPath.startsWith("http://") || imgPath.startsWith("https://")) {
      result.image_url = imgPath;
    } else {
      // ensure no double-slash
      result.image_url = `${BASE}${imgPath.startsWith("/") ? "" : "/"}${imgPath}`;
    }
  }

  // Some backends may also return detailed boxes/labels/scores; attempt to normalize
  // Accept either: { boxes: [...], labels: [...], scores: [...] } or combined payload.boxes array
  if (Array.isArray(payload.boxes) && payload.boxes.length > 0 && typeof payload.boxes[0] === "object") {
    // payload.boxes likely is [{label, score, ...}, ...]
    result.boxes = payload.boxes.map((b) => ({
      label: b.label ?? (b.lbl ?? b.class ?? ""),
      score: typeof b.score === "number" ? b.score : (b.confidence ?? b.conf ?? 0),
      raw: b,
    }));
    result.boxes_count = result.boxes.length;
  } else if (Array.isArray(payload.labels) && Array.isArray(payload.scores)) {
    // separate arrays
    const labels = payload.labels;
    const scores = payload.scores;
    result.boxes = labels.map((lab, i) => ({ label: String(lab), score: scores[i] ?? 0 }));
    result.boxes_count = result.boxes.length;
  } else if (payload.counts && Object.keys(payload.counts).length > 0) {
    // If only counts are provided, create boxes array from counts for display (no per-box scores)
    const boxes = [];
    for (const [lbl, cnt] of Object.entries(payload.counts)) {
      for (let i = 0; i < cnt; ++i) boxes.push({ label: lbl, score: 1.0 });
    }
    result.boxes = boxes;
    result.boxes_count = boxes.length;
  } else {
    // fallback to raw_boxes_count
    result.boxes_count = payload.raw_boxes_count ?? 0;
  }

  return result;
}