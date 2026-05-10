// State -------------------------------------------------------------------
async function saveToFileSystem(blob, defaultName, pickerId) {
  if (window.showSaveFilePicker) {
    try {
      const options = {
        suggestedName: defaultName,
        types: [
          defaultName.endsWith(".gz") 
            ? { description: 'GZIP Files', accept: { 'application/gzip': ['.gz'] } }
            : { description: 'JSON Files', accept: { 'application/json': ['.json'] } }
        ],
      };
      if (pickerId) options.id = pickerId;
      const handle = await window.showSaveFilePicker(options);
      const writable = await handle.createWritable();
      await writable.write(blob);
      await writable.close();
      return true;
    } catch (e) {
      if (e.name === "AbortError") return false;
      throw e;
    }
  } else {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = defaultName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    return true;
  }
}

async function loadFromFileSystem(accept) {
  return new Promise((resolve, reject) => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = accept;
    input.onchange = (e) => {
      const file = e.target.files[0];
      resolve(file || null);
    };
    input.onerror = () => reject(new Error("File selection failed"));
    input.click();
  });
}

const state = {
  videoId: null,
  analysisId: null,
  duration: 0,
  segments: [], // {start_s, end_s, is_on}
  media: [],
  analyses: [],
  pollingAnalyses: false,
  analysis: null,
  poseData: null,
  hitStudyData: null,
  impacts: [], // [{time_s, validated, snr, ...}] from pose artifact's rally section
  strikeLabels: {}, // keyed by `${source}|${time_s.toFixed(3)}`
  includeRejectedInNav: false,
  auditionEndS: null, // when set, player auto-pauses at this time
  audioPreviewRange: { start: 0, end: 0 },
  audioPreviewReady: false,
  timelineView: { start: 0, end: null },
  slowLabelMode: false,
  normalPlaybackRate: 1,
  ballHeatmapEnabled: false,
  ballRoiOverlayEnabled: true,
  ballDiagnostic: null,
  ballDiagnosticLoading: false,
  ballDiagnosticTimer: null,
  ballScan: null,
  ballScanJobId: null,
  knobs: null,
  defaultKnobs: null,
  expandedSegments: new Set(),
  videoUrl: null,
  playMode: "free", // "free" | "segment" | "all_on"
  segmentEndS: null, // when in segment/all_on mode, auto-pause/skip at this time
  allOnQueue: [], // remaining ON segments to play through in "all_on" mode
  calibrationVideoId: null,
  calibrationPoints: [],
};

// Per-detector config schemas — single source of truth for which knobs/badges
// belong to which detector. Used by renderStartConfigControls (filters the Media Pool form).
const _MEDIAN_FRAME_KEYS = [
  "diff_threshold", "motion_threshold",
  "merge_gap_s", "enable_merge_gap",
  "min_segment_s", "enable_min_segment",
  "segment_padding_s", "enable_padding",
  "range_start_s", "range_end_s",
  "duration_s", "target_width", "target_height", "sample_count",
  "detector", "detector_version",
];
const DETECTOR_CONFIGS = {
  median_frame: _MEDIAN_FRAME_KEYS,
  median_court_roi: [
    ..._MEDIAN_FRAME_KEYS,
    "court_weight", "outside_weight", "near_camera_weight",
  ],
  pose_skeleton_yolo: [
    "sample_fps",
    "pose_model", "pose_conf", "pose_imgsz", "model_name",
    "audio_sample_rate", "bandpass_low_hz", "bandpass_high_hz",
    "peak_height_mad_k", "peak_prominence_mult", "min_impact_separation_s",
    "pose_window_s", "wrist_conf_min",
    "min_wrist_velocity", "max_gap_s", "min_hits_per_rally", "rally_padding_s",
    "range_start_s", "range_end_s",
    "duration_s", "sample_count",
    "detector", "detector_version",
  ],
  near_player_hit_study: [
    "sample_fps",
    "pose_model", "pose_conf", "pose_imgsz", "model_name",
    "audio_sample_rate", "bandpass_low_hz", "bandpass_high_hz",
    "peak_height_mad_k", "peak_prominence_mult", "min_impact_separation_s",
    "range_start_s", "range_end_s",
    "duration_s", "sample_count", "feature_window_count", "audio_impact_count",
    "detector", "detector_version",
  ],
};

const HELP_TEXT = {
  sample_fps: "Frames per second sampled for detection. Higher = finer time resolution but slower.",
  median_bg_samples: "Number of frames spread across the video to build the median background. More = cleaner empty-court estimate but slower phase 1.",
  diff_threshold: "Per-pixel intensity difference (0–255) above which a pixel counts as foreground when compared to the median background.",
  motion_threshold: "Fraction of foreground pixels (0–1) at or above which a sample is classified as in-play.",
  merge_gap_s: "Two in-play segments closer than this many seconds get merged. Bridges short gaps inside a real rally.",
  enable_merge_gap: "Toggles segment merging. Disable to keep raw detector output.",
  min_segment_s: "Segments shorter than this many seconds are discarded as transient noise.",
  enable_min_segment: "Toggles short-segment filtering.",
  segment_padding_s: "Seconds of padding added at each end of every segment so cuts don't clip a swing.",
  enable_padding: "Toggles segment padding.",
  court_weight: "Weight applied to foreground pixels inside the marked court polygon (median_court_roi only).",
  outside_weight: "Weight applied to foreground pixels outside the court polygon (median_court_roi only). Lower de-emphasizes background distractions.",
  near_camera_weight: "Weight applied to the near-camera band (median_court_roi only). Set below 1.0 to suppress close-up walking false positives; 0 disables that band entirely.",
  pose_model: "Ultralytics pose model weights file. yolo11n-pose.pt is fastest; larger variants are more accurate but slower.",
  pose_conf: "Minimum YOLO person-detection confidence (0–1). Lower picks up more small/distant players but increases false positives.",
  pose_imgsz: "YOLO inference image size in pixels. Higher helps small/far players but is slower per frame.",
  audio_sample_rate: "Audio sample rate used for impact detection.",
  bandpass_low_hz: "Lower cutoff for impact detection. Raise it to suppress basketball bounce thuds.",
  bandpass_high_hz: "Upper cutoff for impact detection. Keep it high enough to retain racket-ball clicks.",
  peak_height_mad_k: "Peak threshold in median absolute deviations above the local audio floor.",
  peak_prominence_mult: "Peak prominence multiplier. Higher keeps only sharper transients.",
  min_impact_separation_s: "Minimum time between two candidate impacts.",
  min_spectral_centroid_hz: "Reject candidates whose transient centroid is below this frequency. Useful for filtering basketball bounces.",
  pose_window_s: "Seconds around each audio candidate used for pose-motion validation.",
  wrist_conf_min: "Minimum wrist keypoint confidence before wrist velocity is trusted.",
  min_wrist_velocity: "Minimum normalized wrist velocity needed to validate an audio candidate.",
  max_gap_s: "Maximum gap between validated hits inside one rally.",
  min_hits_per_rally: "Minimum validated hits needed to form a rally segment.",
  rally_padding_s: "Seconds added before the first and after the last validated hit in a rally.",
  range_start_s: "Start time within the source video for this analysis run. Use a sub-range when iterating on a slow detector.",
  range_end_s: "End time within the source video for this analysis run.",
  detector: "Detector that produced this analysis.",
  detector_version: "Schema version of the detector's output artifact.",
  duration_s: "Source video duration in seconds.",
  target_width: "Width in pixels at which frames were processed (downscaled for speed).",
  target_height: "Height in pixels at which frames were processed.",
  sample_count: "Total number of frames sampled by the detector.",
  model_name: "Filename of the pose model that was loaded.",
  feature_window_count: "Number of short windows with near-player/audio features precomputed for label evaluation.",
  audio_impact_count: "Number of detected audio impact peaks retained for the hit study.",
};

// DOM helpers -------------------------------------------------------------
const $ = (id) => document.getElementById(id);
const SECTIONS = ["library-section", "analyses-section", "upload-section", "status-section", "editor-section", "calibration-section"];
function showOnly(...ids) {
  for (const sec of SECTIONS) {
    $(sec).hidden = !ids.includes(sec);
  }
}
function fmt(t) {
  if (!isFinite(t)) return "0:00";
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}
function fmtPrecise(t) {
  if (!isFinite(t)) return "0:00.0";
  const m = Math.floor(t / 60);
  const s = (t % 60).toFixed(1);
  return `${m}:${s.padStart(4, "0")}`;
}

// Library -----------------------------------------------------------------
async function refreshLibrary() {
  const [media, analyses] = await Promise.all([
    fetch("/media").then((r) => r.json()),
    fetch("/analyses").then((r) => r.json()),
  ]);
  state.media = media;
  state.analyses = analyses;
  if (!media.length) {
    $("library-section").hidden = true;
    $("analyses-section").hidden = true;
    showOnly("upload-section");
    return media;
  }
  renderLibrary(media);
  renderAnalyses(analyses);
  
  const pathId = window.location.pathname.replace(/^\/+/, '');
  if (pathId && !state.analysisId) {
    const targetA = analyses.find(a => a.analysis_id === pathId);
    if (targetA) {
      openAnalysis(targetA);
      return media;
    }
  }
  
  showOnly("library-section", "analyses-section");
  if (analyses.some((a) => a.status === "pending" || a.status === "analyzing")) {
    pollAnalyses();
  }
  return media;
}

window.addEventListener("popstate", () => {
  const pathId = window.location.pathname.replace(/^\/+/, '');
  if (pathId) {
    const targetA = state.analyses.find(a => a.analysis_id === pathId);
    if (targetA && targetA.analysis_id !== state.analysisId) {
      openAnalysis(targetA);
    }
  } else {
    if ($("player")) $("player").pause();
    state.analysisId = null;
    showOnly("library-section", "analyses-section");
  }
});

function renderLibrary(videos) {
  const tbody = $("videos-tbody");
  tbody.innerHTML = "";
  for (const v of videos) {
    const tr = document.createElement("tr");

    const fileTd = document.createElement("td");
    fileTd.textContent = v.filename;

    const durTd = document.createElement("td");
    durTd.textContent = v.duration_s ? fmt(v.duration_s) : "—";

    const countTd = document.createElement("td");
    countTd.textContent = `${v.analysis_count || 0}`;

    const actTd = document.createElement("td");
    const analyze = document.createElement("button");
    analyze.className = "primary";
    analyze.textContent = "Start analysis";
    analyze.addEventListener("click", () => startAnalysis(v.video_id));
    actTd.appendChild(analyze);
    const calibrate = document.createElement("button");
    calibrate.textContent = v.has_court_calibration ? "Edit court" : "Mark court";
    calibrate.style.marginLeft = "6px";
    calibrate.addEventListener("click", () => openCalibration(v));
    actTd.appendChild(calibrate);
    const del = document.createElement("button");
    del.textContent = "✕";
    del.title = "Delete";
    del.style.marginLeft = "6px";
    del.addEventListener("click", () => deleteVideo(v));
    actTd.appendChild(del);

    tr.append(fileTd, durTd, countTd, actTd);
    tbody.appendChild(tr);
  }
  updateRangeDefaults();
}

function renderAnalyses(analyses) {
  const tbody = $("analyses-tbody");
  tbody.innerHTML = "";
  $("analyses-section").hidden = !analyses.length;
  for (const a of analyses) {
    const tr = document.createElement("tr");
    const fileTd = document.createElement("td");
    fileTd.textContent = a.filename;
    const algoTd = document.createElement("td");
    algoTd.textContent = a.algorithm;
    const statusTd = document.createElement("td");
    const badge = document.createElement("span");
    badge.className = "badge badge-" + a.status;
    badge.textContent = a.status;
    statusTd.appendChild(badge);
    if (a.error_msg) {
      const err = document.createElement("div");
      err.className = "muted";
      err.textContent = a.error_msg;
      statusTd.appendChild(err);
    }
    const progressTd = document.createElement("td");
    if (a.status === "analyzing" || a.status === "pending") {
      const p = document.createElement("progress");
      p.className = "mini-progress";
      p.value = a.progress_percent || 0;
      p.max = 100;
      progressTd.appendChild(p);
      const msg = document.createElement("div");
      msg.className = "muted";
      const eta = typeof a.progress_eta_s === "number" ? ` • ETA ${fmtEta(a.progress_eta_s)}` : "";
      msg.textContent = (a.progress_message || `${Math.round(a.progress_percent || 0)}%`) + eta;
      progressTd.appendChild(msg);
    } else {
      progressTd.textContent = a.status === "done" ? "100%" : "—";
    }
    const actTd = document.createElement("td");
    if (a.status === "done") {
      const open = document.createElement("button");
      open.className = "primary";
      open.textContent = "Open";
      open.addEventListener("click", () => openAnalysis(a));
      actTd.appendChild(open);
    }
    if (a.status === "error") {
      const del = document.createElement("button");
      del.textContent = "Delete";
      del.addEventListener("click", () => deleteAnalysis(a));
      actTd.appendChild(del);
    }
    const range = document.createElement("div");
    range.className = "muted";
    if (typeof a.range_start_s === "number" && typeof a.range_end_s === "number") {
      range.textContent = `${fmtPrecise(a.range_start_s)}-${fmtPrecise(a.range_end_s)}`;
      progressTd.appendChild(range);
    }
    tr.append(fileTd, algoTd, statusTd, progressTd, actTd);
    tbody.appendChild(tr);
  }
}

async function deleteAnalysis(a) {
  if (!confirm(`Delete failed analysis "${a.algorithm}" for "${a.filename}"?`)) return;
  const r = await fetch(`/analyses/${a.analysis_id}`, { method: "DELETE" });
  if (!r.ok) {
    alert("Could not delete analysis: " + (await r.text()));
    return;
  }
  await refreshLibrary();
}

function fmtEta(seconds) {
  if (!isFinite(seconds)) return "unknown";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const mins = seconds / 60;
  if (mins < 60) return `${mins.toFixed(1)}m`;
  return `${(mins / 60).toFixed(1)}h`;
}

function updateRangeDefaults() {
  if (!state.media.length) return;
  const duration = Math.max(...state.media.map((m) => m.duration_s || 0));
  for (const id of ["start-range-start-slider", "start-range-end-slider"]) {
    const el = $(id);
    el.max = duration;
  }
  if (!$("start-range-end").dataset.touched) {
    $("start-range-end").value = fmtPrecise(duration);
    $("start-range-end-slider").value = duration;
  }
}

async function openVideo(v) {
  state.videoId = v.video_id;
  state.duration = v.duration_s || 0;
  if (v.status === "done") {
    await loadEditor();
  } else if (v.status === "analyzing") {
    showOnly("status-section");
    pollStatus();
  } else {
    await reanalyzeVideo(v.video_id);
  }
}

async function reanalyzeVideo(videoId) {
  state.videoId = videoId;
  await fetch(`/process/${videoId}`, { method: "POST" });
  showOnly("status-section");
  pollStatus();
}

async function startAnalysis(videoId) {
  const media = state.media.find((m) => m.video_id === videoId);
  const algorithm = $("start-algorithm").value;
  if (algorithm === "median_court_roi" && !(media && media.has_court_calibration)) {
    alert("Mark the court for this video before starting the Court ROI detector.");
    return;
  }
  const knobs = {
    diff_threshold: 25,
    motion_threshold: 0.02,
    merge_gap_s: 3.0,
    min_segment_s: 5.0,
    segment_padding_s: 1.0,
    sample_fps: parseFloat($("start-sample-fps").value || "2"),
    median_bg_samples: parseInt($("start-median-bg-samples").value || "80", 10),
    enable_merge_gap: true,
    enable_min_segment: true,
    enable_padding: true,
    court_weight: parseFloat($("start-court-weight").value || "1"),
    outside_weight: parseFloat($("start-outside-weight").value || "0.15"),
    near_camera_weight: parseFloat($("start-near-camera-weight").value || "0"),
  };
  const r = await fetch(`/process/${videoId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      algorithm,
      knobs,
      config: readStartConfig(algorithm),
      range_start_s: parseTime($("start-range-start").value) ?? 0,
      range_end_s: parseTime($("start-range-end").value) ?? media.duration_s,
    }),
  });
  if (!r.ok) {
    alert("Could not start analysis: " + (await r.text()));
    return;
  }
  await refreshLibrary();
  pollAnalyses();
}

function readStartConfig(algorithm) {
  if (algorithm === "pose_skeleton_yolo" || algorithm === "near_player_hit_study") {
    return {
      pose_model: $("start-pose-model").value || "yolo11n-pose.pt",
      pose_conf: parseFloat($("start-pose-conf").value || "0.25"),
      pose_imgsz: parseInt($("start-pose-imgsz").value || "640", 10),
      sample_fps: parseFloat($("start-sample-fps").value || "2"),
    };
  }
  return {};
}

async function deleteVideo(v) {
  if (!confirm(`Delete "${v.filename}" and all its segments / exports?`)) return;
  await fetch(`/videos/${v.video_id}`, { method: "DELETE" });
  await refreshLibrary();
}

async function openCalibration(v) {
  state.calibrationVideoId = v.video_id;
  state.calibrationPoints = v.court_calibration ? [...v.court_calibration.points] : [];
  const urlR = await fetch(`/video-file/${v.video_id}`).then((r) => r.json());
  $("calibration-video").src = urlR.url;
  showOnly("calibration-section");
  setupCalibrationCanvas();
}

let calibrationCanvas, calibrationCtx;
function setupCalibrationCanvas() {
  calibrationCanvas = $("calibration-canvas");
  calibrationCtx = calibrationCanvas.getContext("2d");
  const video = $("calibration-video");
  const resize = () => {
    const rect = video.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    calibrationCanvas.width = Math.max(1, rect.width * dpr);
    calibrationCanvas.height = Math.max(1, rect.height * dpr);
    calibrationCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawCalibration();
  };
  resize();
  video.addEventListener("loadedmetadata", resize, { once: true });
  if (!calibrationCanvas.dataset.bound) {
    calibrationCanvas.addEventListener("click", (e) => {
      if (state.calibrationPoints.length >= 4) return;
      const rect = calibrationCanvas.getBoundingClientRect();
      state.calibrationPoints.push({
        x: (e.clientX - rect.left) / rect.width,
        y: (e.clientY - rect.top) / rect.height,
      });
      drawCalibration();
    });
    calibrationCanvas.dataset.bound = "1";
  }
}

function drawCalibration() {
  if (!calibrationCtx) return;
  const canvas = $("calibration-canvas");
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  calibrationCtx.clearRect(0, 0, w, h);
  const pts = state.calibrationPoints;
  if (!pts.length) return;
  calibrationCtx.strokeStyle = "#f2cc60";
  calibrationCtx.fillStyle = "rgba(242, 204, 96, 0.18)";
  calibrationCtx.lineWidth = 2;
  calibrationCtx.beginPath();
  pts.forEach((p, i) => {
    const x = p.x * w, y = p.y * h;
    if (i === 0) calibrationCtx.moveTo(x, y);
    else calibrationCtx.lineTo(x, y);
  });
  if (pts.length === 4) calibrationCtx.closePath();
  calibrationCtx.stroke();
  if (pts.length === 4) calibrationCtx.fill();
  pts.forEach((p, i) => {
    const x = p.x * w, y = p.y * h;
    calibrationCtx.fillStyle = "#cf222e";
    calibrationCtx.beginPath();
    calibrationCtx.arc(x, y, 5, 0, Math.PI * 2);
    calibrationCtx.fill();
    calibrationCtx.fillStyle = "white";
    calibrationCtx.font = "12px sans-serif";
    calibrationCtx.fillText(String(i + 1), x + 7, y - 7);
  });
}

$("reset-calibration-btn").addEventListener("click", () => {
  state.calibrationPoints = [];
  drawCalibration();
});

$("save-calibration-btn").addEventListener("click", async () => {
  if (state.calibrationPoints.length !== 4) {
    alert("Please click exactly 4 court corners.");
    return;
  }
  const r = await fetch(`/media/${state.calibrationVideoId}/court-calibration`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ points: state.calibrationPoints }),
  });
  if (!r.ok) {
    alert("Could not save calibration: " + (await r.text()));
    return;
  }
  await refreshLibrary();
});

$("close-calibration-btn").addEventListener("click", refreshLibrary);

$("show-upload-btn").addEventListener("click", () => {
  showOnly("upload-section");
  $("cancel-upload-btn").hidden = false;
});
$("cancel-upload-btn").addEventListener("click", async () => {
  await refreshLibrary();
});

$("start-algorithm").addEventListener("change", () => {
  if ($("start-algorithm").value === "near_player_hit_study" && $("start-sample-fps").value === "2.0") {
    $("start-sample-fps").value = "4.0";
  }
  renderStartConfigControls();
});

function renderStartConfigControls() {
  const algorithm = $("start-algorithm").value;
  document.querySelectorAll(".start-config").forEach((el) => {
    const algorithms = (el.dataset.algorithms || "").split(/\s+/);
    el.hidden = !algorithms.includes(algorithm);
  });
}
renderStartConfigControls();

// Click-help popover ------------------------------------------------------
// Single floating popover positioned beside whichever .help-dot was clicked.
// Closed by: clicking the same dot again, clicking outside, or pressing Esc.
function showHelpPopover(button) {
  const key = button.dataset.helpKey;
  const text = HELP_TEXT[key];
  const popover = $("help-popover");
  if (!text || !popover) return;
  popover.textContent = text;
  popover.hidden = false;
  // Position below + slightly right of the button, then nudge back into view.
  const rect = button.getBoundingClientRect();
  const top = rect.bottom + window.scrollY + 6;
  let left = rect.left + window.scrollX;
  popover.style.top = `${top}px`;
  popover.style.left = `${left}px`;
  // After it's measured, clamp to viewport.
  const popRect = popover.getBoundingClientRect();
  const overflowRight = popRect.right - (window.innerWidth - 8);
  if (overflowRight > 0) {
    left = Math.max(8, left - overflowRight);
    popover.style.left = `${left}px`;
  }
  popover.dataset.activeKey = key;
}
function hideHelpPopover() {
  const popover = $("help-popover");
  if (!popover) return;
  popover.hidden = true;
  popover.dataset.activeKey = "";
}
document.addEventListener("click", (e) => {
  const dot = e.target.closest(".help-dot");
  const popover = $("help-popover");
  if (dot) {
    e.preventDefault();
    e.stopPropagation();
    if (popover && popover.dataset.activeKey === dot.dataset.helpKey && !popover.hidden) {
      hideHelpPopover();
    } else {
      showHelpPopover(dot);
    }
    return;
  }
  if (popover && !popover.hidden && !popover.contains(e.target)) {
    hideHelpPopover();
  }
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") hideHelpPopover();
});

function bindRangeControls() {
  const startText = $("start-range-start");
  const endText = $("start-range-end");
  const startSlider = $("start-range-start-slider");
  const endSlider = $("start-range-end-slider");
  startSlider.addEventListener("input", () => {
    const end = parseFloat(endSlider.value || "0");
    if (parseFloat(startSlider.value) >= end) startSlider.value = Math.max(0, end - 0.5);
    startText.value = fmtPrecise(parseFloat(startSlider.value || "0"));
    startText.dataset.touched = "1";
  });
  endSlider.addEventListener("input", () => {
    const start = parseFloat(startSlider.value || "0");
    if (parseFloat(endSlider.value) <= start) endSlider.value = start + 0.5;
    endText.value = fmtPrecise(parseFloat(endSlider.value || "0"));
    endText.dataset.touched = "1";
  });
  startText.addEventListener("change", () => {
    const parsed = parseTime(startText.value);
    if (parsed !== null) startSlider.value = parsed;
    startText.dataset.touched = "1";
  });
  endText.addEventListener("change", () => {
    const parsed = parseTime(endText.value);
    if (parsed !== null) endSlider.value = parsed;
    endText.dataset.touched = "1";
  });
}
bindRangeControls();

// Upload ------------------------------------------------------------------
$("file-input").addEventListener("change", () => {
  $("upload-btn").disabled = !$("file-input").files.length;
});

$("upload-btn").addEventListener("click", async () => {
  const file = $("file-input").files[0];
  if (!file) return;

  // Skip upload if we already have a row for this filename.
  try {
    const videos = await fetch("/media").then((r) => r.json());
    const match = videos.find((v) => v.filename === file.name);
    if (match) {
      const reuse = confirm(
        `"${file.name}" is already in the media pool. Reuse it without re-uploading?`,
      );
      if (reuse) {
        await refreshLibrary();
        return;
      }
    }
  } catch (_) {
    // fall through to normal upload
  }

  $("upload-btn").disabled = true;
  $("upload-progress").hidden = false;

  try {
    const data = await uploadWithProgress(file);
    state.videoId = data.video_id;
    state.duration = data.duration_s;
    await refreshLibrary();
  } catch (e) {
    alert("Upload failed: " + e.message);
    $("upload-btn").disabled = false;
  }
});

// Initial load: show library if any videos exist, otherwise upload card.
refreshLibrary().catch((e) => console.error("refreshLibrary failed", e));

function uploadWithProgress(file) {
  return new Promise((resolve, reject) => {
    const fd = new FormData();
    fd.append("file", file);
    const xhr = new XMLHttpRequest();
    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable) {
        const pct = Math.round((e.loaded / e.total) * 100);
        $("progress-bar").value = pct;
        $("progress-label").textContent = pct + "%";
      }
    });
    xhr.addEventListener("load", () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(JSON.parse(xhr.responseText));
      } else {
        reject(new Error(xhr.responseText || `HTTP ${xhr.status}`));
      }
    });
    xhr.addEventListener("error", () => reject(new Error("Network error")));
    xhr.open("POST", "/upload");
    xhr.send(fd);
  });
}

// Status polling ----------------------------------------------------------
async function pollStatus() {
  const labels = {
    pending: "Queued",
    analyzing: "Analyzing video for in-play segments...",
    done: "Done",
    error: "Error",
  };
  while (true) {
    const r = await fetch(`/status/${state.videoId}`);
    const data = await r.json();
    $("status-text").textContent = labels[data.status] || data.status;
    if (data.progress_message) {
      $("status-sub").textContent = data.progress_message;
    }
    if (typeof data.progress_percent === "number") {
      $("analysis-progress").value = data.progress_percent;
    }
    if (data.status === "done") {
      await loadEditor();
      return;
    }
    if (data.status === "error") {
      $("status-text").textContent = "Analysis failed: " + (data.error_msg || "");
      return;
    }
    await new Promise((r) => setTimeout(r, 2000));
  }
}

async function pollAnalyses() {
  if (state.pollingAnalyses) return;
  state.pollingAnalyses = true;
  while (true) {
    const analyses = await fetch("/analyses").then((r) => r.json());
    state.analyses = analyses;
    renderAnalyses(analyses);
    const active = analyses.some((a) => a.status === "pending" || a.status === "analyzing");
    if (!active) {
      state.pollingAnalyses = false;
      await refreshLibrary();
      return;
    }
    await new Promise((r) => setTimeout(r, 2000));
  }
}

// Editor ------------------------------------------------------------------
async function openAnalysis(a) {
  state.videoId = a.video_id;
  state.analysisId = a.analysis_id;
  state.duration = a.duration_s || 0;
  $("editor-title").textContent = `${a.filename} • ${a.algorithm}`;
  
  if (window.location.pathname !== `/${a.analysis_id}`) {
    window.history.pushState({}, "", `/${a.analysis_id}`);
  }
  
  await loadEditor();
}

$("back-to-library-btn").addEventListener("click", async () => {
  $("player").pause();
  state.analysisId = null;
  if (window.location.pathname !== "/") {
    window.history.pushState({}, "", "/");
  }
  showOnly("library-section", "analyses-section");
  // Optional: re-poll library to refresh status if needed, or just use existing state
});

async function loadEditor() {
  const [segR, urlR, analysisR] = await Promise.all([
    fetch(`/analysis-segments/${state.analysisId}`).then((r) => r.json()),
    fetch(`/video-file/${state.videoId}`).then((r) => r.json()),
    fetch(`/analysis/${state.analysisId}`).then((r) => r.ok ? r.json() : null),
  ]);
  state.duration = segR.duration_s;
  state.segments = segR.segments.map((s) => ({ ...s }));
  state.videoUrl = urlR.url;
  state.analysis = analysisR;
  state.poseData = null;
  state.hitStudyData = null;
  if (analysisR && analysisR.metadata && analysisR.metadata.detector === "pose_skeleton_yolo") {
    state.poseData = await fetch(`/pose-data/${state.analysisId}`).then((r) => r.ok ? r.json() : null);
  } else if (analysisR && analysisR.metadata && analysisR.metadata.detector === "near_player_hit_study") {
    state.hitStudyData = await fetch(`/hit-study-data/${state.analysisId}`).then((r) => r.ok ? r.json() : null);
    state.poseData = state.hitStudyData;
  }
  state.knobs = analysisR ? { ...analysisR.knobs } : null;
  state.defaultKnobs = analysisR ? { ...analysisR.defaults } : null;
  state.expandedSegments = new Set();
  state.impacts =
    state.poseData && state.poseData.rally && Array.isArray(state.poseData.rally.impacts)
      ? state.poseData.rally.impacts.slice().sort((a, b) => a.time_s - b.time_s)
      : state.hitStudyData && state.hitStudyData.audio && Array.isArray(state.hitStudyData.audio.impacts)
      ? state.hitStudyData.audio.impacts.slice().sort((a, b) => a.time_s - b.time_s)
      : [];
  state.strikeLabels = {};
  state.includeRejectedInNav = false;
  state.auditionEndS = null;
  state.timelineView = { start: 0, end: null };
  state.slowLabelMode = false;
  state.normalPlaybackRate = 1;
  state.ballHeatmapEnabled = false;
  state.ballDiagnostic = null;
  state.ballDiagnosticLoading = false;
  state.ballScan = null;
  state.ballScanJobId = null;
  clearTimeout(state.ballDiagnosticTimer);
  state.ballDiagnosticTimer = null;
  if ($("slow-label-mode-toggle")) $("slow-label-mode-toggle").checked = false;
  if ($("ball-heatmap-toggle")) $("ball-heatmap-toggle").checked = false;
  if ($("ball-roi-overlay-toggle")) $("ball-roi-overlay-toggle").checked = true;
  if ($("ball-exclude-player-toggle")) $("ball-exclude-player-toggle").checked = true;
  if ($("ball-scan-stop-btn")) $("ball-scan-stop-btn").disabled = true;
  if ($("player")) $("player").playbackRate = 1;
  if ($("strike-include-rejected")) $("strike-include-rejected").checked = false;
  if (state.analysisId && state.impacts.length) {
    try {
      const r = await fetch(`/strike-labels/${state.analysisId}`);
      if (r.ok) {
        const data = await r.json();
        for (const lbl of data.labels || []) {
          state.strikeLabels[labelKey(lbl.source, lbl.time_s)] = lbl;
        }
      }
    } catch (e) {
      console.warn("could not load strike labels", e);
    }
  }
  renderStrikeNav();
  renderStrikeDiagnostic();
  renderStrikeStats();
  renderHitStudyPanel();
  setAudioPreviewRangeFromPlayer();
  state.audioPreviewReady = !!(
    state.analysis &&
    state.analysis.metadata &&
    state.analysis.metadata.detector === "pose_skeleton_yolo"
  );
  setupEditorResizers();

  $("player").src = state.videoUrl;
  showOnly("editor-section");
  renderKnobs();
  renderAll();
  renderPosePanel();
  setupCanvas();
  setupPlayer();
}

function renderAll() {
  drawTimeline();
  renderTable();
  renderSummary();
  renderAnalysisSummary();
  renderPosePanel();
}

function renderSummary() {
  const onSegs = state.segments.filter((s) => s.is_on);
  const onTotal = onSegs.reduce((a, s) => a + (s.end_s - s.start_s), 0);
  $("summary").textContent =
    `${onSegs.length} of ${state.segments.length} ON • ` +
    `${fmt(onTotal)} of ${fmt(state.duration)} (${state.duration > 0 ? Math.round((onTotal / state.duration) * 100) : 0}%)`;
}

function renderTable() {
  const tbody = $("segments-tbody");
  tbody.innerHTML = "";
  state.segments.forEach((seg, i) => {
    const tr = document.createElement("tr");
    if (!seg.is_on) tr.classList.add("off");

    const idxTd = document.createElement("td");
    idxTd.textContent = i + 1;

    const onTd = document.createElement("td");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = seg.is_on;
    cb.addEventListener("change", () => {
      seg.is_on = cb.checked;
      renderAll();
    });
    onTd.appendChild(cb);

    const startTd = numericCell(seg, "start_s", i);
    const endTd = numericCell(seg, "end_s", i);

    const durTd = document.createElement("td");
    durTd.textContent = fmt(seg.end_s - seg.start_s);

    const scoreTd = document.createElement("td");
    scoreTd.className = "score-cell";
    scoreTd.textContent = formatScore(seg);

    const samplesTd = document.createElement("td");
    samplesTd.textContent = typeof seg.sample_count === "number" ? seg.sample_count : "—";

    const stageTd = document.createElement("td");
    stageTd.textContent = seg.decision_stage || "manual";

    const actTd = document.createElement("td");
    const statsBtn = document.createElement("button");
    statsBtn.textContent = state.expandedSegments.has(i) ? "Stats ▴" : "Stats ▾";
    statsBtn.disabled = !(seg.samples && seg.samples.length);
    statsBtn.addEventListener("click", () => {
      if (state.expandedSegments.has(i)) state.expandedSegments.delete(i);
      else state.expandedSegments.add(i);
      renderTable();
    });
    const playBtn = document.createElement("button");
    playBtn.textContent = "▶ Preview";
    playBtn.addEventListener("click", () => playSegment(i));
    const delBtn = document.createElement("button");
    delBtn.textContent = "✕";
    delBtn.title = "Delete";
    delBtn.style.marginLeft = "6px";
    delBtn.addEventListener("click", () => {
      state.segments.splice(i, 1);
      renderAll();
    });
    actTd.appendChild(statsBtn);
    actTd.appendChild(playBtn);
    actTd.appendChild(delBtn);

    tr.append(idxTd, onTd, startTd, endTd, durTd, scoreTd, samplesTd, stageTd, actTd);
    tbody.appendChild(tr);
    if (state.expandedSegments.has(i)) {
      tbody.appendChild(renderSampleRow(seg));
    }
  });
}

function formatScore(seg) {
  if (typeof seg.avg_score !== "number") return "—";
  const min = typeof seg.min_score === "number" ? seg.min_score.toFixed(3) : "—";
  const avg = seg.avg_score.toFixed(3);
  const max = typeof seg.max_score === "number" ? seg.max_score.toFixed(3) : "—";
  return `${avg} (${min}-${max})`;
}

function renderSampleRow(seg) {
  const tr = document.createElement("tr");
  tr.className = "sample-row";
  const td = document.createElement("td");
  td.colSpan = 9;
  const table = document.createElement("table");
  table.className = "sample-table";
  table.innerHTML = `
    <thead>
      <tr>
        <th>Time</th>
        <th>Foreground</th>
        <th>Smoothed</th>
        <th>Threshold</th>
        <th>Final</th>
      </tr>
    </thead>
  `;
  const body = document.createElement("tbody");
  for (const sample of (seg.samples || [])) {
    const row = document.createElement("tr");
    if ("snr" in sample || "spectral_centroid_hz" in sample) {
      row.innerHTML = `
        <td>${fmtPrecise(sample.time_s)}</td>
        <td>${Number(sample.snr || 0).toFixed(2)} SNR</td>
        <td>${Math.round(Number(sample.spectral_centroid_hz || 0))} Hz</td>
        <td>${sample.validated ? "VALID" : "REJECT"}</td>
        <td>${seg.is_on ? "ON" : "OFF"}</td>
      `;
    } else {
      row.innerHTML = `
        <td>${fmtPrecise(sample.time_s)}</td>
        <td>${Number(sample.foreground_ratio).toFixed(4)}</td>
        <td>${Number(sample.smoothed_score).toFixed(4)}</td>
        <td>${sample.threshold_result ? "ON" : "OFF"}</td>
        <td>${seg.is_on ? "ON" : "OFF"}</td>
      `;
    }
    body.appendChild(row);
  }
  table.appendChild(body);
  td.appendChild(table);
  tr.appendChild(td);
  return tr;
}

function numericCell(seg, field, idx) {
  const td = document.createElement("td");
  const input = document.createElement("input");
  input.type = "text";
  input.value = fmtPrecise(seg[field]);
  input.addEventListener("change", () => {
    const parsed = parseTime(input.value);
    if (parsed === null || parsed < 0 || parsed > state.duration) {
      input.value = fmtPrecise(seg[field]);
      return;
    }
    seg[field] = parsed;
    if (seg.end_s <= seg.start_s) {
      seg.end_s = Math.min(state.duration, seg.start_s + 0.1);
    }
    renderAll();
  });
  td.appendChild(input);
  return td;
}

function parseTime(str) {
  // Accepts "M:SS.s" or seconds.
  if (str.includes(":")) {
    const [m, s] = str.split(":");
    const mn = parseFloat(m), sn = parseFloat(s);
    if (isNaN(mn) || isNaN(sn)) return null;
    return mn * 60 + sn;
  }
  const n = parseFloat(str);
  return isNaN(n) ? null : n;
}

// Detector experiment -----------------------------------------------------
let knobTimer = null;
let renderingKnobs = false;
let audioPreviewTimer = null;
let resizersInitialized = false;

const KNOB_IDS = {
  diff_threshold: "knob-diff-threshold",
  motion_threshold: "knob-motion-threshold",
  merge_gap_s: "knob-merge-gap",
  min_segment_s: "knob-min-segment",
  segment_padding_s: "knob-segment-padding",
  sample_fps: "knob-sample-fps",
  pose_model: "knob-pose-model",
  pose_conf: "knob-pose-conf",
  pose_imgsz: "knob-pose-imgsz",
  median_bg_samples: "knob-median-bg-samples",
  court_weight: "knob-court-weight",
  outside_weight: "knob-outside-weight",
  near_camera_weight: "knob-near-camera-weight",
  audio_sample_rate: "knob-audio-sample-rate",
  bandpass_low_hz: "knob-bandpass-low-hz",
  bandpass_high_hz: "knob-bandpass-high-hz",
  peak_height_mad_k: "knob-peak-height-mad-k",
  peak_prominence_mult: "knob-peak-prominence-mult",
  min_spectral_centroid_hz: "knob-min-spectral-centroid-hz",
  min_impact_separation_s: "knob-min-impact-separation",
  min_wrist_velocity: "knob-min-wrist-velocity",
  pose_window_s: "knob-pose-window-s",
  wrist_conf_min: "knob-wrist-conf-min",
  max_gap_s: "knob-max-gap-s",
  min_hits_per_rally: "knob-min-hits-per-rally",
  rally_padding_s: "knob-rally-padding-s",
  enable_merge_gap: "enable-merge-gap",
  enable_min_segment: "enable-min-segment",
  enable_padding: "enable-padding",
};

const AUDIO_KNOB_KEYS = new Set([
  "bandpass_low_hz",
  "bandpass_high_hz",
  "peak_height_mad_k",
  "peak_prominence_mult",
  "min_spectral_centroid_hz",
  "min_impact_separation_s",
  "min_wrist_velocity",
]);

function renderKnobs() {
  renderingKnobs = true;
  const knobs = state.knobs || state.defaultKnobs;
  const isPose = state.analysis && state.analysis.metadata && state.analysis.metadata.detector === "pose_skeleton_yolo";
  const disabled = !knobs || !(state.analysis && state.analysis.metadata);
  for (const [key, id] of Object.entries(KNOB_IDS)) {
    const el = $(id);
    if (!el) continue;
    const poseAudioPreviewKnob = isPose && AUDIO_KNOB_KEYS.has(key);
    const inputDisabled = disabled || (isPose && !poseAudioPreviewKnob);
    if (el.type === "checkbox") {
      if (knobs && knobs[key] !== undefined) {
        el.checked = !!knobs[key];
      }
      el.disabled = inputDisabled;
    } else {
      if (knobs && knobs[key] !== undefined) {
        el.value = knobs[key];
      }
      el.disabled = inputDisabled || key === "sample_fps" || key === "median_bg_samples" ||
        key === "court_weight" || key === "outside_weight" || key === "near_camera_weight";
    }
  }
  renderingKnobs = false;
}

function readKnobs() {
  const base = { ...(state.knobs || state.defaultKnobs || {}) };
  // Pose knobs
  base.sample_fps = parseFloat($("knob-sample-fps").value || base.sample_fps || 4);
  base.pose_model = $("knob-pose-model").value || base.pose_model || "yolo11n-pose.pt";
  base.pose_conf = parseFloat($("knob-pose-conf").value || base.pose_conf || 0.25);
  base.pose_imgsz = parseInt($("knob-pose-imgsz").value || base.pose_imgsz || 640, 10);
  // Audio knobs
  base.audio_sample_rate = parseInt($("knob-audio-sample-rate").value || base.audio_sample_rate || 22050, 10);
  base.bandpass_low_hz = parseFloat($("knob-bandpass-low-hz").value || base.bandpass_low_hz || 1000);
  base.bandpass_high_hz = parseFloat($("knob-bandpass-high-hz").value || base.bandpass_high_hz || 8000);
  base.peak_height_mad_k = parseFloat($("knob-peak-height-mad-k").value || base.peak_height_mad_k || 6);
  base.peak_prominence_mult = parseFloat($("knob-peak-prominence-mult").value || base.peak_prominence_mult || 2);
  base.min_spectral_centroid_hz = parseFloat($("knob-min-spectral-centroid-hz").value || base.min_spectral_centroid_hz || 2500);
  base.min_impact_separation_s = parseFloat($("knob-min-impact-separation").value || base.min_impact_separation_s || 0.15);
  // Logic knobs
  base.diff_threshold = parseInt($("knob-diff-threshold").value || base.diff_threshold || 25, 10);
  base.motion_threshold = parseFloat($("knob-motion-threshold").value || base.motion_threshold || 0.02);
  base.merge_gap_s = parseFloat($("knob-merge-gap").value || base.merge_gap_s || 0);
  base.min_segment_s = parseFloat($("knob-min-segment").value || base.min_segment_s || 0);
  base.segment_padding_s = parseFloat($("knob-segment-padding").value || base.segment_padding_s || 0);
  base.enable_merge_gap = $("enable-merge-gap").checked;
  base.enable_min_segment = $("enable-min-segment").checked;
  base.enable_padding = $("enable-padding").checked;
  base.court_weight = parseFloat($("knob-court-weight").value || base.court_weight || 1);
  base.outside_weight = parseFloat($("knob-outside-weight").value || base.outside_weight || 0.15);
  base.near_camera_weight = parseFloat($("knob-near-camera-weight").value || base.near_camera_weight || 0);
  base.min_wrist_velocity = parseFloat($("knob-min-wrist-velocity").value || base.min_wrist_velocity || 0.4);
  base.pose_window_s = parseFloat($("knob-pose-window-s").value || base.pose_window_s || 0.75);
  base.wrist_conf_min = parseFloat($("knob-wrist-conf-min").value || base.wrist_conf_min || 0.3);
  base.max_gap_s = parseFloat($("knob-max-gap-s").value || base.max_gap_s || 5.0);
  base.min_hits_per_rally = parseInt($("knob-min-hits-per-rally").value || base.min_hits_per_rally || 2, 10);
  base.rally_padding_s = parseFloat($("knob-rally-padding-s").value || base.rally_padding_s || 1.0);
  return base;
}

function renderAnalysisSummary() {
  const el = $("analysis-summary");
  if (!el) return;
  if (!(state.analysis && state.analysis.metadata)) {
    el.textContent = "No cached detector artifact for this analysis. Start a new analysis run to enable instant knob tuning.";
    return;
  }
  const s = state.analysis.summary || {};
  const changed = typeof s.changed_segment_count === "number" ? ` • ${s.changed_segment_count} changed` : "";
  el.textContent =
    `${s.on_count || 0} ON • ${fmt(s.on_duration_s || 0)} kept ` +
    `(${Math.round(s.on_percent || 0)}%) • ${s.sample_count || 0} cached samples${changed}`;
}


function renderPosePanel() {
  const panel = $("pose-panel");
  if (!panel) return;
  if (!state.poseData) {
    panel.hidden = true;
    drawPoseOverlay();
    return;
  }
  panel.hidden = false;
  const summary = state.poseData.summary || {};
  $("pose-summary").textContent =
    `${summary.sample_count || 0} sampled frames • ` +
    `${summary.frames_with_poses || 0} with poses • ` +
    `${Number(summary.avg_detections_per_frame || 0).toFixed(2)} people/frame • ` +
    `avg keypoint confidence ${summary.avg_keypoint_confidence == null ? "—" : Number(summary.avg_keypoint_confidence).toFixed(2)}`;
  renderPoseCurrent();
}

function nearestPoseFrame(t) {
  const frames = state.poseData ? state.poseData.frames || [] : [];
  if (!frames.length) return null;
  let best = frames[0];
  let bestDist = Math.abs(best.time_s - t);
  for (const frame of frames) {
    const dist = Math.abs(frame.time_s - t);
    if (dist < bestDist) {
      best = frame;
      bestDist = dist;
    }
  }
  return best;
}

function renderPoseCurrent() {
  if (!state.poseData) return;
  const player = $("player");
  const t = player ? player.currentTime || 0 : 0;
  const nearest = nearestPoseFrame(t);
  $("pose-current").textContent = nearest
    ? `Nearest pose frame: ${fmtPrecise(nearest.time_s)} • ${nearest.detections.length} people`
    : "No pose frames.";
  renderPosePersonDetails(nearest);
  const tbody = $("pose-frames-tbody");
  tbody.innerHTML = "";
  const frames = state.poseData.frames || [];
  const nearby = frames
    .map((f) => ({ ...f, dist: Math.abs(f.time_s - t) }))
    .sort((a, b) => a.dist - b.dist)
    .slice(0, 12)
    .sort((a, b) => a.time_s - b.time_s);
  for (const frame of nearby) {
    const row = document.createElement("tr");
    const confs = frame.detections.map((d) => d.confidence || 0);
    const kp = [];
    frame.detections.forEach((d) => (d.keypoints || []).forEach((p) => kp.push(p.confidence || 0)));
    row.innerHTML = `
      <td>${fmtPrecise(frame.time_s)}</td>
      <td>${frame.detections.length}</td>
      <td>${confs.length ? Math.max(...confs).toFixed(2) : "—"}</td>
      <td>${kp.length ? (kp.reduce((a, b) => a + b, 0) / kp.length).toFixed(2) : "—"}</td>
    `;
    tbody.appendChild(row);
  }
  drawPoseOverlay();
}

const KEYPOINT_NAMES = [
  "nose", "left_eye", "right_eye", "left_ear", "right_ear",
  "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
  "left_wrist", "right_wrist", "left_hip", "right_hip",
  "left_knee", "right_knee", "left_ankle", "right_ankle",
];

function renderPosePersonDetails(frame) {
  const root = $("pose-person-details");
  root.innerHTML = "";
  if (!frame || !frame.detections.length) {
    root.textContent = "No person skeletons detected for the nearest sampled frame.";
    return;
  }
  frame.detections.forEach((det, idx) => {
    const box = det.box || {};
    const visible = (det.keypoints || []).filter((p) => (p.confidence || 0) >= 0.2);
    const card = document.createElement("div");
    card.className = "pose-person-card";
    const title = document.createElement("div");
    title.innerHTML =
      `<strong>Person ${idx + 1}</strong> ` +
      `box conf ${Number(det.confidence || 0).toFixed(2)} • ` +
      `visible keypoints ${visible.length}/${(det.keypoints || []).length} • ` +
      `box [${fmtNorm(box.x1)}, ${fmtNorm(box.y1)} → ${fmtNorm(box.x2)}, ${fmtNorm(box.y2)}]`;
    card.appendChild(title);

    const table = document.createElement("table");
    table.className = "sample-table";
    table.innerHTML = `
      <thead>
        <tr>
          <th>Keypoint</th>
          <th>x</th>
          <th>y</th>
          <th>Conf</th>
        </tr>
      </thead>
    `;
    const body = document.createElement("tbody");
    for (const kp of det.keypoints || []) {
      const row = document.createElement("tr");
      row.innerHTML = `
        <td>${KEYPOINT_NAMES[kp.index] || `kp_${kp.index}`}</td>
        <td>${fmtNorm(kp.x)}</td>
        <td>${fmtNorm(kp.y)}</td>
        <td>${Number(kp.confidence || 0).toFixed(2)}</td>
      `;
      body.appendChild(row);
    }
    table.appendChild(body);
    card.appendChild(table);
    root.appendChild(card);
  });
}

function fmtNorm(v) {
  return typeof v === "number" && isFinite(v) ? v.toFixed(3) : "—";
}

function schedulePreview() {
  if (
    renderingKnobs ||
    !(state.analysis && state.analysis.metadata) ||
    state.analysis.metadata.detector === "pose_skeleton_yolo"
  ) {
    scheduleAudioPreviewFromKnobs();
    return;
  }
  clearTimeout(knobTimer);
  knobTimer = setTimeout(applyPreviewKnobs, 250);
}

function scheduleAudioPreviewFromKnobs() {
  if (
    renderingKnobs ||
    !state.audioPreviewReady ||
    !(state.analysis && state.analysis.metadata) ||
    state.analysis.metadata.detector !== "pose_skeleton_yolo"
  ) return;
  clearTimeout(audioPreviewTimer);
  const out = $("audio-preview-summary");
  if (out) out.textContent = "Audio knobs changed. Rechecking range...";
  audioPreviewTimer = setTimeout(runAudioPreview, 450);
}

async function applyPreviewKnobs() {
  const knobs = readKnobs();
  try {
    const r = await fetch(`/analysis/${state.analysisId}/preview`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ knobs }),
    });
    if (!r.ok) throw new Error(await r.text());
    const data = await r.json();
    state.knobs = { ...data.knobs };
    state.segments = data.segments.map((s) => ({ ...s }));
    state.analysis = {
      ...(state.analysis || {}),
      knobs: data.knobs,
      summary: data.summary,
    };
    state.expandedSegments = new Set();
    renderKnobs();
    renderAll();
  } catch (e) {
    $("analysis-summary").textContent = "Preview failed: " + e.message;
  }
}

for (const id of Object.values(KNOB_IDS)) {
  const el = $(id);
  if (el) {
    el.addEventListener("input", schedulePreview);
    el.addEventListener("change", schedulePreview);
  }
}

$("reset-knobs-btn").addEventListener("click", () => {
  if (!state.defaultKnobs) return;
  state.knobs = { ...state.defaultKnobs };
  renderKnobs();
  if (state.analysis && state.analysis.metadata && state.analysis.metadata.detector === "pose_skeleton_yolo") {
    $("audio-preview-summary").textContent = "Audio knobs reset. Rechecking range...";
    scheduleAudioPreviewFromKnobs();
  } else {
    applyPreviewKnobs();
  }
});

function setupEditorResizers() {
  if (resizersInitialized) return;
  resizersInitialized = true;
  const grid = document.querySelector(".editor-grid");
  const leftHandle = $("editor-resizer-left");
  const rightHandle = $("editor-resizer-right");
  if (!grid || !leftHandle || !rightHandle) return;

  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
  const applyWidths = (left, right) => {
    grid.style.setProperty("--editor-left-width", `${Math.round(left)}px`);
    grid.style.setProperty("--editor-right-width", `${Math.round(right)}px`);
    localStorage.setItem("editorLeftWidthV2", `${Math.round(left)}`);
    localStorage.setItem("editorRightWidthV2", `${Math.round(right)}`);
  };
  const clampPair = (left, right) => {
    const rect = grid.getBoundingClientRect();
    const minLeft = 220;
    const minCenter = 360;
    const minRight = 300;
    const handleSpace = 20;
    const available = Math.max(0, rect.width - minCenter - handleSpace);
    let nextLeft = clamp(left, minLeft, Math.max(minLeft, available - minRight));
    let nextRight = clamp(right, minRight, Math.max(minRight, available - nextLeft));
    if (nextLeft + nextRight > available) {
      nextRight = Math.max(minRight, available - nextLeft);
    }
    return { left: nextLeft, right: nextRight };
  };

  const savedLeft = parseFloat(localStorage.getItem("editorLeftWidthV2") || "");
  const savedRight = parseFloat(localStorage.getItem("editorRightWidthV2") || "");
  if (Number.isFinite(savedLeft) && Number.isFinite(savedRight)) {
    const initial = clampPair(savedLeft, savedRight);
    applyWidths(initial.left, initial.right);
  } else {
    grid.style.removeProperty("--editor-left-width");
    grid.style.removeProperty("--editor-right-width");
  }

  function startDrag(which, e) {
    e.preventDefault();
    const rect = grid.getBoundingClientRect();
    const startLeft = document.querySelector(".editor-left").getBoundingClientRect().width;
    const startRight = document.querySelector(".editor-right").getBoundingClientRect().width;
    const minLeft = 220;
    const minCenter = 360;
    const minRight = 300;
    const handleSpace = 20;
    const maxSide = Math.max(260, rect.width - minCenter - handleSpace - minLeft);
    const handle = which === "left" ? leftHandle : rightHandle;
    handle.classList.add("dragging");
    document.body.classList.add("editor-resizing");

    const move = (evt) => {
      let nextLeft = startLeft;
      let nextRight = startRight;
      if (which === "left") {
        nextLeft = clamp(evt.clientX - rect.left, minLeft, rect.width - startRight - minCenter - handleSpace);
      } else {
        nextRight = clamp(rect.right - evt.clientX, minRight, rect.width - startLeft - minCenter - handleSpace);
      }
      const clamped = clampPair(
        clamp(nextLeft, minLeft, maxSide),
        clamp(nextRight, minRight, maxSide),
      );
      applyWidths(clamped.left, clamped.right);
    };
    const stop = () => {
      handle.classList.remove("dragging");
      document.body.classList.remove("editor-resizing");
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", stop);
      drawTimeline();
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", stop);
  }

  leftHandle.addEventListener("pointerdown", (e) => startDrag("left", e));
  rightHandle.addEventListener("pointerdown", (e) => startDrag("right", e));
}

// Timeline canvas ---------------------------------------------------------
let canvas, ctx;
function setupCanvas() {
  canvas = $("timeline");
  // Match canvas internal pixel size to its CSS width.
  const ro = new ResizeObserver(() => {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    canvas.width = w * dpr;
    canvas.height = 118 * dpr;
    ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawTimeline();
  });
  ro.observe(canvas);

  canvas.addEventListener("click", (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const t = timelineXToTime(x, rect.width);
    // Find segment under cursor.
    const idx = state.segments.findIndex((s) => t >= s.start_s && t <= s.end_s);
    if (idx >= 0 && e.shiftKey) {
      // Shift+click: toggle on/off
      state.segments[idx].is_on = !state.segments[idx].is_on;
      renderAll();
    } else {
      // Plain click: seek video to that time
      const player = $("player");
      player.currentTime = t;
      state.playMode = "free";
    }
  });
}

function drawTimeline() {
  if (!ctx) return;
  const w = canvas.clientWidth;
  const h = 118;
  // Vertical layout: strike markers, centroid plot, segment band, tick labels.
  const STRIKE_BAND_TOP = 0;
  const STRIKE_BAND_H = 12;
  const CENTROID_TOP = 18;
  const CENTROID_H = 42;
  const SEG_TOP = 68;
  const SEG_BOTTOM = h - 16;
  ctx.clearRect(0, 0, w, h);

  // Background.
  ctx.fillStyle = "#eaeef2";
  ctx.fillRect(0, 0, w, h);

  if (state.duration <= 0) return;

  drawCentroidTrack(w, CENTROID_TOP, CENTROID_H);

  // Draw segments.
  for (const seg of state.segments) {
    const start = Math.max(seg.start_s, timelineStart());
    const end = Math.min(seg.end_s, timelineEnd());
    if (end <= start) continue;
    const x = timeToTimelineX(start, w);
    const segW = Math.max(1, timeToTimelineX(end, w) - x);
    ctx.fillStyle = seg.is_on ? "#1a7f37" : "#b1bac4";
    ctx.fillRect(x, SEG_TOP, segW, SEG_BOTTOM - SEG_TOP);
  }

  // Strike markers (rejected first so validated draw on top).
  if (state.impacts.length) {
    const tickW = 2;
    for (const imp of state.impacts) {
      if (imp.validated) continue;
      if (imp.time_s < timelineStart() || imp.time_s > timelineEnd()) continue;
      const x = timeToTimelineX(imp.time_s, w);
      ctx.fillStyle = "rgba(154, 164, 173, 0.7)";
      ctx.fillRect(x - tickW / 2, STRIKE_BAND_TOP, tickW, STRIKE_BAND_H);
    }
    for (const imp of state.impacts) {
      if (!imp.validated) continue;
      if (imp.time_s < timelineStart() || imp.time_s > timelineEnd()) continue;
      const x = timeToTimelineX(imp.time_s, w);
      ctx.fillStyle = "#0a8e72";
      ctx.fillRect(x - tickW / 2, STRIKE_BAND_TOP, tickW, STRIKE_BAND_H);
    }
    // Overlay user labels: disputed (algorithm wrong) ticks, then missed-strike triangles.
    for (const imp of state.impacts) {
      const lbl = state.strikeLabels[labelKey("candidate", imp.time_s)];
      if (!lbl) continue;
      const algo = lbl.algorithm_validated;
      if ((algo === true && lbl.is_strike === false) ||
          (algo === false && lbl.is_strike === true)) {
        if (imp.time_s < timelineStart() || imp.time_s > timelineEnd()) continue;
        const x = timeToTimelineX(imp.time_s, w);
        ctx.fillStyle = "#d97706";
        ctx.fillRect(x - tickW / 2, STRIKE_BAND_TOP, tickW, STRIKE_BAND_H);
      }
    }
    for (const lbl of Object.values(state.strikeLabels)) {
      if (lbl.source !== "manual") continue;
      if (lbl.time_s < timelineStart() || lbl.time_s > timelineEnd()) continue;
      const x = timeToTimelineX(lbl.time_s, w);
      ctx.fillStyle = "#cf222e";
      ctx.beginPath();
      ctx.moveTo(x - 4, 0);
      ctx.lineTo(x + 4, 0);
      ctx.lineTo(x, 7);
      ctx.closePath();
      ctx.fill();
    }
  }

  // Tick marks.
  ctx.fillStyle = "#57606a";
  ctx.font = "10px sans-serif";
  const span = timelineSpan();
  const tickStep = span <= 10 ? 1 : span <= 30 ? 5 : span <= 120 ? 15 : 60;
  const firstTick = Math.ceil(timelineStart() / tickStep) * tickStep;
  for (let t = firstTick; t <= timelineEnd(); t += tickStep) {
    const x = timeToTimelineX(t, w);
    ctx.fillRect(x, SEG_BOTTOM, 1, 4);
    ctx.fillText(span <= 120 ? fmtPrecise(t) : `${Math.floor(t / 60)}m`, x + 2, h - 2);
  }

  // Playhead.
  const player = $("player");
  if (player && isFinite(player.currentTime)) {
    if (player.currentTime >= timelineStart() && player.currentTime <= timelineEnd()) {
      const x = timeToTimelineX(player.currentTime, w);
      ctx.fillStyle = "#cf222e";
      ctx.fillRect(x - 1, 0, 2, h);
    }
  }
}

function timelineStart() {
  return Math.max(0, Number(state.timelineView.start || 0));
}

function timelineEnd() {
  const end = state.timelineView.end == null ? state.duration : Number(state.timelineView.end);
  return Math.min(state.duration, Math.max(timelineStart() + 0.1, end));
}

function timelineSpan() {
  return Math.max(0.1, timelineEnd() - timelineStart());
}

function timeToTimelineX(t, w) {
  return ((Number(t) - timelineStart()) / timelineSpan()) * w;
}

function timelineXToTime(x, w) {
  return timelineStart() + (x / Math.max(1, w)) * timelineSpan();
}

function setTimelineView(start, end) {
  state.timelineView = {
    start: Math.max(0, Math.min(Number(start) || 0, state.duration)),
    end: Math.max(0, Math.min(Number(end) || state.duration, state.duration)),
  };
  if (state.timelineView.end <= state.timelineView.start) {
    state.timelineView.end = Math.min(state.duration, state.timelineView.start + 1);
  }
  drawTimeline();
  renderStrikeNav();
}

function resetTimelineView() {
  state.timelineView = { start: 0, end: null };
  drawTimeline();
  renderStrikeNav();
}

function impactsWithCentroid() {
  return state.impacts
    .filter((imp) => Number(imp.spectral_centroid_hz || 0) > 0)
    .sort((a, b) => a.time_s - b.time_s);
}

function drawCentroidTrack(w, top, height) {
  const impacts = impactsWithCentroid()
    .filter((imp) => imp.time_s >= timelineStart() && imp.time_s <= timelineEnd());
  if (!impacts.length || state.duration <= 0) {
    ctx.fillStyle = "#8c959f";
    ctx.font = "10px sans-serif";
    ctx.fillText("centroid: run audio recheck to populate", 6, top + height - 5);
    return;
  }
  const knobs = readKnobs();
  const threshold = Number(knobs.min_spectral_centroid_hz || 0);
  const maxCentroid = Math.max(5000, threshold * 1.25, ...impacts.map((imp) => Number(imp.spectral_centroid_hz || 0)));
  const yFor = (centroid) => top + height - (Math.max(0, Math.min(centroid, maxCentroid)) / maxCentroid) * height;

  ctx.strokeStyle = "rgba(9, 105, 218, 0.28)";
  ctx.lineWidth = 1;
  ctx.strokeRect(0, top, w, height);

  if (threshold > 0) {
    const y = yFor(threshold);
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = "#d97706";
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#9a6700";
    ctx.font = "10px sans-serif";
    ctx.fillText(`${Math.round(threshold)} Hz`, 4, Math.max(top + 10, y - 3));
  }

  ctx.strokeStyle = "#0969da";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  impacts.forEach((imp, idx) => {
    const x = timeToTimelineX(imp.time_s, w);
    const y = yFor(Number(imp.spectral_centroid_hz || 0));
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  for (const imp of impacts) {
    const x = timeToTimelineX(imp.time_s, w);
    const y = yFor(Number(imp.spectral_centroid_hz || 0));
    ctx.fillStyle = imp.centroid_pass === false ? "#d97706" : (imp.validated ? "#0a8e72" : "#0969da");
    ctx.beginPath();
    ctx.arc(x, y, 2.2, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.fillStyle = "#57606a";
  ctx.font = "10px sans-serif";
  ctx.fillText("centroid Hz", 6, top + 10);
}

function currentCentroidInfo(t) {
  const impacts = impactsWithCentroid();
  if (!impacts.length) return null;
  let best = null;
  let bestDist = Infinity;
  for (const imp of impacts) {
    const d = Math.abs(Number(imp.time_s) - t);
    if (d < bestDist) {
      best = imp;
      bestDist = d;
    }
  }
  return best ? { imp: best, dist: bestDist } : null;
}

function renderCentroidReadout() {
  const el = $("centroid-readout");
  if (!el) return;
  const player = $("player");
  const t = player ? Number(player.currentTime || 0) : 0;
  const found = currentCentroidInfo(t);
  if (!found || found.dist > 1.0) {
    el.textContent = "Centroid —";
    return;
  }
  el.textContent = `Centroid ${Math.round(Number(found.imp.spectral_centroid_hz || 0))} Hz @ ${fmtPrecise(found.imp.time_s)}`;
}

// Player sync -------------------------------------------------------------
function setupPlayer() {
  const player = $("player");
  player.addEventListener("timeupdate", () => {
    drawTimeline();
    renderPoseCurrent();
    renderStrikeDiagnostic();
    renderCentroidReadout();
    scheduleBallDiagnostic();
    drawPoseOverlay();
    $("missed-strike-time") &&
      ($("missed-strike-time").textContent = fmtPrecise(player.currentTime || 0));
    $("near-hit-time") &&
      ($("near-hit-time").textContent = fmtPrecise(player.currentTime || 0));
    $("player-time").textContent = `${fmt(player.currentTime)} / ${fmt(state.duration)}`;

    if (state.playMode === "segment" && state.segmentEndS !== null) {
      if (player.currentTime >= state.segmentEndS) {
        player.pause();
        state.playMode = "free";
        state.segmentEndS = null;
      }
    } else if (state.playMode === "all_on" && state.segmentEndS !== null) {
      if (player.currentTime >= state.segmentEndS) {
        playNextOnSegment();
      }
    } else if (state.playMode === "audition" && state.auditionEndS !== null) {
      if (player.currentTime >= state.auditionEndS) {
        player.pause();
        state.playMode = "free";
        state.auditionEndS = null;
      }
    }
  });
}

const POSE_EDGES = [
  [5, 7], [7, 9], [6, 8], [8, 10], [5, 6],
  [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [0, 1], [0, 2], [1, 3], [2, 4],
];

function drawPoseOverlay() {
  const canvas = $("pose-overlay");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const video = $("player");
  const rect = video.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, rect.width * dpr);
  canvas.height = Math.max(1, rect.height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (state.poseData && $("pose-overlay-toggle").checked) {
    const frame = nearestPoseFrame(video.currentTime || 0);
    if (frame && Math.abs(frame.time_s - (video.currentTime || 0)) <= 1.0) {
      ctx.lineWidth = 2;
      for (const det of frame.detections || []) {
        const pts = det.keypoints || [];
        ctx.strokeStyle = "#f2cc60";
        for (const [a, b] of POSE_EDGES) {
          const pa = pts[a], pb = pts[b];
          if (!pa || !pb || pa.confidence < 0.2 || pb.confidence < 0.2) continue;
          ctx.beginPath();
          ctx.moveTo(pa.x * w, pa.y * h);
          ctx.lineTo(pb.x * w, pb.y * h);
          ctx.stroke();
        }
        ctx.fillStyle = "#cf222e";
        for (const p of pts) {
          if (p.confidence < 0.2) continue;
          ctx.beginPath();
          ctx.arc(p.x * w, p.y * h, 3, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }
  }
  drawBallHeatmapOverlay(ctx, w, h, video.currentTime || 0);
}

$("pose-overlay-toggle").addEventListener("change", drawPoseOverlay);

// Strike navigation -------------------------------------------------------
function renderStrikeNav() {
  const nav = $("strike-nav");
  if (!nav) return;
  if (!state.impacts.length) {
    nav.hidden = true;
    renderCentroidReadout();
    return;
  }
  nav.hidden = false;
  const validated = state.impacts.filter((i) => i.validated).length;
  const total = state.impacts.length;
  $("strike-count").textContent = `${validated} validated / ${total} candidates`;
  renderCentroidReadout();
}

const AUDITION_PAD_S = 0.7;

function jumpToStrike(direction) {
  if (!state.impacts.length) return;
  const player = $("player");
  const t = player ? player.currentTime || 0 : 0;
  // In review mode (Include rejected), step through every audio candidate so
  // the user can audit rejections too. Default: validated only.
  const pool = state.includeRejectedInNav
    ? state.impacts
    : state.impacts.filter((i) => i.validated);
  const targets = pool.map((i) => i.time_s);
  if (!targets.length) return;
  let dest = null;
  const eps = 0.05; // avoid getting stuck on the current peak
  if (direction === "next") {
    dest = targets.find((ts) => ts > t + eps);
    if (dest === undefined) dest = targets[targets.length - 1];
  } else {
    for (let i = targets.length - 1; i >= 0; i--) {
      if (targets[i] < t - eps) { dest = targets[i]; break; }
    }
    if (dest === null) dest = targets[0];
  }
  // Audition: seek to dest - 0.7s, play, auto-pause at dest + 0.7s.
  const start = Math.max(0, dest - AUDITION_PAD_S);
  player.currentTime = start;
  state.playMode = "audition";
  state.auditionEndS = dest + AUDITION_PAD_S;
  state.segmentEndS = null;
  player.play().catch(() => {});
  renderStrikeDiagnostic();
}

$("prev-strike-btn").addEventListener("click", () => jumpToStrike("prev"));
$("next-strike-btn").addEventListener("click", () => jumpToStrike("next"));
$("strike-include-rejected").addEventListener("change", (e) => {
  state.includeRejectedInNav = !!e.target.checked;
});
document.addEventListener("keydown", (e) => {
  // Skip when typing in inputs.
  const tag = (e.target && e.target.tagName) || "";
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
  if ($("editor-section").hidden) return;
  if (e.key === " ") {
    if (state.slowLabelMode && isHitStudy()) {
      if (!e.repeat) playSlowWhileHeld();
    } else {
      const player = $("player");
      if (player.paused) player.play().catch(() => {});
      else player.pause();
    }
    e.preventDefault();
  } else if (e.key === "[") {
    jumpToStrike("prev");
    e.preventDefault();
  } else if (e.key === "]") {
    jumpToStrike("next");
    e.preventDefault();
  } else if (e.key.toLowerCase() === "h" && isHitStudy()) {
    markNearPlayerHit();
    e.preventDefault();
  }
});
document.addEventListener("keyup", (e) => {
  if (e.key === " " && state.slowLabelMode && isHitStudy()) {
    pauseSlowHold();
    e.preventDefault();
  }
});

// Strike diagnostic panel ------------------------------------------------
function labelKey(source, time_s) {
  return `${source}|${Number(time_s).toFixed(3)}`;
}

function strikeReason(imp, knobs) {
  knobs = knobs || {};
  const minV = Number(knobs.min_wrist_velocity ?? 0.4);
  const poseWindow = Number(knobs.pose_window_s ?? 0.75);
  const v = +Number(imp.max_wrist_v || 0).toFixed(2);
  const b = +Number(imp.max_box_v || 0).toFixed(2);
  if (imp.validated) {
    if (imp.fallback_used) {
      return `Validated via body-motion fallback (max_box_v=${b} ≥ ${(minV * 1.2).toFixed(2)})`;
    }
    return `Validated: max_wrist_v=${v} ≥ ${minV}`;
  }
  if (imp.rejection_reason === "centroid" || imp.centroid_pass === false) {
    const centroid = Math.round(Number(imp.spectral_centroid_hz || 0));
    const threshold = Math.round(Number(imp.centroid_threshold_hz || knobs.min_spectral_centroid_hz || 0));
    return `Rejected: centroid ${centroid} Hz < ${threshold} Hz`;
  }
  if (v === 0 && b === 0) {
    return `Rejected: no person detections in pose window (±${poseWindow}s)`;
  }
  if (v > 0) {
    return `Rejected: max_wrist_v=${v} < ${minV}`;
  }
  return `Rejected: low-confidence wrists; body motion ${b} below fallback threshold`;
}

function nearestImpact(t) {
  if (!state.impacts.length) return null;
  let best = null;
  let bestDist = Infinity;
  for (let i = 0; i < state.impacts.length; i++) {
    const d = Math.abs(state.impacts[i].time_s - t);
    if (d < bestDist) { bestDist = d; best = { imp: state.impacts[i], idx: i, dist: d }; }
  }
  return best;
}

let _renderingDiag = false;

function renderStrikeDiagnostic() {
  const card = $("manual-labels-card");
  if (!card) return;
  card.hidden = !state.impacts.length;
  if (!state.impacts.length) return;

  const root = $("strike-current");
  const player = $("player");
  const t = player ? player.currentTime || 0 : 0;
  const found = nearestImpact(t);
  if (!found || found.dist > 2.0) {
    root.innerHTML = `<div class="strike-empty">No candidate near ${fmtPrecise(t)} — use Prev/Next or seek the timeline to step through strikes.</div>`;
    return;
  }
  const { imp, idx } = found;
  const knobs = (state.poseData && state.poseData.rally && state.poseData.rally.knobs) || {};
  const lbl = state.strikeLabels[labelKey("candidate", imp.time_s)] || null;
  const algo = !!imp.validated;
  // Pre-select judgment radio based on existing label (if any).
  // is_strike == algorithm_validated → "correct"; else "wrong".
  let judgmentInitial = "";
  if (lbl) {
    judgmentInitial = (lbl.is_strike === algo) ? "correct" : "wrong";
  }

  _renderingDiag = true;
  root.innerHTML = `
    <div class="strike-header">
      Candidate at ${fmtPrecise(imp.time_s)} (${idx + 1}/${state.impacts.length}) —
      <span class="${algo ? "strike-status-on" : "strike-status-off"}">${algo ? "VALIDATED" : "REJECTED"}</span>
    </div>
    <div class="strike-reason">${escapeHtml(strikeReason(imp, knobs))}</div>
    <div class="strike-metrics">
      SNR=${Number(imp.snr || 0).toFixed(2)} ·
      amplitude=${Number(imp.amplitude || 0).toFixed(3)} ·
      centroid=${Math.round(Number(imp.spectral_centroid_hz || 0))} Hz ·
      max_wrist_v=${Number(imp.max_wrist_v || 0).toFixed(2)} ·
      max_box_v=${Number(imp.max_box_v || 0).toFixed(2)}${
        imp.player_id !== null && imp.player_id !== undefined
          ? ` · player_id=${imp.player_id}`
          : ""
      }${imp.fallback_used ? " · fallback" : ""}
    </div>
    <div class="strike-judgment">
      <label><input type="radio" name="diag-judge" value="correct" ${judgmentInitial === "correct" ? "checked" : ""} /> Algorithm correct</label>
      <label><input type="radio" name="diag-judge" value="wrong" ${judgmentInitial === "wrong" ? "checked" : ""} /> Algorithm wrong</label>
    </div>
    <textarea class="strike-comment" id="diag-comment" placeholder="Optional comment (e.g. 'shoe squeak', 'mishit on frame')">${escapeHtml(lbl ? (lbl.comment || "") : "")}</textarea>
    <div class="strike-actions">
      <button id="diag-save-btn">${lbl ? "Update label" : "Save label"}</button>
      <button id="diag-save-next-btn">Save &amp; Next ▶</button>
      ${lbl ? '<button id="diag-delete-btn">Delete label</button>' : ""}
      ${lbl ? `<span class="saved-flag">Saved as ${lbl.is_strike === algo ? "correct" : "disputed"}</span>` : ""}
    </div>
  `;
  _renderingDiag = false;

  $("diag-save-btn").addEventListener("click", () => saveCurrentJudgment(imp, false));
  $("diag-save-next-btn").addEventListener("click", () => saveCurrentJudgment(imp, true));
  if (lbl) {
    $("diag-delete-btn").addEventListener("click", () => deleteLabelById(lbl.id));
  }
}

function readJudgment() {
  const radios = document.querySelectorAll('input[name="diag-judge"]');
  for (const r of radios) {
    if (r.checked) return r.value;
  }
  return null;
}

async function saveCurrentJudgment(imp, advance) {
  const judgment = readJudgment();
  if (!judgment) {
    alert("Pick 'Algorithm correct' or 'Algorithm wrong' first.");
    return;
  }
  const algo = !!imp.validated;
  // is_strike: ground truth as the user sees it.
  const is_strike = (judgment === "correct") ? algo : !algo;
  const comment = ($("diag-comment").value || "").trim() || null;
  await saveStrikeLabel({
    source: "candidate",
    time_s: imp.time_s,
    is_strike,
    algorithm_validated: algo,
    comment,
  });
  if (advance) jumpToStrike("next");
}

async function saveStrikeLabel({source, time_s, is_strike, algorithm_validated, comment}) {
  if (!state.analysisId) return null;
  try {
    const r = await fetch(`/strike-labels/${state.analysisId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source, time_s, is_strike, algorithm_validated, comment }),
    });
    if (!r.ok) {
      alert("Save failed: " + (await r.text()));
      return null;
    }
    const lbl = await r.json();
    state.strikeLabels[labelKey(lbl.source, lbl.time_s)] = lbl;
    renderStrikeDiagnostic();
    renderStrikeStats();
    renderHitStudyPanel();
    drawTimeline();
    return lbl;
  } catch (e) {
    alert("Save failed: " + e.message);
    return null;
  }
}

async function deleteLabelById(labelId) {
  try {
    const r = await fetch(`/strike-labels/${labelId}`, { method: "DELETE" });
    if (!r.ok) { alert("Delete failed: " + (await r.text())); return; }
    for (const k of Object.keys(state.strikeLabels)) {
      if (state.strikeLabels[k].id === labelId) delete state.strikeLabels[k];
    }
    renderStrikeDiagnostic();
    renderStrikeStats();
    renderHitStudyPanel();
    drawTimeline();
  } catch (e) {
    alert("Delete failed: " + e.message);
  }
}

function renderStrikeStats() {
  const el = $("strike-stats");
  if (!el) return;
  if (!state.impacts.length) { el.textContent = ""; return; }
  const labels = Object.values(state.strikeLabels);
  if (!labels.length) {
    el.textContent = `0 / ${state.impacts.length} candidates labeled`;
    return;
  }
  const candidateLabels = labels.filter((l) => l.source === "candidate");
  const manualLabels = labels.filter((l) => l.source === "manual");
  const audited = candidateLabels.length;
  const correct = candidateLabels.filter((l) => l.is_strike === l.algorithm_validated).length;
  const validatedTrue = candidateLabels.filter((l) => l.algorithm_validated && l.is_strike).length;
  const validatedFalse = candidateLabels.filter((l) => l.algorithm_validated && !l.is_strike).length;
  const rejectedTrue = candidateLabels.filter((l) => !l.algorithm_validated && l.is_strike).length;
  const missed = manualLabels.filter((l) => l.is_strike).length;
  const validatedSum = validatedTrue + validatedFalse;
  const precision = validatedSum > 0 ? (validatedTrue / validatedSum) : null;
  const recallDenom = validatedTrue + rejectedTrue + missed;
  const recall = recallDenom > 0 ? (validatedTrue / recallDenom) : null;
  const pct = (x) => x === null ? "—" : `${Math.round(x * 100)}%`;
  const agreement = audited > 0 ? `${Math.round((correct / audited) * 100)}%` : "—";
  el.innerHTML =
    `Audited <strong>${audited}</strong>/${state.impacts.length} · ` +
    `agreement ${agreement} · ` +
    `false +ve ${validatedFalse} · false −ve ${rejectedTrue} · ` +
    `missed ${missed} · ` +
    `precision ${pct(precision)} · recall ${pct(recall)}`;
}

function isHitStudy() {
  return !!(state.analysis && state.analysis.metadata && state.analysis.metadata.detector === "near_player_hit_study");
}

function renderHitStudyPanel() {
  const block = $("hit-study-block");
  if (!block) return;
  block.hidden = !isHitStudy();
  const divider = document.querySelector(".hit-study-divider");
  if (divider) divider.hidden = !isHitStudy();
  if (!isHitStudy()) return;
  const labels = Object.values(state.strikeLabels)
    .filter((l) => l.source === "near_player_hit")
    .sort((a, b) => a.time_s - b.time_s);
  const root = $("hit-study-labels");
  root.innerHTML = "";
  if (!labels.length) {
    root.textContent = "No near-player hits marked yet.";
    return;
  }
  const count = document.createElement("span");
  count.textContent = `${labels.length} marked hit${labels.length === 1 ? "" : "s"} `;
  root.appendChild(count);
  const pills = document.createElement("span");
  pills.className = "hit-label-pills";
  for (const label of labels) {
    const pill = document.createElement("span");
    pill.className = "hit-label-pill";
    pill.textContent = fmtPrecise(label.time_s);
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "hit-label-remove";
    btn.title = `Remove hit at ${fmtPrecise(label.time_s)}`;
    btn.textContent = "×";
    btn.addEventListener("click", () => deleteLabelById(label.id));
    pill.appendChild(btn);
    pills.appendChild(pill);
  }
  root.appendChild(pills);
}

async function markNearPlayerHit() {
  const t = $("player").currentTime || 0;
  await saveStrikeLabel({
    source: "near_player_hit",
    time_s: t,
    is_strike: true,
    algorithm_validated: null,
    comment: null,
  });
  renderHitStudyPanel();
}

async function evaluateHitStudy() {
  const out = $("hit-study-report");
  if (!out || !state.analysisId) return;
  out.textContent = "Evaluating labels...";
  try {
    const r = await fetch(`/hit-study/${state.analysisId}/evaluate`, { method: "POST" });
    if (!r.ok) throw new Error(await r.text());
    const data = await r.json();
    if (!data.hit_count) {
      out.textContent = data.message || "Mark near-player hits before evaluating.";
      return;
    }
    const rows = (data.features || []).slice(0, 8);
    out.innerHTML = `
      <div class="muted">${data.hit_count} hits · ${data.positive_window_count} positive windows · ${data.negative_window_count} negative windows</div>
      <table class="sample-table">
        <thead><tr><th>Feature</th><th>Hit mean</th><th>Other mean</th><th>Separation</th></tr></thead>
        <tbody>
          ${rows.map((f) => `
            <tr>
              <td>${escapeHtml(f.feature)}</td>
              <td>${Number(f.positive_mean || 0).toFixed(3)}</td>
              <td>${Number(f.negative_mean || 0).toFixed(3)}</td>
              <td>${Number(f.separation || 0).toFixed(2)}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    `;
  } catch (e) {
    out.textContent = "Evaluation failed: " + e.message;
  }
}

async function runBallDiagnostic() {
  const out = $("hit-study-report");
  const player = $("player");
  if (!out || !state.analysisId || !player) return;
  const timeS = ballDiagnosticCenterTime(player.currentTime || 0);
  out.textContent = `Scanning ball motion around ${fmtPrecise(timeS)}...`;
  try {
    const data = await fetchBallDiagnostic(timeS);
    renderBallDiagnosticReport(data);
    drawPoseOverlay();
  } catch (e) {
    out.textContent = "Ball diagnostic failed: " + e.message;
  }
}

function ballRoiParams() {
  const width = parseFloat($("ball-roi-width") ? $("ball-roi-width").value || "3" : "3");
  const height = parseFloat($("ball-roi-height") ? $("ball-roi-height").value || "1" : "1");
  const threshold = parseFloat($("ball-diff-threshold") ? $("ball-diff-threshold").value || "12" : "12");
  return {
    ball_detector: $("ball-detector-mode") ? $("ball-detector-mode").value : "motion",
    roi_mode: $("ball-roi-mode") ? $("ball-roi-mode").value : "near_player",
    scan_mode: $("ball-scan-mode") ? $("ball-scan-mode").value : "marked_hits",
    mark_before_s: parseFloat($("ball-scan-before") ? $("ball-scan-before").value || "1" : "1"),
    mark_after_s: parseFloat($("ball-scan-after") ? $("ball-scan-after").value || "1" : "1"),
    scan_fps: $("ball-scan-fps") ? $("ball-scan-fps").value : "2",
    tracknet_width: parseInt($("tracknet-width") ? $("tracknet-width").value || "640" : "640", 10),
    tracknet_height: parseInt($("tracknet-height") ? $("tracknet-height").value || "360" : "360", 10),
    tracknet_stack_frames: 3,
    roi_expand_x: width,
    roi_expand_up: height,
    roi_expand_down: Math.max(0.25, height * 0.8),
    exclude_player: $("ball-exclude-player-toggle") ? $("ball-exclude-player-toggle").checked : true,
    diff_threshold: threshold,
  };
}

function ballDiagnosticCenterTime(currentTime) {
  const labels = Object.values(state.strikeLabels)
    .filter((l) => l.source === "near_player_hit")
    .sort((a, b) => Math.abs(a.time_s - currentTime) - Math.abs(b.time_s - currentTime));
  if (labels.length && Math.abs(labels[0].time_s - currentTime) <= 1.5) {
    return labels[0].time_s;
  }
  return currentTime;
}

async function fetchBallDiagnostic(timeS) {
  state.ballDiagnosticLoading = true;
  const params = new URLSearchParams({ time_s: String(timeS) });
  const roi = ballRoiParams();
  params.set("ball_detector", roi.ball_detector);
  params.set("tracknet_width", String(roi.tracknet_width));
  params.set("tracknet_height", String(roi.tracknet_height));
  params.set("roi_mode", roi.roi_mode);
  params.set("roi_expand_x", String(roi.roi_expand_x));
  params.set("roi_expand_up", String(roi.roi_expand_up));
  params.set("roi_expand_down", String(roi.roi_expand_down));
  params.set("exclude_player", String(roi.exclude_player));
  params.set("diff_threshold", String(roi.diff_threshold));
  try {
    const r = await fetch(`/hit-study/${state.analysisId}/ball-diagnostic?${params.toString()}`);
    if (!r.ok) throw new Error(await r.text());
    const data = await r.json();
    state.ballDiagnostic = data;
    return data;
  } finally {
    state.ballDiagnosticLoading = false;
  }
}

function renderBallDiagnosticReport(data) {
  const out = $("hit-study-report");
  if (!out) return;
  const roi = data.roi || { x1: 0, y1: 0, x2: 1, y2: 1 };
  const candidates = (data.candidates || []).slice(0, 220);
  const circles = candidates.map((c) => {
    const before = Number(c.time_s || 0) < Number(data.time_s || 0);
    const opacity = Math.max(0.22, Math.min(0.9, Math.log10(Number(c.score || 1) + 1) / 3.5));
    const radius = Math.max(0.65, Math.min(1.8, Math.sqrt(Number(c.area || 1)) * 0.22));
    return `<circle cx="${Number(c.x || 0) * 100}" cy="${Number(c.y || 0) * 100}" r="${radius.toFixed(2)}" fill="${before ? "#0969da" : "#d97706"}" opacity="${opacity.toFixed(2)}"><title>${fmtPrecise(c.time_s)} · score ${Number(c.score || 0).toFixed(0)}</title></circle>`;
  }).join("");
  out.innerHTML = `
    <div class="muted">
      ${data.ball_detector === "tracknet" ? "TrackNet" : "Motion"} heatmap centered at ${escapeHtml(fmtPrecise(data.time_s))} · ${Number(data.fps || 0).toFixed(1)} fps ·
      ${data.candidate_count || 0} small fast blobs (${data.before_count || 0} before, ${data.after_count || 0} after) ·
      ${data.roi_mode === "full_frame" ? "whole screen ROI" : "near-player ROI"} ·
      ${data.exclude_player ? "player masked" : "player included"} · threshold ${Number(data.diff_threshold || 0).toFixed(0)}
    </div>
    <div class="ball-diagnostic-legend">
      <span><i class="ball-before"></i>before mark</span>
      <span><i class="ball-after"></i>after mark</span>
    </div>
    <svg class="ball-diagnostic-plot" viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet" aria-label="Ball motion diagnostic">
      <rect x="0" y="0" width="100" height="100" class="ball-frame"></rect>
      <rect x="${roi.x1 * 100}" y="${roi.y1 * 100}" width="${(roi.x2 - roi.x1) * 100}" height="${(roi.y2 - roi.y1) * 100}" class="ball-roi"></rect>
      ${circles}
    </svg>
  `;
}

async function startBallScan() {
  const out = $("hit-study-report");
  if (!state.analysisId || !isHitStudy()) return;
  const params = ballRoiParams();
  const scanRange = ballScanRange();
  const start = scanRange.start;
  const end = scanRange.end;
  $("ball-scan-progress").hidden = false;
  $("ball-scan-progress-bar").value = 0;
  $("ball-scan-progress-text").textContent = `Queued ${scanRange.label} ${fmtPrecise(start)} to ${fmtPrecise(end)}`;
  $("ball-scan-stop-btn").disabled = false;
  if (out) out.textContent = `Scanning ROI over ${scanRange.label} from ${fmtPrecise(start)} to ${fmtPrecise(end)}...`;
  const r = await fetch(`/hit-study/${state.analysisId}/ball-scan`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      range_start_s: start,
      range_end_s: end,
      ...params,
    }),
  });
  if (!r.ok) {
    const msg = await r.text();
    $("ball-scan-progress-text").textContent = "Scan failed: " + msg;
    if (out) out.textContent = "Scan failed: " + msg;
    return;
  }
  const data = await r.json();
  state.ballScanJobId = data.job_id;
  pollBallScan(data.job_id);
}

function ballScanRange() {
  const range = state.audioPreviewRange || {};
  const start = Number(range.start);
  const end = Number(range.end);
  if (Number.isFinite(start) && Number.isFinite(end) && end > start + 0.05) {
    return { start, end, label: "audio range" };
  }
  return { start: timelineStart(), end: timelineEnd(), label: "timeline view" };
}

async function pollBallScan(jobId) {
  while (state.ballScanJobId === jobId) {
    const r = await fetch(`/hit-study/ball-scan/${jobId}`);
    if (!r.ok) {
      $("ball-scan-progress-text").textContent = "Scan status failed.";
      return;
    }
    const job = await r.json();
    $("ball-scan-progress").hidden = false;
    $("ball-scan-progress-bar").value = Number(job.progress_percent || 0);
    const eta = typeof job.progress_eta_s === "number" ? ` · ETA ${fmtEta(job.progress_eta_s)}` : "";
    $("ball-scan-progress-text").textContent = `${job.progress_message || job.status}${eta}`;
    if (job.result) {
      state.ballScan = job.result;
      drawPoseOverlay();
    }
    if (job.status === "done" || job.status === "canceled") {
      state.ballScan = job.result;
      state.ballScanJobId = null;
      $("ball-scan-stop-btn").disabled = true;
      if ($("hit-study-report")) {
        const status = job.status === "canceled" ? "Scan stopped" : "Scan complete";
        $("hit-study-report").textContent =
          `${status}: ${job.result.candidate_count} candidates, ${job.result.ball_detector}, ${Number(job.result.fps || 0).toFixed(1)} fps`;
      }
      drawPoseOverlay();
      return;
    }
    if (job.status === "error") {
      state.ballScanJobId = null;
      $("ball-scan-stop-btn").disabled = true;
      if ($("hit-study-report")) $("hit-study-report").textContent = "Scan failed: " + (job.error || "unknown error");
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
}

async function stopBallScan() {
  if (!state.ballScanJobId) return;
  const jobId = state.ballScanJobId;
  $("ball-scan-progress-text").textContent = "Stopping scan...";
  try {
    await fetch(`/hit-study/ball-scan/${jobId}/cancel`, { method: "POST" });
  } catch (e) {
    $("ball-scan-progress-text").textContent = "Stop failed: " + e.message;
  }
}

async function saveBallScan() {
  const out = $("tracknet-summary");
  if (!state.analysisId || !state.ballScan) {
    if (out) out.textContent = "No scan data to save yet.";
    return;
  }
  out.textContent = "Saving ball scan data to server...";
  try {
    const r = await fetch(`/hit-study/${state.analysisId}/ball-scan/save`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_id: state.ballScanJobId, result: state.ballScan }),
    });
    if (!r.ok) throw new Error(await r.text());
    const data = await r.json();
    
    out.textContent = "Choosing folder to save locally...";
    const payload = { result: state.ballScan };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const defaultName = `${state.analysisId}.ball-scan.json`;
    const saved = await saveToFileSystem(blob, defaultName, defaultName.split(".")[1]);
    
    if (saved) {
      if (out) out.textContent = `Saved locally (${state.ballScan.candidate_count || 0} candidates)`;
    } else {
      if (out) out.textContent = `Ball scan data saved on server, but local save was canceled.`;
    }
  } catch (e) {
    if (out) out.textContent = "Save scan failed: " + e.message;
  }
}

async function loadBallScan() {
  const out = $("tracknet-summary");
  if (!state.analysisId) return;
  
  const handleLoad = (data, sourceStr) => {
    state.ballScan = data.result;
    if (out) {
      out.textContent = `Loaded scan ${sourceStr}: ${state.ballScan.candidate_count || 0} candidates`;
    }
    drawPoseOverlay();
  };

  const file = await loadFromFileSystem(".json");
  if (!file) {
    try {
      const r = await fetch(`/hit-study/${state.analysisId}/ball-scan/load`);
      if (!r.ok) throw new Error(await r.text());
      handleLoad(await r.json(), "from server");
    } catch (e) {
      if (out) out.textContent = "Load scan failed: " + e.message;
    }
    return;
  }
  
  try {
    const text = await file.text();
    const payload = JSON.parse(text);
    
    // optionally upload to server so it's cached
    await fetch(`/hit-study/${state.analysisId}/ball-scan/save`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_id: "local_upload", result: payload.result }),
    });
    
    handleLoad(payload, "from local file");
  } catch (e) {
    if (out) out.textContent = "Load scan failed: " + e.message;
  }
}

function scheduleBallDiagnostic() {
  if (!state.ballHeatmapEnabled || state.ballDiagnosticLoading || !state.analysisId || !isHitStudy()) return;
  clearTimeout(state.ballDiagnosticTimer);
  state.ballDiagnosticTimer = setTimeout(() => {
    const player = $("player");
    if (!player || !state.ballHeatmapEnabled) return;
    const labels = Object.values(state.strikeLabels).filter((l) => l.source === "near_player_hit");
    if (!labels.length && !state.ballDiagnostic) return;
    const center = ballDiagnosticCenterTime(player.currentTime || 0);
    if (labels.length && Math.min(...labels.map((l) => Math.abs(l.time_s - (player.currentTime || 0)))) > 1.5) return;
    if (state.ballDiagnostic && Math.abs(Number(state.ballDiagnostic.time_s || 0) - center) < 0.2) return;
    runBallDiagnostic();
  }, 250);
}

function drawBallHeatmapOverlay(ctx, w, h, currentTime) {
  drawBallRoiOverlay(ctx, w, h);
  drawBallScanOverlay(ctx, w, h, currentTime);
  if (!state.ballHeatmapEnabled || !state.ballDiagnostic) return;
  const data = state.ballDiagnostic;
  const center = Number(data.time_s || 0);
  const windowS = Number(data.window_s || 0.8);
  if (Math.abs(currentTime - center) > windowS + 0.25) return;
  ctx.save();
  for (const c of data.candidates || []) {
    const dt = Math.abs(Number(c.time_s || 0) - currentTime);
    if (dt > 0.35) continue;
    const before = Number(c.time_s || 0) < center;
    const alpha = Math.max(0, 1 - dt / 0.35) * 0.65;
    const radius = Math.max(5, Math.min(16, Math.sqrt(Number(c.area || 1)) * 2.0));
    const grad = ctx.createRadialGradient(c.x * w, c.y * h, 1, c.x * w, c.y * h, radius);
    grad.addColorStop(0, before ? `rgba(9, 105, 218, ${alpha})` : `rgba(217, 119, 6, ${alpha})`);
    grad.addColorStop(1, before ? "rgba(9, 105, 218, 0)" : "rgba(217, 119, 6, 0)");
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(c.x * w, c.y * h, radius, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function drawBallScanOverlay(ctx, w, h, currentTime) {
  if (!state.ballScan || !state.ballScan.candidates) return;
  ctx.save();
  for (const c of state.ballScan.candidates) {
    const dt = Math.abs(Number(c.time_s || 0) - currentTime);
    if (dt > 0.18) continue;
    const alpha = Math.max(0, 1 - dt / 0.18) * 0.8;
    const radius = state.ballScan.ball_detector === "tracknet" ? 7 : 5;
    ctx.fillStyle = `rgba(88, 166, 255, ${alpha})`;
    ctx.strokeStyle = `rgba(9, 105, 218, ${alpha})`;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(c.x * w, c.y * h, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }
  ctx.restore();
}

function currentBallRoi() {
  const params = ballRoiParams();
  if (params.roi_mode === "full_frame") return { x1: 0, y1: 0, x2: 1, y2: 1 };
  const player = $("player");
  const frame = player ? nearestPoseFrame(player.currentTime || 0) : null;
  const det = frame && frame.near_player ? frame.near_player : null;
  const box = det && det.box ? det.box : null;
  if (!box) return state.ballDiagnostic && state.ballDiagnostic.roi ? state.ballDiagnostic.roi : null;
  const bw = Math.max(0.04, Number(box.x2 || 1) - Number(box.x1 || 0));
  const bh = Math.max(0.08, Number(box.y2 || 1) - Number(box.y1 || 0));
  return {
    x1: Math.max(0, Number(box.x1 || 0) - params.roi_expand_x * bw),
    y1: Math.max(0, Number(box.y1 || 0) - params.roi_expand_up * bh),
    x2: Math.min(1, Number(box.x2 || 1) + params.roi_expand_x * bw),
    y2: Math.min(1, Number(box.y2 || 1) + params.roi_expand_down * bh),
  };
}

function currentPlayerBox() {
  const player = $("player");
  const frame = player ? nearestPoseFrame(player.currentTime || 0) : null;
  if (frame && frame.near_player && frame.near_player.box) return frame.near_player.box;
  return state.ballDiagnostic && state.ballDiagnostic.player_box ? state.ballDiagnostic.player_box : null;
}

function drawBallRoiOverlay(ctx, w, h) {
  if (!state.ballRoiOverlayEnabled || !isHitStudy()) return;
  const roi = currentBallRoi();
  if (!roi) return;
  const params = ballRoiParams();
  ctx.save();
  ctx.strokeStyle = "#1a7f37";
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 4]);
  ctx.strokeRect(roi.x1 * w, roi.y1 * h, (roi.x2 - roi.x1) * w, (roi.y2 - roi.y1) * h);
  ctx.setLineDash([]);
  ctx.fillStyle = "rgba(26, 127, 55, 0.06)";
  ctx.fillRect(roi.x1 * w, roi.y1 * h, (roi.x2 - roi.x1) * w, (roi.y2 - roi.y1) * h);
  if (params.exclude_player) {
    const box = currentPlayerBox();
    if (box) {
      ctx.fillStyle = "rgba(207, 34, 46, 0.16)";
      ctx.strokeStyle = "rgba(207, 34, 46, 0.85)";
      ctx.lineWidth = 1.5;
      ctx.fillRect(box.x1 * w, box.y1 * h, (box.x2 - box.x1) * w, (box.y2 - box.y1) * h);
      ctx.strokeRect(box.x1 * w, box.y1 * h, (box.x2 - box.x1) * w, (box.y2 - box.y1) * h);
    }
  }
  ctx.restore();
}

function replaceNearHitLabels(labels) {
  for (const key of Object.keys(state.strikeLabels)) {
    if (state.strikeLabels[key].source === "near_player_hit") delete state.strikeLabels[key];
  }
  for (const label of labels || []) {
    state.strikeLabels[labelKey(label.source, label.time_s)] = label;
  }
  renderHitStudyPanel();
  drawTimeline();
}

async function saveNearPlayerHits() {
  const out = $("hit-study-report");
  if (out) out.textContent = "Preparing save...";
  try {
    const r = await fetch(`/hit-study/${state.analysisId}/labels/save`, { method: "POST" });
    if (!r.ok) throw new Error(await r.text());
    const data = await r.json();
    
    const payload = {
      analysis_id: data.analysis_id,
      video_id: data.video_id,
      filename: data.filename,
      algorithm: data.algorithm,
      saved_at: Date.now() / 1000,
      labels: data.labels
    };
    
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const defaultName = `${state.analysisId}.near-player-hit-labels.json`;

    if (window.showSaveFilePicker) {
      try {
        const handle = await window.showSaveFilePicker({
          suggestedName: defaultName,
          types: [{ description: 'JSON Files', accept: { 'application/json': ['.json'] } }],
        });
        const writable = await handle.createWritable();
        await writable.write(blob);
        await writable.close();
        if (out) out.textContent = `Saved ${data.labels.length} marks locally.`;
        return;
      } catch (e) {
        if (e.name === "AbortError") return;
      }
    }
    
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = defaultName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    if (out) out.textContent = `Downloaded ${data.labels.length} marks.`;
  } catch (e) {
    if (out) out.textContent = "Save marks failed: " + e.message;
  }
}

async function loadNearPlayerHits() {
  const out = $("hit-study-report");
  const input = document.createElement("input");
  input.type = "file";
  input.accept = ".json";
  
  input.onchange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    if (out) out.textContent = "Loading marked hits...";
    try {
      const text = await file.text();
      const payload = JSON.parse(text);
      
      const r = await fetch(`/hit-study/${state.analysisId}/labels/upload`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (!r.ok) throw new Error(await r.text());
      const data = await r.json();
      replaceNearHitLabels(data.labels || []);
      if (out) out.textContent = `Loaded ${data.labels.length} marks from ${file.name}`;
    } catch (err) {
      if (out) out.textContent = "Load marks failed: " + err.message;
    }
  };
  input.click();
}

function setSlowLabelMode(enabled) {
  state.slowLabelMode = !!enabled;
  const player = $("player");
  if (!player) return;
  if (state.slowLabelMode) {
    state.normalPlaybackRate = player.playbackRate || 1;
    player.playbackRate = parseFloat($("slow-label-rate").value || "0.5");
    player.pause();
  } else {
    player.playbackRate = state.normalPlaybackRate || 1;
  }
}

function playSlowWhileHeld() {
  const player = $("player");
  if (!player) return;
  player.playbackRate = parseFloat($("slow-label-rate").value || "0.5");
  player.play().catch(() => {});
}

function pauseSlowHold() {
  if (state.slowLabelMode && $("player")) $("player").pause();
}

function segmentAtTime(t) {
  return state.segments.find((s) => t >= s.start_s && t < s.end_s) || null;
}

function setAudioPreviewRange(start, end, { schedule = false } = {}) {
  const max = Math.max(0.1, Number(state.duration || 0));
  const minGap = Math.min(0.5, max);
  let nextStart = Math.max(0, Math.min(Number(start) || 0, max - minGap));
  let nextEnd = Math.max(nextStart + minGap, Math.min(Number(end) || minGap, max));
  if (nextEnd > max) {
    nextEnd = max;
    nextStart = Math.max(0, nextEnd - minGap);
  }
  state.audioPreviewRange = { start: nextStart, end: nextEnd };
  const startRange = $("audio-preview-start-range");
  const endRange = $("audio-preview-end-range");
  if (startRange && endRange) {
    startRange.max = max.toFixed(1);
    endRange.max = max.toFixed(1);
    startRange.value = nextStart.toFixed(1);
    endRange.value = nextEnd.toFixed(1);
  }
  renderAudioPreviewRange();
  if (schedule) scheduleAudioPreviewFromKnobs();
}

function renderAudioPreviewRange() {
  const label = $("audio-preview-range-label");
  if (!label) return;
  const { start, end } = state.audioPreviewRange;
  label.textContent = `${fmtPrecise(start)} to ${fmtPrecise(end)} (${fmt(end - start)})`;
}

function readAudioPreviewRangeFromSliders(changed) {
  const startRange = $("audio-preview-start-range");
  const endRange = $("audio-preview-end-range");
  if (!startRange || !endRange) return state.audioPreviewRange;
  const max = Math.max(0.1, Number(state.duration || 0));
  const minGap = Math.min(0.5, max);
  let start = parseFloat(startRange.value || "0");
  let end = parseFloat(endRange.value || `${max}`);
  if (end - start < minGap) {
    if (changed === "start") start = Math.max(0, end - minGap);
    else end = Math.min(max, start + minGap);
  }
  return { start, end };
}

function setAudioPreviewRangeFromPlayer() {
  const t = $("player") ? $("player").currentTime || 0 : 0;
  const seg = segmentAtTime(t);
  const start = seg ? seg.start_s : Math.max(0, t - 8);
  const end = seg ? seg.end_s : Math.min(state.duration, t + 8);
  setAudioPreviewRange(start, end);
}

function replaceImpactsInRange(start, end, impacts) {
  const incoming = (impacts || []).slice().sort((a, b) => a.time_s - b.time_s);
  state.impacts = state.impacts
    .filter((imp) => imp.time_s < start || imp.time_s >= end)
    .concat(incoming)
    .sort((a, b) => a.time_s - b.time_s);
}

async function runAudioPreview() {
  if (!state.analysisId) return;
  const { start, end } = state.audioPreviewRange;
  const out = $("audio-preview-summary");
  if (end <= start) {
    out.textContent = "Choose a valid recheck range first.";
    return;
  }
  state.audioPreviewReady = true;
  const knobs = readKnobs();
  out.textContent = "Analyzing audio range...";
  try {
    const r = await fetch(`/analysis/${state.analysisId}/audio-preview`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ range_start_s: start, range_end_s: end, knobs }),
    });
    if (!r.ok) throw new Error(await r.text());
    const data = await r.json();
    state.knobs = { ...data.knobs };
    replaceImpactsInRange(data.range_start_s, data.range_end_s, data.impacts || []);
    const summary = data.summary || {};
    out.textContent =
      `${summary.validated_impact_count || 0} validated / ${summary.impact_count || 0} candidates ` +
      `(${summary.centroid_pass_count || 0} pass centroid) ` +
      `from ${fmtPrecise(data.range_start_s)} to ${fmtPrecise(data.range_end_s)} ` +
      `(floor ${Number(summary.noise_floor || 0).toFixed(4)})`;
    renderKnobs();
    renderStrikeNav();
    renderStrikeDiagnostic();
    renderStrikeStats();
    renderCentroidReadout();
    drawTimeline();
  } catch (e) {
    out.textContent = "Audio preview failed: " + e.message;
  }
}

function escapeHtml(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// Missed-strike entry
$("add-missed-strike-btn").addEventListener("click", async () => {
  const t = $("player").currentTime || 0;
  const comment = ($("missed-strike-comment").value || "").trim() || null;
  await saveStrikeLabel({
    source: "manual",
    time_s: t,
    is_strike: true,
    algorithm_validated: null,
    comment,
  });
  $("missed-strike-comment").value = "";
});
$("audio-preview-use-segment-btn").addEventListener("click", setAudioPreviewRangeFromPlayer);
$("audio-preview-run-btn").addEventListener("click", runAudioPreview);
$("mark-near-hit-btn").addEventListener("click", markNearPlayerHit);
$("evaluate-hit-study-btn").addEventListener("click", evaluateHitStudy);
$("ball-diagnostic-btn").addEventListener("click", runBallDiagnostic);
$("ball-scan-btn").addEventListener("click", startBallScan);
$("ball-scan-stop-btn").addEventListener("click", stopBallScan);
$("ball-scan-save-btn").addEventListener("click", saveBallScan);
$("ball-scan-load-btn").addEventListener("click", loadBallScan);
$("ball-heatmap-toggle").addEventListener("change", (e) => {
  state.ballHeatmapEnabled = !!e.target.checked;
  if (state.ballHeatmapEnabled) scheduleBallDiagnostic();
  drawPoseOverlay();
});
$("ball-roi-overlay-toggle").addEventListener("change", (e) => {
  state.ballRoiOverlayEnabled = !!e.target.checked;
  drawPoseOverlay();
});
for (const id of ["ball-roi-width", "ball-roi-height", "ball-diff-threshold", "ball-exclude-player-toggle", "ball-roi-mode", "ball-detector-mode"]) {
  $(id).addEventListener("input", () => {
    state.ballDiagnostic = null;
    scheduleBallDiagnostic();
    drawPoseOverlay();
  });
}
$("save-near-hits-btn").addEventListener("click", saveNearPlayerHits);
$("load-near-hits-btn").addEventListener("click", loadNearPlayerHits);
$("slow-label-mode-toggle").addEventListener("change", (e) => setSlowLabelMode(e.target.checked));
$("slow-label-rate").addEventListener("change", () => {
  if (state.slowLabelMode && $("player")) $("player").playbackRate = parseFloat($("slow-label-rate").value || "0.5");
});
$("timeline-zoom-range-btn").addEventListener("click", () => {
  setTimelineView(state.audioPreviewRange.start, state.audioPreviewRange.end);
});
$("timeline-reset-zoom-btn").addEventListener("click", resetTimelineView);
$("audio-preview-start-range").addEventListener("input", () => {
  const { start, end } = readAudioPreviewRangeFromSliders("start");
  setAudioPreviewRange(start, end, { schedule: state.audioPreviewReady });
});
$("audio-preview-end-range").addEventListener("input", () => {
  const { start, end } = readAudioPreviewRangeFromSliders("end");
  setAudioPreviewRange(start, end, { schedule: state.audioPreviewReady });
});

function playSegment(i) {
  const seg = state.segments[i];
  const player = $("player");
  player.currentTime = seg.start_s;
  state.playMode = "segment";
  state.segmentEndS = seg.end_s;
  player.play();
}

function playNextOnSegment() {
  const player = $("player");
  if (state.allOnQueue.length === 0) {
    player.pause();
    state.playMode = "free";
    state.segmentEndS = null;
    return;
  }
  const seg = state.allOnQueue.shift();
  player.currentTime = seg.start_s;
  state.segmentEndS = seg.end_s;
  player.play();
}

$("play-all-on").addEventListener("click", () => {
  state.allOnQueue = state.segments.filter((s) => s.is_on).map((s) => ({ ...s }));
  state.playMode = "all_on";
  playNextOnSegment();
});

// Add segment at current time --------------------------------------------
$("add-segment-btn").addEventListener("click", () => {
  const t = $("player").currentTime || 0;
  const newSeg = {
    start_s: Math.max(0, t),
    end_s: Math.min(state.duration, t + 5),
    is_on: true,
    source: "manual",
    decision_stage: "manual",
  };
  state.segments.push(newSeg);
  state.segments.sort((a, b) => a.start_s - b.start_s);
  renderAll();
});

// Export ------------------------------------------------------------------
$("export-btn").addEventListener("click", async () => {
  const onSegs = state.segments.filter((s) => s.is_on);
  if (onSegs.length === 0) {
    alert("Select at least one segment to export.");
    return;
  }
  $("export-btn").disabled = true;
  $("export-section").hidden = false;
  $("download-link").hidden = true;
  $("export-text").textContent = "Stitching segments...";
  $("export-spinner").hidden = false;

  const r = await fetch(`/export/${state.videoId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ segments: state.segments }),
  });
  const data = await r.json();
  pollExport(data.export_id);
});

async function pollExport(exportId) {
  while (true) {
    const r = await fetch(`/export-status/${exportId}`);
    const data = await r.json();
    if (data.status === "done") {
      $("export-text").textContent = `Done — ${fmt(data.duration_s)} of footage`;
      $("export-spinner").hidden = true;
      $("download-link").href = `/download/${exportId}`;
      $("download-link").textContent = "Download Video";
      $("download-link").hidden = false;
      $("export-btn").disabled = false;
      return;
    }
    if (data.status === "error") {
      $("export-text").textContent = "Export failed: " + (data.error_msg || "");
      $("export-spinner").hidden = true;
      $("export-btn").disabled = false;
      return;
    }
    await new Promise((r) => setTimeout(r, 1500));
  }
}

// --- Modular Pose Scan ---
async function startPoseScan() {
  if (!state.analysisId || !isHitStudy()) return;
  const out = $("pose-scan-summary");
  const knobs = readKnobs();
  const { start, end } = analysisRange();
  out.textContent = `Starting pose scan (${fmtPrecise(start)} to ${fmtPrecise(end)})...`;
  try {
    const r = await fetch(`/hit-study/${state.analysisId}/pose-scan`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        range_start_s: start,
        range_end_s: end,
        sample_fps: Number(knobs.sample_fps || 4),
        model_name: knobs.pose_model || "yolo11n-pose.pt",
        pose_conf: Number(knobs.pose_conf || 0.25),
        pose_imgsz: Number(knobs.pose_imgsz || 640),
      }),
    });
    if (!r.ok) throw new Error(await r.text());
    const data = await r.json();
    state.poseScanJobId = data.job_id;
    pollPoseScan(data.job_id);
  } catch (e) {
    out.textContent = "Pose scan failed: " + e.message;
  }
}

async function pollPoseScan(jobId) {
  const out = $("pose-scan-summary");
  while (state.poseScanJobId === jobId) {
    const r = await fetch(`/hit-study/pose-scan/${jobId}`);
    if (!r.ok) {
      out.textContent = "Status check failed.";
      return;
    }
    const job = await r.json();
    const eta = typeof job.progress_eta_s === "number" ? ` · ETA ${fmtEta(job.progress_eta_s)}` : "";
    out.textContent = `${job.progress_message || job.status} (${Math.round(job.progress_percent || 0)}%)${eta}`;
    if (job.status === "done") {
      state.poseScanJobId = null;
      if (job.result) {
        state.poseData = {
          metadata: {
            model_name: job.result.model_name,
            conf_threshold: job.result.pose_conf,
            image_size: job.result.pose_imgsz,
            sample_fps: job.result.sample_fps,
            range_start_s: job.result.range_start_s,
            range_end_s: job.result.range_end_s,
          },
          frames: job.result.frames,
          summary: job.result.summary,
        };
        renderPosePanel();
        drawPoseOverlay();
        drawTimeline();
      }
      out.textContent = `Pose scan done: ${job.result?.summary?.sample_count || 0} frames, ${job.result?.summary?.frames_with_poses || 0} with poses`;
      return;
    }
    if (job.status === "error") {
      state.poseScanJobId = null;
      out.textContent = "Pose scan error: " + (job.error || "unknown");
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
}

async function savePoseScan() {
  if (!state.analysisId) return;
  const out = $("pose-scan-summary");
  out.textContent = "Saving pose data to server...";
  try {
    const r = await fetch(`/hit-study/${state.analysisId}/pose-scan/save`, { method: "POST" });
    if (!r.ok) throw new Error(await r.text());
    const data = await r.json();
    
    // Now download it
    out.textContent = "Choosing folder to save locally...";
    const dl = await fetch(`/hit-study/${state.analysisId}/pose-scan/download`);
    if (!dl.ok) throw new Error(await dl.text());
    const blob = await dl.blob();
    const defaultName = `${state.analysisId}.pose-scan.json.gz`;
    const saved = await saveToFileSystem(blob, defaultName, defaultName.split(".")[1]);
    
    if (saved) {
      out.textContent = `Saved locally (${data.frame_count || 0} frames)`;
    } else {
      out.textContent = `Pose data saved on server, but local save was canceled.`;
    }
  } catch (e) {
    out.textContent = "Save failed: " + e.message;
  }
}

async function loadPoseScan() {
  if (!state.analysisId) return;
  const out = $("pose-scan-summary");
  out.textContent = "Loading saved pose data...";
  try {
    const r = await fetch(`/hit-study/${state.analysisId}/pose-scan/load`);
    if (!r.ok) throw new Error(await r.text());
    const payload = await r.json();
    const result = payload.result;
    if (result) {
      state.poseData = {
        metadata: {
          model_name: result.model_name,
          conf_threshold: result.pose_conf,
          image_size: result.pose_imgsz,
          sample_fps: result.sample_fps,
          range_start_s: result.range_start_s,
          range_end_s: result.range_end_s,
        },
        frames: result.frames,
        summary: result.summary,
      };
      renderPosePanel();
      drawPoseOverlay();
      drawTimeline();
      out.textContent = `Loaded: ${result.summary?.sample_count || 0} frames, ${result.summary?.frames_with_poses || 0} with poses`;
    } else {
      out.textContent = "Load returned no data.";
    }
  } catch (e) {
    out.textContent = "Load failed: " + e.message;
  }
}

// --- Modular Audio Scan ---
async function startAudioScan() {
  if (!state.analysisId || !isHitStudy()) return;
  const out = $("audio-preview-summary");
  const knobs = readKnobs();
  const { start, end } = analysisRange();
  out.textContent = `Starting audio scan (${fmtPrecise(start)} to ${fmtPrecise(end)})...`;
  try {
    const r = await fetch(`/hit-study/${state.analysisId}/audio-scan`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        range_start_s: start,
        range_end_s: end,
        audio_sample_rate: Number(knobs.audio_sample_rate || 22050),
        bandpass_low_hz: Number(knobs.bandpass_low_hz || 1000),
        bandpass_high_hz: Number(knobs.bandpass_high_hz || 8000),
        peak_height_mad_k: Number(knobs.peak_height_mad_k || 6),
        peak_prominence_mult: Number(knobs.peak_prominence_mult || 2),
        min_impact_separation_s: Number(knobs.min_impact_separation_s || 0.15),
        min_spectral_centroid_hz: Number(knobs.min_spectral_centroid_hz || 0),
      }),
    });
    if (!r.ok) throw new Error(await r.text());
    const data = await r.json();
    state.audioScanJobId = data.job_id;
    pollAudioScan(data.job_id);
  } catch (e) {
    out.textContent = "Audio scan failed: " + e.message;
  }
}

async function pollAudioScan(jobId) {
  const out = $("audio-preview-summary");
  while (state.audioScanJobId === jobId) {
    const r = await fetch(`/hit-study/audio-scan/${jobId}`);
    if (!r.ok) {
      out.textContent = "Status check failed.";
      return;
    }
    const job = await r.json();
    out.textContent = `${job.progress_message || job.status} (${Math.round(job.progress_percent || 0)}%)`;
    if (job.status === "done" && job.result) {
      state.audioScanJobId = null;
      const impacts = job.result.impacts || [];
      replaceImpactsInRange(job.result.range_start_s, job.result.range_end_s, impacts);
      out.textContent = `Audio scan done: ${impacts.length} impacts (floor ${Number(job.result.noise_floor || 0).toFixed(4)})`;
      renderStrikeNav();
      renderStrikeDiagnostic();
      renderStrikeStats();
      drawTimeline();
      return;
    }
    if (job.status === "error") {
      state.audioScanJobId = null;
      out.textContent = "Audio scan error: " + (job.error || "unknown");
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
}

async function saveAudioScan() {
  if (!state.analysisId) return;
  const out = $("audio-preview-summary");
  out.textContent = "Saving audio data to server...";
  try {
    const r = await fetch(`/hit-study/${state.analysisId}/audio-scan/save`, { method: "POST" });
    if (!r.ok) throw new Error(await r.text());
    const data = await r.json();
    
    out.textContent = "Choosing folder to save locally...";
    const dl = await fetch(`/hit-study/${state.analysisId}/audio-scan/download`);
    if (!dl.ok) throw new Error(await dl.text());
    const blob = await dl.blob();
    const defaultName = `${state.analysisId}.audio-scan.json`;
    const saved = await saveToFileSystem(blob, defaultName, defaultName.split(".")[1]);
    
    if (saved) {
      out.textContent = `Saved locally (${data.impact_count || 0} impacts)`;
    } else {
      out.textContent = `Audio data saved on server, but local save was canceled.`;
    }
  } catch (e) {
    out.textContent = "Save failed: " + e.message;
  }
}

async function loadAudioScan() {
  if (!state.analysisId) return;
  const out = $("audio-preview-summary");
  
  const handleLoad = (payload, sourceStr) => {
    const result = payload.result;
    if (result) {
      const impacts = result.impacts || [];
      replaceImpactsInRange(result.range_start_s, result.range_end_s, impacts);
      out.textContent = `Loaded: ${impacts.length} impacts ${sourceStr}`;
      renderStrikeNav();
      renderStrikeDiagnostic();
      renderStrikeStats();
      drawTimeline();
    } else {
      out.textContent = "Load returned no data.";
    }
  };

  const file = await loadFromFileSystem(".json");
  if (!file) {
    out.textContent = "Loading saved audio data from server...";
    try {
      const r = await fetch(`/hit-study/${state.analysisId}/audio-scan/load`);
      if (!r.ok) throw new Error(await r.text());
      handleLoad(await r.json(), "from server");
    } catch (e) {
      out.textContent = "Load failed: " + e.message;
    }
    return;
  }
  
  out.textContent = "Uploading local audio data...";
  try {
    const formData = new FormData();
    formData.append("file", file);
    const r = await fetch(`/hit-study/${state.analysisId}/audio-scan/upload`, {
      method: "POST",
      body: formData,
    });
    if (!r.ok) throw new Error(await r.text());
    handleLoad(await r.json(), "from local file");
  } catch (e) {
    out.textContent = "Load failed: " + e.message;
  }
}

function analysisRange() {
  const range = state.audioPreviewRange || {};
  const start = Number(range.start);
  const end = Number(range.end);
  if (Number.isFinite(start) && Number.isFinite(end) && end > start + 0.05) {
    return { start, end };
  }
  return { start: timelineStart(), end: timelineEnd() };
}

// Wire up pose card buttons
$("pose-scan-btn").addEventListener("click", startPoseScan);
$("pose-save-btn").addEventListener("click", savePoseScan);
$("pose-load-btn").addEventListener("click", loadPoseScan);
// Wire up audio card buttons — override previous audio-preview-run-btn handler
$("audio-preview-run-btn").removeEventListener("click", runAudioPreview);
$("audio-preview-run-btn").addEventListener("click", startAudioScan);
$("audio-save-btn").addEventListener("click", saveAudioScan);
$("audio-load-btn").addEventListener("click", loadAudioScan);

// --- Card Drag and Drop ---
function loadCardLayout() {
  const saved = localStorage.getItem("editorCardLayout");
  if (!saved) return;
  try {
    const layout = JSON.parse(saved);
    const cols = {
      left: document.querySelector(".editor-left"),
      center: document.querySelector(".editor-center"),
      right: document.querySelector(".editor-right")
    };
    for (const [colName, col] of Object.entries(cols)) {
      if (!layout[colName]) continue;
      layout[colName].forEach(id => {
        const card = document.getElementById(id);
        if (card) col.appendChild(card);
      });
    }
  } catch (e) {
    console.error("Failed to restore layout", e);
  }
}

function initCardDragDrop() {
  const cards = document.querySelectorAll(".card");
  const columns = document.querySelectorAll(".editor-left, .editor-center, .editor-right");
  let draggedCard = null;

  // Global mouseup: clean up draggable on any card that isn't mid-drag.
  // This replaces the per-h2 mouseleave handler which had a race condition:
  // mouseleave fires before dragstart when the mouse exits the small h2 bounds.
  document.addEventListener("mouseup", () => {
    cards.forEach(card => {
      if (card !== draggedCard) {
        card.removeAttribute("draggable");
      }
    });
  });

  cards.forEach(card => {
    if (!card.id) return;

    const header = card.querySelector("h2");
    if (header) {
      header.style.cursor = "grab";
      header.title = "Drag to move card";
      header.addEventListener("mousedown", (e) => {
        // Only trigger on primary mouse button, and not if clicking a button/input inside h2
        if (e.button !== 0) return;
        card.setAttribute("draggable", "true");
      });
    }

    card.addEventListener("dragstart", (e) => {
      draggedCard = card;
      e.dataTransfer.effectAllowed = "move";
      e.dataTransfer.setData("text/plain", card.id);
      setTimeout(() => card.style.opacity = "0.4", 0);
      if (header) header.style.cursor = "grabbing";
    });

    card.addEventListener("dragend", () => {
      card.removeAttribute("draggable");
      if (draggedCard) {
        draggedCard.style.opacity = "1";
        if (header) header.style.cursor = "grab";
        draggedCard = null;
        saveCardLayout();
      }
    });
  });

  columns.forEach(col => {
    col.addEventListener("dragover", (e) => {
      e.preventDefault();
      e.dataTransfer.dropEffect = "move";
      if (!draggedCard) return;
      
      const afterElement = getDragAfterElement(col, e.clientY);
      if (afterElement == null) {
        col.appendChild(draggedCard);
      } else {
        col.insertBefore(draggedCard, afterElement);
      }
    });
  });

  function getDragAfterElement(container, y) {
    const draggableElements = [...container.querySelectorAll('.card:not([style*="opacity: 0.4"])')];
    return draggableElements.reduce((closest, child) => {
      const box = child.getBoundingClientRect();
      const offset = y - box.top - box.height / 2;
      if (offset < 0 && offset > closest.offset) {
        return { offset: offset, element: child };
      } else {
        return closest;
      }
    }, { offset: Number.NEGATIVE_INFINITY }).element;
  }
}

function saveCardLayout() {
  const layout = {
    left: Array.from(document.querySelector(".editor-left").children).filter(c => c.classList.contains('card') && c.id).map(c => c.id),
    center: Array.from(document.querySelector(".editor-center").children).filter(c => c.classList.contains('card') && c.id).map(c => c.id),
    right: Array.from(document.querySelector(".editor-right").children).filter(c => c.classList.contains('card') && c.id).map(c => c.id),
  };
  localStorage.setItem("editorCardLayout", JSON.stringify(layout));
}

// Apply layout on load
loadCardLayout();
initCardDragDrop();

// Test hooks — exposes internals to tests in tests/test_frontend_render.js.
// Has no effect in production beyond setting a few read-only references on window.
if (typeof window !== "undefined") {
  window.DETECTOR_CONFIGS = DETECTOR_CONFIGS;
  window.HELP_TEXT = HELP_TEXT;
  window.state = state;
  window.renderStartConfigControls = renderStartConfigControls;
  window.showHelpPopover = showHelpPopover;
  window.hideHelpPopover = hideHelpPopover;
  window.strikeReason = strikeReason;
  window.renderStrikeDiagnostic = renderStrikeDiagnostic;
  window.renderStrikeStats = renderStrikeStats;
  window.labelKey = labelKey;
  window.nearestImpact = nearestImpact;
}
