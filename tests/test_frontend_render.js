// Runtime DOM tests for the frontend.
//
// These guard the *behavior* that the Python static-analysis tests can't
// catch: that hide/show actually toggles, filtering produces the right
// pills, and the help popover responds to clicks/Esc.
//
// Run with:
//   node tests/test_frontend_render.js
//
// Requires `npm install --no-save jsdom` (already installed locally).

const fs = require("node:fs");
const path = require("node:path");
const assert = require("node:assert/strict");
const { JSDOM } = require("jsdom");

const ROOT = path.resolve(__dirname, "..");
const HTML_PATH = path.join(ROOT, "static", "index.html");
const APP_JS_PATH = path.join(ROOT, "static", "app.js");
const STYLE_CSS_PATH = path.join(ROOT, "static", "style.css");

let testsRun = 0;
let testsFailed = 0;
function it(name, fn) {
  testsRun += 1;
  try {
    fn();
    console.log(`  ok   ${name}`);
  } catch (e) {
    testsFailed += 1;
    console.log(`  FAIL ${name}`);
    console.log(`         ${e.message}`);
    if (process.env.VERBOSE) console.log(e.stack);
  }
}
function suite(name, fn) {
  console.log(`\n# ${name}`);
  fn();
}

// Build a fresh DOM that has loaded index.html, style.css, and app.js.
// We stub fetch so refreshLibrary() doesn't blow up on first load.
function makeDom({ media = [], analyses = [], poseData = null } = {}) {
  const html = fs.readFileSync(HTML_PATH, "utf8");
  const css = fs.readFileSync(STYLE_CSS_PATH, "utf8");
  const js = fs.readFileSync(APP_JS_PATH, "utf8");

  const dom = new JSDOM(html, {
    runScripts: "outside-only",
    pretendToBeVisual: true,
    url: "http://localhost",
  });
  const { window } = dom;

  // Inject stylesheet so CSS rules (specifically [hidden] !important) apply.
  const style = window.document.createElement("style");
  style.textContent = css;
  window.document.head.appendChild(style);

  // Stub fetch for the endpoints app.js touches at startup.
  window.fetch = async (url, opts) => {
    let body;
    if (url === "/media") body = media;
    else if (url === "/analyses") body = analyses;
    else if (url.startsWith("/pose-data/")) body = poseData;
    else body = {};
    return {
      ok: true,
      status: 200,
      json: async () => body,
      text: async () => JSON.stringify(body),
    };
  };

  // Run the app code inside the JSDOM window context.
  window.eval(js);
  return window;
}

function flushTicks() {
  return new Promise((r) => setTimeout(r, 0));
}

// ---- DETECTOR_CONFIGS structure ----------------------------------------

suite("DETECTOR_CONFIGS shape", () => {
  const win = makeDom();
  const cfgs = win.DETECTOR_CONFIGS;

  it("exposes the three detectors", () => {
    assert.ok(cfgs.median_frame, "median_frame missing");
    assert.ok(cfgs.median_court_roi, "median_court_roi missing");
    assert.ok(cfgs.pose_skeleton_yolo, "pose_skeleton_yolo missing");
    assert.ok(cfgs.near_player_hit_study, "near_player_hit_study missing");
  });

  it("median_court_roi is a strict superset of median_frame", () => {
    const median = new Set(cfgs.median_frame);
    const roi = new Set(cfgs.median_court_roi);
    for (const k of median) assert.ok(roi.has(k), `roi missing ${k}`);
    assert.ok(roi.has("court_weight"));
    assert.ok(roi.has("outside_weight"));
    assert.ok(roi.has("near_camera_weight"));
  });

  it("pose detector has imgsz and not motion knobs", () => {
    const pose = new Set(cfgs.pose_skeleton_yolo);
    assert.ok(pose.has("pose_imgsz"));
    assert.ok(pose.has("pose_conf"));
    assert.ok(pose.has("pose_model"));
    assert.ok(!pose.has("diff_threshold"));
    assert.ok(!pose.has("court_weight"));
    assert.ok(!pose.has("median_bg_samples"));
  });
});

// ---- renderStartConfigControls ----------------------------------------

suite("renderStartConfigControls hides irrelevant inputs", () => {
  const win = makeDom();

  function visibleLabelsFor(algorithm) {
    win.document.getElementById("start-algorithm").value = algorithm;
    win.renderStartConfigControls();
    return Array.from(
      win.document.querySelectorAll(".start-config")
    ).filter(
      (el) => !el.hidden && win.getComputedStyle(el).display !== "none"
    );
  }

  it("YOLO selection hides court_weight / median_bg_samples", () => {
    const visible = visibleLabelsFor("pose_skeleton_yolo");
    const labels = visible.map((el) =>
      el.querySelector("input")?.id || ""
    );
    assert.ok(
      !labels.includes("start-court-weight"),
      `court_weight still visible: ${labels.join(",")}`,
    );
    assert.ok(
      !labels.includes("start-outside-weight"),
      `outside_weight still visible`,
    );
    assert.ok(
      !labels.includes("start-near-camera-weight"),
      `near_camera_weight still visible`,
    );
    assert.ok(
      !labels.includes("start-median-bg-samples"),
      `median_bg_samples still visible`,
    );
    // pose-specific inputs should be visible
    assert.ok(labels.includes("start-pose-model"));
    assert.ok(labels.includes("start-pose-conf"));
    assert.ok(labels.includes("start-pose-imgsz"));
  });

  it("median_court_roi shows court weights + median knobs, hides pose", () => {
    const labels = visibleLabelsFor("median_court_roi").map(
      (el) => el.querySelector("input")?.id || ""
    );
    assert.ok(labels.includes("start-court-weight"));
    assert.ok(labels.includes("start-outside-weight"));
    assert.ok(labels.includes("start-near-camera-weight"));
    assert.ok(labels.includes("start-median-bg-samples"));
    assert.ok(!labels.includes("start-pose-model"));
    assert.ok(!labels.includes("start-pose-conf"));
  });

  it("median_frame hides court and pose knobs", () => {
    const labels = visibleLabelsFor("median_frame").map(
      (el) => el.querySelector("input")?.id || ""
    );
    assert.ok(!labels.includes("start-court-weight"));
    assert.ok(!labels.includes("start-pose-model"));
    assert.ok(labels.includes("start-median-bg-samples"));
    assert.ok(labels.includes("start-sample-fps"));
  });

  it("near_player_hit_study starts without library-page knobs", () => {
    const labels = visibleLabelsFor("near_player_hit_study").map(
      (el) => el.querySelector("input")?.id || ""
    );
    assert.deepStrictEqual(labels, []);
  });
});

// ---- Hit study modular knobs --------------------------------------------

suite("hit study modular knobs", () => {
  it("allows editing audio knobs in a new lightweight hit-study workspace", () => {
    const win = makeDom();
    win.state.analysis = { algorithm: "near_player_hit_study", metadata: null };
    win.state.knobs = null;
    win.state.defaultKnobs = {
      sample_fps: 4,
      pose_model: "yolo11n-pose.pt",
      pose_conf: 0.25,
      pose_imgsz: 640,
      audio_sample_rate: 22050,
      bandpass_low_hz: 1000,
      bandpass_high_hz: 8000,
      peak_height_mad_k: 6,
      peak_prominence_mult: 2,
      min_spectral_centroid_hz: 2500,
      min_impact_separation_s: 0.15,
    };
    win.renderKnobs();

    for (const id of [
      "knob-audio-sample-rate",
      "knob-bandpass-low-hz",
      "knob-bandpass-high-hz",
      "knob-peak-height-mad-k",
      "knob-peak-prominence-mult",
      "knob-min-spectral-centroid-hz",
      "knob-min-impact-separation",
    ]) {
      assert.equal(win.document.getElementById(id).disabled, false, `${id} should be editable`);
    }
  });
});

// ---- Analysis range ------------------------------------------------------

suite("analysis range", () => {
  it("can default the audio analysis range to the whole timeline", () => {
    const win = makeDom();
    win.state.duration = 435;
    win.state.timelineView = { start: 0, end: null };
    win.setAudioPreviewRange(0, win.state.duration);
    assert.equal(win.state.audioPreviewRange.start, 0);
    assert.equal(win.state.audioPreviewRange.end, 435);
    assert.equal(win.document.getElementById("audio-preview-start-range").value, "0.0");
    assert.equal(win.document.getElementById("audio-preview-end-range").value, "435.0");
  });

  it("still supports manually setting the audio range near the current player time", () => {
    const win = makeDom();
    win.state.duration = 435;
    const player = win.document.getElementById("player");
    Object.defineProperty(player, "currentTime", { value: 44, configurable: true });
    win.setAudioPreviewRangeFromPlayer();
    assert.equal(win.state.audioPreviewRange.start, 36);
    assert.equal(win.state.audioPreviewRange.end, 52);
  });
});

// ---- Pose overlay --------------------------------------------------------

suite("pose overlay", () => {
  it("keeps the pose data card visible in a new hit-study workspace", () => {
    const win = makeDom();
    win.state.analysis = { algorithm: "near_player_hit_study", metadata: null };
    win.state.poseData = null;
    win.renderPosePanel();
    assert.equal(win.document.getElementById("pose-panel").hidden, false);
    assert.match(win.document.getElementById("pose-summary").textContent, /No pose skeleton data/i);
  });

  it("lets the YOLO pose card hide and show the pose data card", () => {
    const win = makeDom();
    win.state.analysis = { algorithm: "near_player_hit_study", metadata: null };
    const toggle = win.document.getElementById("pose-panel-toggle");
    const panel = win.document.getElementById("pose-panel");
    win.renderPosePanel();
    assert.equal(panel.hidden, false);
    toggle.checked = false;
    toggle.dispatchEvent(new win.Event("change", { bubbles: true }));
    assert.equal(panel.hidden, true);
    toggle.checked = true;
    toggle.dispatchEvent(new win.Event("change", { bubbles: true }));
    assert.equal(panel.hidden, false);
  });

  it("can auto-enable the overlay after pose data is available", () => {
    const win = makeDom();
    const toggle = win.document.getElementById("pose-overlay-toggle");
    assert.equal(toggle.checked, false);
    win.enablePoseOverlay();
    assert.equal(toggle.checked, true);
  });
});

// ---- Card drag-drop setup -----------------------------------------------

suite("card drag-and-drop initialization", () => {
  it("every card with an h2 gets cursor:grab on the h2", () => {
    const win = makeDom();
    const cards = win.document.querySelectorAll(".card[id]");
    let cardsWithH2 = 0;
    let cardsWithGrab = 0;
    cards.forEach(card => {
      const h2 = card.querySelector("h2");
      if (h2) {
        cardsWithH2++;
        if (h2.style.cursor === "grab") cardsWithGrab++;
      }
    });
    assert.ok(cardsWithH2 > 0, "should have at least one card with h2");
    assert.strictEqual(cardsWithGrab, cardsWithH2, `only ${cardsWithGrab}/${cardsWithH2} cards have grab cursor on h2`);
  });

  it("h2 elements have a drag title", () => {
    const win = makeDom();
    const cards = win.document.querySelectorAll(".card[id]");
    cards.forEach(card => {
      const h2 = card.querySelector("h2");
      if (h2) {
        assert.ok(h2.title.length > 0, `h2 in #${card.id} has no title`);
      }
    });
  });

  it("restores saved card columns and order", () => {
    const win = makeDom();
    win.localStorage.setItem("editorCardLayout", JSON.stringify({
      left: ["audio-processing-card", "yolo-pose-card"],
      center: ["candidate-validation-card", "timeline-card", "player-card"],
      right: ["segments-card"],
    }));
    win.loadCardLayout();
    const leftIds = Array.from(win.document.querySelector(".editor-left").children)
      .filter((el) => el.classList.contains("card"))
      .map((el) => el.id);
    const centerIds = Array.from(win.document.querySelector(".editor-center").children)
      .filter((el) => el.classList.contains("card"))
      .map((el) => el.id);
    assert.equal(leftIds[0], "audio-processing-card");
    assert.equal(leftIds[1], "yolo-pose-card");
    assert.equal(centerIds[0], "candidate-validation-card");
    assert.equal(centerIds[1], "timeline-card");
    assert.equal(centerIds[2], "player-card");
  });
});

// ---- Help popover ------------------------------------------------------

suite("help popover shows/hides on click", () => {
  // Each test gets a fresh DOM so popover state doesn't leak across cases.
  function freshClickHelp(key) {
    const w = makeDom();
    const dot = w.document.querySelector(`.help-dot[data-help-key="${key}"]`);
    assert.ok(dot, `no help dot for ${key}`);
    dot.dispatchEvent(new w.MouseEvent("click", { bubbles: true }));
    return w;
  }

  it("clicking a ? dot shows the popover with the right text", () => {
    const w = freshClickHelp("sample_fps");
    const popover = w.document.getElementById("help-popover");
    assert.ok(!popover.hidden, "popover should be visible after click");
    assert.match(popover.textContent, /[Ff]rames per second/);
  });

  it("clicking the same dot toggles it closed", () => {
    const w = freshClickHelp("sample_fps"); // open
    const dot = w.document.querySelector('.help-dot[data-help-key="sample_fps"]');
    dot.dispatchEvent(new w.MouseEvent("click", { bubbles: true })); // close
    const popover = w.document.getElementById("help-popover");
    assert.ok(popover.hidden, "popover should toggle closed");
  });

  it("Escape closes the popover", () => {
    const w = freshClickHelp("sample_fps");
    w.document.dispatchEvent(
      new w.KeyboardEvent("keydown", { key: "Escape" }),
    );
    const popover = w.document.getElementById("help-popover");
    assert.ok(popover.hidden, "Escape should close popover");
  });

  it("clicking outside closes the popover", () => {
    const w = freshClickHelp("sample_fps");
    // Click on something neutral (the body), not on a help-dot.
    w.document.body.dispatchEvent(
      new w.MouseEvent("click", { bubbles: true }),
    );
    const popover = w.document.getElementById("help-popover");
    assert.ok(popover.hidden, "outside click should close popover");
  });
});

// ---- Strike diagnostic panel ------------------------------------------

suite("strikeReason classifies impact decisions", () => {
  const win = makeDom();

  it("validated normal case: reports max_wrist_v ≥ threshold", () => {
    const reason = win.strikeReason(
      { validated: true, fallback_used: false, max_wrist_v: 0.55, max_box_v: 0.2 },
      { min_wrist_velocity: 0.4 },
    );
    assert.match(reason, /Validated.*0\.55.*0\.4/);
  });

  it("validated via fallback mentions body motion", () => {
    const reason = win.strikeReason(
      { validated: true, fallback_used: true, max_wrist_v: 0, max_box_v: 0.7 },
      { min_wrist_velocity: 0.4 },
    );
    assert.match(reason, /fallback/i);
    assert.match(reason, /max_box_v=0\.7/);
  });

  it("rejected with no detections explains pose window", () => {
    const reason = win.strikeReason(
      { validated: false, fallback_used: false, max_wrist_v: 0, max_box_v: 0 },
      { min_wrist_velocity: 0.4, pose_window_s: 0.75 },
    );
    assert.match(reason, /no person detections/i);
    assert.match(reason, /0\.75/);
  });

  it("rejected with low velocity explains threshold gap", () => {
    const reason = win.strikeReason(
      { validated: false, fallback_used: false, max_wrist_v: 0.21, max_box_v: 0.05 },
      { min_wrist_velocity: 0.4 },
    );
    assert.match(reason, /Rejected.*0\.21.*0\.4/);
  });
});

suite("renderStrikeDiagnostic shows the card and computes the nearest impact", () => {
  it("hides the validation card when no impacts are loaded", () => {
    const win = makeDom();
    win.state.impacts = [];
    win.renderStrikeDiagnostic();
    const card = win.document.getElementById("candidate-validation-card");
    assert.ok(card.hidden, "card should be hidden when impacts empty");
  });

  it("hides the validation card for raw audio-only candidates", () => {
    const win = makeDom();
    win.state.impacts = [
      { time_s: 10.0, snr: 5, amplitude: 0.2 },
    ];
    win.renderStrikeDiagnostic();
    const card = win.document.getElementById("candidate-validation-card");
    assert.ok(card.hidden, "card should be hidden before candidate validation results exist");
  });

  it("shows the validation card and renders the nearest candidate", () => {
    const win = makeDom();
    win.state.impacts = [
      { time_s: 10.0, validated: true, max_wrist_v: 0.6, max_box_v: 0.1, snr: 5, amplitude: 0.2, fallback_used: false, player_id: 0 },
      { time_s: 20.0, validated: false, max_wrist_v: 0.2, max_box_v: 0.1, snr: 4, amplitude: 0.1, fallback_used: false, player_id: null },
    ];
    win.state.poseData = { rally: { knobs: { min_wrist_velocity: 0.4, pose_window_s: 0.75 } } };
    win.state.strikeLabels = {};
    // Stub player.currentTime: jsdom doesn't actually run video, so we set it manually.
    Object.defineProperty(win.document.getElementById("player"), "currentTime", {
      configurable: true,
      get: () => 9.7,
    });
    win.renderStrikeDiagnostic();
    const card = win.document.getElementById("candidate-validation-card");
    assert.ok(!card.hidden, "card should be visible");
    const text = win.document.getElementById("strike-current").textContent;
    assert.match(text, /VALIDATED/);
    assert.match(text, /1\/2/, `index marker missing: ${text}`);
  });

  it("explains rejection for the nearest rejected candidate", () => {
    const win = makeDom();
    win.state.impacts = [
      { time_s: 30.0, validated: false, max_wrist_v: 0.15, max_box_v: 0.0, snr: 3, amplitude: 0.05, fallback_used: false, player_id: null },
    ];
    win.state.poseData = { rally: { knobs: { min_wrist_velocity: 0.4, pose_window_s: 0.75 } } };
    Object.defineProperty(win.document.getElementById("player"), "currentTime", {
      configurable: true,
      get: () => 30.0,
    });
    win.renderStrikeDiagnostic();
    const text = win.document.getElementById("strike-current").textContent;
    assert.match(text, /REJECTED/);
    assert.match(text, /max_wrist_v=0\.15/);
  });
});

suite("strike stats", () => {
  it("counts agreement, false positives, false negatives, and missed", () => {
    const win = makeDom();
    win.state.impacts = [{ time_s: 1, validated: true }, { time_s: 2, validated: false }, { time_s: 3, validated: true }];
    win.state.strikeLabels = {
      "candidate|1.000": { source: "candidate", time_s: 1.0, algorithm_validated: true, is_strike: true },
      "candidate|2.000": { source: "candidate", time_s: 2.0, algorithm_validated: false, is_strike: true },
      "candidate|3.000": { source: "candidate", time_s: 3.0, algorithm_validated: true, is_strike: false },
      "manual|7.500":   { source: "manual",   time_s: 7.5, algorithm_validated: null,  is_strike: true },
    };
    win.renderStrikeStats();
    const text = win.document.getElementById("strike-stats").textContent;
    assert.match(text, /Audited.*3\/3/);
    assert.match(text, /false \+ve\s*1/i);  // 1 validated-but-not-a-strike
    assert.match(text, /false −ve\s*1/);    // 1 rejected-but-was-a-strike
    assert.match(text, /missed\s*1/);
  });
});

// ---- Active modular file labels -----------------------------------------

suite("active modular file labels", () => {
  it("keeps saved modular filenames after a full render", () => {
    const win = makeDom();
    win.state.poseData = { source_file: "old.pose-scan.json.gz" };
    win.state.hitStudyData = { pose_source_file: "old.pose-scan.json.gz" };
    win.state.ballScan = { source_file: "old.ball-scan.json" };
    win.state.audio_source_file = "old.audio-scan.json";
    win.state.label_source_file = "old.near-player-hit-labels.json";

    win.setLoadedSourceFile("pose", "20260514_120000_analysis.pose-scan.json.gz");
    win.setLoadedSourceFile("ball", "20260514_120001_analysis.ball-scan.json");
    win.setLoadedSourceFile("audio", "20260514_120002_analysis.audio-scan.json");
    win.setLoadedSourceFile("label", "20260514_120003_analysis.near-player-hit-labels.json");
    win.renderAll();

    assert.match(win.document.getElementById("pose-loaded-status").textContent, /20260514_120000_analysis\.pose-scan\.json\.gz/);
    assert.match(win.document.getElementById("ball-loaded-status").textContent, /20260514_120001_analysis\.ball-scan\.json/);
    assert.match(win.document.getElementById("audio-loaded-status").textContent, /20260514_120002_analysis\.audio-scan\.json/);
    assert.match(win.document.getElementById("label-loaded-status").textContent, /20260514_120003_analysis\.near-player-hit-labels\.json/);
    assert.equal(win.state.poseData.source_file, "20260514_120000_analysis.pose-scan.json.gz");
    assert.equal(win.state.ballScan.source_file, "20260514_120001_analysis.ball-scan.json");
    assert.equal(win.state.audio_source_file, "20260514_120002_analysis.audio-scan.json");
    assert.equal(win.state.label_source_file, "20260514_120003_analysis.near-player-hit-labels.json");
  });
});

// ---- Near-player hit export ---------------------------------------------

suite("near-player hit export", () => {
  it("uses the compact save-as schema without DB row fields", () => {
    const win = makeDom();
    win.state.analysisId = "analysis-1";
    win.state.videoId = "video-1";
    win.state.analysis = { filename: "clip.mov" };
    win.state.strikeLabels = {
      "near_player_hit|3.000": {
        id: "db-id",
        analysis_id: "analysis-1",
        source: "near_player_hit",
        time_s: 3,
        is_strike: true,
        algorithm_validated: null,
        comment: "nice",
        created_at: 1,
        updated_at: 2,
      },
      "near_player_hit|7.000": {
        source: "near_player_hit",
        time_s: 7,
        is_strike: false,
        comment: "ignored",
      },
    };
    const payload = win.nearPlayerHitExportPayload();
    assert.equal(payload.labels.length, 1);
    assert.deepStrictEqual(Object.keys(payload.labels[0]), ["time_s", "source", "is_strike", "comment"]);
    assert.deepStrictEqual(JSON.parse(JSON.stringify(payload.labels[0])), {
      time_s: 3,
      source: "near_player_hit",
      is_strike: true,
      comment: "nice",
    });
  });
});

// ---- TrackNet target selection ------------------------------------------

suite("TrackNet target selection", () => {
  it("uses audio candidates as target times within the selected range", () => {
    const win = makeDom();
    win.state.impacts = [{ time_s: 2 }, { time_s: 5.1234 }, { time_s: 9 }];
    assert.deepStrictEqual(JSON.parse(JSON.stringify(win.ballScanTargetTimes("audio_candidates", 3, 8))), [5.123]);
  });

  it("uses marked hits as target times within the selected range", () => {
    const win = makeDom();
    win.state.strikeLabels = {
      "near_player_hit|2.000": { source: "near_player_hit", time_s: 2, is_strike: true },
      "near_player_hit|5.000": { source: "near_player_hit", time_s: 5, is_strike: true },
      "near_player_hit|7.000": { source: "near_player_hit", time_s: 7, is_strike: false },
      "manual|6.000": { source: "manual", time_s: 6, is_strike: true },
    };
    assert.deepStrictEqual(JSON.parse(JSON.stringify(win.ballScanTargetTimes("marked_hits", 3, 8))), [5]);
  });
});

// ---- exit ---------------------------------------------------------------

console.log(`\n${testsRun} tests, ${testsFailed} failed`);
process.exit(testsFailed > 0 ? 1 : 0);
