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
  it("hides the card when no impacts are loaded", () => {
    const win = makeDom();
    win.state.impacts = [];
    win.renderStrikeDiagnostic();
    const card = win.document.getElementById("manual-labels-card");
    assert.ok(card.hidden, "card should be hidden when impacts empty");
  });

  it("shows the card and renders the nearest candidate", () => {
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
    const card = win.document.getElementById("manual-labels-card");
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

// ---- exit ---------------------------------------------------------------

console.log(`\n${testsRun} tests, ${testsFailed} failed`);
process.exit(testsFailed > 0 ? 1 : 0);
