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

// ---- renderAnalysisConfigs filters pills by detector --------------------

suite("renderAnalysisConfigs filters by detector and adds tooltips", () => {
  const win = makeDom();

  function pillsFor(analysis, poseData = null) {
    win.state.analysis = analysis;
    win.state.poseData = poseData;
    win.renderAnalysisConfigs();
    return Array.from(
      win.document.querySelectorAll("#analysis-configs .config-pill")
    );
  }

  it("pose analysis: only pose-relevant pills, with imgsz", () => {
    const pose = pillsFor(
      {
        detector: "pose_skeleton_yolo",
        knobs: {
          sample_fps: 2,
          pose_model: "yolo11n-pose.pt",
          pose_conf: 0.25,
          pose_imgsz: 640,
          motion_threshold: 0.02, // noise that should be filtered
          court_weight: 1, // noise
        },
        summary: {
          config: {
            detector: "pose_skeleton_yolo",
            duration_s: 765.9,
            sample_count: 120,
          },
        },
      },
      {
        metadata: {
          model_name: "yolo11n-pose.pt",
          conf_threshold: 0.25,
          image_size: 640,
          sample_fps: 2,
          range_start_s: 0,
          range_end_s: 60,
        },
      },
    );
    const keys = pose.map((p) => p.textContent.split(":")[0]);
    assert.ok(keys.includes("pose_imgsz"), "pose_imgsz should be visible");
    assert.ok(keys.includes("pose_conf"));
    assert.ok(keys.includes("pose_model"));
    assert.ok(!keys.includes("motion_threshold"), `motion_threshold leaked: ${keys}`);
    assert.ok(!keys.includes("court_weight"), `court_weight leaked: ${keys}`);
  });

  it("median_frame analysis: motion knobs visible, pose hidden", () => {
    const pills = pillsFor({
      detector: "median_frame",
      knobs: {
        sample_fps: 2,
        diff_threshold: 25,
        motion_threshold: 0.02,
        court_weight: 1, // not in median_frame
        pose_imgsz: 640, // not in median_frame
      },
      summary: { config: { detector: "median_frame" } },
    });
    const keys = pills.map((p) => p.textContent.split(":")[0]);
    assert.ok(keys.includes("motion_threshold"));
    assert.ok(keys.includes("diff_threshold"));
    assert.ok(!keys.includes("court_weight"));
    assert.ok(!keys.includes("pose_imgsz"));
  });

  it("pills with help text get a title attribute", () => {
    const pills = pillsFor({
      detector: "median_frame",
      knobs: { sample_fps: 2, diff_threshold: 25 },
      summary: { config: { detector: "median_frame" } },
    });
    const sampleFps = pills.find((p) => p.textContent.startsWith("sample_fps"));
    assert.ok(sampleFps, "sample_fps pill missing");
    assert.ok(sampleFps.title.length > 0, "sample_fps pill has no title");
  });

  it("unknown detector falls back to showing everything", () => {
    const pills = pillsFor({
      detector: "future_detector_xyz",
      knobs: { whatever: 1, sample_fps: 2 },
      summary: { config: { detector: "future_detector_xyz" } },
    });
    const keys = pills.map((p) => p.textContent.split(":")[0]);
    assert.ok(keys.includes("whatever"));
    assert.ok(keys.includes("sample_fps"));
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

// ---- exit ---------------------------------------------------------------

console.log(`\n${testsRun} tests, ${testsFailed} failed`);
process.exit(testsFailed > 0 ? 1 : 0);
