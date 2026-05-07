# Tennis Highlight Extractor

A web app that removes dead time from amateur tennis practice videos. Upload
footage from a fixed tripod camera, the app analyzes motion (no audio
required) and proposes a timeline of in-play vs dead-time segments. You
review the proposal in an interactive editor — toggle individual segments,
nudge boundaries, preview each one — then export a single trimmed MP4 with
only the rallies you kept.

## Requirements

- **Python 3.12+** (developed on 3.14)
- **ffmpeg** on your `PATH` (`brew install ffmpeg` on macOS)
- **Node 18+** *(optional, only for running the frontend DOM tests)*

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the server

```bash
uvicorn app.main:app --port 8000
# then open http://localhost:8000
```

Set `VERBOSE=true` for debug-level logs:

```bash
VERBOSE=true uvicorn app.main:app --port 8000
```

## Run the tests

A single command runs both suites:

```bash
bash scripts/test.sh
```

That runs:

1. **Python tests** (`unittest`, no extra deps) — schema/contract tests over
   the JS, HTML, and CSS, plus pipeline tests for the motion and pose
   analyzers. ~23 tests.
2. **Frontend DOM tests** (`Node` + `jsdom`) — runtime tests that load
   `static/index.html` + `static/app.js` into a simulated browser and
   exercise the algorithm-filter form, the per-detector pill renderer,
   and the help popover. ~14 tests.

The DOM tests are auto-skipped if `jsdom` isn't installed. Enable them
with:

```bash
npm install --no-save jsdom
```

If you only want one suite:

```bash
.venv/bin/python -m unittest discover tests/ -v   # Python only
node tests/test_frontend_render.js                # Frontend DOM only
```

## Project layout

```
app/
  main.py              # FastAPI app
  config.py            # env-driven settings
  database.py          # SQLite schema + helpers
  routers/             # HTTP endpoints (upload, process, segments, export, ...)
  pipeline/
    motion_analysis.py # median-frame + court-ROI detectors
    pose_analysis.py   # YOLO pose detector
    video_editor.py    # ffmpeg extract + concat for export
    orchestrator.py    # background-task coordinator
static/
  index.html           # single-page UI
  app.js               # library, status, editor, popover, all behavior
  style.css
tests/
  test_motion_analysis.py
  test_pose_analysis.py
  test_frontend_schemas.py    # static-analysis tests over JS/HTML/CSS
  test_frontend_render.js     # Node + jsdom runtime tests
scripts/
  test.sh              # run both test suites
data/                  # uploads, exports, sqlite db (gitignored)
```

## Detectors

| Key | What it does | When to use |
|---|---|---|
| `median_frame` | Per-pixel median across the whole video → "empty court" image. Each frame's deviation from that image is the foreground. | Default for fixed tripod shots. No warmup, no learn-in. |
| `median_court_roi` | Same as median_frame, but weights foreground inside/outside a user-marked court polygon (and a near-camera band). | When close-up walking past the camera causes false positives. |
| `pose_skeleton_yolo` | YOLO pose model picks up player skeletons frame-by-frame. | When motion-based detection misses small/distant players. |

Algorithm and per-detector knobs are picked from the Media Pool form on the
landing page. Hover or click the `?` dots beside each control for inline
help.
