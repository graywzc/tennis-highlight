import re

with open('static/index.html', 'r') as f:
    content = f.read()

# 1. Replace the old detector-experiment-card with the new modular left-column cards
old_detector_card = re.search(r'<div class="card" id="detector-experiment-card">.*?</div>\s*</div>\s*<div class="editor-resizer" id="editor-resizer-left"', content, re.DOTALL)

new_left_cards = """
          <div class="card" id="yolo-pose-card">
            <div class="row-between">
              <h2>YOLO Pose Detection</h2>
            </div>
            <div class="muted" style="margin-bottom: 8px">Detects players and keypoints. Needed by TrackNet ROI.</div>
            <div class="knob-grid">
              <label>
                <span>sample_fps</span>
                <input id="knob-sample-fps" type="number" value="30" />
              </label>
              <label>
                <span>pose_conf</span>
                <input id="knob-pose-conf" type="number" min="0" max="1" step="0.05" value="0.25" />
              </label>
            </div>
            <div class="hit-study-actions" style="margin-top: 12px">
              <button id="pose-scan-btn">Scan Range</button>
              <button id="pose-save-btn">Save Data</button>
              <button id="pose-load-btn">Load Data</button>
            </div>
            <div id="pose-scan-summary" class="muted" style="margin-top: 4px;"></div>
          </div>

          <div class="card" id="audio-processing-card">
            <div class="row-between">
              <h2>Audio Processing</h2>
            </div>
            <div class="muted" style="margin-bottom: 8px">Detects sharp transients (racket impacts).</div>
            <div class="knob-grid">
              <label class="audio-knob">
                <span>bandpass_low_hz</span>
                <input id="knob-bandpass-low-hz" type="number" min="20" step="50" />
              </label>
              <label class="audio-knob">
                <span>bandpass_high_hz</span>
                <input id="knob-bandpass-high-hz" type="number" min="100" step="50" />
              </label>
              <label class="audio-knob">
                <span>peak_height_mad_k</span>
                <input id="knob-peak-height-mad-k" type="number" min="0" step="0.5" />
              </label>
              <label class="audio-knob">
                <span>peak_prominence_mult</span>
                <input id="knob-peak-prominence-mult" type="number" min="0" step="0.25" />
              </label>
            </div>
            <div class="hit-study-actions" style="margin-top: 12px">
              <button id="audio-preview-run-btn">Scan Range</button>
              <button id="audio-save-btn">Save Data</button>
              <button id="audio-load-btn">Load Data</button>
            </div>
            <div id="audio-preview-summary" class="muted" style="margin-top: 4px;">Tune audio knobs, then analyze the selected range.</div>
          </div>

          <div class="card" id="tracknet-card">
            <div class="row-between">
              <h2>TrackNet Ball Tracking</h2>
            </div>
            <div class="muted" style="margin-bottom: 8px">Scans ROI around players for tennis balls. (Requires YOLO Pose).</div>
            <div class="ball-roi-controls">
              <label class="ball-roi-check">
                <input id="ball-exclude-player-toggle" type="checkbox" checked />
                Exclude player body
              </label>
              <label>
                ROI mode
                <select id="ball-roi-mode">
                  <option value="near_player" selected>Near player</option>
                  <option value="full_frame">Whole screen</option>
                </select>
              </label>
              <label>
                ROI width
                <input id="ball-roi-width" type="range" min="1" max="6" step="0.25" value="3" />
              </label>
              <label>
                ROI height
                <input id="ball-roi-height" type="range" min="0.5" max="3" step="0.25" value="1" />
              </label>
            </div>
            <details class="tracknet-scan-config" open style="margin-top: 8px;">
              <summary>Advanced TrackNet settings</summary>
              <div class="ball-roi-controls">
                <label>
                  Scan FPS
                  <select id="ball-scan-fps">
                    <option value="2" selected>2 fps</option>
                    <option value="4">4 fps</option>
                    <option value="8">8 fps</option>
                    <option value="15">15 fps</option>
                    <option value="30">30 fps</option>
                    <option value="original">Original fps</option>
                  </select>
                </label>
                <label>
                  Width
                  <select id="tracknet-width">
                    <option value="640" selected>640</option>
                    <option value="1280">1280</option>
                    <option value="1920">1920</option>
                    <option value="3840">3840</option>
                  </select>
                </label>
                <label>
                  Height
                  <select id="tracknet-height">
                    <option value="360" selected>360</option>
                    <option value="720">720</option>
                    <option value="1080">1080</option>
                    <option value="2160">2160</option>
                  </select>
                </label>
              </div>
            </details>
            <div class="ball-roi-controls" style="margin-top: 8px;">
              <label class="muted hit-study-slow-toggle">
                <input id="ball-heatmap-toggle" type="checkbox" />
                Ball heatmap
              </label>
              <label class="muted hit-study-slow-toggle">
                <input id="ball-roi-overlay-toggle" type="checkbox" checked />
                Show ROI
              </label>
              <button id="ball-diagnostic-btn">Refresh heatmap</button>
            </div>
            <div id="ball-scan-progress" class="ball-scan-progress" hidden style="margin-top: 12px;">
              <progress id="ball-scan-progress-bar" value="0" max="100"></progress>
              <span id="ball-scan-progress-text" class="muted">Queued</span>
            </div>
            <div class="hit-study-actions" style="margin-top: 12px">
              <button id="ball-scan-btn">Scan Range</button>
              <button id="ball-scan-stop-btn" disabled>Stop</button>
              <button id="ball-scan-save-btn">Save Data</button>
              <button id="ball-scan-load-btn">Load Data</button>
            </div>
            <div id="tracknet-summary" class="muted" style="margin-top: 4px;"></div>
          </div>

          <div class="card" id="logic-filters-card">
            <div class="row-between">
              <h2>Logic & Filters</h2>
              <button id="reset-knobs-btn">Reset</button>
            </div>
            <div class="muted" style="margin-bottom: 8px">Rules for combining signals into valid segments.</div>
            <div class="knob-grid">
              <label>
                <span>min_wrist_velocity</span>
                <input id="knob-min-wrist-velocity" type="number" min="0" step="0.05" />
              </label>
              <label>
                <span>diff_threshold</span>
                <input id="knob-diff-threshold" type="number" min="0" max="255" step="1" />
              </label>
              <label>
                <span>motion_threshold</span>
                <input id="knob-motion-threshold" type="number" min="0" step="0.001" />
              </label>
              <label>
                <span>merge_gap_s</span>
                <span class="inline-control">
                  <input id="enable-merge-gap" type="checkbox" />
                  <input id="knob-merge-gap" type="number" min="0" step="0.1" />
                </span>
              </label>
              <label>
                <span>min_segment_s</span>
                <span class="inline-control">
                  <input id="enable-min-segment" type="checkbox" />
                  <input id="knob-min-segment" type="number" min="0" step="0.1" />
                </span>
              </label>
              <label>
                <span>segment_padding_s</span>
                <span class="inline-control">
                  <input id="enable-padding" type="checkbox" />
                  <input id="knob-segment-padding" type="number" min="0" step="0.1" />
                </span>
              </label>
            </div>
            <div class="hit-study-actions" style="margin-top: 12px">
              <button id="evaluate-hit-study-btn">Evaluate Pipeline</button>
            </div>
            <div id="logic-summary" class="muted" style="margin-top: 4px;"></div>
          </div>
        </div>

        <div class="editor-resizer" id="editor-resizer-left" """

content = content[:old_detector_card.start()] + new_left_cards + content[old_detector_card.end():]

# 2. Replace strike-diagnostic-card with manual-labels-card
old_strike_card = re.search(r'<div class="card" id="strike-diagnostic-card" hidden>.*?<div class="strike-missed-controls">.*?</div>\s*</div>\s*</div>', content, re.DOTALL)

new_center_card = """<div class="card" id="manual-labels-card" hidden>
            <div class="row-between">
              <h2>Manual Labels & Ground Truth</h2>
              <span class="muted" id="strike-stats">No labels yet</span>
            </div>
            <div id="strike-current" class="strike-current"></div>
            
            <hr class="strike-divider" />
            
            <div class="hit-study-block" id="hit-study-block">
              <strong>Mark Near-Player Hits</strong>
              <div class="muted">Press H to mark a hit. Space toggles play/pause while you are not typing.</div>
              <div class="hit-study-actions" style="margin-top: 8px;">
                <button id="mark-near-hit-btn">Mark hit at <span id="near-hit-time">0:00.0</span></button>
                <button id="save-near-hits-btn">Save Marks</button>
                <button id="load-near-hits-btn">Load Marks</button>
              </div>
              <div class="hit-study-actions" style="margin-top: 8px;">
                <label class="muted hit-study-slow-toggle">
                  <input id="slow-label-mode-toggle" type="checkbox" />
                  Slow hold mode
                </label>
                <label class="muted hit-study-rate">
                  Speed
                  <select id="slow-label-rate">
                    <option value="0.25">0.25x</option>
                    <option value="0.5" selected>0.5x</option>
                    <option value="0.75">0.75x</option>
                  </select>
                </label>
              </div>
              <div id="hit-study-labels" class="muted" style="margin-top: 8px;"></div>
              <div id="hit-study-report" class="hit-study-report"></div>
            </div>

            <hr class="strike-divider" />
            
            <div class="strike-missed-block">
              <strong>Missed a strike?</strong>
              <div class="muted" style="margin: 4px 0 6px">
                Pause the player at the moment the algorithm missed a real strike, then add it here.
              </div>
              <div class="strike-missed-controls">
                <button id="add-missed-strike-btn">+ Add missed strike at <span id="missed-strike-time">0:00.0</span></button>
                <input id="missed-strike-comment" type="text" placeholder="Optional comment" />
              </div>
            </div>
          </div>"""

if old_strike_card:
    content = content[:old_strike_card.start()] + new_center_card + content[old_strike_card.end():]
else:
    print("Could not find strike diagnostic card")

with open('static/index.html', 'w') as f:
    f.write(content)

