import re

with open('static/index.html', 'r') as f:
    content = f.read()

# 1. Remove analysis-config-card
content = re.sub(
    r'<div class="card" id="analysis-config-card">.*?</div>\s*(<div class="card" id="yolo-pose-card">)',
    r'\1',
    content,
    flags=re.DOTALL
)

# 2. Add knobs to yolo-pose-card
yolo_knobs = """              <label>
                <span>sample_fps</span>
                <input id="knob-sample-fps" type="number" value="30" />
              </label>
              <label>
                <span>pose_model</span>
                <input id="knob-pose-model" type="text" value="yolo11n-pose.pt" />
              </label>
              <label>
                <span>pose_conf</span>
                <input id="knob-pose-conf" type="number" min="0" max="1" step="0.05" value="0.25" />
              </label>
              <label>
                <span>pose_imgsz</span>
                <input id="knob-pose-imgsz" type="number" step="32" value="640" />
              </label>"""
content = re.sub(
    r'<label>\s*<span>sample_fps</span>.*?</label>\s*<label>\s*<span>pose_conf</span>.*?</label>',
    yolo_knobs,
    content,
    flags=re.DOTALL
)

# 3. Add audio_sample_rate to audio-processing-card
audio_knobs = """              <label class="audio-knob">
                <span>audio_sample_rate</span>
                <input id="knob-audio-sample-rate" type="number" value="22050" step="1000" />
              </label>
              <label class="audio-knob">
                <span>bandpass_low_hz</span>"""
content = content.replace(
    """<label class="audio-knob">
                <span>bandpass_low_hz</span>""",
    audio_knobs
)

with open('static/index.html', 'w') as f:
    f.write(content)

