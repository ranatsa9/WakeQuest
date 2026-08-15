"""WakeQuest: alarm missions powered by computer vision.

Run with: streamlit run app.py
For deployed use, the browser must grant camera and audio permissions.
"""

from __future__ import annotations

import math
import random
import tempfile
import threading
import time
import wave
from datetime import datetime
from pathlib import Path

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer


APP_DIR = Path(__file__).resolve().parent
EYE_MODEL_PATH = APP_DIR / "models" / "eye_model.keras"
OBJECT_MODEL_PATH = APP_DIR / "models" / "best.pt"
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


def draw_text(frame, text, y, color=(255, 255, 255), scale=0.65):
    cv2.putText(frame, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(frame, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 2, cv2.LINE_AA)


class MissionProcessor:
    def __init__(self):
        self.lock = threading.Lock()
        self.complete = False
        self.status = "Starting camera..."

    def snapshot(self):
        with self.lock:
            return self.complete, self.status

    def set_status(self, status, complete=None):
        with self.lock:
            self.status = status
            if complete is not None:
                self.complete = complete


class StableValue:
    def __init__(self, frames=10):
        self.frames = frames
        self.value = None
        self.count = 0

    def update(self, value):
        if value is None:
            self.value, self.count = None, 0
            return None
        if value == self.value:
            self.count += 1
        else:
            self.value, self.count = value, 1
        return value if self.count >= self.frames else None


def make_question(difficulty):
    if difficulty == "Easy":
        symbol = random.choice(("+", "-"))
        a, b = random.randint(2, 20), random.randint(1, 15)
    elif difficulty == "Medium":
        symbol = random.choice(("+", "-", "x"))
        a, b = random.randint(5, 50), random.randint(2, 15)
    else:
        symbol = random.choice(("x", "/"))
        if symbol == "/":
            answer, b = random.randint(3, 15), random.randint(2, 12)
            a = answer * b
        else:
            a, b = random.randint(8, 25), random.randint(5, 18)
    if symbol == "-" and b > a:
        a, b = b, a
    answer = {"+": a + b, "-": a - b, "x": a * b, "/": a // b}[symbol]
    choices = {answer}
    span = max(5, min(25, abs(answer) // 3 + 4))
    while len(choices) < 4:
        candidate = answer + random.choice((-1, 1)) * random.randint(1, span)
        if candidate >= 0:
            choices.add(candidate)
    choices = list(choices)
    random.shuffle(choices)
    return f"{a} {symbol} {b} = ?", choices, choices.index(answer) + 1


class MathProcessor(MissionProcessor):
    def __init__(self, correct_option):
        super().__init__()
        self.correct_option = correct_option
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1, min_detection_confidence=0.65,
            min_tracking_confidence=0.65
        )
        self.stable = StableValue(12)
        self.locked = False
        self.empty_frames = 0

    @staticmethod
    def gesture(landmarks):
        tips, pips = (8, 12, 16, 20), (6, 10, 14, 18)
        palm_height = abs(landmarks[0].y - landmarks[9].y)
        margin = max(0.015, palm_height * 0.12)
        states = tuple(landmarks[t].y < landmarks[p].y - margin
                       for t, p in zip(tips, pips))
        return {
            (True, False, False, False): 1,
            (True, True, False, False): 2,
            (True, True, True, False): 3,
            (True, True, True, True): 4,
        }.get(states)

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        result = self.hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        choice = None
        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            mp.solutions.drawing_utils.draw_landmarks(
                image, hand, mp.solutions.hands.HAND_CONNECTIONS
            )
            choice = self.gesture(hand.landmark)

        if self.locked:
            self.empty_frames = self.empty_frames + 1 if choice is None else 0
            if self.empty_frames >= 6:
                self.locked = False
                self.stable.update(None)
                self.set_status("Try again: show the correct option")
        else:
            accepted = self.stable.update(choice)
            if choice:
                self.set_status(f"Detected option {choice}: hold steady")
            if accepted:
                self.locked = True
                if accepted == self.correct_option:
                    self.set_status(f"Correct! Option {accepted}", True)
                else:
                    self.set_status(f"Option {accepted} is incorrect. Lower your hand.")

        complete, status = self.snapshot()
        draw_text(image, status, 32, (0, 255, 0) if complete else (0, 220, 255))
        return av.VideoFrame.from_ndarray(image, format="bgr24")


def angle(a, b, c):
    ba, bc = np.asarray(a) - np.asarray(b), np.asarray(c) - np.asarray(b)
    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator < 1e-6:
        return 180.0
    return math.degrees(math.acos(np.clip(np.dot(ba, bc) / denominator, -1, 1)))


@st.cache_resource
def load_pose_model():
    from ultralytics import YOLO
    return YOLO("yolo11n-pose.pt")


class SquatProcessor(MissionProcessor):
    def __init__(self, goal):
        super().__init__()
        self.goal = goal
        self.count = 0
        self.phase = "UP"
        self.model = load_pose_model()

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        result = self.model(image, imgsz=256, conf=0.5, max_det=1, verbose=False)[0]
        knee_angle = None
        if result.keypoints is not None and len(result.keypoints.data):
            points = result.keypoints.data[0].cpu().numpy()
            angles = []
            for hip, knee, ankle in ((11, 13, 15), (12, 14, 16)):
                if min(points[hip, 2], points[knee, 2], points[ankle, 2]) >= 0.4:
                    angles.append(angle(points[hip, :2], points[knee, :2], points[ankle, :2]))
            if angles:
                knee_angle = sum(angles) / len(angles)
                if self.phase == "UP" and knee_angle <= 100:
                    self.phase = "DOWN"
                    self.set_status("Depth reached: stand up")
                elif self.phase == "DOWN" and knee_angle >= 160:
                    self.phase = "UP"
                    self.count += 1
                    if self.count >= self.goal:
                        self.set_status("Squat goal complete!", True)
                    else:
                        self.set_status(f"Rep counted: {self.count}/{self.goal}")
                elif self.phase == "UP":
                    self.set_status("Squat until your knee angle is below 100 degrees")
        else:
            self.set_status("Show your full body, including ankles")

        complete, status = self.snapshot()
        draw_text(image, f"Squats: {self.count}/{self.goal}", 32, (0, 255, 0))
        draw_text(image, status, 64, (0, 255, 0) if complete else (0, 220, 255), 0.55)
        if knee_angle is not None:
            draw_text(image, f"Knee angle: {knee_angle:.0f}", 94, (255, 255, 255), 0.55)
        return av.VideoFrame.from_ndarray(image, format="bgr24")


@st.cache_resource
def load_object_model():
    from ultralytics import YOLO
    if not OBJECT_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing object model: {OBJECT_MODEL_PATH}")
    return YOLO(str(OBJECT_MODEL_PATH))


class ObjectProcessor(MissionProcessor):
    def __init__(self, target):
        super().__init__()
        self.target = target
        self.model = load_object_model()
        self.stable = 0

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        result = self.model(image, imgsz=320, conf=0.45, verbose=False)[0]
        detected = []
        if result.boxes is not None:
            for box in result.boxes:
                name = result.names[int(box.cls[0])]
                detected.append(name)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = (0, 255, 0) if name == self.target else (255, 180, 0)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                draw_text(image, name, max(20, y1 - 7), color, 0.5)
        if self.target in detected:
            self.stable += 1
            self.set_status(f"Found {self.target}: hold steady {self.stable}/8")
            if self.stable >= 8:
                self.set_status(f"Object found: {self.target}", True)
        else:
            self.stable = 0
            self.set_status(f"Show this object: {self.target}")
        complete, status = self.snapshot()
        draw_text(image, status, 32, (0, 255, 0) if complete else (0, 220, 255))
        return av.VideoFrame.from_ndarray(image, format="bgr24")


@st.cache_resource
def load_eye_model():
    if not EYE_MODEL_PATH.exists():
        return None
    import keras
    return keras.models.load_model(EYE_MODEL_PATH, compile=False)


class BlinkProcessor(MissionProcessor):
    def __init__(self, goal, score_one_is_open, preprocessing):
        super().__init__()
        self.goal = goal
        self.score_one_is_open = score_one_is_open
        self.preprocessing = preprocessing
        self.model = load_eye_model()
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.6, min_tracking_confidence=0.6
        )
        self.closed_frames = 0
        self.open_frames = 0
        self.was_closed = False
        self.blinks = 0

    def preprocess(self, crop):
        image = cv2.resize(crop, (224, 224)).astype(np.float32)
        if self.preprocessing == "0 to 1":
            image /= 255.0
        else:
            image = image / 127.5 - 1.0
        return image[None, ...]

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        if self.model is None:
            self.set_status("Missing models/eye_model.keras")
            return av.VideoFrame.from_ndarray(image, format="bgr24")
        result = self.mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        label, score = None, None
        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0].landmark
            scores = []
            # Classify each eye separately because eye-state datasets normally
            # contain single-eye crops rather than the whole face.
            for indices in ((33, 133, 159, 145), (362, 263, 386, 374)):
                xs = [landmarks[i].x * image.shape[1] for i in indices]
                ys = [landmarks[i].y * image.shape[0] for i in indices]
                width, height = max(xs) - min(xs), max(ys) - min(ys)
                pad_x, pad_y = width * 0.35, max(height * 1.8, width * 0.22)
                x1 = max(0, int(min(xs) - pad_x))
                x2 = min(image.shape[1], int(max(xs) + pad_x))
                y1 = max(0, int(min(ys) - pad_y))
                y2 = min(image.shape[0], int(max(ys) + pad_y))
                crop = image[y1:y2, x1:x2]
                if crop.size:
                    rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    scores.append(float(self.model.predict(
                        self.preprocess(rgb_crop), verbose=0
                    )[0][0]))
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if scores:
                score = sum(scores) / len(scores)
                is_open = score >= 0.5 if self.score_one_is_open else score < 0.5
                label = "OPEN" if is_open else "CLOSED"
                if is_open:
                    self.open_frames += 1
                    self.closed_frames = 0
                    if self.was_closed and self.open_frames >= 3:
                        self.blinks += 1
                        self.was_closed = False
                        if self.blinks >= self.goal:
                            self.set_status("Blink goal complete!", True)
                else:
                    self.closed_frames += 1
                    self.open_frames = 0
                    if self.closed_frames >= 3:
                        self.was_closed = True
                if not self.complete:
                    self.set_status(f"Blinks: {self.blinks}/{self.goal} - eyes {label}")
        else:
            self.set_status("Face not found: look toward the camera")
        complete, status = self.snapshot()
        draw_text(image, status, 32, (0, 255, 0) if complete else (0, 220, 255))
        if score is not None:
            draw_text(image, f"Model score: {score:.3f}", 62, (255, 255, 255), 0.5)
        return av.VideoFrame.from_ndarray(image, format="bgr24")


@st.cache_resource
def alarm_sound():
    path = Path(tempfile.gettempdir()) / "wakequest_alarm.wav"
    if not path.exists():
        rate, seconds = 22050, 2
        samples = []
        for i in range(rate * seconds):
            t = i / rate
            active = int(t * 4) % 2 == 0
            value = int(16000 * math.sin(2 * math.pi * 880 * t)) if active else 0
            samples.append(value)
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes(np.asarray(samples, dtype=np.int16).tobytes())
    return path


def initialize_state():
    defaults = {
        "alarm_active": False,
        "mission_complete": False,
        "mission_key": 0,
        "question": None,
        "correct_option": None,
        "squat_goal": None,
        "object_target": None,
        "blink_goal": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def begin_mission(mission, difficulty):
    st.session_state.alarm_active = True
    st.session_state.mission_complete = False
    st.session_state.mission_key += 1
    st.session_state.active_mission = mission
    if mission == "Math Gesture":
        st.session_state.question = make_question(difficulty)
    elif mission == "Squats":
        st.session_state.squat_goal = random.randint(3, 7)
    elif mission == "Object Hunt":
        object_model = load_object_model()
        st.session_state.object_target = random.choice(list(object_model.names.values()))
    else:
        st.session_state.blink_goal = random.randint(3, 6)


st.set_page_config(page_title="WakeQuest · Mission Alarm", page_icon="⏰", layout="wide",
                   initial_sidebar_state="expanded")
initialize_state()
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --cyan:#52e5ff; --violet:#9b7cff; --lime:#8fffa8; --ink:#070a13; }
.stApp {
  background:
    radial-gradient(circle at 12% 5%, rgba(82,229,255,.13), transparent 28rem),
    radial-gradient(circle at 88% 2%, rgba(155,124,255,.15), transparent 30rem),
    linear-gradient(145deg, #070a13 0%, #0b1020 46%, #080b14 100%);
  color:#eef4ff; font-family:'DM Sans',sans-serif;
}
[data-testid="stSidebar"] { background:rgba(8,12,24,.88); border-right:1px solid rgba(255,255,255,.08); }
[data-testid="stSidebar"] > div { padding-top:1.25rem; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.025em; }
.block-container { max-width:1180px; padding-top:2rem; }
.wake-hero { position:relative; overflow:hidden; padding:2rem 2.2rem; border:1px solid rgba(255,255,255,.10);
  border-radius:28px; background:linear-gradient(120deg,rgba(255,255,255,.075),rgba(255,255,255,.025));
  box-shadow:0 25px 80px rgba(0,0,0,.32); backdrop-filter:blur(20px); margin-bottom:1.4rem; }
.wake-kicker { color:var(--cyan); font-size:.75rem; font-weight:700; letter-spacing:.18em; text-transform:uppercase; }
.wake-title { font-family:'Space Grotesk',sans-serif; font-size:clamp(2.5rem,6vw,5rem); line-height:.95;
  font-weight:700; margin:.55rem 0 .8rem; letter-spacing:-.065em; }
.wake-title span { background:linear-gradient(90deg,var(--cyan),#b9a6ff); -webkit-background-clip:text; color:transparent; }
.wake-sub { color:#aebbd1; max-width:650px; font-size:1.02rem; }
.mission-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:.8rem; margin:1.15rem 0 1.5rem; }
.mission-card { padding:1rem; min-height:112px; border-radius:18px; border:1px solid rgba(255,255,255,.09);
  background:rgba(255,255,255,.035); }
.mission-card b { display:block; margin:.65rem 0 .2rem; font-family:'Space Grotesk',sans-serif; }
.mission-card small { color:#91a0b8; }
.mission-icon { font-size:1.35rem; }
.status-chip { display:inline-flex; align-items:center; gap:.45rem; padding:.45rem .8rem; border-radius:999px;
  background:rgba(143,255,168,.10); color:var(--lime); border:1px solid rgba(143,255,168,.25); font-size:.82rem; font-weight:700; }
.status-dot { width:7px; height:7px; border-radius:50%; background:currentColor; box-shadow:0 0 12px currentColor; }
div[data-testid="stButton"] button { border-radius:14px; min-height:46px; font-weight:700; border:1px solid rgba(255,255,255,.14); }
div[data-testid="stButton"] button[kind="primary"] { background:linear-gradient(100deg,#28bfd7,#7560ee); border:0; box-shadow:0 12px 30px rgba(65,116,238,.25); }
div[data-testid="stSelectbox"] > div > div, div[data-testid="stTimeInput"] > div > div { border-radius:13px; }
[data-testid="stAlert"] { border-radius:16px; }
video { border-radius:22px !important; border:1px solid rgba(255,255,255,.12); box-shadow:0 24px 70px rgba(0,0,0,.35); }
@media(max-width:800px){ .mission-grid{grid-template-columns:repeat(2,1fr)} .wake-hero{padding:1.5rem} }
</style>
<section class="wake-hero">
  <div class="wake-kicker">Computer vision alarm system</div>
  <div class="wake-title">Wake up.<br><span>Prove it.</span></div>
  <div class="wake-sub">An alarm that does not negotiate. Complete a live vision mission to silence it and start the day fully awake.</div>
</section>
<div class="mission-grid">
  <div class="mission-card"><span class="mission-icon">✋</span><b>Math Gesture</b><small>Solve, then answer with your hand.</small></div>
  <div class="mission-card"><span class="mission-icon">🏋️</span><b>Squat Sprint</b><small>Pose-verified movement challenge.</small></div>
  <div class="mission-card"><span class="mission-icon">🔎</span><b>Object Hunt</b><small>Find the requested real-world item.</small></div>
  <div class="mission-card"><span class="mission-icon">👁️</span><b>Blink Check</b><small>Wakefulness verified through vision.</small></div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⏰ Mission control")
    st.caption("Configure the challenge that stands between you and silence.")
    alarm_time = st.time_input("Alarm time")
    mission = st.selectbox("Mission", ("Math Gesture", "Squats", "Object Hunt", "Eye Blinks"))
    difficulty = st.selectbox("Math difficulty", ("Easy", "Medium", "Hard"), disabled=mission != "Math Gesture")
    eye_mapping = st.toggle("Eye score ≥ 0.5 means OPEN", value=True,
                            help="Switch this if open eyes are displayed as CLOSED.")
    eye_preprocessing = st.selectbox("Eye-model preprocessing", ("0 to 1", "-1 to 1"),
                                     help="Use the setting used during model training.")
    start_now = st.button("Test mission now", type="primary", use_container_width=True)
    arm_alarm = st.button("Set alarm", use_container_width=True)

if start_now:
    begin_mission(mission, difficulty)
if arm_alarm:
    st.session_state.alarm_active = False
    st.session_state.mission_complete = False
    st.session_state.armed_for = alarm_time.strftime("%H:%M")
    st.session_state.armed_mission = mission
    st.session_state.armed_difficulty = difficulty
    st.success(f"Alarm armed for {st.session_state.armed_for}")

if not st.session_state.alarm_active:
    @st.fragment(run_every=1)
    def alarm_clock():
        now = datetime.now().strftime("%H:%M:%S")
        st.caption(f"Current time: {now}")
        if (st.session_state.get("armed_for")
                and now[:5] == st.session_state.armed_for):
            selected = st.session_state.get("armed_mission", mission)
            selected_difficulty = st.session_state.get("armed_difficulty", difficulty)
            begin_mission(selected, selected_difficulty)
            st.session_state.armed_for = None
            st.rerun()

    alarm_clock()
    st.markdown('<span class="status-chip"><span class="status-dot"></span> SYSTEM READY</span>', unsafe_allow_html=True)
    st.info("Set an alarm or click **Test mission now** to launch the selected challenge.")
    if st.session_state.get("armed_for"):
        st.write(f"Alarm is armed for **{st.session_state.armed_for}**. Refresh near that time if this page is idle.")
    st.stop()

mission = st.session_state.get("active_mission", mission)
st.audio(str(alarm_sound()), autoplay=True, loop=True)
st.warning("Alarm active — complete the selected mission to stop it.")

key = str(st.session_state.mission_key)
if mission == "Math Gesture":
    prompt, choices, correct_option = st.session_state.question
    st.subheader(prompt)
    st.write(" ".join(f"**{i}.** {value}" for i, value in enumerate(choices, 1)))
    processor_factory = lambda: MathProcessor(correct_option)
elif mission == "Squats":
    st.subheader(f"Complete {st.session_state.squat_goal} squats")
    processor_factory = lambda: SquatProcessor(st.session_state.squat_goal)
elif mission == "Object Hunt":
    st.subheader(f"Show a {st.session_state.object_target} to the camera")
    processor_factory = lambda: ObjectProcessor(st.session_state.object_target)
else:
    st.subheader(f"Blink {st.session_state.blink_goal} times")
    processor_factory = lambda: BlinkProcessor(
        st.session_state.blink_goal, eye_mapping, eye_preprocessing
    )

ctx = webrtc_streamer(
    key=f"wakequest-{mission}-{key}",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=processor_factory,
    async_processing=True,
)

@st.fragment(run_every=1)
def mission_status():
    if ctx.video_processor:
        complete, status = ctx.video_processor.snapshot()
        st.write(status)
        if complete:
            st.session_state.mission_complete = True
            st.session_state.alarm_active = False
            st.session_state.mission_key += 1
            st.success("Mission complete — alarm stopped!")
            st.balloons()

mission_status()
