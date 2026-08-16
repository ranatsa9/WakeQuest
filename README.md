<div align="center">

# 🌙 WakeQuest

### Wake up. Prove it. Earn the silence.

**A mission-based alarm clock powered by real-time computer vision.**

Set an alarm, choose a challenge, and complete it through your webcam to stop
the alarm. WakeQuest turns waking up into a short interactive quest instead of
another button you can press half-asleep.

</div>

---

## ✨ The idea

Traditional alarms are easy to dismiss and ignore. WakeQuest makes dismissal
intentional: when the alarm rings, the user must complete a live camera mission
that demonstrates movement, attention, or visual interaction.

The application combines a polished **12-hour alarm interface**, browser-based
camera access, and four computer-vision missions in one Streamlit experience.
There is also a **Test mission** button, so every challenge can be demonstrated
without waiting for an alarm.

## 🎯 Available missions

| Mission | What the user does | How WakeQuest verifies it |
|:---|:---|:---|
| ✋ **Math Gesture** | Solve a multiple-choice equation, then hold up 1, 2, 3, or 4 fingers | MediaPipe Hands detects 21 landmarks; finger geometry selects an option, confirmed after it is held steadily |
| 🏋️ **Squat Sprint** | Complete the required number of squats | A YOLO pose model tracks body keypoints and leg movement to count repetitions |
| 🔎 **Object Hunt** | Show the requested everyday item | A YOLO object detector checks incoming frames for the requested object |
| 👁️ **Blink Check** | Blink the requested number of times | Eye localization and a Keras eye-state model identify open/closed transitions |

For Math Gesture, **Easy**, **Medium**, and **Hard** generate different equation
types. The correct result is mixed with three plausible wrong answers, so the
hand gesture represents an answer position rather than its numeric value.

## 🕹️ User flow

1. Click the clock to choose an **hour** and **minute**.
2. Select **AM** or **PM**.
3. Choose one of the four missions.
4. Select a difficulty when using Math Gesture.
5. Click **Set alarm** and keep the page open.
6. When it rings, allow camera access and complete the challenge.
7. WakeQuest verifies the result and silences the alarm. 🎉

> Use **Test mission** to start the selected challenge immediately.

## 🧠 How it works

```text
Interactive alarm clock
          │
          ▼
  Alarm time reached
          │
          ▼
 Browser webcam stream
          │
          ├── MediaPipe hand landmarks ──► gesture option 1–4
          ├── YOLO pose keypoints ───────► completed squat
          ├── YOLO object detection ─────► requested object found
          └── Eye-state model ───────────► completed blink
                                              │
                                              ▼
                                      Mission confirmed
                                              │
                                              ▼
                                        Alarm stopped
```

Camera frames come from `streamlit-webrtc`. This means a deployed app processes
the **visitor's browser camera**, not a camera attached to the server. Models
are loaded only when needed to reduce startup work and memory usage.

## 🧰 Technology stack

- **Python + Streamlit** — application logic and interface
- **streamlit-webrtc + PyAV** — real-time browser video
- **OpenCV** — image preparation and drawing
- **MediaPipe** — hand and facial landmark analysis
- **Ultralytics YOLO** — pose estimation and object detection
- **TensorFlow/Keras** — eye-state classification
- **HTML, CSS, JavaScript** — custom clickable alarm-clock component

## 📁 Project structure

```text
wakequest/
├── app.py                       # Main Streamlit application
├── clock_picker/
│   └── index.html               # Interactive 12-hour clock
├── models/
│   ├── best.pt                  # Object-detection weights
│   └── eye_model.keras          # Open/closed-eye classifier
├── yolo11n-pose.pt              # Pose model for squats
├── requirements.txt             # Python dependencies
├── packages.txt                 # Linux cloud packages
├── setup_and_run.bat            # One-click Windows setup
├── .gitignore
└── README.md
```

The additional `app_*` files in the development folder are design backups and
experiments. The application entry point is always **`app.py`**.

## 🚀 Run locally

### Windows — easiest method

1. Download and extract the project ZIP.
2. Open the extracted `wakequest` folder.
3. Double-click **`setup_and_run.bat`**.
4. Wait for the first installation to finish.
5. Open `http://localhost:8501` if it does not open automatically.
6. Grant camera and audio permission in the browser.

### Windows — PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### macOS or Linux

Python **3.11 or 3.12** is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Then visit `http://localhost:8501` and approve camera/audio access.

> The first installation can take several minutes because TensorFlow,
> MediaPipe, and Ultralytics are large computer-vision dependencies.

## ☁️ Deploy on Streamlit Community Cloud

1. Upload the project to a GitHub repository.
2. Keep `app.py`, `clock_picker/`, `models/`, `yolo11n-pose.pt`,
   `requirements.txt`, and `packages.txt` in the repository.
3. Do **not** upload `.venv`, caches, or local logs.
4. Create an app in Streamlit Community Cloud from the repository.
5. Select `app.py` as the main file.
6. Wait for dependencies and models to install.
7. Open the app and grant browser camera/audio permission.

Local execution is usually faster than free cloud hosting because frames do not
travel to a remote server. Performance also depends on the browser, lighting,
internet connection, and available CPU.

## 👁️ Eye model notes

The included classifier expects:

- RGB eye crops
- `224 × 224` input images
- pixel values normalized to `0–1`
- score `≥ 0.5` interpreted as **open**
- score `< 0.5` interpreted as **closed**

A blink is a complete **open → closed → open** transition. This prevents one
closed-eye frame from being counted several times.

## 🛠️ Troubleshooting

### The camera keeps loading

- Enable camera permission for the site.
- Close other programs using the webcam.
- Choose the correct camera under **Select device**.
- Refresh after changing browser permission.
- For the smoothest demo, run locally in Chrome or Edge.

### `cv2` fails to import on Streamlit Cloud

Use `opencv-python-headless`, already listed in `requirements.txt`. Desktop
OpenCV can require graphical Linux libraries unavailable on a headless server.

### A model is not found

Confirm these paths still exist:

```text
models/best.pt
models/eye_model.keras
yolo11n-pose.pt
clock_picker/index.html
```

GitHub paths are case-sensitive when deployed on Linux.

### Detection is slow or inaccurate

- Use bright, even lighting.
- Keep the relevant hand, face, body, or object clearly in frame.
- Avoid a busy background when possible.
- Close CPU-heavy applications.
- Run locally for lower latency.

## 🔐 Privacy

WakeQuest uses the webcam only while a mission is active. It does not include a
feature for saving or uploading recordings. A deployed version processes frames
for live inference, so users should run it only on infrastructure they trust.

## 🎓 Project purpose

WakeQuest is a team computer-vision project demonstrating how several AI
approaches can cooperate inside one practical application:

- landmark-based gesture recognition
- human pose estimation
- object detection
- image classification
- real-time video processing
- interactive application design

The result is more than four separate models: an alarm triggers a mission,
computer vision evaluates the user, and that result controls the app state.

---

<div align="center">

### 🌅 Good morning — your alarm has a quest for you.

**WakeQuest · Computer Vision Alarm System**

</div>
