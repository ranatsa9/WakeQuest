# 🌙 WakeQuest

### An alarm you cannot snooze—you have to prove you are awake.

WakeQuest is a computer-vision alarm built with Streamlit. Set a time, choose a
challenge, and when the alarm rings, complete the live camera mission to make it
stop. No sleepy button tapping allowed. 😴➡️😎

> **Wake up. Prove it. Start your day.**

## ✨ What makes it different?

WakeQuest combines a 12-hour alarm interface with four interactive missions.
Each mission uses live webcam input and a different computer-vision technique.

| Mission | Your quest | Computer vision |
|---|---|---|
| ✋ **Math Gesture** | Solve a multiple-choice equation and show option 1–4 with your hand | MediaPipe Hands + geometric finger detection |
| 🏋️ **Squat Sprint** | Complete the requested number of squats | YOLO pose estimation + knee-angle tracking |
| 🔎 **Object Hunt** | Find and show the requested everyday object | Team-trained YOLO object detector |
| 👁️ **Blink Check** | Blink the requested number of times | MediaPipe face landmarks + MobileNetV2 eye classifier |

## 🎮 How it works

1. Choose an hour, minute, and **AM/PM**.
2. Select one of the four WakeQuest missions.
3. Press **Set alarm** and keep the browser page open.
4. When the alarm starts, allow camera access.
5. Complete the mission to silence the alarm. 🎉

You can also press **Test mission** to launch a challenge immediately.

## 🧠 Under the hood

```text
Browser camera
      │
      ▼
streamlit-webrtc
      │
      ├── MediaPipe Hands ─────► Math option
      ├── YOLO Pose ───────────► Squat count
      ├── Custom YOLO model ───► Object match
      └── MediaPipe + CNN ─────► Blink count
                                  │
                                  ▼
                           Mission complete
                                  │
                                  ▼
                            Alarm silenced
```

The models are loaded only when their mission needs them. Camera frames are
processed through the browser with `streamlit-webrtc`, so a deployed app uses
the user's camera rather than the server's camera.

## 🚀 Run locally

### Quick start on Windows

Double-click:

```text
setup_and_run.bat
```

Or run these commands in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Then open `http://localhost:8501` and grant camera/audio permission.

## 📁 Project structure

```text
WakeQuest/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── setup_and_run.bat
└── models/
    ├── best.pt
    └── eye_model.keras
```

## 👁️ Eye-model configuration

The eye classifier expects:

- RGB eye crops
- `224 × 224` input images
- pixel values normalized from `0` to `1`
- score `≥ 0.5` = **open eyes**
- score `< 0.5` = **closed eyes**

The developer sidebar keeps calibration switches available in case a different
eye-model export is tested later.

## ☁️ Deploy with Streamlit Community Cloud

1. Push the project files to GitHub. Do not upload `.venv`.
2. Connect the repository to Streamlit Community Cloud.
3. Select `app.py` as the entry file.
4. Wait for the Python dependencies and model files to install.
5. Open the deployed page and grant browser camera/audio permission.

TensorFlow, MediaPipe, and Ultralytics make the first installation relatively
large. If a free deployment runs out of memory, converting the eye model to
TensorFlow Lite is the first recommended optimization.

## 🛠️ Built with

- Python and Streamlit
- OpenCV
- MediaPipe
- Ultralytics YOLO
- TensorFlow / Keras and MobileNetV2
- streamlit-webrtc

## 🎓 Project note

WakeQuest was created as a team computer-vision project. It demonstrates how
classification, pose estimation, object detection, facial landmarks, and
real-time browser video can work together inside one playful application.

---

<div align="center">
  <strong>Good morning. Your alarm has a quest for you. 🌅</strong>
</div>
