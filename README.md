# WakeQuest — advanced four-mission alarm

WakeQuest is a Streamlit alarm that can be dismissed only after completing one
of four live computer-vision missions.

## Missions

1. **Math Gesture** — solve a multiple-choice equation and show 1–4 fingers.
2. **Squat Sprint** — complete a random squat goal verified using YOLO pose.
3. **Object Hunt** — show Book, Bottle, or Phone using the team's `best.pt`.
4. **Blink Check** — complete a blink goal using MediaPipe eye crops and the
   team's Keras eye classifier.

## Start on Windows

Double-click `setup_and_run.bat`, or run it from PowerShell:

```powershell
.\setup_and_run.bat
```

It creates a local virtual environment, installs the dependencies, and opens
the app at `http://localhost:8501`. Grant camera and audio permissions.

Manual alternative:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Eye-model calibration

The eye-model creator did not provide the binary label direction or training
normalization. Use the two temporary sidebar controls while testing:

- **Eye score ≥ 0.5 means OPEN**
- **Eye-model preprocessing:** `0 to 1` or `-1 to 1`

Choose the combination that displays OPEN with open eyes and CLOSED with closed
eyes. Once confirmed, those values can be fixed in code and the controls hidden.

## Architecture

Models load lazily only when their mission is selected. Browser camera frames
arrive through `streamlit-webrtc`; deployed code never calls the server's local
camera. Alarm settings and generated targets live in `st.session_state`.

## Deployment

Test locally first. Then push all files except `.venv` to a private GitHub
repository and deploy `app.py` on Streamlit Community Cloud. The first build is
heavy because TensorFlow, MediaPipe, and Ultralytics must all install. If the
free deployment exceeds memory, the eye mission may need TensorFlow Lite or a
larger hosting plan.

