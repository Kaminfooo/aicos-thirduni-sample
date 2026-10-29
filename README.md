# AICOS · Safety Operations

AICOS helps camera operators inspect potential safety incidents, review the associated image/clip, record a human decision and retain a traceable audit. It is a development prototype, not a certified emergency-response system.

## Try the product

Open [AICOS Safety Operations](https://aicos-safety-operations.onrender.com/), then choose **Start guided walkthrough → Analyse supplied recording**. This runs fresh inference on licensed source footage; it is different from the older standalone recorded-output sample. No account, camera installation or private camera URL is needed for this path.

Read [the usage guide](docs/REVIEWER-GUIDE.md) for the exact workflow. The demo is isolated from production accounts and dispatch systems. Data and signing identity reset when the demo service restarts. Render Free can take about a minute to wake; recheck the connection after it loads.

## What works and what remains limited

- The dashboard connects sampled video/camera inputs to selected detectors, evidence, review, separate approval/dismissal, audit and camera-position mapping.
- The surveillance wall previews up to nine browser-compatible sources. Only one selected source runs detection at a time. Nine preview tiles is not measured detection throughput.
- Twelve detector categories are integrated. Person-down and fire/smoke have limited reference-test evidence. Violence and several interaction detectors remain research previews, and indoor smoke misses remain.
- Two recent real-time surveillance runs missed the controlled person-down event, despite a passing sampled reference replay. Physical-camera reliability is not established. These failures are retained in [validation status](docs/VALIDATION-STATUS.md); software test passes are not detection accuracy.
- RTSP/ONVIF cameras need an external browser-compatible streaming gateway. No NVR recorder or multi-camera inference scheduler is provided.
- Map pins identify configured camera positions, not exact person coordinates. Review guidance is rule-based and uses demonstration procedures. No medical cause or criminal intent is established, and no responders are dispatched.

## Run locally — Windows

Use Python 3.12 and run these commands from a new checkout. Create the environment only if it does not already exist:

```powershell
python -m venv .venv-inference
.\.venv-inference\Scripts\python.exe -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv-inference\Scripts\python.exe -m pip install -r deploy/requirements-hosted.txt
.\.venv-inference\Scripts\python.exe -I -B tools/fetch_reference_models.py
.\.venv-inference\Scripts\python.exe -m uvicorn deploy.judge_workspace:create_app --factory --host 127.0.0.1 --port 8084 --workers 1 --no-access-log
```

Open `http://127.0.0.1:8084/`. Stop this foreground server with Ctrl+C. The worker uses `.venv-inference` on Windows; it never falls back to an unrelated system interpreter there.

## Run locally — Linux/macOS

```sh
python3.12 -m venv .venv-inference
.venv-inference/bin/python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
.venv-inference/bin/python -m pip install -r deploy/requirements-hosted.txt
.venv-inference/bin/python -I -B tools/fetch_reference_models.py
.venv-inference/bin/python -m uvicorn deploy.judge_workspace:create_app --factory --host 127.0.0.1 --port 8084 --workers 1 --no-access-log
```

Linux OpenCV may need system `libgl1` and `libglib2.0-0`. The Dockerfile supplies them. Apple users may need platform-specific PyTorch installation; the hosted release targets Linux CPU and the development environment is Windows CPU. A fresh Linux/container deployment must be verified separately from Windows tests.

## Render deployment

The isolated full workspace uses `deploy/Dockerfile.workspace`, one process/worker, `/health`, and `$PORT` on `0.0.0.0`. `deploy/render-workspace.yaml` selects the Free plan and disables automatic redeploys. CPU dependencies and four SHA-256-pinned reference models are installed at build time. Runtime inference does not auto-download weights.

`RENDER_EXTERNAL_HOSTNAME` is supplied by Render and accepted as one exact host. Hosted session cookies use Secure, HttpOnly and SameSite=Strict. The production API is not mounted in the anonymous workspace. There are bounded requests, session limits, frame/time limits and same-origin write checks.

Free hosting has limited memory/CPU, cold starts and ephemeral storage. A passing local run is not proof that all models fit a hosted instance. Treat only actual hosted acceptance as deployment success; do not silently replace inference with recorded output if a deployment fails. Do not enable paid resources without the owner's approval.

## Tests

Install test-only dependencies into the environment:

```powershell
.\.venv-inference\Scripts\python.exe -m pip install -r deploy/requirements-test.txt
.\.venv-inference\Scripts\python.exe -I -B tools/run_starter_review_checks.py --suite application --output .runtime/check-application-01
.\.venv-inference\Scripts\python.exe -I -B tools/run_starter_review_checks.py --suite focused --output .runtime/check-focused-01
.\.venv-inference\Scripts\python.exe -I -B tools/run_starter_review_checks.py --suite starter --output .runtime/check-starter-01
.\.venv-inference\Scripts\python.exe -I -B tools/run_starter_review_checks.py --suite completion --output .runtime/check-completion-01
node tools/check_judge_dashboard.cjs
node tools/check_coordinates.cjs
node tools/check_location_controller.cjs
node tools/check_camera_wall.cjs
```

Use a new output directory on every run. Application checks include workspace checks in a separate process; don't count them twice. Node checks use DOM/media doubles. Neither is a benchmark of recognition accuracy or physical-camera permission behavior.

## Source structure

`backend/`: original authenticated incident/evidence/RBAC services. `edge/`: detection, tracking, calibration and camera workers. `dashboard/`: operations UI and earlier platform pages. `deploy/`: isolated inference service, pinned models/bootstrap and licensed demo assets. `tools/` and `tests/`: setup, evaluation and software checks. `config/`: example/sanitized fixtures only. `models/pretrained/`: provenance manifests; model binaries are fetched separately.

The original authenticated backend remains available for engineering review (`backend/main.py`, `.env.example`, `config/cameras.example.yaml`). It requires explicit operator/device configuration and is separate from the anonymous review workspace. Do not expose production administrator access to reviewers.

## Who built what, earlier work and AI use

Kamal Belkheiri — solo project owner: product direction, camera setup, evaluation recordings, operator testing and project decisions. AICOS predates ThirdUni. Earlier work included camera ingestion, incident storage, media, maps, permissions and audit. October work added/improved replay evaluation, pose/temporal confirmation, operational health, evidence-linked review/approval, the current operations dashboard, map, surveillance previews and integration tests.

The owner reports that Claude assisted the initial starter; ChatGPT/Codex assisted later implementation, debugging, tests, evaluation and documentation. Pretrained models, open-source libraries and licensed research footage are third-party resources, not team-trained models or invented customer evidence. No external language model is required by the current review workflow.

See [third-party notices](THIRD-PARTY-NOTICES.md), the model manifests and media license/credits. This publication does not assign a new blanket license to the owner's code or override third-party licenses. Preserve all notices and review applicable license obligations before redistribution or commercial deployment.
