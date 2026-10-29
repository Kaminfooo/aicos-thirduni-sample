# AICOS — AI City Operating System · Government Edition v6.0.2

**Private project root — do not publish this folder.** The reviewed public-repository candidate is the separate `thirduni-public-sample/` folder, packaged only by `tools/prepare_minimal_public_sample.py`. Earlier full-source publication candidates were withdrawn. Keep the platform implementation, runtime and private data private.

**ThirdUni October 2026 prototype:** start with [the competition README](docs/THIRDUNI-COMPETITION-README.md) for the demonstrated workflow, validation limits, October contributions and reproducible test commands. The version below describes the pre-existing foundation; it is not an outdoor detection-accuracy certification.

> **Current release:** v6.0.2 is the hardened launch-candidate foundation for sovereign pilots. It adds
> local RBAC operator accounts, secure session-authenticated dashboards/WebSockets, camera
> fleet health + view-integrity/drift monitoring, ONVIF discovery/probing, durable VMS/SIEM
> integrations, stronger evidence-chain export/verification and cross-camera forensic journey
> candidates on top of v5 signed licensing/certification. Start with
> [`docs/GOVERNMENT-EDITION-v6.md`](docs/GOVERNMENT-EDITION-v6.md).
>
> **Status:** pilot/launch candidate, not a claim of ONVIF/ISO/government certification.
> Formal conformity, independent penetration testing and site-specific model validation remain
> deployment gates.


This kit runs the AICOS event pipeline end-to-end on real hardware:

```
[ IP cameras / RTSP ] --> [ edge worker ] --> [ backend API + DB ] --> [ live dashboard ]
                            (detects)          (stores + streams)       (operators act)
```

It ships with a motion-based demo detector so you can prove the full
pipeline with any RTSP camera or a laptop webcam TODAY, then plug
production detection models (violence / weapon / crash / crowd) into
the same interface.

## Repository layout

```
aicos-starter-kit/
├── backend/
│   ├── main.py              FastAPI event API + WebSocket stream + SQLite
│   ├── requirements.txt        Development minimums
│   ├── requirements.lock.txt   Tested backend pins used by Docker
│   ├── auth_users.py         Local sovereign RBAC sessions
│   ├── camera_registry.py    Fleet health / liveness registry
│   ├── evidence.py           Signed evidence ledger / exports
│   ├── forensics.py          Incident/object/journey forensics
│   ├── integrations.py       Durable VMS/SIEM webhook outbox
│   └── Dockerfile
├── edge/
│   ├── preflight.py         Verify a deployment before running it
│   ├── survey.py            Batch-test cameras, see which detectors each supports
│   ├── onvif_probe.py       ONVIF discovery/capability probe (no conformance claim)
│   ├── camera_integrity.py  View drift/obscuration integrity checks
│   ├── integrity_baseline.py Site-approved camera baseline tool
│   ├── geo.py               Ground calibration: image position -> real coordinates
│   ├── calibrate.py         Grid overlay + calibration checker
│   ├── rtsp_worker.py       RTSP capture -> detector -> POST events
│   ├── outbox_admin.py      Inspect/export/replay quarantined edge events
│   └── requirements.txt
├── config/
│   └── cameras.example.yaml Camera list + worker settings
├── dashboard/
│   ├── login.html           Local secure operator sign-in
│   ├── live-ops.html        Real-time ops dashboard (WebSocket client)
│   ├── fleet.html           Camera liveness/integrity console
│   ├── forensics.html       Incident/object forensic search
│   └── aicos-command-center.jsx  Full simulated command-center demo (React)
├── deploy/
│   ├── aicos-edge.service   Hardened reference systemd unit
│   └── edge.env.example     Edge-node environment template
├── docker-compose.yml
├── .env.example
└── README.md
```

## Quickstart — run it on your own machine (no Docker)

Fastest way to see it working, and it runs fine on a laptop:

```bash
pip install fastapi "uvicorn[standard]" requests
cd backend
export AICOS_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export AICOS_ALLOW_SHARED_OPERATOR_KEYS=false
export AICOS_BOOTSTRAP_ADMIN_USER=aicosadmin
export AICOS_BOOTSTRAP_ADMIN_PASSWORD='replace-with-a-strong-admin-password'
python -m uvicorn main:app --host 0.0.0.0 --port 8080
# After first sign-in, remove AICOS_BOOTSTRAP_ADMIN_PASSWORD from the environment and restart.
```

Then open `http://localhost:8080/dashboard/live-ops.html`.

## Quickstart — backend (any Linux server or PC)

Requirements: Docker + Docker Compose.

```bash
cp .env.example .env          # then EDIT .env; set a strong device key + bootstrap admin password
docker compose up -d --build
curl http://localhost:8080/api/health
```

Open `https://<server>/dashboard/login.html`, sign in with the bootstrapped local admin, create named operator/viewer accounts, then remove the bootstrap password. Shared control-room keys are disabled by default in the v6 production template.

### Send a test event (no cameras needed)

```bash
curl -X POST http://localhost:8080/api/events \
  -H "Content-Type: application/json" \
  -H "X-AICOS-KEY: <your AICOS_API_KEY>" \
  -d '{"source_id":"TEST-01","agent":"security","type":"Robbery alarm","severity":4,"confidence":0.91,"district":"Central"}'
```

It appears on the dashboard instantly, with APPROVE / DISMISS buttons
(severity 4–5 events are the human-in-the-loop queue).

## Which version am I running?

```bash
cat /root/aicos-starter-kit/VERSION
```

Check this after every upload — transferring an older copy by mistake is
the most common cause of "the file you told me about does not exist".

## Camera Survey — which cameras can do what

`edge/survey.py` batch-tests a list of camera sources and reports, per
camera, whether it works and WHICH DETECTORS it can support. Point it at
a client's camera list during a site survey, or at public test cameras.

```bash
cat > cams.txt << 'EOF'
Market Street,/root/videos/street2.mp4
Traffic cam,https://example.gov/cameras/cam42.jpg
Front gate,rtsp://user:pass@192.168.1.50:554/Streaming/Channels/101
EOF

python survey.py cams.txt --out ../config/cameras-survey.yaml
```

The recommendations come from measurements, not guesses:

- **person size** — pose keypoints get unreliable below ~0.18 of frame
  height, so `person_down` is only offered above that
- **continuity** — snapshot cameras deliver an image every few seconds,
  so `fight` and `panic` (which need frame-to-frame motion) are never
  offered for them
- **population** — a camera that never sees 8 people gets a lower
  suggested `crowd.warn`

`--out` writes a ready cameras.yaml with per-camera detector settings
already filled in. Set district and coordinates before using it.

## Preflight — run this before anything else

`edge/preflight.py` verifies a deployment in seconds instead of after a
long download or a twenty-minute test run. It checks the virtualenv,
every dependency group, the config file (including zone coordinates and
duplicate camera ids), that every camera source is actually reachable,
the backend AND the API key, model weights, measured CPU capacity with a
`process_every_n_frames` recommendation, and disk space.

```bash
cd edge && source .venv/bin/activate
python preflight.py --config ../config/cameras.yaml \
                    --backend http://127.0.0.1:8080 \
                    --key "$(grep '^AICOS_API_KEY=' ../.env | cut -d= -f2)"
```

Exit code is 0 when nothing failed, so it can gate a deploy script. Run
it after every config change — it catches the mistakes that otherwise
only surface as silence in the dashboard.

## Edge delivery recovery / dead-letter queue

Permanent backend rejections (for example a rotated key, license policy or schema
mismatch) are **not deleted**. They are moved to a local dead-letter queue so an
operator can repair the configuration and replay them. Under storage pressure,
AICOS removes bulky clip/snapshot media before dropping low-severity queue rows;
severity 4/5 event metadata is protected from automatic eviction.

```bash
cd edge
python outbox_admin.py --db /var/lib/aicos/aicos-outbox.db status
python outbox_admin.py --db /var/lib/aicos/aicos-outbox.db list-dead
python outbox_admin.py --db /var/lib/aicos/aicos-outbox.db export-dead dead-letter-export.json
# after fixing the root cause:
python outbox_admin.py --db /var/lib/aicos/aicos-outbox.db requeue-all
```

The edge heartbeat includes delivery spool statistics so a control-room fleet view
can surface queued/quarantined incidents.

## Quickstart — edge worker (Jetson, mini-PC, or laptop)

```bash
# system prerequisites (headless servers need these for OpenCV)
sudo apt -y install python3-venv libgl1 libglib2.0-0t64

cd edge
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp ../config/cameras.example.yaml ../config/cameras.yaml
# EDIT cameras.yaml: put your real RTSP URLs (or "0" for a webcam test)

export BACKEND_URL=http://<server-ip>:8080
export AICOS_API_KEY=<same key as the backend .env>
python rtsp_worker.py --config ../config/cameras.yaml
```

Wave at the camera — a "Unusual activity (motion)" event lands on the
dashboard. The pipeline is proven; production models slot into
`rtsp_worker.py` via the `Detector` interface (example included).

## Exact Incident Location (ground calibration)

Start with the helper — it removes the guesswork:

```bash
python calibrate.py "http://192.168.1.12:8080/video"   # writes calibration-grid.png
# ... pick landmarks, edit cameras.yaml, then verify:
python calibrate.py --check ../config/cameras-phone.yaml
```

The grid image labels every line with its fraction of the frame, so you
read landmark positions straight off it. `--check` reports how well each
camera's calibration fits and shows where sample image points land.

**Indoors, or anywhere a tape measure beats satellite view.** Pick one
floor point as the origin, measure the others from it in metres, and
anchor the set at the building's coordinates:

```bash
python calibrate.py --anchor 34.87133,3.31422 \
  --pair "0.20,0.88 = 0,0" \
  --pair "0.80,0.88 = 4,0" \
  --pair "0.70,0.42 = 4,3" \
  --pair "0.30,0.42 = 0,3"
```

Left of `=` is the image position from the grid; right of `=` is metres
east,north of the anchor. It prints a ready calibration block and checks
the fit. Relative accuracy is as good as your tape — better than GPS
indoors, where a blue dot drifts several metres.


By default an event is reported at its CAMERA's coordinates — accurate to
the size of the field of view, 20-100 m. Add a ground calibration and it
is reported where the person actually is, to a few metres.

Add 4+ landmarks per camera in `cameras.yaml`:

```yaml
    calibration:
      - image: [0.15, 0.90]      # fraction of frame width/height
        world: [34.87120, 3.31400]   # [lat, lon] from satellite view
      - image: [0.85, 0.90]
        world: [34.87120, 3.31444]
      - image: [0.70, 0.35]
        world: [34.87147, 3.31438]
      - image: [0.30, 0.35]
        world: [34.87147, 3.31406]
```

How to pick them: freeze a frame, choose things on the GROUND that are
easy to find again — a building corner, a lamppost base, the end of a road
marking. Spread them across the view; four points in a line cannot define
the transform and are rejected. Then find the same spots in satellite view
and read their coordinates.

On startup the worker logs how well the calibration fits
(`worst fit 0.8 m`). Over 10 m means a landmark was mis-identified or is
not on the ground — a rooftop corner will not work, because the maths
assumes one flat surface.

Every event carries `extra.located`: `calibrated` or `camera`, so you can
always tell which cameras have been surveyed. A broken calibration is
logged and ignored — the camera keeps running on camera coordinates.

## Incident Map — free, no API key

`dashboard/incident-map.html` plots every incident on OpenStreetMap using
Leaflet. No API key, no billing account, works immediately:

```
http://<server-ip>:8080/dashboard/incident-map.html
```

Each marker is placed at the coordinates of the camera that saw the
event, sized and coloured by severity. Click one for the details, the
distance to the nearest response station, a link to open the spot in a
full map, and the frame that triggered it. Severity 4-5 pans the map
automatically as they arrive.

Set `CONFIG.CITY_CENTER` and `CONFIG.STATIONS` at the top of the file,
and make sure each camera in `cameras.yaml` has real `lat`/`lon` —
events from a camera with 0,0 are skipped, because an incident with no
location is worse than none.

## Optional: Traffic & Dispatch Map (Google Maps)

`dashboard/dispatch-map.html` adds live traffic + emergency routing:
real-time traffic layer, incident markers from the backend, and one-tap
**fastest route in current traffic** from the nearest police / fire /
ambulance station to the incident (SEV-5 events route automatically).

Activate it:

1. Google Cloud Console → create a project → enable **Maps JavaScript
   API** and **Directions API** → create an API key (restrict it to
   your dashboard domain/IP).
2. Open `dashboard/dispatch-map.html` and paste the key into
   `CONFIG.GOOGLE_MAPS_API_KEY`.
3. Edit `CONFIG.CITY_CENTER` and `CONFIG.STATIONS` with your real
   coordinates (police / fire / ambulance).
4. Rebuild: `docker compose up -d --build`, then open
   `http://<server-ip>:8080/dashboard/dispatch-map.html`

Google's free monthly usage tier comfortably covers a pilot ops room.
Note: Google Maps requires Internet access from the ops workstation —
for fully air-gapped deployments, a self-hosted OpenStreetMap + OSRM
stack can replace it (no live traffic; camera-derived congestion from
the Traffic agent fills that gap).

## Real Behavior Detection (YOLO pack)

`edge/detectors.py` upgrades the edge worker from the motion demo to
real abnormal-behavior detection — one shared YOLO-pose inference per
frame, per-camera tracking, wall-clock thresholds:

| Behavior | Event | Sev |
|---|---|---|
| Person collapsed / immobile | Person down / possible medical emergency | 4 |
| Sustained agitated close contact | Possible fight / altercation | 3 |
| 8+ / 15+ people in view | Crowd build-up / Large crowd forming | 2-3 |
| 3+ people running at once | Panic movement detected | 4 |
| Person inside a restricted zone | Intrusion — <zone> (4 at night hours) | 3-4 |
| Person too long in a zone | Loitering — <zone> | 2 |
| Bag/case left with nobody near | Unattended object left behind | 3 |
| Contact, then one flees at speed | Possible snatch theft | 4 |
| Vehicle motionless on the road | Vehicle stopped on the road | 3 |
| Vehicle against configured flow | Vehicle travelling against traffic | 4 |
| Many near-motionless vehicles | Traffic congestion building | 2 |
| Trained model sees a fight | Possible violence detected (model) | 3 |

The last two are **opt-in** (`abandoned_object` and `snatch`) — enable
them per camera after checking they behave on that specific view.

Enable it (on the edge machine):

```bash
sudo apt -y install libgl1 libglib2.0-0t64   # required, or cv2 fails on import
cd edge && source .venv/bin/activate
# CPU-only servers: the CPU torch build is far smaller than the CUDA default
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-yolo.txt
# Jetson / GPU: install NVIDIA's torch wheel for your JetPack INSTEAD of the CPU line
```

Any camera can override the global detector settings with its own
`detectors:` block — a busy square and a quiet alley need different
thresholds. Only the keys you name are overridden; the rest are
inherited:

```yaml
  - id: "CAM-002"
    name: "Market Street"
    rtsp_url: "/root/videos/street2.mp4"
    detectors:
      yolo:
        crowd: { warn: 6, critical: 10 }
        person_down: { enabled: true }
```

Then in `config/cameras.yaml` set `settings.detectors.yolo.enabled: true`
and (optionally) draw `zones:` per camera — coordinates are fractions of
the frame (see cameras.example.yaml). First run downloads the model
weights (~7 MB) automatically. Tune thresholds per site during week 1
of the pilot; that tuning IS the "Operate & tune" phase.

## Trained Fall Model

`edge/falldet.py` runs a model trained ON fallen people rather than
inferring a fall from box geometry. It removes the blind spot the
geometric detector has — a body lying along the camera's line of sight
looks narrow and tall, exactly like standing.

```bash
pip install huggingface_hub
```
```yaml
      fall_model:
        enabled: true
        repo: "melihuzunoglu/human-fall-detection"
        seconds: 3
```

Weights download once on first run. Run it ALONGSIDE `person_down`, not
instead of it: the two fail differently, so agreement between them is a
strong signal and either alone still catches what the other misses.

See **`docs/FALL-MODEL-GUIDE.md`** for the model comparison, published
accuracy figures, and how to fine-tune on your own footage.

## Trained Violence Model

Two files, for two different jobs. Free pretrained weights exist — you
do not have to train anything to start.

**Measure first — `edge/violence_probe.py`.** Runs a pretrained
crime/violence classifier over your own footage and reports how often it
fires. On ordinary street footage every "Fighting" prediction is a false
positive, so that count IS your false-alarm rate. An afternoon's work,
zero cost, before you spend a GPU-hour.

```bash
pip install transformers torch torchvision av
python violence_probe.py /root/videos/street2.mp4
```

**Then run it live — `edge/violence.py`.** A trained model alongside the
motion heuristics; they fail differently, so two independent signals on
one incident beat either alone.

```yaml
      violence:
        enabled: true
        weights: "/path/to/fight_yolov8n.pt"
        violence_class: 1
        persistence: 3       # consecutive positives before alerting
```

Two safeguards are deliberate: a single positive frame never alerts
(that is model noise, not violence), and output is severity 3 "Possible
violence detected" — a recommendation an operator confirms, never an
autonomous trigger.

**`docs/VIOLENCE-MODEL-GUIDE.md`** covers which free models to use and
which to avoid, free GPU (Kaggle: ~30 h/week, no credit card), free
datasets, the fine-tuning recipe, and how to validate honestly.

Honesty by design: "fight" and "panic" are motion heuristics meant to
pull an operator's eyes to the feed fast — they confirm, the AI flags.
That is the human-in-the-loop model governments are told about in the
deck.

How fight detection discriminates (v0.7, after real street-footage
testing showed the first version firing on ordinary pedestrians): a pair
must stay engaged for several seconds AND both be moving more than
detector jitter AND have a low path ratio — moving without travelling
anywhere — AND show an opening/closing gap between them. Walking pairs
fail the path-ratio test; standing conversations fail the gap test.
Tunable per camera under `fight:` (`proximity`, `min_speed`,
`max_path_ratio`, `min_gap_variation`, `seconds`).

## Live Camera Sources

The worker accepts any of these as `rtsp_url` in `cameras.yaml`:

| Source | Value to use |
|---|---|
| IP camera (RTSP) | `rtsp://user:pass@ip:554/...` |
| Phone as a camera | `http://phone-lan-ip:8080/video` (IP Webcam app) |
| Laptop/USB webcam | `"0"` |
| Video file (loops) | `/root/videos/clip.mp4` |
| Remote video clip | `https://.../clip.mp4` (paced to real time) |
| Traffic-cam snapshot (JPEG) | `https://.../camera123.jpg` |
| MJPEG stream | `http://host/mjpg/video.mjpg` |
| Public live stream | `https://www.youtube.com/watch?v=...` |

**Snapshot cameras.** Many municipal and public traffic cameras publish a
single JPEG that refreshes instead of a continuous stream. The worker
detects a `.jpg` / `.jpeg` URL automatically (or set `snapshot: true` on
the camera) and polls it on `settings.snapshot_interval_seconds`
(default 2s). At multi-second intervals, count- and persistence-based
behaviors work — crowd, person down, intrusion, loitering — while the
motion heuristics that need frame continuity (fight, panic) will not
fire. Point those at a real stream instead.

**YouTube live cameras.** These need yt-dlp on the edge machine:

```bash
pip install yt-dlp
```

The worker resolves the stream URL automatically, and re-resolves on every
reconnect (those URLs expire after a few hours). Caveat: YouTube blocks
most datacenter IPs with a bot check, so this usually works from a home
or office connection but NOT from a cloud VPS. Use a snapshot camera,
an RTSP stream, or a local webcam on cloud servers.

Check any source before wiring detectors to it:

```bash
python probe.py "<file path, rtsp URL, snapshot URL, or live stream URL>"

# snapshot cameras: control how many images and how far apart
python probe.py "https://.../cam.jpg" --polls 12 --interval 5
```

The probe reports how many people the model sees and how large they are,
then prints a VERDICT with the right `imgsz` / `conf` for that camera.

Note: for client-facing demos and recordings, use cameras you control or
licensed footage — third-party live streams are fine for your own
technical testing, but they are other people's feeds.

## Event Snapshots and Lead-up Clips

An alert fires only after a behavior has persisted — a fall is not a fall
until the person stays down. By then the moment has passed, so a frame
grabbed at alert time shows the aftermath, not the incident.

The worker therefore keeps the last few seconds of every camera in memory
and discards them continuously. When an event fires it claims that
history and sends two things:

- **a snapshot from when the incident actually began**, not when the
  alert fired (the detector reports how far back that was)
- **a short animated GIF of the lead-up**, so an operator sees the person
  walk in and fall rather than a body already on the floor

GIF because it plays in any browser with no codec or plugin — evidence
has to open on whatever machine the operator is using.

Tap **CLIP** on the dashboard, or click a map marker.

```yaml
  event_clips: true
  clip_seconds: 6      # lead-up kept in memory
  clip_fps: 3
  clip_width: 480
```

Set `event_clips: false` to save bandwidth and disk; snapshots continue.
Clips are pruned on the same retention schedule as snapshots.

## Event Snapshots (details)

Every alert carries the frame that triggered it. The dashboard shows a
thumbnail on each event row; tap it for the full image. Operators can
therefore verify before acting, and you can measure precision honestly
instead of guessing.

Controlled under `settings` in cameras.yaml:

```yaml
  event_snapshots: true
  event_snapshot_quality: 70
```

Images are stored as files beside the database (never inside it) and
pruned automatically — `SNAPSHOT_RETENTION_DAYS` in `.env`, default 14.

## Night Rules

The same behavior means something different at 03:00. Set night hours and
security events are escalated automatically inside that window
(emergency and traffic events are left alone):

```yaml
      night: { hours: "21:00-06:00", severity_boost: 1 }
```

Times use the local clock of the machine running the worker — check its
timezone before relying on this.

## Escalation (who gets told)

AICOS does **not** call emergency services automatically. A detection is
a recommendation; an operator confirms it, and that confirmation is the
dispatch signal. Configure channels in `.env`:

```bash
AICOS_NOTIFY_ON=approved          # detected | approved | both | off
AICOS_NOTIFY_MIN_SEVERITY=4
AICOS_NOTIFY_WEBHOOK_URL=         # any dispatch system, n8n, Zapier...
AICOS_TELEGRAM_BOT_TOKEN=
AICOS_TELEGRAM_CHAT_ID=
AICOS_SMTP_HOST=
AICOS_SMTP_TO=
```

Unset channels are skipped, failures are logged and never block an event.
Alerts carry the incident id, camera, severity and a map link.

## AI Incident Reports

Every event row on the live dashboard has a **REPORT** button. It
generates a complete incident report: details, narrative, related
events in the same area (±30 min), and the full operator audit trail —
ready to copy into the official record.

- Works fully **offline** by default (template mode).
- Optional AI-polished prose: set `AICOS_LLM_API_KEY` in `.env`
  (Anthropic key). `AICOS_REPORT_LANG=ar` produces Arabic reports.
  AI mode sends event **metadata only — never video** — to the AI
  provider; leave the key unset for air-gapped deployments.
- API: `GET /api/events/{id}/report?format=text` (`X-AICOS-OPERATOR-KEY` header, operator or viewer key).

## Production notes (pilot hardening)

- Set independent long random `AICOS_API_KEY` (edge ingest) and `AICOS_OPERATOR_KEY` (operator actions).
- Optionally set a separate read-only `AICOS_VIEWER_KEY`.
- Put the backend behind a reverse proxy with TLS (Caddy or Nginx). Cross-origin consoles must be explicitly listed in `AICOS_CORS_ORIGINS`; same-origin dashboards need no CORS entry.
- Edge delivery is durable: failed events are retained in `aicos-outbox.db` and retried until acknowledged. The backend deduplicates retries by `event_uuid`.
- Keep cameras on an isolated VLAN; only the edge worker reaches them.
- Video never needs to leave the site — only JSON events travel.
- Full deployment steps, hardware sizing, and acceptance tests are in
  the companion document: **AICOS Architecture & Installation Guide**.

## API summary

| Method | Path                     | Auth        | Purpose                    |
|--------|--------------------------|-------------|----------------------------|
| POST   | /api/events              | X-AICOS-KEY | Ingest a detection event   |
| GET    | /api/events              | operator/viewer | Query recent events     |
| POST   | /api/events/{id}/status  | operator    | approve/dismiss/resolve    |
| GET    | /api/stats               | operator/viewer | Counters                |
| GET    | /api/health              | none        | Liveness                   |
| WS     | /ws/events               | operator/viewer token | Live JSON event stream |
| GET    | /api/events/{id}/report  | operator/viewer | Incident report (json/text) |

— AICOS hardened kit 4.2.0 / backend API 0.5.0
