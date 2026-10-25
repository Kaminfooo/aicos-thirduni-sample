# AICOS — standalone ThirdUni replay sample

**Author / solo entrant:** Kamal Belkheiri ([kaminfooo](https://github.com/kaminfooo)). ChatGPT/Codex assisted implementation, debugging and documentation.

This is a small runnable demonstration of **recorded alert → evidence → demonstration procedure → human review → incident decision → sample audit**. It is independently implemented for public review. **It is not the complete AICOS source or production backend.**

The private AICOS platform runs the camera/detector and more advanced permissions, evidence and operational features. Those implementations, tuning, model runners, credentials and private footage are deliberately absent here. The two application files are `app.py` and `web/index.html`.

## Honest demonstration scope

- The alert is recorded output from the private evaluation, not new cloud inference. No model download, camera connection, external AI call or video upload is offered.
- The licensed source is a controlled laboratory simulation, composed from two views of the same activity. No medical cause or fainting diagnosis is established.
- Each browser gets a separate anonymous demo session. It is not a verified real operator identity. Nothing is dispatched.
- Review and incident approval are separate actions. Only a demonstration procedure is supplied.
- The sample log links entries using SHA-256; verification checks consistency and media hashes. It is **not digitally signed** and is not the private AICOS evidence system. A server operator could rewrite the sample database.
- Free hosting can sleep, cold-start or restart. Session state and sample logs reset on process restart. This is not a persistent production safety service.

## Run locally

Use Python 3.12 and a project-local environment. Do not replace an installed interpreter. In a new checkout:

```powershell
Set-Location 'C:\path\to\aicos-thirduni-sample'
New-Item -ItemType Directory -Force '.runtime\tmp' | Out-Null
$env:TEMP = Join-Path (Get-Location) '.runtime\tmp'
$env:TMP = $env:TEMP
& 'C:\path\to\Python312\python.exe' -m venv .venv
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app:create_app --factory --host 127.0.0.1 --port 8084 --workers 1
```

Open `http://127.0.0.1:8084/`. No login/password is needed for this intentionally public sample. Stop the foreground server using Ctrl+C. Databases stay inside `.runtime/boot-*`; no private AICOS folder is opened.

For tests, install test-only `httpx==0.28.1` into that same environment, then run `python -B -m unittest -v test_sample`. Tests create disposable data only in this sample's `.runtime` folder. No models or cameras are used.

## Free deployment

`render.yaml` selects one Free Python service, no paid disk/database, no automatic deploys and one worker. Start command: `python -m uvicorn app:create_app --factory --host 0.0.0.0 --port $PORT --workers 1 --no-access-log`. Health endpoint: `/health`.

Cloud build and public-link testing are still pending. Do not attach payment methods or paid resources for this zero-additional-cost plan. Existing Hostinger Business Web Hosting is untouched. No live URL or competition acceptance is claimed before verification.

## Recorded evaluation context

The private candidate alerted in 4/4 controlled fall clips and 0/9 ordinary-activity clips, with an extra same-source alert in composed footage. Only 62.7 seconds of ordinary activity were tested. All came from one research dataset, and some were used for development. These results **do not establish 99% accuracy**, outdoor reliability or performance in every circumstance. The public sample only replays the selected lateral example; it cannot reproduce the private detector benchmark.

October contributions to the full platform include timestamp-aware replay, pose/temporal posture confirmation, camera operational-health reporting, evidence/procedure-linked triage and operator interface improvements. The existing platform predates ThirdUni. The submission video should demonstrate the full private platform and distinguish it from this simplified public sample.

## Media attribution and rights

Footage: [ASSIST-IoT multimodal fall detection dataset](https://github.com/ASSIST-IoT-SRIPAS/multimodal-fall-detection-dataset), **CC BY 4.0**. Authors: Anastasiya Danilenka, Piotr Sowiński, Monika Kobus, Anna Dąbrowska, Kajetan Rachwał, Karolina Bogacka and Krzysztof Baszczyński. EU Horizon 2020 grant 957258; CIOP-PIB facilities/equipment. No endorsement implied.

Original MP4s are unchanged. The JPEG is extracted; the GIF is resized/captioned. Terms: `assets/LICENSE.txt`; source/output hashes and full credit: `assets/record.json`. Model weights and private inference code are not redistributed.

No permissive open-source license is automatically assigned to this newly written sample. See `NOTICE.md`. A public GitHub repository can still be viewed and forked under GitHub's platform terms; keeping the main platform private protects its unpublished source, not the public sample from copying.

## Submission requirement to confirm

The supplied ThirdUni page requires a public repository with a README but does not establish whether a representative sample may replace the full proprietary project source. Confirm that point with ThirdUni before relying on this arrangement. Suggested question: “May I submit a public, runnable representative workflow sample while keeping the full proprietary detector/backend private, with that distinction clearly disclosed and the full platform shown in the video?” No message has been sent on the owner's behalf.
