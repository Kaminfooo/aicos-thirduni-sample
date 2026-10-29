# AICOS platform and ThirdUni readiness — 29 October 2026

**Verdict: the local prototype has working integration, but the full platform and submission are not ready to call complete.** Passing software checks do not certify detection reliability. The new real-time surveillance acceptance misses below are release concerns.

## Surveillance detector wiring

The selected camera uses the same detector settings and inference worker as Detection lab. All twelve available categories are forwarded: person down, fire/smoke, crowd, zone entry, zone dwell, traffic, unattended object, violence cues, altercation movement, running group, contact/flight and image motion. Selecting a preview does not start inference. Use Analyse selected feed; other wall tiles remain previews. Detection remains one feed at a time.

The wall now displays the selected detector names, a Configure detectors shortcut, and actual analysis frame/alert counts. This addresses confusion between connected preview and active detection. The browser controller has an additional regression check that surveillance forwards all twelve selected IDs and the selected camera metadata, including valid zero coordinates. No model weights or thresholds were changed.

A real stream-ending usability issue was fixed: ending a finite stream now finishes the run and loads retained evidence instead of dropping the review state into a generic playback error. The browser acceptance below exercised this path.

## Current verification

- Python: **207 passed**, zero failures/errors/skips: original application plus workspace 71, focused 104, starter 22, completion 10. The workspace's 12 checks are already included in 71; do not count them twice.
- JavaScript: **92 passed**, zero failures: dashboard 24, coordinate parsing 41, map/location 11, camera wall 16. DOM/media doubles are not physical camera acceptance.
- Fresh local HTTP integration loaded the four reference model weights. All 12 adapters returned results on bounded blank frames without detector errors. Fresh person-down reference alerted after 27 samples, walking stayed quiet for 17, fire alerted after 15, smoke after 6, configured zone entry after 9. Every retained alert passed evidence/media retrieval, review and separate approval checks.
- The known violent clip remained a **miss** after 15 samples. The script records it as a known recognition limitation, not a successful violence detection. The other case-level integration passes must not conceal this outcome.
- Weight hashes matched all four pinned reference values. No new models, changed thresholds or public service modifications were used in this audit.

### Actual surveillance browser acceptance

A licensed controlled-fall recording was opened as an explicitly labelled HTTP test source, using a synthetic public camera position rather than the owner's home location.

| Selection | Frames | Alerts | Person-down result | Runtime errors |
|---|---:|---:|---|---:|
| All 12 categories | 39 | 2: configured zone entry and image motion | **Miss** | 0 |
| Core: person down, fire/smoke, crowd | 46 | 0 | **Miss** | 0 |

The all-category run displayed all twelve result rows, correctly mapped the two retained alerts and verified evidence. Its person-down decisions contained six intermittent confirming samples and 33 unsupported-pose samples; this did not satisfy the existing persistence gate. This is an observation, not proof of the underlying cause. The core comparison also missed. It does not support blaming only the number of enabled detectors. Sampling/pose variability, frame transport and real-time timing require further diagnosis. Preserve the passing file/reference acceptance and these failed real-time cases together.

The owner also reported a physical-camera miss. That attempt's frames and detector decisions have not been inspected, so its exact cause remains unverified. No new personal camera footage or location permission was captured during this audit. Do not ask the owner or reviewers to perform an actual fall. The safest diagnostic material is a permitted recording or ordinary activity/controlled posture with full-body visibility.

Evidence: `.runtime/platform-readiness-20261029/READINESS-RESULT.json`, per-suite logs, `fresh-inference/report.json`, `surveillance-all12.json` and `surveillance-core.json`. The fresh reference cases and browser recordings are integration/development examples, not a representative independent accuracy benchmark.

## ThirdUni rules review

This assessment uses the owner-supplied announcement recorded in `THIRDUNI-ANNOUNCEMENT-REVIEW-20261025.md` and the supplied form screenshots. It is not organizer approval of source scope.

| Required item | Current state | Readiness |
|---|---|---|
| Project name and summary, at most 20 words | AICOS; prepared summary is 18 words | Draft meets length |
| Description, at most 150 words | Updated descriptions are 103 words for full inference and 112 for the separate replay sample | Draft meets length; choose the one matching the actual submitted product link |
| Accessible working demo, or the stated deployment/video fallback | Render sample loaded without sign-in after a cold start; full current inference/map/surveillance workspace remains local | Partial; replay must not be presented as live full-platform inference |
| Public GitHub code and README | Existing sample URL returned 404 while logged out; acceptance of representative code scope is unresolved | **Not satisfied as verified** |
| Accessible product video, 2–4 minutes | Storyboard/narration/assets exist; no final accessible video verified | **Unfinished** |
| Team names/roles and registration before deadline | Solo owner draft exists; organizer registration and final form save are not verified | **Unverified** |

The checked sample repository URL is `https://github.com/kaminfooo/aicos-thirduni-sample`. A logged-out 404 proves that this link did not provide public code in this check; it does not distinguish private, renamed or nonexistent repositories. No repository visibility was changed. Do not expose the full private platform without the owner's publication decision and license review. The announcement directs source-scope questions to the public group; no message has been sent.

The public replay URL is `https://aicos-thirduni-sample.onrender.com/`. Its current content explicitly excludes fresh model inference and the full private backend. It still has the older sample branding and layout. Local neutral branding does not update that remote service. The supplied rules do not require competition branding inside the product; prior-work, third-party and AI disclosures remain necessary in submission material. Use English or French and preserve accurate October contributions.

The supplied deadline is **31 October 2026, 23:59 Algiers**. Aim for final logged-out link QA and save confirmation on 30 October. Do not rely on a green form URL indicator as proof that code/video/demo are accessible or eligible.

## Ordered remaining work

1. Diagnose and improve the real-time person-down acceptance gap using the preserved cases and permitted input; do not weaken thresholds merely to turn a test green. Retest positive and ordinary-activity cases before promoting any change.
2. Establish the authorized zero-cost public access path for the intended submitted product. Full inference still needs deliberate Linux startup/host/cookie/memory/session checks; deploying just the HTML does not deploy the models.
3. Resolve public source scope and provide an actually accessible eligible repository, while preserving the owner's private-source preference.
4. Record the final 2–4 minute walkthrough of the actual current product, including honest detection limits, and verify access without login.
5. Verify team registration, select matching submission copy, fill the actual links and confirm final save/submission before the deadline.

The submission draft was updated to distinguish fresh local inference from the hosted recorded-output sample. No source was published, no hosting purchased, no external message sent and no ThirdUni form submitted during this review.
