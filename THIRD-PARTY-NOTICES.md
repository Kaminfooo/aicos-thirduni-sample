# Third-party notices — local review draft, 26 October 2026

This inventory is not a project license or a finding that publication is cleared.
No new source license has been assigned. Owner reports the original starter was
developed using Claude; ChatGPT/Codex assisted subsequent work. This disclosure is
not a warranty that generated code contains no third-party material.

## Installed Python environments

The backend environment has 32 distributions and inference has 37 (overlapping
groups, not 69 unique libraries). Exact versions, metadata license labels,
project URLs and SHA-256 hashes of retained notice files are recorded in:

- `docs/dependencies/backend/installed-dependencies.json`
- `docs/dependencies/inference/installed-dependencies.json`
- Each environment's `license-texts/` subtree retains installed notices verbatim.

54 backend and 66 inference notice artifacts were retained. Every installed
distribution had an embedded license/notice artifact. Dependencies' binaries are
not bundled. Some wheels include other software with their own notices; preserve
the complete texts (not just the top-level license label). Installed environment
inventories include optional and packaging tools; they are not a minimal SBOM of
only libraries used at runtime. Install recipes exclude pip/setuptools/wheel.

Core components include FastAPI/Uvicorn/Pydantic, Requests, HTTPX, PyYAML,
Cryptography, Pillow, OpenCV, NumPy, Torch/torchvision and optional Ultralytics.
Consult the exact retained notices rather than assuming one license covers all.

## Models — binaries excluded

`models/pretrained/MANIFEST.json` and `THIRDUNI-EXTENSIONS.json` retain model names,
source URLs, revisions where supplied, sizes and hashes. These records do not
independently verify all training-data rights.

| Component | Evidence and unresolved point |
|---|---|
| Ultralytics 8.4.0 and YOLO pose/object assets | Installed metadata identifies AGPL-3.0; publisher offers AGPL/Enterprise alternatives. Decide the actual full-project licensing path before distribution or proprietary deployment. Merely omitting weights does not remove software-license obligations. |
| Community fall model | Local manifest reports AGPL-3.0. Optional, excluded; training provenance not independently established. |
| Musawer14 fight model | Local record/publisher metadata says MIT, but underlying Ultralytics/model provenance must also be reviewed. Pinned revision card could not be retrieved in this review. Not cleared for unrestricted distribution. |
| seawsurf fire model | Publisher card labels CC BY 4.0 and identifies YOLO26 as its base. That card alone does not resolve underlying Ultralytics rights; pinned revision card could not be retrieved. Not cleared for unrestricted distribution. |

Publisher guidance (checked 26 October): https://www.ultralytics.com/license
Optional model cards: https://huggingface.co/Musawer14/fight_detection_yolov8
and https://huggingface.co/seawsurf/fire_yolo . No inference service purchase,
commercial license or general legal clearance is implied by this inventory.

## Demonstration footage

ASSIST-IoT multimodal fall-detection dataset, CC BY 4.0. Credits retained in
`deploy/free-demo-assets/LICENSE.txt` and `credits.html`, including the original
authors, source, license, project/facility acknowledgements and derivative-media
changes. Original MP4 bytes are unchanged; JPEG/GIF derived examples are labelled.
No personal room recording is included. No endorsement is implied.
Source: https://github.com/ASSIST-IoT-SRIPAS/multimodal-fall-detection-dataset

## Frontend source and map dependencies

- `dashboard/incident-map.html` references Leaflet 1.9.4 from unpkg. Its upstream
  license is BSD-2-Clause. The source attribution is retained; the map page needs
  external network access and is not an offline guarantee. Preserve OpenStreetMap
  contributor attribution and comply with the chosen tile provider's terms.
  https://github.com/Leaflet/Leaflet/blob/v1.9.4/LICENSE
- `dashboard/aicos-command-center.jsx` imports React (MIT) and lucide-react
  (ISC, with inherited notices in the upstream license). This is a separate
  simulated JSX prototype, not the built FastAPI dashboard or a pinned/tested
  Node application. It is not part of the runnable judge quickstart. Exact
  versions/build and any notices for those versions require review if deployed.
  https://github.com/react/react/blob/main/LICENSE
  https://github.com/lucide-icons/lucide/blob/main/LICENSE

## Remaining publication decisions

1. ThirdUni's public-source scope confirmation and owner's exact upload approval.
2. Full-project license choice and compatibility, particularly Ultralytics.
3. Rights/attribution for any other externally sourced material the owner identifies.
4. Do not replace upstream notices with a blanket license for AICOS.

Documentation, observed pins and a working local demo do not by themselves close
these gates. The release remains private and marked for review.


## Current release preparation — 29 October 2026

The owner requested full-code publication for ThirdUni. Source, example configuration and attributed laboratory assets are prepared; private credentials, environments, databases and personal footage are excluded. No new blanket source license is assigned by this preparation. Model binaries are not redistributed in Git; the setup script obtains pinned reference files from their publishers and verifies SHA-256. Publisher labels and underlying license obligations remain applicable. Leaflet 1.9.4 is now locally served with its BSD license at dashboard/map-assets/LEAFLET-LICENSE.txt. Claude, ChatGPT and Codex assistance is disclosed in the README.
