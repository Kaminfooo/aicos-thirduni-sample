# Camera map and capacity — 29 October 2026

## Map in the current operations workspace

Coordinate-entry fix: both the map and surveillance setup now accept decimal degrees and degrees/minutes/seconds, including hemisphere suffixes, Unicode degree/prime symbols and compact DMS notation. Previously, numeric-only browser fields could reject pasted DMS text or expose an empty value, leaving the position unknown. Applying a valid location now normalizes the fields to decimal degrees, confirms the coordinates and centres the pin. Validation errors are separate from street-tile loading messages, so map loads cannot overwrite an invalid-entry error. A next-analysis camera pin also remains visible when an older run has different location settings.

Verification of this fix: 41 coordinate cases, 11 location checks, 16 surveillance checks, 23 dashboard checks and 12 workspace backend checks passed (103 targeted automated checks). Browser acceptance verified DMS conversion, pin placement, invalid-minute rejection without erasing the saved location, and surveillance setup using DMS without opening a camera. Only public test coordinates were used; the owner's home coordinates were not used for browser verification. The stopped local service was restarted and reported ready.

Use **Camera & event map** in the sidebar. Enter a camera name, optional site label and latitude/longitude, then apply. A map click fills coordinates; applying them is a separate action. Device geolocation is optional, permission-based, and suitable only when the browser device is physically at the camera. Browser coordinates are not automatically the position of a remote camera or source recording.

Location settings apply to the next analysis. They are copied into its server-side session and each actual alert, and included in the incident list and JSON run export. Edits after completion do not change existing event coordinates. Missing locations are explicit and are not plotted at the database's legacy zero-coordinate default; a deliberately configured (0,0) remains valid. Latitude/longitude must be finite, bounded and supplied together.

Blue pins show configured camera positions; orange pins show detected events at the associated camera. The event list opens the corresponding evidence/review, and centre/zoom controls locate the source. Several event types can share a camera coordinate; the list preserves separate access to their evidence. These are camera positions, not calibrated person positions. The existing edge calibration module remains available in the private multi-camera worker; it is not silently applied to browser inputs without calibration data.

Leaflet 1.9.4 is served locally with its BSD license. At the owner's request, street tiles now load automatically when the dashboard opens. The map discloses the third-party area/IP request and keeps OpenStreetMap attribution visible. **Reload map** retries a failed layer. No bulk tile download, offline tile archive, external geocoder, satellite imagery or API key is used. Local coordinate editing and the event list remain usable if street tiles fail. Settings reset on page reload; this anonymous workspace is not a persistent production camera registry. Automatic loading was verified in a fresh browser page with 12 rendered tiles; all 34 controller checks passed again.

## What camera capacity is actually established

Update: `CAMERA-WALL-20261029.md` describes the new multi-preview wall (up to nine tiles). Preview display and selected-feed detection are separate; the following single-analysis and unbenchmarked-throughput limits remain applicable.

The current browser operations workspace processes **one selected input per browser run**. Its inference worker is serialized, and concurrent frame requests from another run can receive busy responses. Eight retained sessions is an isolation/capacity guard, not a certified eight-camera processing limit. It does not open many RTSP streams.

The older edge service in `edge/rtsp_worker.py` starts a CameraWorker per configured camera and creates separate detector state per camera. The backend has camera health/registry support and the earlier dashboard has fleet/incident-map views. This architecture supports multiple configured cameras, but no sustained multi-camera throughput benchmark establishes a supported maximum on the present hardware.

Latest browser measurements before this map change averaged about 528 ms per sampled frame for the three core groups and 949 ms with all twelve categories. Those figures exclude transport/capture and vary by scene, warm-up, hardware and enabled models. They cannot justify a claim of 10, 50 or 100 live cameras. Full-source 21-second recording analysis can take longer than real time when several models run on this CPU.

Before quoting a deployment limit, benchmark 1, 2 and 4 permitted recorded streams using the intended models and target sampling rate; measure p95 frame age/latency, achieved analysis FPS per camera, drops, CPU/GPU/RAM, reconnect behavior and event delivery. Set acceptance thresholds from the site's detection timing requirements. Do not use a high camera count achieved by silently reducing coverage as a reliability claim.

## Verification

12 workspace backend tests passed, including new location validation, explicit unknown/zero coordinates, persistence into event metadata and bounded asset access. Existing 23 dashboard controller checks and 11 new location controller checks passed. Browser acceptance uses a clearly labelled synthetic public map coordinate, not the actual location of the licensed source recording. Physical browser geolocation grants are not automated or claimed as tested.

This change is local. The old public Render replay sample has not acquired this map or live inference.

Provider references: https://leafletjs.com/reference.html and https://operations.osmfoundation.org/policies/tiles/

Final browser acceptance: the supplied recording raised one fresh person-down alert after 27 sampled frames. Its explicitly synthetic Algiers map position appeared in the event list, linked to the correct evidence, and remained present in the downloaded JSON export. Street tiles rendered with attribution. Desktop/mobile checks found no horizontal overflow; the mobile navigation closed after selecting the map. Proof files: `.runtime/judge-redesign-20261028/event-map-desktop-20261029.png`, `event-map-mobile-20261029.png` and `map-run-20261029.json`. All 104 focused regression checks also passed.
