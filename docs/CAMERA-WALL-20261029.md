# Camera surveillance wall — 29 October 2026

The operations dashboard now has a Camera surveillance section and sidebar link. Add up to nine independent preview tiles, each with a camera name, site/location label, optional coordinates, source type and connection state. Grid controls offer one, two or three columns and adapt to a single column on small screens. Select, Connect, Disconnect, Remove and Disconnect all feeds control each source. Device access begins only when the operator selects Connect; audio is disabled.

Supported input paths:

- Browser cameras through getUserMedia and the available camera-device selector.
- Browser-playable HTTP(S) video streams. Cross-origin streams must permit CORS. Codec/HLS support depends on the browser. URLs with embedded usernames/passwords and non-HTTP protocols are rejected, as are insecure streams from an HTTPS dashboard.
- Explicitly labelled reference playback of the two existing licensed recordings, for verifying the wall without capturing personal camera footage. Reference tiles are excluded from the connected-camera counter and cannot be represented as live camera analysis.

Raw RTSP/ONVIF feeds still require a compatible external streaming gateway. The backend event/health registry is not a video relay. No gateway, transcoder, persistent NVR recording, multi-camera inference scheduler or new hosted service was implemented here.

Analyse selected feed uses the existing single-input detector pipeline. For browser cameras, the analysis owns cloned video tracks; stopping analysis leaves the wall preview independent. For compatible network sources, the analysis opens the selected URL with CORS enabled. Selected camera metadata and position become frozen run settings and are retained in actual incidents and map markers. Only one feed is analysed at a time, for at most 90 seconds. The wall displays an analysing state for that selected source. Ordinary source and reference recording analysis still work through Detection lab.

Preview tiles detect ended streams and mark an 8-second lack of advancing frames. Detection also stops on ended/paused/frozen input rather than repeatedly analysing a frozen image. Camera tracks are released on disconnect/removal/page exit; an outstanding permission request that resolves after cancellation has its tracks stopped. Disconnect all also requests stopping an active detector run. There is no silent retry of event writes.

Camera-wall configuration is page-local and resets on reload. Source URLs and device IDs are not copied into incident exports or sent to the model API. Camera locations appear on the existing map. Nine tiles is a UI ceiling, not a measured claim about nine simultaneous camera detections or network-stream capacity.

## Verification

- 16 wall-controller checks passed: URL/coordinate validation, no automatic capture, reference-versus-camera status, source-specific reconnect, device cleanup/cancellation boundaries, active-run locks, selected-source metadata and nine-slot guard. Physical devices in these checks are doubles.
- 23 existing dashboard checks, 11 location-controller checks and all 12 workspace backend tests passed after the final changes, including asset allowlisting. Total targeted checks: 62.
- Actual browser: two distinct reference recordings and a third same-origin HTTP test recording decoded and played concurrently. Source names, location labels, selection, 3-column/focus layouts and disconnect-all worked. These recordings are not physical live cameras.
- Selected HTTP test source produced a fresh person-down alert after 22 samples at 9.156 server-elapsed seconds; signed evidence verified and the map displayed the selected camera name and explicitly synthetic coordinates. This acceptance case is not a new accuracy benchmark or RTSP acceptance.
- Desktop 1440×1000 and mobile 390×844 layouts had no horizontal page overflow. Mobile tiles stacked and navigation closed correctly. Browser error log was empty during acceptance. All test previews were disconnected afterward.

Proof images: `.runtime/judge-redesign-20261028/camera-wall-desktop-20261029.png` and `camera-wall-mobile-20261029.png`. Physical multi-device grants, specific IP-camera gateways, sustained throughput and deployment remain acceptance work. This update is local; the public Render replay sample is unchanged.
