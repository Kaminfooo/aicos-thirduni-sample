# Third-party model inventory

This file is a due-diligence checklist for AICOS deployments and acquisitions. It intentionally does **not** claim that a model is commercially licensed unless that license has been verified from the model publisher at the exact revision used.

## Models referenced by the current source

| Component | Default / source | Purpose | Action before commercial deployment |
|---|---|---|---|
| Pose / person detection | Ultralytics YOLOv8 pose family | person boxes + COCO-17 keypoints | Record exact package/model version, model file SHA-256, and applicable Ultralytics/model license. |
| Fall model | Hugging Face repository configured in `fall_model.repo` | fallen/sitting/standing classification | Pin repository revision; archive license text; record weights SHA-256 and training-data provenance if published. |
| Violence model | Hugging Face repository or local weights configured in `violence` settings | violence/non-violence confirmation | Pin repository revision; archive license text; record weights SHA-256 and dataset provenance. |

## Required acquisition / deployment record

For every model actually enabled, record: repository or vendor, exact revision/tag, downloaded filename, SHA-256, license identifier and full license text, commercial-use permission, redistribution permission, attribution obligations, dataset provenance, and any export/privacy restrictions.

Do not transfer third-party weights to a buyer unless the applicable license permits redistribution. When redistribution is not permitted, transfer only the integration code and a reproducible download/pinning procedure.
