# Practical Implementation Demo

## AI-Assisted Dental Radiology Workflow

This repository now includes a practical educational implementation showing how AI can be used in dental radiology as a support tool.

The demo is not a diagnostic device. It uses a synthetic bitewing-style image and a toy machine-learning classifier to demonstrate the workflow used by many real radiology AI systems:

1. Load or create a radiographic image.
2. Divide the image into small patches.
3. Extract image features from each patch.
4. Use a trained classifier to estimate the probability of suspicious radiolucency.
5. Convert patch predictions into a heatmap.
6. Highlight regions for clinician review.

## Why This Is Practical

Real AI tools in dental radiology often work by detecting patterns in radiographs and presenting suspicious areas to the dentist. The dentist remains responsible for final interpretation.

This demo mirrors that concept in a simplified and transparent way.

## What The Script Does

The script `src/ai_radiology_demo.py`:

- generates synthetic normal and suspicious radiographic patches
- trains a Random Forest classifier
- creates a synthetic bitewing-style image
- scans the image with a sliding window
- produces an AI probability heatmap
- marks suspicious regions for review

## Output

The main output is:

```text
figures/ai_radiology_demo_workflow.png
```

It shows:

- original synthetic radiograph-like image
- AI probability heatmap
- flagged regions for dentist review

## Important Clinical Note

This implementation is for education and portfolio demonstration only. It does not use real patient data and must not be used for clinical diagnosis.

For clinical use, a dental AI model would require:

- real annotated radiographic datasets
- validation on external clinical data
- calibration and performance testing
- prospective evaluation
- privacy and ethics review
- regulatory approval where required
- dentist oversight in the clinical workflow

## How This Connects To Research

This implementation can support a discussion of:

- caries detection on bitewing radiographs
- periapical radiolucency detection
- periodontal bone-loss measurement
- AI as a second-reader tool
- limitations of model generalization
- explainability and clinician trust

## Run The Demo

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run:

```bash
python src/ai_radiology_demo.py
```
