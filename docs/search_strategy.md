# Search Strategy

## Databases and Sources

Initial searches will use:

- PubMed
- Google Scholar
- ScienceDirect
- BMC Oral Health
- Frontiers
- MDPI Dentistry Journal / Diagnostics / Journal of Clinical Medicine
- PMC full-text articles when available

## Core Search Terms

```text
artificial intelligence dentistry radiology
AI dental radiography systematic review
deep learning dental radiographs caries detection
artificial intelligence caries bitewing radiographs systematic review
AI periodontal bone loss dental radiographs systematic review
deep learning periodontal bone loss panoramic radiograph
AI periapical lesion detection dental radiograph CBCT systematic review
artificial intelligence cone beam computed tomography dentistry review
AI oral maxillofacial radiology review
FDA cleared AI dental imaging review
```

## Example PubMed Search String

```text
("artificial intelligence" OR "machine learning" OR "deep learning" OR "convolutional neural network")
AND
("dental radiograph" OR "dental radiography" OR "bitewing" OR "periapical" OR "panoramic" OR "CBCT" OR "cone beam computed tomography")
AND
("caries" OR "periodontal bone loss" OR "periapical lesion" OR "oral maxillofacial")
```

## Screening Plan

1. Screen titles for dental imaging relevance.
2. Screen abstracts for AI method and clinical focus.
3. Prioritize systematic reviews and meta-analyses.
4. Add high-quality primary studies if they fill a gap.
5. Extract key information into `data/evidence_table.csv`.

## Extraction Fields

- Citation
- Year
- Study type
- Dental application
- Imaging modality
- AI task
- Key finding
- Clinical relevance
- Limitations
- Link
