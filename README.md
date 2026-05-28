# Strabismus Screening System

A lightweight, client-side web application for automatic strabismus 
screening using MediaPipe Face Mesh. No server infrastructure or 
special hardware required.

## Live Demo

[Open via GitHub Pages](https://checilliaabigail.github.io/Strabismus-Detection/)

## How to Use

1. Open the application in any modern browser
2. Upload a clear frontal facial image (JPG, PNG, or WEBP)
3. The system will automatically:
   - Extract 10 eye-region landmarks via MediaPipe Face Mesh
   - Apply rotation correction for head tilt
   - Compute normalized horizontal (dx_norm) and vertical (|dy_norm|) deviation parameters
   - Classify the result using statistical thresholds

## Classification Output

**Horizontal (3 classes):**
- Exotropia — outward (temporal) deviation
- Normal — no significant horizontal deviation
- Esotropia — inward (nasal) deviation

**Vertical (2 classes):**
- Normal — no significant vertical deviation
- Vertical Strabismus — upward or downward deviation (hypertropia/hypotropia)

## Performance

| Axis | Accuracy | Sensitivity | Specificity |
|------|----------|-------------|-------------|
| Horizontal | 87.20% | 86.24% | 89.00% |
| Vertical | 80.00% | 79.49% | 81.00% |

## Method

Thresholds were derived offline from 484 labelled cropped 
eye-region images using:
- Gaussian log-likelihood intersection for horizontal deviation
- Gamma log-likelihood intersection for vertical deviation

Distribution models were selected empirically via the Shapiro-Wilk 
normality test.

## Important Notice

This system is intended **solely as a preliminary screening aid** 
and should not replace professional medical diagnosis. Clinical 
validation is required before deployment in medical practice.

## Related Paper

> Abighail C.J., Primulando R., Sulungbudi J.V., Fidiani E. 
> *Computer Vision for Strabismus Detection*. 2025.

## Dataset

Moorthy A. (2024). *Strabismus Detection Dataset*. Kaggle.  
https://www.kaggle.com/datasets/ananthamoorthya/strabismus

## License

MIT License
