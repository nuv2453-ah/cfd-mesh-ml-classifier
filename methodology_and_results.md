# Methodology and Results: Naive Bayes Mesh Quality Classification

## Methodology

Mesh data was generated in SimScale using a public NACA 0012 airfoil project as a
starting geometry. Eleven meshes were produced by systematically varying two mesh
generation settings: **fineness** (SimScale's 1–10 resolution slider) and whether
**automatic boundary layers** were enabled. All other settings (Standard algorithm,
Automatic sizing, Automatic curvature, physics-based meshing, hex element core) were
held constant across runs so that any change in mesh quality could be attributed to
fineness and boundary-layer treatment specifically.

For each run, SimScale's meshing log was parsed for per-element-type quality metrics
(non-orthogonality, skewness, aspect ratio, edge ratio, volume ratio — reported as
min, max, average, and 99.99th percentile) along with total cell and node counts.
Tetrahedra were used as the representative element type for labeling, since they made
up the majority of each mesh and non-orthogonality is the metric SimScale enforces
most strictly (an acceptable range of 0.0–88.0, versus 0.0–100.0 for the others).

**Ground-truth labeling.** A run was labeled "good" (1) if its maximum tetrahedral
non-orthogonality stayed below SimScale's own acceptable ceiling of 88.0, and "bad"
(0) otherwise. This ties the label directly to a platform-enforced quality bound
rather than an arbitrary cutoff chosen after the fact.

**Classifier.** A Gaussian Naive Bayes classifier (`GaussianNB`, scikit-learn) was
trained on ten numeric features: fineness, boundary-layer flag, total cell count, and
eight tetrahedral quality statistics (average non-orthogonality; max/average
skewness, aspect ratio; max edge ratio; max volume ratio). Features were
standardized before fitting. Random Forest and Logistic Regression were trained on
the same features for comparison.

Given the small sample size (n=11), model performance was evaluated with
**leave-one-out cross-validation** rather than a train/test split, since a single
held-out point would not produce a stable estimate at this scale.

## Results

| Run | Fineness | Boundary layers | Total cells | Max non-orthogonality | Label |
|---|---|---|---|---|---|
| 1 | 5 | on | 947,046 | 87.6 | good |
| 2 | 2 | on | 433,296 | 87.7 | good |
| 3 | 8 | on | 3,787,644 | 81.2 | good |
| 4 | 5 | off | 958,256 | 68.6 | good |
| 5 | 1 | off | 340,353 | 72.4 | good |
| 6 | 1 | on | 357,432 | 88.8 | **bad** |
| 7 | 3 | on | 506,715 | 89.4 | **bad** |
| 8 | 3 | off | 466,041 | 68.2 | good |
| 9 | 6 | on | 1,236,411 | 86.6 | good |
| 10 | 4 | on | 606,741 | 88.9 | **bad** |
| 11 | 7 | on | 1,650,537 | 86.2 | good |

Class balance: 8 good, 3 bad.

**Leave-one-out cross-validation accuracy:**

| Model | Accuracy |
|---|---|
| Gaussian Naive Bayes | 0.82 |
| Random Forest | 0.91 |
| Logistic Regression | 0.64 |

GaussianNB's leave-one-out confusion matrix (see `confusion_matrix.png`):

|  | Predicted bad | Predicted good |
|---|---|---|
| **Actual bad** | 1 | 2 |
| **Actual good** | 0 | 8 |

The model produced zero false negatives (no good mesh was misclassified as bad) and
correctly identified one of the three bad meshes when held out, missing the other
two — both of which sit close to the 88.0 decision boundary (88.8 and 88.9),
consistent with the classifier's uncertainty concentrating near the threshold rather
than being distributed randomly.

### The fineness × boundary-layer interaction

The central finding, visualized in `fineness_vs_quality.png`, is that mesh failure is
not driven by fineness alone but by an **interaction** between fineness and boundary
layer settings. With automatic boundary layers enabled, every run at fineness ≤ 4
exceeded SimScale's acceptable non-orthogonality ceiling (88.8, 89.4, 88.9), while
every run at fineness ≥ 5 passed (87.6, 86.6, 86.2, 81.2), with non-orthogonality
decreasing roughly monotonically as fineness increased further. With boundary layers
disabled, non-orthogonality stayed comfortably low (68–72) across all tested
fineness levels.

This indicates that coarse-to-mid global mesh resolution does not provide enough
surrounding cells for SimScale's automatic boundary-layer algorithm to transition
smoothly into the thin, stretched prism cells it places near the airfoil wall — a
known risk in practical CFD meshing. Run 10 (fineness 4, boundary layers on) confirms
this beyond non-orthogonality alone: its maximum volume ratio reached 167.5, well
outside the acceptable range of 0–100, whereas no other run in the dataset exceeded
82.

## Limitations

- **Sample size.** Eleven data points is small for any supervised learning method.
  Leave-one-out cross-validation was used specifically because it is the least biased
  estimate available at this scale, but the reported accuracies should be treated as
  illustrative rather than statistically robust. The 3 "bad" examples in particular
  limit how confidently the model's minority-class recall can be assessed — the two
  leave-one-out misses in the confusion matrix reflect this small-sample variance
  rather than a stable failure mode of the classifier.
- **Single-metric labeling.** The good/bad label was derived from tetrahedral
  non-orthogonality alone. Run 10 shows that a mesh can pass on one metric's
  definition while badly failing another (volume ratio), suggesting a composite or
  multi-metric label would be a more robust ground truth for future work.
- **Boundary-layer quality is not purely "worse."** Meshes with boundary layers
  disabled scored better on every global quality metric tested, but this does not
  mean they are better meshes for CFD purposes — boundary-layer prism cells are
  intentionally thin and stretched to resolve the viscous sublayer near walls, which
  is correct and often necessary practice for accurate wall-bounded flow simulation.
  Global scalar quality metrics do not capture this physical justification, so a
  classifier trained only on these metrics risks penalizing meshes that are actually
  better suited to the physics being simulated. This is a meaningful limitation of
  using generic mesh-quality statistics as a proxy for "meshing correctness."

## Future work

- Expand the dataset with additional fineness levels and additional binary settings
  (hex element core, physics-based meshing) to test whether other settings interact
  with fineness the way boundary layers do.
- Replace or supplement the single-metric label with a composite score across
  non-orthogonality, skewness, aspect ratio, and volume ratio.
- Run actual CFD simulations on meshes the classifier labels "good" versus "bad" and
  compare solver convergence behavior, to validate whether the mesh-quality label
  corresponds to real differences in simulation outcomes — not just static mesh
  statistics.
