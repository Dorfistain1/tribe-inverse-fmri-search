# TRIBE v2 Shared Runtime Architecture

## Purpose

This project should provide a reusable local environment for running **TRIBE v2** brain-response predictions.

The intention is to configure TRIBE once and then use it from many independent experiments without duplicating:

* model downloads,
* model initialization,
* dependency setup,
* preprocessing logic,
* cortical metadata,
* caching,
* prediction serialization,
* visualization utilities,
* or basic validation.

TRIBE should therefore be treated as shared infrastructure.

Individual experiments should only contain the logic specific to the question being investigated.

---

# Core Principle

The overall structure should conceptually be:

```text
                    TRIBE Runtime
                         │
            ┌────────────┼────────────┐
            │            │            │
            ↓            ↓            ↓
       Experiment A  Experiment B  Experiment C
```

The TRIBE runtime knows how to transform sensory stimuli into predicted brain responses.

Experiments decide what to do with those predictions.

For example:

```text
Audio
  ↓
Shared TRIBE Runtime
  ↓
Predicted Brain Activity
  ↓
Experiment-Specific Analysis
```

The shared layer should know nothing about concepts such as:

* psychedelic similarity,
* evolutionary optimization,
* music generation,
* stimulus ranking,
* classification,
* behavioral hypotheses,
* or specific scientific experiments.

Those belong entirely inside individual experiment directories.

---

# Proposed Repository Structure

```text
tribe-lab/
│
├── README.md
│
├── config/
│   ├── tribe.yaml
│   └── paths.yaml
│
├── tribe_core/
│   ├── runtime/
│   ├── inputs/
│   ├── outputs/
│   ├── preprocessing/
│   ├── brain/
│   ├── visualization/
│   ├── cache/
│   └── validation/
│
├── models/
│
├── cache/
│
├── shared_data/
│
├── experiments/
│   │
│   ├── example_audio_test/
│   │
│   ├── example_video_test/
│   │
│   └── ...
│
├── outputs/
│
└── docs/
```

The exact filenames are not important.

The important architectural distinction is:

```text
tribe_core/
```

contains reusable infrastructure,

while:

```text
experiments/
```

contains independent research projects.

---

# Shared TRIBE Runtime

The central component should expose TRIBE through a simple internal interface.

Conceptually:

```text
stimulus
   ↓
TRIBE Runtime
   ↓
Prediction
```

The rest of the project should not need to understand how TRIBE itself is initialized or configured.

The runtime should be responsible for:

* loading the pretrained TRIBE model,
* selecting the appropriate compute device,
* maintaining the model cache,
* accepting supported stimulus modalities,
* invoking TRIBE inference,
* retaining timing information,
* exposing cortical predictions in a consistent format,
* attaching relevant metadata,
* and returning standardized results.

This creates a boundary between TRIBE itself and everything built on top of it.

---

# Supported Inputs

TRIBE v2 can operate on naturalistic:

```text
VIDEO
AUDIO
TEXT
```

The shared runtime should therefore expose these as standardized stimulus types.

Conceptually:

```text
Stimulus
│
├── Video
├── Audio
└── Text
```

Experiments should not implement separate TRIBE loading logic for each modality.

Instead:

```text
experiment
    ↓
shared input representation
    ↓
TRIBE
```

---

# Standard Stimulus Representation

Every input should have a common conceptual representation containing information such as:

```text
Stimulus
│
├── identifier
├── modality
├── source
├── duration
├── preprocessing information
└── arbitrary experiment metadata
```

This allows experiments to attach their own information without modifying TRIBE itself.

For example:

```text
stimulus_id: candidate_0042
modality: audio

experiment metadata:
    generation: 17
    parent: candidate_0031
```

The shared runtime should preserve this metadata but should not interpret it.

---

# Standard Prediction Representation

TRIBE predictions should also have a consistent project-wide representation.

Conceptually:

```text
BrainPrediction
│
├── stimulus metadata
├── cortical activity
├── temporal information
├── cortical-space information
├── TRIBE model information
├── processing information
└── prediction metadata
```

The primary data produced by TRIBE can conceptually be represented as:

```text
time × cortical vertices
```

TRIBE's pretrained prediction space uses the `fsaverage5` cortical surface with roughly 20,000 vertices.

Every experiment should receive the same standardized representation rather than directly depending on TRIBE's internal objects.

---

# Temporal Information

Brain-response predictions should always retain their relationship with stimulus time.

Conceptually:

```text
Stimulus

0s ───────────────────────────── 30s

        ↓

TRIBE prediction

t0
t1
t2
t3
...
tn
```

TRIBE accounts for the delayed nature of the fMRI BOLD response, so the shared system should preserve TRIBE's timing conventions rather than allowing each experiment to reinterpret them independently.

Timing information is scientifically important and should therefore remain part of the prediction itself.

---

# Cortical Coordinate System

All predictions should retain information describing where predicted activity exists on the cortical surface.

The shared system should provide one canonical cortical representation.

Conceptually:

```text
vertex 0
vertex 1
vertex 2
...
vertex ~20,000
```

Experiments may later transform these predictions into:

```text
vertices
    ↓
regions
    ↓
networks
    ↓
experiment-specific features
```

But those transformations should not alter the original prediction.

The original cortical prediction should remain available.

---

# Raw vs Derived Data

A strict distinction should exist between:

```text
RAW TRIBE PREDICTION
```

and:

```text
DERIVED EXPERIMENT DATA
```

For example:

```text
TRIBE
  ↓
20k cortical vertices
  ↓
stored prediction
  ↓
experiment
  ↓
DMN activity
connectivity
entropy
similarity score
etc.
```

The original TRIBE result should not be overwritten when producing derived metrics.

This makes results reusable across experiments.

---

# Prediction Cache

TRIBE inference can be computationally expensive.

Therefore the project should treat predictions as reusable assets.

Conceptually:

```text
stimulus
   ↓
Has this exact stimulus already been processed?

        YES ─────→ cached prediction

        NO
        ↓
      TRIBE
        ↓
   save prediction
```

The cache should identify predictions using both:

```text
stimulus identity
+
model/configuration identity
```

This prevents accidentally reusing a prediction generated under incompatible settings.

---

# Shared Model Storage

The pretrained model should exist once for the entire project.

It should not be downloaded independently inside every experiment.

Conceptually:

```text
models/
   │
   └── TRIBE v2
          │
          ├── experiment A
          ├── experiment B
          ├── experiment C
          └── experiment D
```

Likewise, large supporting assets should be stored globally rather than copied into experiment directories.

---

# Experiment Isolation

Every experiment should behave as its own small project.

Example:

```text
experiments/
│
├── psychedelic_audio/
│   ├── README.md
│   ├── config/
│   ├── data/
│   ├── src/
│   ├── results/
│   └── notes/
│
├── movie_comparison/
│   ├── README.md
│   ├── config/
│   ├── data/
│   ├── src/
│   └── results/
│
└── language_comparison/
    ├── README.md
    ├── config/
    ├── data/
    ├── src/
    └── results/
```

Experiments may depend on:

```text
tribe_core
```

but should not modify it to accommodate experiment-specific requirements unless the change is genuinely reusable.

---

# Experiment Contract

An experiment should conceptually operate through a very small boundary:

```text
INPUT

stimulus
    ↓

SHARED SYSTEM

TRIBE prediction
    ↓

EXPERIMENT

analysis
    ↓
result
```

This separation is important.

A psychedelic experiment should therefore never need to know:

> How do I load TRIBE?

It should only need to know:

> Give me the TRIBE prediction for this stimulus.

---

# Shared Data

Some datasets may eventually be useful across multiple experiments.

These should live outside individual experiment directories.

For example:

```text
shared_data/
│
├── cortical_atlases/
├── network_definitions/
├── reference_fmri/
└── metadata/
```

Experiment-specific datasets remain inside the relevant experiment.

The distinction should be:

```text
used by many projects
        ↓
shared_data/


used by one experiment
        ↓
experiments/<experiment>/data/
```

---

# Brain-Space Utilities

A reusable neuroscience layer should sit between raw TRIBE output and experiment-specific analysis.

Conceptually:

```text
TRIBE vertices
      ↓
brain utilities
      ↓
regions / networks / cortical masks
      ↓
experiment
```

This layer can eventually provide standardized concepts such as:

```text
vertex selections
regions of interest
cortical atlases
brain networks
spatial averaging
temporal averaging
surface metadata
```

These are generic neuroscience operations and therefore belong outside individual experiments.

---

# Visualization

Visualization should also be reusable.

Experiments should be able to request representations such as:

```text
brain activity at time T
```

or:

```text
average activity over interval T1 → T2
```

without implementing cortical rendering themselves.

The visualization layer should consume standardized prediction objects.

Conceptually:

```text
BrainPrediction
      ↓
Visualization
      ↓
cortical surface / timeline / region representation
```

Visualization must remain separate from analysis.

---

# Output Organization

Shared predictions and experiment results should not be mixed together.

Conceptually:

```text
outputs/
│
├── tribe_predictions/
│
└── experiments/
    │
    ├── psychedelic_audio/
    ├── movie_comparison/
    └── ...
```

This allows a single TRIBE prediction to be referenced by several different experiments.

---

# Provenance

Every prediction should retain enough metadata to determine where it came from.

At minimum, the conceptual provenance chain should be recoverable:

```text
stimulus
   ↓
stimulus preprocessing
   ↓
TRIBE version
   ↓
model configuration
   ↓
prediction
   ↓
experiment
   ↓
derived result
```

Scientific results should never exist without knowing which prediction produced them.

---

# Reproducibility

The shared runtime should prioritize deterministic and reproducible behavior wherever possible.

Every prediction should be associated with:

```text
model identity
software version
configuration
input identity
processing configuration
timestamp
```

Experiments should separately record their own:

```text
experiment version
parameters
random seeds
analysis configuration
```

This produces two independent layers of reproducibility:

```text
TRIBE reproducibility
        +
experiment reproducibility
```

---

# Configuration Philosophy

Global TRIBE settings should live at the project level.

Experiment-specific settings should live inside experiments.

For example:

```text
GLOBAL

model
device
cache
brain surface
shared paths
```

versus:

```text
EXPERIMENT

hypothesis
dataset
analysis parameters
fitness metrics
generation parameters
etc.
```

An experiment should never silently redefine global TRIBE behavior.

---

# Validation

The shared runtime should have a minimal validation capability.

Its purpose is not to answer a scientific question.

Its purpose is simply to establish:

```text
TRIBE loads
      ↓
stimulus is accepted
      ↓
prediction is produced
      ↓
dimensions are valid
      ↓
metadata is preserved
```

This provides a known-good baseline.

When an experiment fails later, it should therefore be possible to distinguish:

```text
TRIBE/runtime problem
```

from:

```text
experiment problem
```

---

# Dependency Boundary

TRIBE has substantial machine-learning and neuroscience dependencies.

These dependencies should belong to the shared environment rather than being repeatedly managed by every experiment.

Experiments may add additional dependencies for their own purposes, but TRIBE's underlying environment should remain stable.

The shared runtime should therefore behave almost like an internal platform.

```text
Operating System
      ↓
Shared Python / ML Environment
      ↓
TRIBE
      ↓
tribe_core
      ↓
Experiments
```

---

# Version Stability

TRIBE should not automatically change underneath existing experiments.

A particular project state should correspond to a known TRIBE version.

When TRIBE changes:

```text
old experiments
       ↓
should remain reproducible
```

while new experiments may deliberately move to the newer version.

This is particularly important because TRIBE v2 is currently a relatively new research codebase rather than a mature stable software package.

The surrounding project should therefore isolate experiments from unnecessary upstream changes.

---

# Model Abstraction

Although the project is initially designed around TRIBE v2, experiment code should ideally depend on a conceptual:

```text
Brain Encoder
```

rather than tightly depending on TRIBE internals.

Conceptually:

```text
Stimulus
   ↓
Brain Encoder
   ↓
BrainPrediction
```

Today:

```text
Brain Encoder = TRIBE v2
```

In the future:

```text
Brain Encoder = TRIBE v3
```

or another model.

If the prediction interface remains stable, experiments could potentially be reused without major redesign.

---

# Scientific Boundary

The runtime should make a strong distinction between:

```text
predicted brain response
```

and:

```text
measured brain response
```

TRIBE produces predicted fMRI responses from sensory stimuli.

It does not measure the brain of the person experiencing that stimulus.

Likewise, its default pretrained model represents an average-subject prediction rather than an individualized brain simulation.

Every downstream experiment should preserve this distinction.

---

# Repository Philosophy

The repository should ultimately behave like a small neuroscience laboratory.

```text
                   tribe-lab
                       │
        ┌──────────────┼───────────────┐
        │              │               │
        ↓              ↓               ↓
     TRIBE          Shared Brain     Shared Data
     Runtime          Tools
        │
        │
        ├─────────────────────────────────────┐
        ↓                  ↓                  ↓
   Experiment 1       Experiment 2       Experiment 3
        │                  │                  │
        ↓                  ↓                  ↓
     Results            Results            Results
```

TRIBE is infrastructure.

Experiments are disposable and independent.

Predictions are reusable.

Scientific transformations are explicit.

Original data remains preserved.

---

# Desired End State

Once the shared environment exists, creating a new experiment should conceptually require only:

```text
Create experiment directory
        ↓
Provide stimulus
        ↓
Request TRIBE prediction
        ↓
Perform experiment-specific analysis
```

Everything concerning model initialization, preprocessing, caching, cortical representation, prediction storage, and basic visualization should already be handled by the shared system.

The core design objective is therefore:

> **Set up TRIBE once, expose it through a stable internal abstraction, and make every future neuroscience idea an independent experiment built on top of that common foundation.**
