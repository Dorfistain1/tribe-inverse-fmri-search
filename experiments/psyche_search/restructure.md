# Useful Research Findings for Neural-State Synthesis

## Purpose

This document summarizes research findings that are directly useful for building a system that searches for sensory stimuli whose predicted neural effects resemble a chosen target brain state.

The focus is not on academic background or implementation details.

The goal is to extract practical conceptual lessons from existing work in:

* psychedelic neuroimaging,
* inverse neuroscience,
* closed-loop stimulus optimization,
* evolutionary stimulus generation,
* neural-response modeling,
* affective stimulus synthesis,
* and brain-state representation.

The broader system can be described conceptually as:

```text
Target brain state
        ↑
     similarity
        ↑
Predicted brain response
        ↑
Generated stimulus
        ↑
search / evolution
        ↺
```

The initial example target is a psychedelic state, particularly psilocybin-related brain activity during music.

---

# 1. A Psychedelic Target Should Not Be a Single Activation Map

One of the most important conclusions from recent psychedelic neuroimaging research is that psychedelic states should not simply be represented as:

```text
region A = more active
region B = less active
region C = more active
```

The stronger and more reproducible findings involve changes in relationships between brain networks.

These include changes in:

* functional connectivity,
* network integration,
* network segregation,
* communication between sensory systems and higher-order systems,
* global functional connectivity,
* temporal dynamics,
* and signal diversity.

A target should therefore represent the **organization and dynamics of the brain**, rather than only the amplitude of activity at individual cortical locations.

Conceptually:

```text
BAD TARGET

vertex 00001   +0.4
vertex 00002   -0.1
vertex 00003   +0.8
...
```

A more meaningful target might instead resemble:

```text
BETTER TARGET

Visual ↔ Default Mode          increased
Visual ↔ Frontoparietal        increased
Somatomotor ↔ Frontoparietal   increased
Network segregation            decreased
Global integration             increased
Signal complexity              altered
```

---

# 2. Cross-Drug Psychedelic Research Provides a General Target

A large 2026 psychedelic fMRI mega-analysis combined data from multiple datasets and several psychedelic compounds.

The analysis included:

* psilocybin,
* LSD,
* DMT,
* ayahuasca,
* and mescaline.

Across datasets, one of the strongest shared findings involved greater coupling between higher-order networks and sensory or motor networks.

Examples include stronger communication between:

```text
Default Mode Network
          ↕
Visual / sensory systems
```

and:

```text
Frontoparietal networks
          ↕
Visual / somatomotor systems
```

This suggests that a broadly defined psychedelic target could represent common network-level effects rather than the peculiarities of a single drug or experiment.

---

## Confidence-Weighted Targets

An especially useful idea from large-scale analyses is that not all measured effects should be treated as equally reliable.

Instead of constructing a fitness function where every brain feature contributes equally:

```text
fitness =
feature_1 +
feature_2 +
feature_3
```

the target could conceptually incorporate confidence:

```text
fitness =
Σ(
    similarity_to_feature
    ×
    confidence_in_feature
)
```

Features that consistently appear across several studies would influence the target more strongly.

Features that only appear in isolated studies would contribute less.

This reduces the chance that optimization discovers stimuli that exploit noise or study-specific artifacts.

---

# 3. PsiConnect Is Especially Relevant for Music

The PsiConnect dataset is particularly useful because participants were measured both:

```text
baseline
```

and:

```text
under psilocybin
```

while performing several different activities.

These included:

```text
rest
meditation
music
movie
```

This allows psychedelic effects to be studied while controlling for the sensory context.

---

# 4. Psychedelic Brain States Depend on Context

PsiConnect found that the psychedelic brain state changes significantly depending on what the participant is doing.

The brain under psilocybin while listening to music is not simply:

```text
generic psychedelic state
+
music
```

The interaction between the drug and the sensory environment itself appears to matter.

This suggests that the best initial target for a music synthesis experiment would not necessarily be:

```text
generic psilocybin brain
```

Instead:

```text
PSILOCYBIN + MUSIC
        -
BASELINE + MUSIC
```

would provide a more relevant target.

The same principle could later apply to other modalities.

For example:

```text
psilocybin + movie
      -
baseline + movie
```

could become a target for generated audiovisual stimuli.

---

# 5. The Same Neural Features Should Be Extracted From Both Sides

TRIBE produces a predicted cortical response to a stimulus.

A psychedelic dataset contains measured neural activity.

Directly comparing these two raw representations may not be meaningful.

Instead, both should conceptually be transformed into equivalent neural features.

For example:

```text
                 TRIBE RESPONSE
                       ↓
              predicted time series
                       ↓
        ┌──────────────┼──────────────┐
        ↓              ↓              ↓
 connectivity      integration      dynamics
        ↓              ↓              ↓
        └──────────────┼──────────────┘
                       ↓
               feature representation
                       ↕
               psychedelic target
```

The same analysis should ideally be applied to:

```text
real psychedelic fMRI
```

and:

```text
TRIBE-generated fMRI prediction
```

This makes the comparison conceptually cleaner.

---

# 6. PsiConnect Already Contains Useful Analysis Approaches

The public PsiConnect analysis includes methods for examining:

* functional connectivity,
* connectivity differences,
* cortical global functional connectivity,
* network modularity,
* cortical surface projections,
* neural complexity,
* and state-dependent brain dynamics.

These approaches provide useful examples of how a psychedelic state can be represented without reducing it to a single activation map.

The existing analysis can therefore help define the target space for neural-state synthesis.

---

# 7. Search-Space Design May Matter More Than the Evolutionary Algorithm

Research involving evolutionary generation of emotional stimuli provides an important warning.

Naively searching the full latent space of a generative model may fail to converge.

A modern generator can contain an enormous number of possible states.

Most random changes may:

* produce meaningless outputs,
* produce extremely large perceptual changes,
* move into regions poorly represented by training data,
* or provide almost no useful gradient toward the target.

Therefore:

```text
huge unconstrained generator space
```

may be a poor evolutionary environment.

---

# 8. Constrained Latent Spaces Can Improve Evolution

A recent synthetic-image evolution study initially attempted to evolve stimuli directly within a large generator representation.

The search did not converge reliably.

The researchers instead constructed a lower-dimensional representation based on meaningful examples.

They used dimensionality reduction to create a manageable search manifold.

Evolution could then move through this space more effectively.

Conceptually:

```text
FULL GENERATOR SPACE

enormous
sparse
difficult to navigate
```

became:

```text
MEANINGFUL MANIFOLD

            ●
        ●       ●
     ●      ●      ●
          ↑
        parent
       ↙  ↓  ↘
   child child child
```

Nearby positions now corresponded to meaningfully related stimuli.

---

# 9. Intermediate-Dimensional Representations May Be Best

The same work found that extremely small representations lost too much useful information.

Extremely large representations became unstable and difficult to search.

Intermediate-dimensional representations provided a better balance between:

```text
expressiveness
```

and:

```text
searchability
```

This suggests a general principle for neural-state synthesis:

> Evolution should operate in a representation where small mutations produce meaningful but limited sensory changes.

This is likely more important than choosing between several similar evolutionary algorithms.

---

# 10. Raw Audio Is Probably the Wrong Search Space

For music synthesis, the equivalent warning is that evolution should probably not search directly through raw waveform samples.

Raw audio exists in an extremely high-dimensional space.

A small waveform mutation often has no meaningful musical interpretation.

The search space should instead represent higher-level properties where nearby solutions remain perceptually related.

Conceptually:

```text
BAD

millions of waveform values
```

versus:

```text
BETTER

compact meaningful representation
        ↓
generated audio
```

The exact representation can change over time.

The important theoretical principle is that the evolutionary space should preserve useful sensory structure.

---

# 11. Evolution Can Exploit the Fitness Function

Evolutionary optimization does not care whether the resulting stimulus is meaningful, pleasant, musical, interesting, or safe.

It only cares about increasing the target score.

Therefore, if an unintended stimulus causes a large predicted response, evolution may repeatedly exploit it.

This is a form of:

```text
reward hacking
```

For example:

```text
unpleasant repetitive noise
        ↓
large neural response
        ↓
high fitness
        ↓
evolution favors it
```

Even if the result has little connection to the intended subjective experience.

---

# 12. Neural Similarity Does Not Guarantee Subjective Similarity

The synthetic fear-evolution research provides one of the most important warnings for the entire project.

Researchers compared:

```text
subjective fear
```

with a physiological signal related to arousal.

The physiological signal did not reliably correspond to how frightened people actually reported feeling.

In other words:

```text
physiological similarity
```

did not necessarily mean:

```text
subjective similarity
```

This has a direct analogue for neural-state synthesis.

Even if a stimulus produces:

```text
very high psychedelic neural similarity
```

it does not automatically follow that a listener will report:

```text
very psychedelic subjective experience
```

---

# 13. This Difference Is Scientifically Interesting

A disagreement between neural and subjective similarity would not automatically mean the project failed.

It could instead reveal that:

```text
measured neural signature
```

and:

```text
subjective phenomenology
```

are only partially related.

Possible outcomes could include:

```text
Neural similarity:      0.93
Subjective similarity:  0.15
```

or:

```text
Neural similarity:      0.81
Subjective similarity:  0.74
```

Both outcomes would be informative.

The system should therefore preserve the distinction between:

```text
NEURAL TARGET MATCHING
```

and:

```text
SUBJECTIVE EXPERIENCE
```

---

# 14. Genetic Algorithms Are a Reasonable Starting Point

The XDREAM project provides strong precedent for evolutionary stimulus optimization.

Its basic structure was:

```text
generate stimuli
       ↓
measure neural response
       ↓
assign fitness
       ↓
selection
       ↓
mutation / crossover
       ↓
new generation
       ↺
```

This approach was able to discover generated stimuli that strongly activated selected neural targets.

The work also compared several different black-box optimization approaches.

Simple genetic algorithms performed surprisingly well and were relatively robust to noisy neural measurements.

This supports the idea that a basic evolutionary system is sufficient for early experimentation.

---

# 15. Random Search Must Be Kept as a Control

Evolution should always be compared with a simple baseline.

For example:

```text
RANDOM SEARCH

best score = 0.47
```

versus:

```text
EVOLUTION

best score = 0.82
```

would provide evidence that evolution is actually discovering structure.

But:

```text
RANDOM SEARCH

best score = 0.75
```

versus:

```text
EVOLUTION

best score = 0.77
```

would suggest that evolution may not provide much benefit.

Without a random-search baseline, an increasing fitness curve can be misleading.

---

# 16. Multiple Independent Evolutionary Runs Are Important

One successful evolutionary sequence is weak evidence.

It may simply discover one unusual exploit of the fitness function.

A stronger experiment would repeatedly start from unrelated initial populations.

Conceptually:

```text
Run A → solution A
Run B → solution B
Run C → solution C
Run D → solution D
Run E → solution E
```

The key question becomes:

> Which properties independently evolve again and again?

If unrelated searches converge toward similar structures, those recurring properties may reflect something real about the target neural state.

---

# 17. Convergent Evolution May Be More Interesting Than the Best Stimulus

Suppose independent audio evolution repeatedly develops characteristics such as:

```text
slow rhythmic structures
long temporal transitions
particular spectral distributions
high spatial movement
unusual repetition
low predictability
```

That recurring structure may be more scientifically interesting than any one resulting song.

The main output of the project may therefore ultimately be:

```text
features repeatedly favored by the neural target
```

rather than:

```text
one optimized piece of music
```

---

# 18. Bayesian Optimization Is Another Useful Search Strategy

The Automatic Neuroscientist demonstrated a closely related form of inverse neuroscience.

Instead of asking:

```text
What neural response does this stimulus cause?
```

the experiment asked:

```text
What stimulus moves brain activity toward this desired target?
```

Bayesian optimization was used to choose promising future stimuli efficiently.

Conceptually:

```text
target neural response
       ↑
     distance
       ↑
measured brain response
       ↑
     stimulus
       ↑
optimizer
       ↺
```

This provides an alternative to evolutionary algorithms.

Eventually it may be useful to compare:

```text
genetic evolution
vs
Bayesian optimization
vs
random search
```

rather than assuming a single optimizer is optimal.

---

# 19. TRIBE Introduces an Important Coverage Limitation

The released TRIBE v2 model predicts cortical activity.

Its output corresponds approximately to:

```text
time × 20,484 cortical vertices
```

on the `fsaverage5` cortical surface.

However, psychedelic neuroimaging also finds relevant effects involving subcortical structures.

Examples may include structures such as:

```text
caudate
putamen
thalamic systems
other subcortical regions
```

These are not represented by the standard cortical TRIBE prediction.

Therefore:

```text
REAL PSYCHEDELIC BRAIN STATE

        cortex
          ↓
      observable
      with TRIBE

        +

      subcortex
          ↓
    not represented
    by current TRIBE
```

The first version of the project should therefore explicitly use a:

```text
CORTICAL PSYCHEDELIC TARGET
```

rather than claiming to reproduce the complete psychedelic brain state.

---

# 20. Observable and Full Targets Should Be Kept Separate

The project should conceptually distinguish:

```text
FULL BIOLOGICAL TARGET
```

from:

```text
MODEL-OBSERVABLE TARGET
```

For example:

```text
Full psychedelic state
├── cortical effects
└── subcortical effects
```

while:

```text
TRIBE-observable state
└── cortical effects
```

This distinction prevents the project from accidentally treating model limitations as biological absence.

---

# 21. Temporal Dynamics May Matter as Much as Static Connectivity

A psychedelic state is not necessarily one fixed brain configuration.

Brain activity moves through different configurations over time.

Therefore, reducing a full session to:

```text
one connectivity matrix
```

may remove important information.

A richer representation could preserve the neural trajectory.

Conceptually:

```text
time 1 → state A
time 2 → state B
time 3 → state C
time 4 → state D
...
```

This forms a path through neural-state space.

---

# 22. Low-Dimensional Neural-State Spaces May Be Useful

PsiConnect used neural representation techniques to map high-dimensional fMRI activity into lower-dimensional trajectories.

Instead of representing a psychedelic state as thousands of independent measurements, the state becomes something closer to:

```text
                 neural-state space

                     ●
                  ●
               ●
            ●
         ●
```

The psychedelic brain moves through this space over time.

Different sensory contexts can produce different trajectories.

This creates another possible future target representation.

---

# 23. The Ultimate Fitness Could Compare Neural Trajectories

Instead of:

```text
TRIBE(song)
      ↓
average connectivity
      ↓
compare with psychedelic average
```

a richer approach could eventually become:

```text
TRIBE(song)
      ↓
predicted neural time series
      ↓
shared neural embedding
      ↓
trajectory
      ↕
psychedelic trajectory
```

The fitness would then reflect similarity between patterns of neural movement rather than similarity between static averages.

This may preserve:

* transitions,
* temporal organization,
* state switching,
* context sensitivity,
* and other dynamic properties.

This is likely more complex than necessary for an initial system but potentially more meaningful later.

---

# 24. A Multi-Level Target May Be Better Than One Score

Rather than reducing everything immediately to:

```text
psychedelic_score = 0.83
```

the project could retain several independent components.

Conceptually:

```text
Candidate stimulus

Visual ↔ DMN coupling          0.79
Network integration           0.72
Network modularity            0.63
Temporal similarity           0.84
Global connectivity           0.70
Classifier probability        0.76
```

A combined score can still be produced for evolution.

However, retaining the individual components allows researchers to see *why* a candidate performed well.

This also helps detect reward hacking.

---

# 25. Multiple Independent Neural Targets Can Provide Validation

A useful strategy is to optimize against one dataset and test the result against another.

For example:

```text
TRAIN / OPTIMIZE TARGET

PsiConnect
psilocybin + music
```

then test against:

```text
VALIDATION TARGET

cross-drug psychedelic signature
```

If evolved stimuli improve only against PsiConnect, they may be exploiting some dataset-specific property.

If they also become more similar to independently measured psychedelic effects across:

```text
LSD
DMT
mescaline
ayahuasca
psilocybin
```

the result becomes more interesting.

---

# 26. Different Psychedelics Can Become Separate Targets

The project does not have to assume that all psychedelics produce the same neural state.

Eventually the target library could contain:

```text
Psilocybin
LSD
DMT
Mescaline
Ayahuasca
```

The system could then ask:

> Do independently optimized stimuli differ depending on the drug-derived target?

Potentially:

```text
psilocybin target
      ↓
stimulus family A

LSD target
      ↓
stimulus family B

DMT target
      ↓
stimulus family C
```

Or all targets might converge toward similar stimulus properties.

Either outcome would be interesting.

---

# 27. Context Can Also Become Part of the Target

A future neural target could represent not just a drug but a drug-context combination.

For example:

```text
psilocybin + music
psilocybin + meditation
psilocybin + movie
psilocybin + rest
```

This potentially creates multiple related but distinct neural destinations.

The same idea could later apply outside psychedelics.

---

# 28. The System Is Better Understood as Inverse Neuroscience

Traditional neuroscience commonly follows:

```text
Stimulus
   ↓
Brain response
```

This project reverses the direction conceptually:

```text
Desired brain response
        ↓
Search
        ↓
Stimulus
```

The stimulus is treated as the unknown variable.

The neural state becomes the design objective.

---

# 29. Existing Research Supports the General Idea

Several research traditions independently support pieces of this concept.

## Automatic Neuroscientist

Demonstrated closed-loop optimization toward desired fMRI responses.

```text
desired response
      ↑
brain measurement
      ↑
stimulus
      ↑
optimization
```

---

## XDREAM

Demonstrated evolutionary generation of stimuli using neural responses as fitness.

```text
generated stimulus
      ↓
neural response
      ↓
fitness
      ↓
evolution
```

---

## Synthetic Fear Evolution

Demonstrated evolution of generated stimuli toward subjective or physiological emotional responses.

Also demonstrated that:

```text
physiological objective
```

does not necessarily equal:

```text
subjective experience
```

---

## TRIBE

Provides an in-silico model capable of predicting cortical fMRI responses from naturalistic sensory stimuli.

Conceptually:

```text
stimulus
   ↓
TRIBE
   ↓
predicted cortical response
```

This allows many candidate stimuli to be evaluated without placing a human inside an fMRI scanner for every generation.

---

# 30. TRIBE Changes the Scale of the Experiment

Traditional closed-loop fMRI optimization is expensive because every candidate requires an actual participant and scanner time.

With a neural emulator:

```text
candidate
   ↓
TRIBE
   ↓
predicted response
```

thousands of candidates could theoretically be evaluated computationally.

This changes the problem from:

```text
few expensive biological experiments
```

to:

```text
large computational search
```

Human testing can then be reserved for a much smaller set of high-performing candidates.

---

# 31. The Model Must Not Be Confused With a Real Brain

TRIBE predicts neural activity.

It does not measure the listener's actual brain.

Therefore:

```text
TRIBE prediction
```

must remain conceptually separate from:

```text
actual fMRI response
```

A high-performing candidate should therefore initially be described as:

> A stimulus whose predicted cortical response resembles selected properties of the target neural state.

Not:

> A stimulus that puts the listener into that brain state.

---

# 32. Neural Optimization and Human Validation Are Separate Stages

The computational system could eventually produce:

```text
thousands of generated candidates
          ↓
TRIBE evaluation
          ↓
optimization
          ↓
small number of finalists
```

Only the finalists would need subjective testing.

This preserves the distinction between:

```text
computational neural similarity
```

and:

```text
human phenomenology
```

---

# 33. Blind Human Evaluation Would Be Particularly Valuable

If human testing is eventually performed, participants should ideally not know what neural target produced the stimulus.

For example:

```text
Stimulus A
Stimulus B
Stimulus C
Stimulus D
```

rather than:

```text
"PSILOCYBIN SIMULATION"
```

The latter would strongly bias subjective reports.

Afterward, participants could describe experiences across dimensions such as:

```text
absorption
emotional intensity
time distortion
visual imagery
sense of self
anxiety
awe
unusual meaning
dissociation
```

Only afterward would the neural target be revealed.

This would allow comparison between:

```text
neural target
```

and:

```text
reported phenomenology
```

without strong expectation effects.

---

# 34. The Psychedelic Experiment Can Become One Case of a General System

The broader concept is not limited to psychedelics.

Any sufficiently characterized neural state could theoretically become a target.

For example:

```text
meditation
fear
awe
sleep
dreaming
anesthesia
specific emotional states
attention
memory states
end-of-life neural states
```

Each would create a different inverse-neuroscience problem.

Conceptually:

```text
                   TARGET LIBRARY

                  Psilocybin
                      │
                      ├── LSD
                      ├── Fear
                      ├── Meditation
                      ├── Sleep
                      ├── Awe
                      └── ...
                           ↓
                    choose target
                           ↓
                    neural synthesis
```

This broader interpretation can be thought of as:

```text
NEURAL-STATE SYNTHESIS
```

rather than merely psychedelic music generation.

---

# 35. Current Preferred Conceptual Architecture

Based on the existing research, the initial psychedelic-music experiment could conceptually be organized as:

```text
                       PsiConnect

          ┌────────────────┴────────────────┐
          │                                 │
 baseline + music                  psilocybin + music
          │                                 │
          └────────────────┬────────────────┘
                           ↓
                 psychedelic difference
                           ↓
             cortical/network representation
                           ↓
                    TARGET MODEL
                           ↑
                           │
                    similarity score
                           │
                           ↑
                    TRIBE prediction
                           ↑
                           │
                    generated audio
                           ↑
                           │
              meaningful searchable space
                           ↑
                           │
              selection / variation
                           │
                           └──────────────↺
```

A separate cross-drug dataset could then provide external validation:

```text
             evolved candidate
                     ↓
             TRIBE prediction
                     ↓
       cross-drug psychedelic target
                     ↓
          independent similarity score
```

---

# 36. Primary Lessons From Existing Research

The most important lessons are:

1. **Do not reduce psychedelic states to a simple activation map.**

2. **Network connectivity and organization are likely more meaningful targets.**

3. **Use context-specific psychedelic data when possible.**

4. **Psilocybin + music versus baseline + music is particularly relevant for a music experiment.**

5. **Extract equivalent neural features from real fMRI and TRIBE output.**

6. **Keep raw neural predictions separate from derived metrics.**

7. **Search through a meaningful low-dimensional stimulus representation.**

8. **Do not assume the generator's full latent space is searchable.**

9. **Evolutionary algorithms are a reasonable starting point.**

10. **Always compare evolution against random search.**

11. **Run many independent evolutionary searches.**

12. **Look for convergent features across solutions.**

13. **Expect reward hacking.**

14. **Neural similarity does not imply subjective similarity.**

15. **Preserve individual fitness components instead of relying entirely on one score.**

16. **Use independent datasets for validation.**

17. **TRIBE currently models cortex rather than the full brain.**

18. **Separate the biological target from the model-observable target.**

19. **Temporal neural trajectories may eventually be more informative than static averages.**

20. **Treat TRIBE output as prediction rather than measured neural activity.**

---

# 37. Suggested Research Resources

## PsiConnect

Public psilocybin dataset and analysis involving:

* music,
* meditation,
* movie viewing,
* rest,
* connectivity,
* network organization,
* cortical activity,
* and neural dynamics.

Repository:

https://github.com/razilab/psiconnect

Dataset:

https://openneuro.org/datasets/ds006110

---

## BOLD Psychedelics Consortium

Large multi-study, multi-drug psychedelic fMRI analysis.

Useful for identifying robust effects shared across several psychedelics.

Repository:

https://github.com/banilo/BOLD_psychedelics_consortium

---

## TRIBE v2

Foundation model predicting cortical fMRI activity from naturalistic stimuli.

Repository:

https://github.com/facebookresearch/tribev2

---

## XDREAM

Evolutionary stimulus optimization driven by neural responses.

Useful as precedent for:

* genetic search,
* mutation,
* selection,
* crossover,
* optimizer comparison,
* and neural fitness.

Repository:

https://github.com/willwx/XDream

---

## Synthetic Image Evolution for Affective Science

Useful precedent for:

* evolving generated sensory stimuli,
* searching generator representations,
* dimensionality reduction,
* subjective fitness,
* physiological fitness,
* random controls,
* and distinguishing objective signals from subjective experience.

Code:

https://github.com/dariusvalevicius/image-evolution

---

## The Automatic Neuroscientist

Early demonstration of inverse neuroscience using closed-loop fMRI and Bayesian optimization.

Conceptually important for the idea:

```text
desired neural response
        ↓
discover stimulus
```

---

# Final Concept

The core scientific premise is:

> If a target brain state can be represented quantitatively, and a model can predict how sensory stimuli influence neural activity, then generated stimuli can be searched or evolved according to how closely their predicted neural effects approach that target.

For psychedelic synthesis specifically:

```text
real psychedelic neuroimaging
           ↓
define cortical neural target
           ↓
generated sensory stimulus
           ↓
TRIBE prediction
           ↓
extract comparable neural features
           ↓
measure similarity
           ↓
search / evolution
           ↺
```

The most interesting outcome may not ultimately be a single optimized stimulus.

It may instead be discovering:

> **Which sensory structures repeatedly emerge when evolution is asked to move predicted cortical activity toward a particular altered neural state?**
