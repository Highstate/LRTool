#  LRTool

<p align="center">
LRTool is a mathematically structured learning rate engine for diffusion-based LoRA training. It replaces heuristic tuning with analytically structured, model-aware scaling.
</p>

---

<p align="center">
<img width="500" height="780" alt="LRTool" src="https://github.com/Highstate/LRTool/blob/main/assets/LRTool.png"/>
</p>

##  Overview

LRTool provides analytically derived learning rate recommendations for diffusion-based LoRA training. It models training as a structured energy system, scaling the learning rate according to exposure (steps and effective batch), capacity (rank–alpha ratio), resolution via hybrid scaling, scheduler RMS integration, and optimizer modifiers. A curvature-based stability model is then used to estimate deviation risk, promoting consistent convergence behavior across diverse architectures and training configurations.

<br>

## Features

- **Analytically Derived Learning Rates**  
  Computes learning rates from a structured energy-preserving formulation rather than empirical presets.

- **Model-Aware Scaling**  
  Architecture-dependent calibration constants and hybrid resolution scaling ensure consistent behavior across supported diffusion models.

- **Exposure Normalization**  
  Automatically scales learning rate according to steps, effective batch size, and dataset size using √-based exposure modeling.

- **LoRA Capacity Modeling**  
  Incorporates rank–alpha relationships into the learning rate calculation to maintain consistent parameter scaling.

- **Scheduler RMS Integration**  
  Numerically integrates scheduler curves (including warmup) using RMS formulation to preserve true effective magnitude.

- **Curvature-Based Stability Score**  
  Provides a continuous stability metric anchored at 100% at the analytical optimum, with asymmetric overshoot penalties.

- **Noise-Aware Overshoot Modeling**  
  Incorporates effective batch size into curvature scaling to reflect gradient noise dampening effects.

- **Interactive Offset Control**  
  Adjustable learning rate offset slider with real-time stability feedback.

- **Training Efficiency Indicator**  
  Secondary efficiency score estimating convergence behavior relative to exposure and capacity.

- **Profile Save / Load System**  
  Save and restore full configurations, including computed statistics.

<br>

## Supported Models

LRTool includes calibrated base energy constants, fragility parameters, and architecture-dependent resolution scaling for the following diffusion models:

- **SD 1.5**
- **SDXL**
- **SDXL Lightning**
- **SDXL Turbo**
- **Pony XL**
- **FLUX.1**
- **FLUX.2 Dev**
- **Z-Image**

<br>

##  Core Learning Rate Model

LRTool derives the recommended learning rate from an energy-preserving formulation:

$$
LR = \frac{TargetEnergy}
{SchedulerFactor \cdot Exposure \cdot Capacity \cdot ResolutionScale \cdot OptimizerModifier}
$$

Rather than treating learning rate as an isolated hyperparameter, the model assumes that stable training occurs when total injected optimization “energy” remains approximately invariant across configurations. The denominator decomposes this energy into measurable components:


- Exposure - Modeling dataset-normalized update magnitude

$$
Exposure = \sqrt{\frac{steps \cdot effective\_batch}{images}}
$$

- Capacity - Modeling LoRA parameter scaling

$$
Capacity = \sqrt{\frac{rank}{\alpha}}
$$

- ResolutionScale - Where \( p \) is an architecture-dependent hybrid scaling exponent

$$
ResolutionScale = \left(\frac{resolution}{native\_resolution}\right)^{p}
$$

- SchedulerFactor - The scheduler contribution is computed as the RMS-integrated value of the learning rate curve (including warmup):

$$
SchedulerFactor = \sqrt{\frac{1}{T} \sum_{t=1}^{T} s(t)^2}
$$

- OptimizerModifier — A static scaling correction applied to non-self-adjusting optimizers.

Target energy is derived from calibrated per-architecture base energy constants, optionally modulated by objective type. This formulation ensures that when configuration variables change (batch size, resolution, rank, scheduler, etc.), the learning rate adjusts analytically to preserve training stability rather than relying on empirical presets.

<br>

## Energy Reconstruction & Stability Model

After computing the learning rate, LRTool reconstructs the delivered optimization energy to verify consistency:


$$
Energy = LR \cdot SchedulerFactor \cdot Exposure \cdot Capacity \cdot ResolutionScale \cdot OptimizerModifier
$$


Deviation from the calibrated target energy is measured as:

$$
deviation = \frac{|Energy - TargetEnergy|}{TargetEnergy}
$$

When the slider offset is zero, the formulation is algebraically symmetric and deviation evaluates to zero by construction.

### Stability Function

Stability is modeled using a curvature-based exponential decay:

$$
Stability = 100 \cdot \exp\left(-k \cdot deviation^2\right)
$$

This produces a smooth, continuous penalty curve centered at the analytically optimal learning rate.

### Asymmetric Curvature

Undershoot and overshoot are treated differently:

- **Undershoot** uses a fixed universal curvature constant.
- **Overshoot** scales curvature according to model fragility, objective sensitivity, and gradient noise.

For overshoot:

$$
k = model\_fragility \cdot objective\_sensitivity \cdot noise\_factor
$$

Where gradient noise is approximated as:

$$
noise\_factor = \max\left(0.35,\sqrt{\frac{1}{effective\_batch}}\right)
$$

This introduces controlled asymmetry: aggressive overshoot degrades stability more rapidly than conservative undershoot, while larger effective batch sizes dampen curvature growth.

The result is a stability model that is analytically anchored at 100% at the computed optimum and degrades smoothly as configuration deviates from energy-preserving conditions.

<br>

## Calibration Methodology

LRTool is calibrated using an energy-invariance principle rather than matching fixed preset learning rates.

### 1. Base Energy Constants

Each supported architecture is assigned a calibrated base energy constant. These constants are chosen to preserve relative training behavior across models rather than replicate any single empirical configuration.

Calibration focuses on:

- Stable convergence at typical LoRA ranks (16–32)
- Common dataset sizes (20–100 images)
- Standard resolutions (native model resolution)
- Cosine-based schedulers with moderate warmup

The goal is not to “fit” a specific preset, but to ensure that equivalent configurations across architectures produce comparable effective optimization energy.

---

### 2. Resolution Scaling Exponent (Hybrid Model)

Resolution scaling uses an architecture-dependent hybrid exponent:

$$
ResolutionScale = \left(\frac{resolution}{native\_resolution}\right)^p
$$

The exponent \( p \) is derived from latent-space signal-to-noise considerations rather than pure pixel area scaling.  

- \( p = 1.0 \) corresponds to edge-length scaling  
- \( p = 2.0 \) corresponds to area scaling  
- LRTool uses calibrated intermediate values to better approximate diffusion SNR behavior  

This prevents over-aggressive LR reduction at high resolutions while preserving stability.

---

### 3. Fragility & Curvature Constants

Each model includes a fragility modifier used in overshoot curvature:

$$
k = model\_fragility \cdot objective\_sensitivity \cdot noise\_factor
$$

Fragility constants are tuned to reflect relative sensitivity of architectures to aggressive learning rates.  
Higher-fragility models penalize overshoot more strongly.

Objective sensitivity parameters introduce additional curvature separation between:

- Style training  
- Character training  
- Concept training  
- Fidelity-focused training  

This preserves meaningful differentiation without altering the core energy formulation.

---

### 4. Scheduler RMS Integration

Schedulers are integrated numerically using RMS formulation:

$$
SchedulerFactor = \sqrt{\frac{1}{T} \sum_{t=1}^{T} s(t)^2}
$$

This ensures that learning rate scaling reflects true effective magnitude rather than peak value, particularly when warmup is used.

---

### 5. Noise Dampening Cap

Gradient noise is approximated via effective batch size:

$$
noise\_factor = \max\left(0.35,\sqrt{\frac{1}{effective\_batch}}\right)
$$

The lower bound prevents curvature collapse at very large batch sizes while still modeling diminishing gradient variance.

---

Calibration is intentionally conservative. The system prioritizes internal mathematical consistency and relative scaling behavior over aggressive maximum-throughput tuning.

LRTool is therefore best interpreted as an analytically consistent baseline from which informed adjustments can be made.

<br>

## Assumptions & Limitations

LRTool is built on structured analytical assumptions rather than empirical curve-fitting. While the model is internally consistent, it operates under the following constraints:

### Analytical Assumptions

- Training stability is approximated via energy invariance across configurations.
- Gradient noise is modeled as a function of effective batch size.
- Resolution scaling follows a hybrid exponent derived from latent-space SNR considerations.
- LoRA capacity effects are approximated using √(rank / α) scaling.
- Scheduler influence is represented by RMS integration rather than peak magnitude.

These approximations are designed to preserve relative behavior across configurations, not to perfectly model every diffusion training dynamic.

---

### Scope Limitations

- The tool is designed specifically for LoRA training, not full-model fine-tuning.
- It does not account for optimizer-internal adaptive dynamics beyond static modifiers.
- Dataset quality, caption entropy, and regularization techniques are not modeled.
- Extremely unconventional training setups (e.g., very large ranks, exotic schedulers, highly imbalanced datasets) may deviate from predicted behavior.
- Stability scoring reflects modeled curvature risk, not guaranteed convergence outcomes.

---

### Calibration Limitations

Calibration constants are chosen to preserve relative scaling consistency rather than optimize for maximum possible learning rate. As such:

- The recommended learning rate should be interpreted as a stable analytical baseline.
- Aggressive experimentation beyond the computed optimum is possible but may incur increased instability risk.

<br>

## License

This project is licensed under the MIT License — see the LICENSE file for details.
