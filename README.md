#  LRTool

<p align="center">
LRTool is simple utility designed to calculate stable, high-quality learning rates for LoRA training workflows.
</p>

---

<p align="center">
<img width="500" height="780" alt="LRTool" src="https://github.com/Highstate/LRTool/blob/main/images/LRTool.png"/>
</p>

##  📌 Overview

**Supported Models**:
-   ✅ SD1.5
-   ✅ SDXL
-   ✅ FLUX.1
-   ✅ FLUX.2 Dev
-   ✅ Z-Image

**Features**:
-   ✅ Empirical sweet-spot modeling
-   ✅ Mathematical exposure modeling
-   ✅ Model-aware resolution scaling
-   ✅ Optimizer-aware adjustment
-   ✅ Dual scoring diagnostics

The goal: eliminate LR guesswork and prevent:

-   🔴 Overcooked textures
-   🔴 Plastic faces
-   🔴 Identity drift
-   🔴 Underfitting
-   🔴 Instability



##  🚀 Installation

-  Python 3.9+
-  Windows executable provided (No python required)


##  🏗 Application Modes

### **Standard Mode (Default)** - Empirically centered.

Learning rate is determined by:

-   Base Model
-   Training Objective
-   Scheduler
-   Resolution
-   Training length regime  



### **Advanced Mode** - Full mathematical modeling.

Advanced Mode models:

-   Exposure scaling
-   Rank/Alpha scaling
-   Optimizer behavior
-   Warmup steps
-   Resolution
-   Training length regimes  



### **Standard vs Advanced**

| Feature                | Standard | Advanced              |
| ---------------------- | -------- | --------------------- |
| Empirically centered   | ✅        | ❌                     |
| Model-aware resolution | ✅        | ✅                     |
| Objective shift        | ✅        | ❌                     |
| Optimizer scaling      | ❌        | ✅                     |
| Warmup modeling        | ❌        | ✅                     |
| Exposure scaling       | ❌        | ✅                     |
| Rank/Alpha scaling     | ❌        | ✅                     |
| Safe defaults          | ✅        | ⚠ Depends              |
