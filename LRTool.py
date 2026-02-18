import math
import tkinter as tk
from tkinter import ttk

# ==========================================================
# MODEL DATA
# ==========================================================

MODEL_DATA = {
    "SD1.5": {"native_res": 512, "cosine_center": 5.75e-5},
    "SDXL": {"native_res": 1024, "cosine_center": 6.5e-5},
    "FLUX.1": {"native_res": 1024, "cosine_center": 6.0e-5},
    "FLUX.2 Dev": {"native_res": 1024, "cosine_center": 6.0e-5},
    "Z-Image": {"native_res": 1024, "cosine_center": 5.5e-5},
}

MODEL_ENERGY_TARGETS = {
    "SD1.5": 0.0045,
    "SDXL": 0.0050,
    "FLUX.1": 0.0053,
    "FLUX.2 Dev": 0.0053,
    "Z-Image": 0.0046,
}

MODEL_QUALITY_SENSITIVITY = {
    "SD1.5": 120,
    "SDXL": 120,
    "FLUX.1": 115,
    "FLUX.2 Dev": 115,
    "Z-Image": 130,
}

OBJECTIVE_SHIFTS = {
    "Style": -0.08,
    "Concept": -0.03,
    "Character": 0.0,
    "Sharpness": 0.06,
}

OPTIMIZER_MODS = {
    "Adafactor":0.75,
    "Adagrad":1.08,
    "Adagrad (8-Bit)":1.10,
    "Adam":1.02,
    "Adam (8-Bit)":1.05,
    "AdamW":1.00,
    "AdamW (8-Bit)":1.05,
    "AdEMAMix":0.97,
    "AdEMAMix (8-Bit)":1.00,
    "Adopt":0.90,
    "LAM":0.85,
    "LAM (8-Bit)":0.90,
    "LARS":0.85,
    "LARS (8-Bit)":0.90,
    "LION":0.90,
    "LION (8-Bit)":0.95,
    "Muon":0.80
}

K0 = 0.004

# ==========================================================
# SCORING
# ==========================================================

def compute_stability(lr, exposure, capacity, model):
    energy = lr * exposure * capacity
    target = MODEL_ENERGY_TARGETS[model]
    deviation = abs(energy - target) / target
    score = max(0, 100 - deviation * 110)
    color = "green" if score > 85 else "orange" if score > 65 else "red"
    return score, color

def compute_quality(lr, model, objective):
    model_center = MODEL_DATA[model]["cosine_center"]
    objective_shift = OBJECTIVE_SHIFTS[objective]
    center = model_center * (1 + objective_shift)

    sensitivity = MODEL_QUALITY_SENSITIVITY[model]

    deviation = abs(lr - center) / center
    score = max(0, 100 - deviation * sensitivity)
    color = "green" if score > 85 else "orange" if score > 65 else "red"
    return score, color

# ==========================================================
# STANDARD MODE CENTER
# ==========================================================

def compute_standard_center(model, objective, scheduler, resolution, steps, rank):

    native_res = MODEL_DATA[model]["native_res"]
    cosine_center = MODEL_DATA[model]["cosine_center"]

    cosine_center *= (1 + OBJECTIVE_SHIFTS[objective])

    base_center = cosine_center if scheduler == "Cosine" else cosine_center * 0.8

    resolution_shift = 1 / (math.sqrt(resolution / native_res) ** 0.5)

    if steps < 1000:
        step_shift = 1.08
    elif steps > 3000:
        step_shift = 0.92
    else:
        step_shift = 1.0

    # Rank micro adjustment
    rank_factor = 1 + ((32 - rank) / 32) * 0.05

    return base_center * resolution_shift * step_shift * rank_factor

# ==========================================================
# MAIN CALCULATION
# ==========================================================

def calculate_lr(*args):
    try:
        steps = float(steps_var.get())
        batch = float(batch_var.get())
        images = float(images_var.get())
        rank = float(rank_var.get())
        alpha = float(alpha_var.get())
        resolution = float(resolution_var.get())
        grad_accum = float(grad_accum_var.get())

        model = model_var.get()
        objective = objective_var.get()
        optimizer = optimizer_var.get()
        scheduler = scheduler_var.get()
        warmup = warmup_var.get()

        exposure = (steps * (batch * grad_accum)) / images
        capacity = math.sqrt(rank / alpha)

        if advanced_mode_var.get():

            scheduler_mod = 1.25 if scheduler == "Cosine" else 1.0
            warmup_mod = {"None":1.0, "5%":1.05, "10%":1.1}[warmup]
            resolution_mod = math.sqrt(
                resolution / MODEL_DATA[model]["native_res"]
            )

            step_mod = 1.15 if steps < 1000 else 1.0 if steps <= 3000 else 0.9

            rank_factor = 1 + ((32 - rank) / 32) * 0.05

            lr = (
                K0 *
                scheduler_mod *
                warmup_mod *
                OPTIMIZER_MODS[optimizer] *
                resolution_mod *
                step_mod *
                rank_factor
            ) / (exposure * capacity)

            lr *= (1 + slider_offset.get())

        else:

            lr = compute_standard_center(
                model, objective, scheduler, resolution, steps, rank
            ) * (1 + slider_offset.get())

        if show_effective_var.get() and scheduler == "Cosine":
            display_lr = lr * 0.63
        else:
            display_lr = lr

        result_text.set(
            f"Recommended LR:\n{display_lr:.2e} ({display_lr:.8f})"
        )

        stability_score, stability_color = compute_stability(
            lr, exposure, capacity, model
        )

        quality_score, quality_color = compute_quality(
            lr, model, objective
        )

        stability_label.config(
            text=f"Stability Score: {stability_score:.1f}%",
            foreground=stability_color
        )

        quality_label.config(
            text=f"Quality Score: {quality_score:.1f}%",
            foreground=quality_color
        )

    except:
        pass

# ==========================================================
# GUI
# ==========================================================

root = tk.Tk()
root.title("LRTool v1.0.1")
root.geometry("500x800")
root.configure(bg="#c8c8c8")

style = ttk.Style()
style.theme_use("clam")
style.configure("TEntry", fieldbackground="#e0e0e0")
style.configure("TCombobox", fieldbackground="white", background="white")

main = ttk.Frame(root, padding=20)
main.pack(fill="both", expand=True)

def add_entry(label, var):
    ttk.Label(main, text=label).pack(anchor="w")
    entry = ttk.Entry(main, textvariable=var)
    entry.pack(fill="x", pady=4)
    var.trace_add("write", calculate_lr)

steps_var = tk.StringVar(value="1200")
batch_var = tk.StringVar(value="2")
images_var = tk.StringVar(value="40")
rank_var = tk.StringVar(value="32")
alpha_var = tk.StringVar(value="16")
resolution_var = tk.StringVar(value="1024")
grad_accum_var = tk.StringVar(value="1")

add_entry("Total Steps:", steps_var)
add_entry("Batch Size:", batch_var)
add_entry("Gradient Accumulation:", grad_accum_var)
add_entry("Number of Images:", images_var)
add_entry("LoRA Rank:", rank_var)
add_entry("LoRA Alpha:", alpha_var)
add_entry("Resolution:", resolution_var)

model_var = tk.StringVar(value="SDXL")
ttk.Label(main, text="Base Model").pack(anchor="w")
ttk.Combobox(main, textvariable=model_var,
             values=list(MODEL_DATA.keys()),
             state="readonly").pack(fill="x")
model_var.trace_add("write", calculate_lr)

objective_var = tk.StringVar(value="Character")
ttk.Label(main, text="Training Objective").pack(anchor="w")
ttk.Combobox(main, textvariable=objective_var,
             values=list(OBJECTIVE_SHIFTS.keys()),
             state="readonly").pack(fill="x")
objective_var.trace_add("write", calculate_lr)

optimizer_var = tk.StringVar(value="AdamW")
ttk.Label(main, text="Optimizer").pack(anchor="w")
ttk.Combobox(main, textvariable=optimizer_var,
             values=sorted(OPTIMIZER_MODS.keys()),
             state="readonly").pack(fill="x")
optimizer_var.trace_add("write", calculate_lr)

scheduler_var = tk.StringVar(value="Cosine")
ttk.Label(main, text="Scheduler").pack(anchor="w")
ttk.Combobox(main, textvariable=scheduler_var,
             values=["Constant", "Cosine"],
             state="readonly").pack(fill="x")
scheduler_var.trace_add("write", calculate_lr)

warmup_var = tk.StringVar(value="5%")
ttk.Label(main, text="Warmup").pack(anchor="w")
ttk.Combobox(main, textvariable=warmup_var,
             values=["None", "5%", "10%"],
             state="readonly").pack(fill="x")
warmup_var.trace_add("write", calculate_lr)

advanced_mode_var = tk.BooleanVar(value=False)
ttk.Checkbutton(main,
    text="Advanced Mode (Full Mathematical Modeling)",
    variable=advanced_mode_var,
    command=calculate_lr).pack(pady=10)

show_effective_var = tk.BooleanVar(value=False)
ttk.Checkbutton(main,
    text="Show Effective Average LR (Scheduler Adjusted)",
    variable=show_effective_var,
    command=calculate_lr).pack()

ttk.Label(main, text="Adjustment (±50%)").pack(anchor="w")
slider_offset = tk.DoubleVar(value=0.0)
slider_offset.trace_add("write", calculate_lr)

slider = ttk.Scale(main,
                   from_=-0.50,
                   to=0.50,
                   variable=slider_offset)
slider.pack(fill="x")

def reset_slider(event=None):
    slider_offset.set(0.0)
    calculate_lr()

slider.bind("<Double-Button-1>", reset_slider)

percent_frame = ttk.Frame(main)
percent_frame.pack(fill="x")
ttk.Label(percent_frame, text="-50%").pack(side="left")
ttk.Label(percent_frame, text="0%").pack(side="left", expand=True)
ttk.Label(percent_frame, text="+50%").pack(side="right")

result_text = tk.StringVar()
ttk.Label(main, textvariable=result_text).pack(pady=10)

stability_label = ttk.Label(main)
stability_label.pack()

quality_label = ttk.Label(main)
quality_label.pack()

calculate_lr()
root.mainloop()
