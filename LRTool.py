import math
import json
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
from datetime import datetime
import sys
import os
results = {}

# GLOBAL CONSTANTS

ACCUM_EFFICIENCY = 0.7
UNIVERSAL_UNDERSHOOT_K = 0.65
EFFICIENCY_SENSITIVITY = 2.0
COSINE_FLOOR = 0.10
REX_DECAY_K = 4.0

if getattr(sys, 'frozen', False):
    # Running as EXE
    RESOURCE_DIR = sys._MEIPASS
    APP_DIR = os.path.dirname(sys.executable)
else:
    # Running as script
    RESOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
    APP_DIR = RESOURCE_DIR

DEFAULT_PROFILE_PATH = os.path.join(APP_DIR, "LRTool.default.json")
WINDOW_STATE_PATH = os.path.join(APP_DIR, "LRTool.state")

# MODEL DATA

ARCH_MODIFIERS = {
    "classic_ldm": 1.0,
    "sdxl_backbone": 1.2,
    "flux": 1.25,
    "distilled_sdxl": 1.45,
    "lightning_sdxl": 1.55,
    "highly_compressed": 2.2
}

ARCH_RESOLUTION_EXPONENT = {
    "classic_ldm": 0.95,
    "sdxl_backbone": 0.925,
    "flux": 0.91,
    "distilled_sdxl": 0.90,
    "lightning_sdxl": 0.875,
    "highly_compressed": 0.85
}

MODEL_DATA = {
    "SD1.5": {"native_res": 512, "base_energy": 0.000630, "arch": "classic_ldm"},
    "SD2.1": {"native_res": 768, "base_energy": 0.000665, "arch": "classic_ldm"},
    "SDXL": {"native_res": 1024, "base_energy": 0.000712, "arch": "sdxl_backbone"},
    "Pony (SDXL)": {"native_res": 1024, "base_energy": 0.000690, "arch": "sdxl_backbone"},
    "SDXL Turbo": {"native_res": 1024, "base_energy": 0.000660, "arch": "distilled_sdxl"},
    "SDXL Lightning": {"native_res": 1024, "base_energy": 0.000650, "arch": "lightning_sdxl"},
    "FLUX.1": {"native_res": 1024, "base_energy": 0.000657, "arch": "flux"},
    "FLUX.2 Dev": {"native_res": 1024, "base_energy": 0.000657, "arch": "flux"},
    "Z-Image": {"native_res": 1024, "base_energy": 0.000602, "arch": "highly_compressed"},
}

REFERENCE_ENERGY = sum(
    m["base_energy"] for m in MODEL_DATA.values()
) / len(MODEL_DATA)

# OPTIMIZER MODIFIERS

OPTIMIZER_MODS = {
    "AdamW": 1.00, "AdamW (8-Bit)": 1.02,
    "Adam": 1.02, "Adam (8-Bit)": 1.04,
    "Adagrad": 1.08, "Adagrad (8-Bit)": 1.10,
    "RMSprop": 1.05, "RMSprop (8-Bit)": 1.07,
    "Adafactor": 0.90,
    "AdEMAMix": 0.97, "AdEMAMix (8-Bit)": 1.00,
    "Simplified AdEMAMix": 0.98,
    "SGD": 0.85, "SGD (8-Bit)": 0.88,
    "Lars": 0.88, "Lars (8-Bit)": 0.90,
    "Lam": 0.90, "Lam (8-Bit)": 0.92,
    "Lion": 0.90, "Lion (8-Bit)": 0.93,
    "Muon": 0.80, "AdaMuon": 0.85,
    "CAME": 0.95, "CAME (8-Bit)": 0.98,
    "Adopt": 0.92,
    "Tiger": 0.88
}

OBJECTIVE_ENERGY_MOD = {
    "Style": 0.93,
    "Concept": 0.97,
    "Character": 1.00,
    "Fidelity": 1.05,
}

OBJECTIVE_SENSITIVITY = {
    "Style": 0.6,
    "Concept": 1.0,
    "Character": 1.4,
    "Fidelity": 2.5
}

# SCHEDULER

def scheduler_multiplier(scheduler, step, total_steps):
    t = step / total_steps

    if scheduler == "Constant":
        return 1.0
    elif scheduler == "Linear":
        if total_steps <= 1:
            return 1.0
        return max(0.0, 1 - t)
    elif scheduler == "Cosine":
        return COSINE_FLOOR + (1 - COSINE_FLOOR) * math.cos(math.pi * t)
    elif scheduler == "Cosine (Restarts)":
        cycle = max(1, total_steps // 2)
        local_step = step % cycle
        return COSINE_FLOOR + (1 - COSINE_FLOOR) * math.cos(math.pi * local_step / cycle)
    elif scheduler == "Cosine (Hard Restarts)":
        cycle = max(1, total_steps // 2)
        local_step = step % cycle
        return max(0.0, math.cos(math.pi * local_step / cycle))
    elif scheduler == "Rex":
        return math.exp(-REX_DECAY_K * t)
    elif scheduler == "Adafactor":
        return 1 / math.sqrt(step)
    else:
        return 1.0


def compute_scheduler_rms(scheduler, total_steps, warmup_fraction):
    warmup_steps = int(total_steps * warmup_fraction)
    sum_sq = 0.0

    for step in range(1, total_steps + 1):
        base = scheduler_multiplier(scheduler, step, total_steps)

        if warmup_steps > 0 and step <= warmup_steps:
            warmup_factor = step / warmup_steps
        else:
            warmup_factor = 1.0

        s = warmup_factor * base
        sum_sq += s**2

    return math.sqrt(sum_sq / total_steps)

# CALCULATION

def calculate_lr(*args):
    try:
        steps = max(float(steps_var.get()), 1)
        batch = max(float(batch_var.get()), 1)
        grad_accum = max(float(grad_accum_var.get()), 1)
        images = max(float(images_var.get()), 1)
        rank = max(float(rank_var.get()), 1)
        alpha = max(float(alpha_var.get()), 1e-6)
        resolution = max(float(resolution_var.get()), 1)

        model = model_var.get()
        objective = objective_var.get()
        optimizer = optimizer_var.get()
        scheduler = scheduler_var.get()

        warmup_percent = int(warmup_var.get().replace("%", ""))
        warmup_fraction = warmup_percent / 100.0

        model_info = MODEL_DATA[model]

        effective_batch = batch + ACCUM_EFFICIENCY * (grad_accum - 1)
        exposure = ((steps * effective_batch) / images) ** 0.5
        capacity = math.sqrt(rank / alpha)
        
        p = ARCH_RESOLUTION_EXPONENT[model_info["arch"]]
        resolution_scale = (resolution / model_info["native_res"]) ** p

        scheduler_factor = compute_scheduler_rms(
            scheduler, int(steps), warmup_fraction
        )
        
        target_energy = (
            model_info["base_energy"] *
            OBJECTIVE_ENERGY_MOD[objective]
        )

        base_lr = target_energy / (
            scheduler_factor *
            exposure *
            capacity *
            resolution_scale *
            OPTIMIZER_MODS[optimizer]
        )

        offset = slider_offset.get()
        adjusted_lr = base_lr * (1 + offset)

        lr = adjusted_lr
        slider_percent_label.config(text=f"{offset*100:+.0f}%")

        energy = (
            lr *
            scheduler_factor *
            exposure *
            capacity *
            resolution_scale *
            OPTIMIZER_MODS[optimizer]
        )

        deviation = abs(energy - target_energy) / target_energy

        # Stability
        curvature = math.sqrt(model_info["base_energy"] / REFERENCE_ENERGY)
        arch_modifier = ARCH_MODIFIERS[model_info["arch"]]
        k_model = 1.0 * curvature * arch_modifier
        noise_factor = max(0.35, math.sqrt(1 / effective_batch))

        if energy > target_energy:
            k = k_model * OBJECTIVE_SENSITIVITY[objective] * noise_factor
        else:
            k = UNIVERSAL_UNDERSHOOT_K

        stability = 100 * math.exp(-k * deviation**2)

        # Efficiency
        rho = energy / target_energy
        efficiency = 100 * math.exp(-EFFICIENCY_SENSITIVITY * (rho - 1)**2)

        # Update UI
        lr_label.config(text=f"{lr:.2e}")
        lr_numeric_label.config(text=f"({lr:.6f})")
        
        if stability >= 85:
            stability_label.config(text=f"Stability Confidence: {stability:.1f}%", foreground="green")
        elif stability >= 65:
            stability_label.config(text=f"Stability Confidence: {stability:.1f}%", foreground="orange")
        else:
            stability_label.config(text=f"Stability Confidence: {stability:.1f}%", foreground="red")

        if efficiency >= 90:
            efficiency_label.config(text=f"Convergence Efficiency: {efficiency:.1f}%", foreground="green")
        elif efficiency >= 60:
            efficiency_label.config(text=f"Convergence Efficiency: {efficiency:.1f}%", foreground="orange")
        else:
            efficiency_label.config(text=f"Convergence Efficiency: {efficiency:.1f}%", foreground="red")
        global results

        results = {
            # Core LR
            "recommended_lr": base_lr,
            "adjusted_lr": adjusted_lr,
            "base_lr": base_lr,

            # Energy
            "target_energy": target_energy,
            "delivered_energy": energy,
            "energy_ratio": rho,
            "deviation": deviation,

            # Exposure / Capacity
            "effective_batch": effective_batch,
            "exposure": exposure,
            "capacity": capacity,
            "resolution_scale": resolution_scale,

            # Scheduler / Optimizer
            "scheduler_factor": scheduler_factor,
            "optimizer_modifier": OPTIMIZER_MODS[optimizer],
            "warmup_fraction": warmup_fraction,
            "warmup_steps": int(steps * warmup_fraction),

            # Stability
            "curvature_factor": curvature,
            "objective_sensitivity": OBJECTIVE_SENSITIVITY[objective],
            "undershoot_k": UNIVERSAL_UNDERSHOOT_K,
            "overshoot_k": k_model * OBJECTIVE_SENSITIVITY[objective] * noise_factor,

            # Scores
            "stability_score": stability,
            "efficiency_score": efficiency,

            # Bands
            "risk_band": (
                "Safe" if stability >= 85
                else "Caution" if stability >= 65
                else "Risky"
            ),
            "efficiency_band": (
                "Optimal" if efficiency >= 90
                else "Suboptimal" if efficiency >= 60
                else "Inefficient"
            ),
        }
        
    except Exception as e:
        print("Calculation error:", e)
        lr_label.config(text="Error in input")

# SAVE/LOAD PROFILE

def build_full_profile():
    return {
        "meta": {
            "app": "LRTool",
            "version": "1.0.0"
        },
        "configuration": {
            "steps": steps_var.get(),
            "batch": batch_var.get(),
            "grad_accum": grad_accum_var.get(),
            "images": images_var.get(),
            "rank": rank_var.get(),
            "alpha": alpha_var.get(),
            "resolution": resolution_var.get(),
            "model": model_var.get(),
            "objective": objective_var.get(),
            "optimizer": optimizer_var.get(),
            "scheduler": scheduler_var.get(),
            "warmup": warmup_var.get(),
            "slider_offset": slider_offset.get()
        },
        "statistics": results.copy(),
    }

def save_profile():
    try:
        profile = build_full_profile()

        file_path = filedialog.asksaveasfilename(
            initialdir=APP_DIR,
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json")],
            title="Save Profile"
        )

        if not file_path:
            return

        with open(file_path, "w") as f:
            json.dump(profile, f, indent=4)

    except Exception as e:
        messagebox.showerror("Save Error", str(e))

def load_profile():
    try:
        file_path = filedialog.askopenfilename(
            initialdir=APP_DIR,
            filetypes=[("JSON Files", "*.json")],
            title="Load Profile"
        )

        if not file_path:
            return

        with open(file_path, "r") as f:
            profile = json.load(f)
            
        config = profile.get("configuration", {})

        steps_var.set(config.get("steps", steps_var.get()))
        batch_var.set(config.get("batch", batch_var.get()))
        grad_accum_var.set(config.get("grad_accum", grad_accum_var.get()))
        images_var.set(config.get("images", images_var.get()))
        rank_var.set(config.get("rank", rank_var.get()))
        alpha_var.set(config.get("alpha", alpha_var.get()))
        resolution_var.set(config.get("resolution", resolution_var.get()))
        model_var.set(config.get("model", model_var.get()))
        objective_var.set(config.get("objective", objective_var.get()))
        optimizer_var.set(config.get("optimizer", optimizer_var.get()))
        scheduler_var.set(config.get("scheduler", scheduler_var.get()))
        warmup_var.set(config.get("warmup", warmup_var.get()))
        slider_offset.set(config.get("slider_offset", 0.0))

        calculate_lr()

    except Exception as e:
        messagebox.showerror("Load Error", str(e))

def load_default_profile_if_exists():
    if not os.path.exists(DEFAULT_PROFILE_PATH):
        return

    try:
        with open(DEFAULT_PROFILE_PATH, "r") as f:
            profile = json.load(f)

        config = profile.get("configuration", {})

        steps_var.set(config.get("steps", steps_var.get()))
        batch_var.set(config.get("batch", batch_var.get()))
        grad_accum_var.set(config.get("grad_accum", grad_accum_var.get()))
        images_var.set(config.get("images", images_var.get()))
        rank_var.set(config.get("rank", rank_var.get()))
        alpha_var.set(config.get("alpha", alpha_var.get()))
        resolution_var.set(config.get("resolution", resolution_var.get()))
        model_var.set(config.get("model", model_var.get()))
        objective_var.set(config.get("objective", objective_var.get()))
        optimizer_var.set(config.get("optimizer", optimizer_var.get()))
        scheduler_var.set(config.get("scheduler", scheduler_var.get()))
        warmup_var.set(config.get("warmup", warmup_var.get()))
        slider_offset.set(config.get("slider_offset", 0.0))

        calculate_lr()

    except Exception as e:
        print("Failed to load default profile:", e)

# WINDOW STATE

def save_window_state():
    try:
        geometry = root.geometry()

        with open(WINDOW_STATE_PATH, "w") as f:
            json.dump({"geometry": geometry}, f)

    except Exception as e:
        print("Failed to save window state:", e)

def restore_window_state():
    if not os.path.exists(WINDOW_STATE_PATH):
        return False

    try:
        with open(WINDOW_STATE_PATH, "r") as f:
            data = json.load(f)

        geometry = data.get("geometry")
        if geometry:
            root.geometry(geometry)
            return True

    except Exception as e:
        print("Failed to restore window state:", e)

    return False

# UI

root = tk.Tk()
root.withdraw()

restored = restore_window_state()

root.title("LRTool v1.0.0")
root.iconbitmap(os.path.join(RESOURCE_DIR, "LRTool.ico"))
root.minsize(300, 670)

main = ttk.Frame(root, padding=20)
main.pack(fill="both", expand=True)

form_frame = ttk.Frame(main)
form_frame.pack()

current_row = 0

def add_row(label, widget):
    global current_row
    ttk.Label(form_frame, text=label, anchor="e", width=12).grid(
        row=current_row, column=0, sticky="e", padx=(0,12), pady=4)
    widget.grid(row=current_row, column=1, sticky="w", pady=4)
    current_row += 1

# VARIABLES

resolution_var = tk.StringVar(value="1024")
images_var = tk.StringVar(value="40")
steps_var = tk.StringVar(value="2000")
batch_var = tk.StringVar(value="1")
grad_accum_var = tk.StringVar(value="2")
rank_var = tk.StringVar(value="32")
alpha_var = tk.StringVar(value="16")
model_var = tk.StringVar(value="SDXL")
optimizer_var = tk.StringVar(value="AdamW")
scheduler_var = tk.StringVar(value="Cosine")
warmup_var = tk.StringVar(value="10%")
objective_var = tk.StringVar(value="Character")
slider_offset = tk.DoubleVar(value=0.0)

# INPUTS

add_row("Base Model:", ttk.Combobox(form_frame, textvariable=model_var,
        values=list(MODEL_DATA.keys()), state="readonly", width=16))
add_row("Resolution:", ttk.Entry(form_frame, textvariable=resolution_var, width=18))
add_row("Total Steps:", ttk.Entry(form_frame, textvariable=steps_var, width=18))
add_row("Images:", ttk.Entry(form_frame, textvariable=images_var, width=18))
add_row("Batch Size:", ttk.Entry(form_frame, textvariable=batch_var, width=18))
add_row("Grad Accum:", ttk.Entry(form_frame, textvariable=grad_accum_var, width=18))
add_row("Rank:", ttk.Entry(form_frame, textvariable=rank_var, width=18))
add_row("Alpha:", ttk.Entry(form_frame, textvariable=alpha_var, width=18))
add_row("Optimizer:", ttk.Combobox(form_frame, textvariable=optimizer_var,
        values=list(OPTIMIZER_MODS.keys()), state="readonly", width=16))
add_row("Scheduler:", ttk.Combobox(form_frame, textvariable=scheduler_var,
        values=["Constant","Linear","Cosine",
                "Cosine (Restarts)","Cosine (Hard Restarts)",
                "Rex","Adafactor"],
        state="readonly", width=16))
add_row("Warmup:", ttk.Combobox(form_frame, textvariable=warmup_var,
        values=["0%","5%","10%","15%","20%","25%"], state="readonly", width=16))
add_row("Objective:", ttk.Combobox(form_frame, textvariable=objective_var,
        values=list(OBJECTIVE_ENERGY_MOD.keys()), state="readonly", width=16))

ttk.Separator(main).pack(fill="x", pady=12)

# SLIDER

slider = ttk.Scale(
    main,
    from_=-1.0,
    to=1.0,
    variable=slider_offset,
    command=lambda e: calculate_lr()
)

range_frame = ttk.Frame(main)
range_frame.pack(fill="x")

left_label = ttk.Label(range_frame, text="-100%")
left_label.pack(side="left")

right_label = ttk.Label(range_frame, text="+100%")
right_label.pack(side="right")

slider.pack(fill="x", pady=4)
slider_percent_label = ttk.Label(main, font=("Segoe UI", 9))
slider_percent_label.pack()

# SLIDER RESET

def reset_slider(event=None):
    slider_offset.set(0.0)
    calculate_lr()

slider.bind("<Double-Button-1>", reset_slider)

# OUTPUT

lr_frame = ttk.Frame(main)
lr_frame.pack(pady=6)

lr_label = ttk.Label(lr_frame, font=("Segoe UI", 11, "bold"))
lr_label.pack()

lr_numeric_label = ttk.Label(lr_frame, font=("Segoe UI", 9))
lr_numeric_label.pack()

stability_label = ttk.Label(main)
stability_label.pack(pady=2)

efficiency_label = ttk.Label(main)
efficiency_label.pack(pady=2)

# PROFILE BUTTONS

button_frame = ttk.Frame(main)
button_frame.pack(pady=10)

save_button = ttk.Button(button_frame, text="Save Profile",
    command=save_profile, width=15)
save_button.pack(side="left", padx=2)

load_button = ttk.Button(button_frame, text="Load Profile",
    command=load_profile, width=15)
load_button.pack(side="left", padx=2)

# DEFAULT BUTTONS

def save_as_default_profile():
    try:
        profile = build_full_profile()

        with open(DEFAULT_PROFILE_PATH, "w") as f:
            json.dump(profile, f, indent=4)

        print("Default profile saved.")

    except Exception as e:
        print("Failed to save default profile:", e)

defaults_frame = ttk.Frame(main)
defaults_frame.pack(pady=0)

save_defaults_button = ttk.Button(
    defaults_frame,
    text="Set Defaults",
    command=save_as_default_profile,
    width=15
)
save_defaults_button.pack(side="left", padx=2)

load_defaults_button = ttk.Button(
    defaults_frame,
    text="Load Defaults",
    command=load_default_profile_if_exists,
    width=15
)
load_defaults_button.pack(side="right", padx=2)

# BIND CALCULATION

for var in [steps_var,batch_var,grad_accum_var,images_var,
            rank_var,alpha_var,resolution_var,
            model_var,objective_var,optimizer_var,
            scheduler_var,warmup_var]:
    var.trace_add("write", calculate_lr)
    
#WINDOW POSITION

def center_window(root):
    root.update_idletasks()

    width = root.winfo_width()
    height = root.winfo_height()

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)

    root.geometry(f"{width}x{height}+{x}+{y}")

# STARTUP

if not restored:
    center_window(root)

load_default_profile_if_exists()
calculate_lr()

root.protocol("WM_DELETE_WINDOW", lambda: (save_window_state(), root.destroy()))

root.update()
root.deiconify()
root.mainloop()