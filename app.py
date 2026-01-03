import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Thermal Engineering Lab", layout="wide")

# --- COMPREHENSIVE MATERIAL DATABASE (Engineering Grade) ---
# alpha: Thermal Diffusivity (mm^2/s) | k: Thermal Conductivity (W/m·K)
MATERIALS = {
    "--- AEROSPACE SUPERALLOYS ---": {"alpha": 0, "k": 0},
    "Inconel 718 (Nickel-Chrome)": {"alpha": 2.9, "k": 11.4},
    "Haynes 230 (Ni-Cr-W-Mo)": {"alpha": 3.2, "k": 12.0},
    "Titanium 6Al-4V (Grade 5)": {"alpha": 2.9, "k": 6.7},
    "Hastelloy X": {"alpha": 2.8, "k": 9.1},
    "--- REFRACTORY METALS ---": {"alpha": 0, "k": 0},
    "Tungsten (W)": {"alpha": 67.0, "k": 173.0},
    "Molybdenum (Mo)": {"alpha": 54.0, "k": 138.0},
    "Tantalum (Ta)": {"alpha": 24.0, "k": 57.5},
    "--- PURE METALS & HEAT SPREADERS ---": {"alpha": 0, "k": 0},
    "Diamond (C)": {"alpha": 1120.0, "k": 2200.0},
    "Copper (Oxygen Free)": {"alpha": 117.0, "k": 391.0},
    "Aluminum 6061": {"alpha": 64.0, "k": 167.0},
    "--- STRUCTURAL & INSULATION ---": {"alpha": 0, "k": 0},
    "Carbon Steel (AISI 1020)": {"alpha": 14.7, "k": 51.9},
    "Alumina Ceramic (Al2O3)": {"alpha": 8.1, "k": 30.0},
    "Zirconia (ZrO2)": {"alpha": 0.6, "k": 2.0},
    "Silica Aerogel": {"alpha": 0.01, "k": 0.02}
}

# --- HEADER ---
st.title(" Aerospace & High-Temp Thermal Solver")
st.markdown("""
This simulator solves the **2D Transient Heat Equation** using the Finite Difference Method. 
It maps heat propagation through specialized alloys and insulators used in extreme environments.
""")

# --- SIDEBAR: ENGINEERING PARAMETERS ---
st.sidebar.header(" Engineering Controls")
selectable_materials = [m for m in MATERIALS.keys() if "---" not in m]
selected_name = st.sidebar.selectbox("Select Specimen Material", selectable_materials)

alpha = MATERIALS[selected_name]["alpha"]
k_val = MATERIALS[selected_name]["k"]

st.sidebar.info(f"**Properties of {selected_name}:**\n- α: {alpha} mm²/s\n- k: {k_val} W/m·K")

# Simulation Tuning
grid_size = 50
time_steps = st.sidebar.slider("Simulation Steps", 50, 2000, 500)
source_temp = st.sidebar.slider("Heat Source Temp (°C)", 100, 2500, 1200)
ambient_temp = 25.0

# --- ENGINEERING OVERSIGHT: NUMERICAL STABILITY CHECK ---
# Fourier Number (Fo) must be <= 0.25 for 2D stability
dx = 1.0
dt = 0.001
fo_number = (alpha * dt) / (dx**2)

if fo_number > 0.25:
    st.sidebar.error(f" Numerical Instability! Fourier Number: {fo_number:.3f} (Max 0.25). Reduce 'alpha' or 'dt'.")
else:
    st.sidebar.success(f" Simulation Stable. Fourier Number: {fo_number:.3f}")

# --- SOLVER ENGINE ---
def run_simulation(steps, alpha_val, source_t):
    T = np.full((grid_size, grid_size), ambient_temp)
    # Define a complex heat source (Internal component)
    T[20:30, 20:30] = source_t
    
    for _ in range(steps):
        Tn = T.copy()
        # Laplace Operator (Centered Difference)
        T[1:-1, 1:-1] = Tn[1:-1, 1:-1] + alpha_val * dt * (
            (Tn[1:-1, 2:] - 2*Tn[1:-1, 1:-1] + Tn[1:-1, 0:-2]) +
            (Tn[2:, 1:-1] - 2*Tn[1:-1, 1:-1] + Tn[0:-2, 1:-1])
        )
        # Boundary Conditions
        T[20:30, 20:30] = source_t # Maintain Heat Source
        T[0,:]=T[1,:]; T[-1,:]=T[-2,:]; T[:,0]=T[:,1]; T[:,-1]=T[:,-2] # Insulated edges
    return T

# --- EXECUTION & VISUALIZATION ---
if st.button(" Execute Analysis"):
    T_final = run_simulation(time_steps, alpha, source_temp)
    
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("2D Thermal Gradient & Heat Flux")
        grad_y, grad_x = np.gradient(T_final)
        flux_x, flux_y = -k_val * grad_x, -k_val * grad_y
        
        fig_heat, ax_heat = plt.subplots(figsize=(10, 7))
        mesh = ax_heat.pcolormesh(T_final, cmap='hot', shading='auto')
        plt.colorbar(mesh, label="Temp (°C)")
        
        # Heat Flux Quiver
        skip = (slice(None, None, 4), slice(None, None, 4))
        ax_heat.quiver(np.arange(grid_size)[skip[1]], np.arange(grid_size)[skip[0]], 
                      flux_x[skip], flux_y[skip], color='cyan', alpha=0.3)
        st.pyplot(fig_heat)

    with col2:
        st.subheader("Cross-Sectional Analysis")
        # Center-line Temperature Profile
        center_line = T_final[grid_size // 2, :]
        fig_line, ax_line = plt.subplots(figsize=(6, 8.5))
        ax_line.plot(center_line, color='red', lw=2)
        ax_line.set_xlabel("Node Index (Distance)")
        ax_line.set_ylabel("Temperature (°C)")
        ax_line.grid(True, alpha=0.3)
        ax_line.set_title("Horizontal Center-line Profile")
        st.pyplot(fig_line)

    # --- ENGINEERING OVERSIGHT REPORT ---
    st.markdown("---")
    st.subheader(" Engineering Oversight Report")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Peak Specimen Temp", f"{np.max(T_final):.1f} °C")
    c2.metric("Mean System Temp", f"{np.mean(T_final):.1f} °C")
    c3.metric("Max Heat Flux", f"{np.max(np.sqrt(flux_x**2 + flux_y**2)):.0e} W/m²")

    if np.max(T_final) > 1300 and "Steel" in selected_name:
        st.warning(f" ALERT: Temperature exceeds melting point for {selected_name}!")
    elif alpha < 1.0:
        st.write(f" {selected_name} is acting as an effective thermal barrier.")
