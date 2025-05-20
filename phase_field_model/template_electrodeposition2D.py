import streamlit as st
import numpy as np
from scipy.signal import convolve2d
import plotly.graph_objects as go
import pyvista as pv
from pathlib import Path
import shutil
import tempfile
import os
import zipfile
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from io import BytesIO

# Cached simulation function
@st.cache_data
def run_simulation(Lx, Ly, Nx, Ny, epsilon, y0, M, dt, t_max, c_bulk, Phi_anode, D, z, F, R, T, k, alpha, i0, c_ref, M_Cu, rho_Cu, beta, a_index, h, psi):
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)
    
    # Initialize phi (electrodeposit = 1, electrolyte = 0)
    phi = (1 - psi) * 0.5 * (1 - np.tanh((Y - y0) / epsilon))
    c = c_bulk * (Y / Ly) * (1 - phi) * (1 - psi)  # Concentration: 0 in electrodeposit/template
    phi_l = (Y / Ly) * Phi_anode  # Electrolyte potential

    # Kernels for derivatives
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]) / (dx**2)
    grad_x_kernel = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]]) / (2 * dx)
    grad_y_kernel = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]]) / (2 * dy)

    times_to_plot = np.arange(0, t_max + 1, 1.0)
    phi_history = []
    c_history = []
    phi_l_history = []
    time_history = []

    # Time evolution loop
    for t in np.arange(0, t_max + dt, dt):
        phi_x = convolve2d(phi, grad_x_kernel, mode='same', boundary='symm')
        phi_y = convolve2d(phi, grad_y_kernel, mode='same', boundary='symm')
        grad_phi_mag = np.sqrt(phi_x**2 + phi_y**2)
        delta_int = 6 * phi * (1 - phi) * (1 - psi) * grad_phi_mag  # Interface indicator
        phi_xx = convolve2d(phi, laplacian_kernel, mode='same', boundary='symm')
        
        # Free energy derivatives
        f_prime_electrodeposit = beta * 2 * phi * (1 - phi) * (1 - 2 * phi)
        f_prime_template = beta * 2 * (phi - h)
        f_prime_total = ((1 + a_index) / 8) * (1 - psi) * f_prime_electrodeposit + ((1 - a_index) / 8) * psi * f_prime_template
        
        mu = -epsilon**2 * phi_xx + f_prime_total - alpha * c
        mu_xx = convolve2d(mu, laplacian_kernel, mode='same', boundary='symm')
        
        # Butler-Volmer current
        eta = -phi_l
        c_mol_m3 = c * 1e6 * (1 - phi) * (1 - psi)
        i_loc = i0 * (np.exp(1.5 * F * eta / (R * T)) * c_mol_m3 / c_ref - np.exp(-0.5 * F * eta / (R * T)))
        i_loc = i_loc * delta_int
        
        # Velocity (in y-direction)
        u = -(i_loc / (2 * F)) * (M_Cu / rho_Cu) * 1e-2  # cm to m
        advection = u * (1 - psi) * phi_y
        phi += dt * (M * mu_xx - advection)
        
        # Concentration update
        c_eff = (1 - phi) * (1 - psi) * c
        c_eff_xx = convolve2d(c_eff, laplacian_kernel, mode='same', boundary='symm')
        c_eff_y = convolve2d(c_eff, grad_y_kernel, mode='same', boundary='symm')
        migration = (z * F * D / (R * T)) * (Phi_anode / Ly) * c_eff_y
        sink = -i_loc * delta_int / (2 * F * 1e6)
        c_t = D * c_eff_xx + migration + sink
        c += dt * c_t
        c[:, 0] = 0  # Zero at y=0
        c[:, -1] = c_bulk  # Bulk at y=Ly

        if np.any(np.isclose(t, times_to_plot, atol=dt/2)):
            phi_history.append(phi.copy())
            c_history.append(c.copy())
            phi_l_history.append(phi_l.copy())
            time_history.append(t)

    return x, y, phi_history, c_history, phi_l_history, time_history, psi

# Template geometry function
def create_template(Lx, Ly, Nx, Ny, template_type, radius, side_length, param1, param2, param_func):
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)
    psi = np.zeros((Ny, Nx))
    
    if template_type == "Semicircle":
        center = (Lx / 2, 0)
        dist = np.sqrt((X - center[0])**2 + Y**2)
        psi = np.where((dist <= radius) & (Y >= 0), 1, 0)
    elif template_type == "Square":
        center = (Lx / 2, 0)
        psi = np.where((np.abs(X - center[0]) <= side_length / 2) & (Y >= 0) & (Y <= side_length), 1, 0)
    elif template_type == "Parametric":
        try:
            g = eval(param_func, {"x": X - Lx / 2, "y": Y, "p1": param1, "p2": param2, "np": np})
            psi = np.where(g <= 0, 1, 0)
        except Exception as e:
            st.error(f"Invalid parametric function: {e}")
            psi = np.zeros((Ny, Nx))
    
    return psi

# Streamlit app configuration
st.title("Template Electrodeposition Phase Field Simulation")
st.markdown("""
    This app simulates 2D copper deposition with a fixed template phase (ψ = 1) immiscible with electrodeposit (φ = 1) and electrolyte (φ = 0). Features include template geometry selection (semicircle, square, parametric), adjustable free energy parameters, VTS downloads, and customizable visualizations.
""")

# Sidebar for parameters
st.sidebar.header("Simulation Parameters")
Lx = st.sidebar.slider("Domain Width (Lx, cm)", 0.1, 2.0, 1.0, 0.1)
Ly = st.sidebar.slider("Domain Height (Ly, cm)", 0.1, 2.0, 1.0, 0.1)
Nx = st.sidebar.slider("Grid Points X (Nx)", 50, 200, 100, 10)
Ny = st.sidebar.slider("Grid Points Y (Ny)", 50, 200, 100, 10)
epsilon = st.sidebar.slider("Interface Width (epsilon, cm)", 0.005, 0.3, 0.05, 0.005)
y0 = st.sidebar.slider("Initial Interface Position (y0, cm)", 0.01, 0.5, 0.1, 0.01)
M = st.sidebar.number_input("Mobility (M, cm²/s)", 1e-6, 1e-4, 1e-5, 1e-6)
dt = st.sidebar.number_input("Time Step (dt, s)", 0.001, 0.1, 0.01, 0.001)
t_max = st.sidebar.number_input("Total Time (t_max, s)", 1.0, 20.0, 10.0, 1.0)
c_bulk = st.sidebar.number_input("Bulk Ion Concentration (c_bulk, mol/cm³)", 0.0001, 0.01, 0.001, 0.0001)
Phi_anode = st.sidebar.slider("Anode Potential (Phi_anode, V)", 0.1, 2.0, 1.0, 0.1)
D = st.sidebar.number_input("Diffusion Coefficient (D, cm²/s)", 1e-6, 1e-5, 1e-5, 1e-6)
z = st.sidebar.number_input("Ion Charge (z)", 1, 3, 2)
F = 96485
R = 8.314
T = 298
k = st.sidebar.number_input("Deposition Rate Constant (k, s^-1)", 0.001, 0.1, 0.05, 0.001)
alpha = st.sidebar.number_input("Coupling Constant (alpha)", 0.0, 1.0, 0.1, 0.01)
i0 = st.sidebar.number_input("Exchange Current Density (i0, A/m²)", 0.1, 10.0, 1.0, 0.1)
c_ref = st.sidebar.number_input("Reference Concentration (c_ref, mol/m³)", 100, 2000, 1000, 100)
M_Cu = 0.06355
rho_Cu = 8960
beta = st.sidebar.slider("Free Energy Scale (beta)", 0.1, 10.0, 1.0, 0.1)
a_index = st.sidebar.slider("Free Energy Weight (a_index)", -1.0, 1.0, 0.0, 0.1)
h = st.sidebar.slider("Hydrophobicity Index (h)", 0.0, 1.0, 0.5, 0.1)

# Template geometry selection
st.sidebar.header("Template Geometry")
template_type = st.sidebar.selectbox("Template Type", ["Semicircle", "Square", "Parametric"])
radius = st.sidebar.slider("Semicircle Radius (cm)", 0.05, 0.5, 0.2, 0.05) if template_type == "Semicircle" else 0.2
side_length = st.sidebar.slider("Square Side Length (cm)", 0.05, 0.5, 0.2, 0.05) if template_type == "Square" else 0.2
param_func = st.sidebar.text_input("Parametric Function g(x, y, p1, p2)", "(x/p1)**2 + (y/p2)**2 - 1", help="Define g(x, y; p1, p2) ≤ 0 for ψ = 1, e.g., ellipse: (x/p1)^2 + (y/p2)^2 - 1") if template_type == "Parametric" else ""
param1 = st.sidebar.slider("Parametric p1", 0.05, 0.5, 0.2, 0.05) if template_type == "Parametric" else 0.2
param2 = st.sidebar.slider("Parametric p2", 0.05, 0.5, 0.2, 0.05) if template_type == "Parametric" else 0.2

# Create template
psi = create_template(Lx, Ly, Nx, Ny, template_type, radius, side_length, param1, param2, param_func)

# Output directory
output_dir = Path("electrodeposition_outputs")
output_dir.mkdir(exist_ok=True)

# Session state
if "simulation_results" not in st.session_state:
    st.session_state.simulation_results = None

# Run simulation
if st.sidebar.button("Run Simulation"):
    with st.spinner("Running simulation..."):
        x, y, phi_history, c_history, phi_l_history, time_history, psi = run_simulation(
            Lx, Ly, Nx, Ny, epsilon, y0, M, dt, t_max, c_bulk, Phi_anode, D, z, F, R, T, k, alpha, i0, c_ref, M_Cu, rho_Cu, beta, a_index, h, psi
        )
        st.session_state.simulation_results = {
            "x": x, "y": y, "phi_history": phi_history, "c_history": c_history,
            "phi_l_history": phi_l_history, "time_history": time_history, "psi": psi
        }
    st.success("Simulation complete!")

# Display results
if st.session_state.simulation_results:
    results = st.session_state.simulation_results
    x, y, phi_history, c_history, phi_l_history, time_history, psi = (
        results["x"], results["y"], results["phi_history"], results["c_history"],
        results["phi_l_history"], results["time_history"], results["psi"]
    )

    # Time slider
    st.subheader("Simulation Results")
    time_index = st.slider("Select Time Step", 0, len(time_history) - 1, 0, format="t=%.1f s")
    t = time_history[time_index]
    phi = phi_history[time_index]
    c = c_history[time_index]
    phi_l = phi_l_history[time_index]

    # Color scheme
    #color_schemes = ['Viridis', 'Plasma', 'Magma', 'Inferno', 'Cividis']
    color_schemes = [
    'Viridis', 'Plasma', 'Magma', 'Inferno', 'Cividis', 'Jet', 'Rainbow',
    'Turbo', 'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
    'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu',
    'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn', 'cubehelix', 'brg', 'bwr', 'seismic',
    'twilight', 'twilight_shifted', 'hsv', 'nipy_spectral', 'gist_earth',
    'gist_stern', 'ocean', 'terrain', 'gist_rainbow', 'gnuplot', 'gnuplot2',
    'CMRmap', 'cubehelix', 'flag', 'prism', 'spring', 'summer', 'autumn',
    'winter', 'Accent', 'Dark2', 'Paired', 'Pastel1', 'Pastel2',
    'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c',
    'Picnic', 'Portland', 'Blackbody', 'Electric', 'Hot', 'Cool',
    'IceFire', 'Edge', 'HSV', 'Turbo', 'Viridis_r', 'Plasma_r',
    'Magma_r', 'Inferno_r', 'Cividis_r', 'Jet_r', 'Rainbow_r']
    selected_scheme = st.selectbox("Select Plotly Color Scheme", color_schemes)
    cmap = cm.get_cmap(selected_scheme.lower())
    colors = [f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in cmap(np.linspace(0, 1, 3))]

    # Contour plots
    st.write("**Plotly Contour Plots**")
    for data, title in [
        (phi, "Phase Field (φ)"), (c, "Concentration (c, mol/cm³)"), (phi_l, "Potential (φ_l, V)"), (psi, "Template (ψ)")
    ]:
        fig = go.Figure(data=go.Contour(z=data, x=x, y=y, colorscale=selected_scheme))
        fig.update_layout(title=f"{title} at t = {t:.1f} s", xaxis_title="x (cm)", yaxis_title="y (cm)", height=400)
        st.plotly_chart(fig)

    # Matplotlib line plots at x = Lx/2
    st.write("**Matplotlib Line Plots at x = Lx/2**")
    mid_x = Nx // 2
    for data, label, color in [
        (phi[:, mid_x], "φ (Phase Field)", "blue"),
        (c[:, mid_x], "c (Concentration, mol/cm³)", "red"),
        (phi_l[:, mid_x], "φ_l (Potential, V)", "green")
    ]:
        fig, ax = plt.subplots()
        ax.plot(y, data, color=color, label=label)
        ax.set_xlabel("y (cm)")
        ax.set_ylabel(label)
        ax.set_title(f"{label} at x = Lx/2, t = {t:.1f} s")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        plt.close(fig)

    # Additional visualizations
    st.write("**Additional Visualizations**")
    vis_type = st.selectbox("Select Visualization Type", [
        "None", "Heatmap for φ", "Heatmap for c", "Heatmap for φ_l", "Heatmap for ψ",
        "3D Surface for φ", "3D Surface for c", "3D Surface for φ_l", "3D Surface for ψ"
    ])
    if vis_type != "None":
        if "Heatmap" in vis_type:
            if "ψ" in vis_type:
                data = psi
                title = "Heatmap for ψ"
            else:
                data = np.array(phi_l_history).transpose((1, 2, 0))[:, :, time_index] if "φ_l" in vis_type else \
                       np.array(phi_history).transpose((1, 2, 0))[:, :, time_index] if "φ" in vis_type else \
                       np.array(c_history).transpose((1, 2, 0))[:, :, time_index]
                title = f"Heatmap for {'φ' if 'φ' in vis_type else 'c' if 'c' in vis_type else 'φ_l'}"
            fig_heatmap = go.Figure(data=go.Heatmap(z=data, x=x, y=y, colorscale=selected_scheme))
            fig_heatmap.update_layout(title=title, xaxis_title="x (cm)", yaxis_title="y (cm)")
            st.plotly_chart(fig_heatmap)
        elif "3D Surface" in vis_type:
            if "ψ" in vis_type:
                data = psi
                title = "3D Surface for ψ"
            else:
                data = np.array(phi_l_history).transpose((1, 2, 0))[:, :, time_index] if "φ_l" in vis_type else \
                       np.array(phi_history).transpose((1, 2, 0))[:, :, time_index] if "φ" in vis_type else \
                       np.array(c_history).transpose((1, 2, 0))[:, :, time_index]
                title = f"3D Surface for {'φ' if 'φ' in vis_type else 'c' if 'c' in vis_type else 'φ_l'}"
            fig_surface = go.Figure(data=[go.Surface(z=data, x=x, y=y, colorscale=selected_scheme)])
            fig_surface.update_layout(title=title, scene=dict(xaxis_title="x (cm)", yaxis_title="y (cm)", zaxis_title=vis_type.split()[2]), height=600)
            st.plotly_chart(fig_surface)

    # VTR download for selected timestep
    st.write("**Download VTR File for Selected Timestep**")
    with tempfile.TemporaryDirectory() as tmpdirname:
        vtr_path = Path(tmpdirname) / f"electrodeposition_t{t:.1f}.vtr"
        grid = pv.RectilinearGrid(x, y, [0])
        grid.point_data['phi'] = phi.T.ravel(order='F')
        grid.point_data['c'] = c.T.ravel(order='F')
        grid.point_data['phi_l'] = phi_l.T.ravel(order='F')
        grid.point_data['psi'] = psi.T.ravel(order='F')
        grid.save(vtr_path)
        with open(vtr_path, "rb") as f:
            vtr_data = f.read()
        st.download_button(
            label=f"Download VTR for t = {t:.1f} s",
            data=vtr_data,
            file_name=f"electrodeposition_t{t:.1f}.vtr",
            mime="application/octet-stream",
            key=f"vtr_download_{time_index}"
        )

    # Download all VTR files
    st.write("**Download All VTR Files**")
    if st.button("Download All VTR Files as ZIP"):
        with tempfile.TemporaryDirectory() as tmpdirname:
            zip_path = Path(tmpdirname) / "electrodeposition_vtr_files.zip"
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for idx, t in enumerate(time_history):
                    vtr_path = Path(tmpdirname) / f"electrodeposition_t{t:.1f}.vtr"
                    grid = pv.RectilinearGrid(x, y, [0])
                    grid.point_data['phi'] = phi_history[idx].T.ravel(order='F')
                    grid.point_data['c'] = c_history[idx].T.ravel(order='F')
                    grid.point_data['phi_l'] = phi_l_history[idx].T.ravel(order='F')
                    grid.point_data['psi'] = psi.T.ravel(order='F')
                    grid.save(vtr_path)
                    zipf.write(vtr_path, f"electrodeposition_t{t:.1f}.vtr")
            with open(zip_path, "rb") as f:
                zip_data = f.read()
            st.download_button(
                label="Download ZIP of All VTR Files",
                data=zip_data,
                file_name="electrodeposition_vtr_files.zip",
                mime="application/zip",
                key="vtr_zip_download"
            )

# Cleanup
if st.button("Clear Output Files"):
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(exist_ok=True)
    st.success("Output files cleared!")

st.markdown("""
### Parametric Template Design Guideline
Define a function \( g(x, y; p_1, p_2) \leq 0 \) for \(\psi = 1\), centered at \((L_x/2, 0)\):
- **Formula**: \(\psi = 1\) if \( g(x - L_x/2, y; p_1, p_2) \leq 0 \), else \(\psi = 0\).
- **Examples**:
  - Ellipse: \((x/p_1)^2 + (y/p_2)^2 - 1\)
  - Star: \(\sqrt{x^2 + y^2} - p_1 (1 + 0.2 \cos(p_2 \arctan2(y, x)))\)
- **Input**: Enter \( g(x, y, p1, p2) \) in the text box, use \( p1 \), \( p2 \) as parameters.
- **Tips**: Ensure \( g \) is continuous and centered at \((0, 0)\) in shifted coordinates.

### Instructions
1. Adjust parameters in the sidebar.
2. Select template geometry and parameters.
3. Click 'Run Simulation'.
4. Use time slider to view results.
5. Choose Plotly color scheme.
6. View Matplotlib plots at \( x = L_x/2 \).
7. Download VTR files (single or all timesteps).
8. Select additional visualizations, including ψ.
9. Clear files with 'Clear Output Files'.

### Notes
- **φ = 1** (electrodeposit), **φ = 0** (electrolyte), **ψ = 1** (template, fixed).
- Concentration is zero in electrodeposit and template.
- Advection affects only electrodeposit/electrolyte interface.
- VTR files are compatible with Paraview.
""")
