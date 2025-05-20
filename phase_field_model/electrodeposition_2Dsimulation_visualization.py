import streamlit as st
import numpy as np
from scipy.signal import convolve
import plotly.graph_objects as go
import pyvista as pv
from pathlib import Path
import shutil
import tempfile
import os
import zipfile
import matplotlib.pyplot as plt
from io import BytesIO
import matplotlib.cm as cm

# Cached simulation function
@st.cache_data
def run_simulation(Ly, N, epsilon, y0, M, dt, t_max, c_bulk, Phi_anode, D, z, F, R, T, k, alpha, i0, c_ref, M_Cu, rho_Cu, beta):
    dy = Ly / (N - 1)
    y = np.linspace(0, Ly, N)
    phi = 0.5 * (1 - np.tanh((y - y0) / epsilon))  # Initial phi: 1 at cathode, 0 at anode
    c = c_bulk * (y / Ly) * (1 - phi)  # Initial c: 0 in electrodeposit, gradient in electrolyte
    grad_Phi = Phi_anode / Ly  # Potential gradient
    phi_l = y * grad_Phi  # Electrolyte potential

    # Define kernels for derivatives
    laplacian_kernel = np.array([1, -2, 1]) / dy**2
    grad_kernel = np.array([-1, 0, 1]) / (2 * dy)

    times_to_plot = np.arange(0, t_max + 1, 1.0)
    phi_history = []
    c_history = []
    phi_l_history = []
    time_history = []

    # Time evolution loop
    for t in np.arange(0, t_max + dt, dt):
        phi_y = convolve(phi, grad_kernel, mode='same')
        delta_int = 6 * phi * (1 - phi) * np.abs(phi_y)  # Phase field delta function
        phi_xx = convolve(phi, laplacian_kernel, mode='same')
        f_prime = beta * 2 * phi * (1 - phi) * (1 - 2 * phi)  # Double-well derivative
        mu = -epsilon**2 * phi_xx + f_prime - alpha * c  # Chemical potential
        mu_xx = convolve(mu, laplacian_kernel, mode='same')
        # Butler-Volmer current density
        eta = -phi_l  # Overpotential (phi_s = 0, E_eq = 0)
        c_mol_m3 = c * 1e6 * (1 - phi)  # Convert mol/cm³ to mol/m³, zero in electrodeposit
        i_loc = i0 * (np.exp(1.5 * F * eta / (R * T)) * c_mol_m3 / c_ref - np.exp(-0.5 * F * eta / (R * T)))
        i_loc = i_loc * delta_int  # Apply only at interface
        # Velocity from current density
        u = -(i_loc / (2 * F)) * (M_Cu / rho_Cu) * 1e-2  # Convert cm to m
        # Advection term
        advection = u * phi_y
        phi += dt * (M * mu_xx - advection)  # Update phase field
        mu_xx[0] = mu_xx[1]  # Zero-flux boundary conditions
        mu_xx[-1] = mu_xx[-2]

        # Concentration update with electrolyte volume fraction
        c_eff = (1 - phi) * c
        c_eff_xx = convolve(c_eff, laplacian_kernel, mode='same')
        c_eff_y = convolve(c_eff, grad_kernel, mode='same')
        migration = (z * F * D / (R * T)) * grad_Phi * c_eff_y
        sink = -i_loc * delta_int / (2 * F * 1e6)  # Convert to mol/cm³
        c_t = D * c_eff_xx + migration + sink
        c += dt * c_t
        c[0] = 0  # Zero at cathode (electrodeposit)
        c[-1] = c_bulk  # Bulk at anode (electrolyte)

        if np.any(np.isclose(t, times_to_plot, atol=dt/2)):
            phi_history.append(phi.copy())
            c_history.append(c.copy())
            phi_l_history.append(phi_l.copy())
            time_history.append(t)

    return y, phi_history, c_history, phi_l_history, time_history

# Streamlit app configuration
st.title("Copper Deposition Phase Field Simulation with Butler-Volmer Kinetics")
st.markdown("""
    This app simulates copper deposition using a phase field model (φ = 1 for electrodeposit, φ = 0 for electrolyte), incorporating Butler-Volmer kinetics, an advection term, and adjustable ε and β. Concentration flux is from the anode to the interface, with zero concentration in the electrodeposit. Features VTS download for all timesteps, customizable Plotly colors, and Matplotlib plots.
""")

# Sidebar for parameter inputs
st.sidebar.header("Simulation Parameters")
Ly = st.sidebar.slider("Domain Length (Ly, cm)", 0.1, 2.0, 1.0, 0.1)
N = st.sidebar.slider("Number of Grid Points (N)", 50, 200, 100, 10)
epsilon = st.sidebar.slider("Interface Width (epsilon, cm)", 0.005, 0.3, 0.05, 0.005, help="Controls interface thickness; higher values stabilize the interface.")
y0 = st.sidebar.slider("Initial Interface Position (y0, cm)", 0.01, 0.5, 0.1, 0.01)
M = st.sidebar.number_input("Mobility (M, cm²/s)", 1e-6, 1e-4, 1e-5, 1e-6)
dt = st.sidebar.number_input("Time Step (dt, s)", 0.001, 0.1, 0.01, 0.001)
t_max = st.sidebar.number_input("Total Time (t_max, s)", 1.0, 20.0, 10.0, 1.0)
c_bulk = st.sidebar.number_input("Bulk Ion Concentration (c_bulk, mol/cm³)", 0.0001, 0.01, 0.001, 0.0001)
Phi_anode = st.sidebar.slider("Anode Potential (Phi_anode, V)", 0.1, 2.0, 1.0, 0.1)
D = st.sidebar.number_input("Diffusion Coefficient (D, cm²/s)", 1e-6, 1e-5, 1e-5, 1e-6)
z = st.sidebar.number_input("Ion Charge (z)", 1, 3, 2)
F = 96485  # Faraday constant (C/mol)
R = 8.314  # Gas constant (J/(mol·K))
T = 298    # Temperature (K)
k = st.sidebar.number_input("Deposition Rate Constant (k, s^-1)", 0.001, 0.1, 0.05, 0.001)
alpha = st.sidebar.number_input("Coupling Constant (alpha)", 0.0, 1.0, 0.1, 0.01)
i0 = st.sidebar.number_input("Exchange Current Density (i0, A/m²)", 0.1, 10.0, 1.0, 0.1)
c_ref = st.sidebar.number_input("Reference Concentration (c_ref, mol/m³)", 100, 2000, 1000, 100)
M_Cu = 0.06355  # Molar mass of copper (kg/mol)
rho_Cu = 8960   # Density of copper (kg/m³)
beta = st.sidebar.slider("Double-Well Factor (beta)", 0.1, 10.0, 1.0, 0.1, help="Scales bulk energy; higher values stabilize φ at 0, 1.")

# Create temporary directory for outputs
output_dir = Path("electrodeposition_outputs")
output_dir.mkdir(exist_ok=True)

# Initialize session state for simulation results
if "simulation_results" not in st.session_state:
    st.session_state.simulation_results = None

# Run simulation on button click
if st.sidebar.button("Run Simulation"):
    with st.spinner("Running simulation..."):
        y, phi_history, c_history, phi_l_history, time_history = run_simulation(
            Ly, N, epsilon, y0, M, dt, t_max, c_bulk, Phi_anode, D, z, F, R, T, k, alpha, i0, c_ref, M_Cu, rho_Cu, beta
        )
        st.session_state.simulation_results = {
            "y": y,
            "phi_history": phi_history,
            "c_history": c_history,
            "phi_l_history": phi_l_history,
            "time_history": time_history
        }
    st.success("Simulation complete!")

# Display results if available
if st.session_state.simulation_results:
    results = st.session_state.simulation_results
    y = results["y"]
    phi_history = results["phi_history"]
    c_history = results["c_history"]
    phi_l_history = results["phi_l_history"]
    time_history = results["time_history"]

    # Time slider for selecting time step
    st.subheader("Simulation Results")
    time_index = st.slider("Select Time Step", 0, len(time_history) - 1, 0, format="t=%.1f s")

    # Selected time
    t = time_history[time_index]
    phi = phi_history[time_index]
    c = c_history[time_index]
    phi_l = phi_l_history[time_index]

    # Color scheme selection
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
    # Sample three colors from the selected colormap
    cmap = cm.get_cmap(selected_scheme.lower())
    colors = [f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in cmap(np.linspace(0, 1, 3))]

    # Plot phase field (φ), concentration (c), and potential (φ_l) with Plotly
    st.write("**Plotly: Phase Field, Concentration, and Electrolyte Potential**")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y, y=phi, mode='lines', name='φ (Phase Field)', line=dict(color=colors[0])))
    fig.add_trace(go.Scatter(x=y, y=c, mode='lines', name='c (Concentration)', line=dict(color=colors[1]), yaxis='y2'))
    fig.add_trace(go.Scatter(x=y, y=phi_l, mode='lines', name='φ_l (Potential)', line=dict(color=colors[2]), yaxis='y3'))
    fig.update_layout(
        title=f'Simulation at t = {t:.1f} s',
        xaxis_title='Position (y, cm)',
        yaxis_title='Phase Field (φ)',
        yaxis2=dict(title='Concentration (c, mol/cm³)', overlaying='y', side='right', position=0.85),
        yaxis3=dict(title='Potential (φ_l, V)', overlaying='y', side='right'),
        legend=dict(x=0.7, y=1.0),
        height=500
    )
    st.plotly_chart(fig)

    # Matplotlib plots for each variable
    st.write("**Matplotlib: Individual Variable Plots**")
    # Plot for φ
    fig_phi, ax_phi = plt.subplots()
    ax_phi.plot(y, phi, color='blue', label='φ (Phase Field)')
    ax_phi.set_xlabel('Position (y, cm)')
    ax_phi.set_ylabel('Phase Field (φ)')
    ax_phi.set_title(f'Phase Field at t = {t:.1f} s')
    ax_phi.legend()
    ax_phi.grid(True)
    st.pyplot(fig_phi)
    plt.close(fig_phi)

    # Plot for c
    fig_c, ax_c = plt.subplots()
    ax_c.plot(y, c, color='red', label='c (Concentration)')
    ax_c.set_xlabel('Position (y, cm)')
    ax_c.set_ylabel('Concentration (c, mol/cm³)')
    ax_c.set_title(f'Concentration at t = {t:.1f} s')
    ax_c.legend()
    ax_c.grid(True)
    st.pyplot(fig_c)
    plt.close(fig_c)

    # Plot for φ_l
    fig_phi_l, ax_phi_l = plt.subplots()
    ax_phi_l.plot(y, phi_l, color='green', label='φ_l (Potential)')
    ax_phi_l.set_xlabel('Position (y, cm)')
    ax_phi_l.set_ylabel('Potential (φ_l, V)')
    ax_phi_l.set_title(f'Electrolyte Potential at t = {t:.1f} s')
    ax_phi_l.legend()
    ax_phi_l.grid(True)
    st.pyplot(fig_phi_l)
    plt.close(fig_phi_l)

    # Additional visualizations
    st.write("**Additional Visualizations**")
    vis_type = st.selectbox("Select Visualization Type", ["None", "Heatmap for φ", "Heatmap for c", "Heatmap for φ_l", "3D Surface for φ", "3D Surface for c", "3D Surface for φ_l"])

    if vis_type != "None":
        if "Heatmap" in vis_type:
            if "φ_l" in vis_type:
                data = np.array(phi_l_history).T
                title = "Heatmap for φ_l over Space and Time"
            else:
                data = np.array(phi_history).T if "φ" in vis_type else np.array(c_history).T
                title = f"Heatmap for {'φ' if 'φ' in vis_type else 'c'} over Space and Time"
            fig_heatmap = go.Figure(data=go.Heatmap(z=data, x=time_history, y=y, colorscale=selected_scheme))
            fig_heatmap.update_layout(title=title, xaxis_title='Time (s)', yaxis_title='Position (y, cm)')
            st.plotly_chart(fig_heatmap)
        elif "3D Surface" in vis_type:
            if "φ_l" in vis_type:
                data = np.array(phi_l_history).T
                title = "3D Surface for φ_l"
            else:
                data = np.array(phi_history).T if "φ" in vis_type else np.array(c_history).T
                title = f"3D Surface for {'φ' if 'φ' in vis_type else 'c'}"
            fig_surface = go.Figure(data=[go.Surface(z=data, x=time_history, y=y, colorscale=selected_scheme)])
            fig_surface.update_layout(
                title=title,
                scene=dict(xaxis_title='Time (s)', yaxis_title='Position (y, cm)', zaxis_title=vis_type.split()[2]),
                height=600
            )
            st.plotly_chart(fig_surface)

    # VTS file download for selected timestep
    st.write("**Download VTS File for Selected Timestep**")
    with tempfile.TemporaryDirectory() as tmpdirname:
        vts_path = Path(tmpdirname) / f"electrodeposition_t{t:.1f}.vts"
        points = np.array([[yi, ti, 0] for ti in time_history for yi in y])
        grid = pv.StructuredGrid()
        grid.points = points
        grid.dimensions = (len(y), len(time_history), 1)
        phi_2d = np.array(phi_history).T
        c_2d = np.array(c_history).T
        phi_l_2d = np.array(phi_l_history).T
        grid.point_data['phi'] = phi_2d.ravel(order='C')
        grid.point_data['c'] = c_2d.ravel(order='C')
        grid.point_data['phi_l'] = phi_l_2d.ravel(order='C')
        grid.save(vts_path)
        with open(vts_path, "rb") as f:
            vts_data = f.read()
        st.download_button(
            label=f"Download VTS for t = {t:.1f} s",
            data=vts_data,
            file_name=f"electrodeposition_t{t:.1f}.vts",
            mime="application/octet-stream",
            key=f"vts_download_{time_index}"
        )

    # Download all VTS files as ZIP
    st.write("**Download All VTS Files**")
    if st.button("Download All VTS Files as ZIP"):
        with tempfile.TemporaryDirectory() as tmpdirname:
            zip_path = Path(tmpdirname) / "electrodeposition_vts_files.zip"
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for idx, t in enumerate(time_history):
                    vts_path = Path(tmpdirname) / f"electrodeposition_t{t:.1f}.vts"
                    grid = pv.StructuredGrid()
                    grid.points = points
                    grid.dimensions = (len(y), len(time_history), 1)
                    grid.point_data['phi'] = phi_2d.ravel(order='C')
                    grid.point_data['c'] = c_2d.ravel(order='C')
                    grid.point_data['phi_l'] = phi_l_2d.ravel(order='C')
                    grid.save(vts_path)
                    zipf.write(vts_path, f"electrodeposition_t{t:.1f}.vts")
            with open(zip_path, "rb") as f:
                zip_data = f.read()
            st.download_button(
                label="Download ZIP of All VTS Files",
                data=zip_data,
                file_name="electrodeposition_vts_files.zip",
                mime="application/zip",
                key="vts_zip_download"
            )

# Cleanup temporary directory
if st.button("Clear Output Files"):
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(exist_ok=True)
    st.success("Output files cleared!")

st.markdown("""
### Instructions
1. Adjust parameters in the sidebar (e.g., epsilon = 0.05–0.1 cm, beta = 2.0–5.0).
2. Click 'Run Simulation' to start.
3. Use the time slider to view φ, c, and φ_l.
4. Select a Plotly color scheme (e.g., Viridis, Plasma).
5. View Matplotlib plots for individual variables.
6. Download VTS files:
   - For the selected timestep via the time slider.
   - For all timesteps as a ZIP archive.
7. Select visualizations (heatmaps or 3D surfaces).
8. Clear temporary files with 'Clear Output Files'.

### Notes
- **φ = 1** (electrodeposit), **φ = 0** (electrolyte).
- Concentration is zero in electrodeposit, with flux from anode to interface.
- Uses Butler-Volmer kinetics, zero-flux boundary conditions for φ, and adjustable ε, β.
- VTS files contain φ, c, and φ_l for all timesteps in the ZIP archive.
""")
