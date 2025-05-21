import streamlit as st
import plotly.graph_objects as go
import pickle
import numpy as np
import os
from functools import lru_cache

# Streamlit page config
st.set_page_config(layout="wide", page_title="Phase-Field Electrodeposition Visualization")

# Cache solution loading
@st.cache_data
def load_solution(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Cache model evaluation (memoized to avoid recomputation)
@lru_cache(maxsize=32)
def get_solution_data(Ly, phi_anode):
    filename = f"pinn_solutions/solution_ly_{Ly:.4f}_phi_{phi_anode:.1f}.pkl"
    if not os.path.exists(filename):
        return None
    sol = load_solution(filename)
    return (
        sol['X'], sol['Y'], sol['phi_preds'], sol['c_preds'],
        sol['phi_l_preds'], sol['psi'], sol['J_preds'], sol['times'],
        sol['params']
    )

# Main app
st.title("Phase-Field Electrodeposition Visualization")

# Sidebar for parameter selection
st.sidebar.header("Simulation Parameters")
Ly_options = [0.003, 0.006, 0.012]
phi_anode_options = [0.5, 0.6, 0.7, 0.8]
Ly = st.sidebar.selectbox("Domain Height (Ly, cm)", Ly_options, format_func=lambda x: f"{x*1e4:.0f} μm")
phi_anode = st.sidebar.selectbox("Anode Potential (φ_anode, V)", phi_anode_options)
phi_anode = float(phi_anode)  # Ensure phi_anode is a float

# Load solution
solution_data = get_solution_data(Ly, phi_anode)
if solution_data is None:
    st.error(f"Solution file for Ly={Ly:.4f}, φ_anode={phi_anode:.1f} not found in pinn_solutions/")
    st.stop()

X, Y, phi_preds, c_preds, phi_l_preds, psi, J_preds, times, params = solution_data
Lx = params['Lx']
c_bulk = params.get('c_bulk', 2.5e-3)  # Fallback if not in params
if isinstance(c_bulk, np.ndarray):
    c_bulk = float(c_bulk.item())  # Convert to float if NumPy array

# Time slider
time_idx = st.slider("Select Time (s)", 0, len(times)-1, 0, format="%d")
t_val = times[time_idx]

# Compute global color scales
phi_min, phi_max = np.min(phi_preds), np.max(phi_preds)
c_min, c_max = np.min(c_preds), np.max(c_preds)
phi_l_min, phi_l_max = np.min(phi_l_preds), np.max(phi_l_preds)

# Base layout dictionary
base_layout = {
    'margin': dict(l=50, r=50, t=50, b=50),
    'hovermode': 'closest',
    'showlegend': True,
    'xaxis': dict(title="x (cm)", range=[0, Lx], scaleanchor="y", scaleratio=1),
    'yaxis': dict(title="y (cm)", range=[0, Ly], autorange=True),
    'width': 600,
    'height': 600
}

# Plot heatmaps
st.header("Field Variables")

col1, col2 = st.columns(2)

# Phi (phase-field)
with col1:
    fig_phi = go.Figure(
        data=go.Heatmap(
            x=X[:,0], y=Y[0,:], z=phi_preds[time_idx],
            colorscale='Viridis', zmin=phi_min, zmax=phi_max,
            colorbar=dict(title="φ")
        ),
        layout=go.Layout(
            title=f"Phase-Field (φ) at t={t_val:.2f}s",
            **base_layout
        )
    )
    st.plotly_chart(fig_phi, use_container_width=True)

# Concentration (c)
with col2:
    fig_c = go.Figure(
        data=go.Heatmap(
            x=X[:,0], y=Y[0,:], z=c_preds[time_idx],
            colorscale='Plasma', zmin=c_min, zmax=c_max,
            colorbar=dict(title="c (mol/cm³)")
        ),
        layout=go.Layout(
            title=f"Concentration (c) at t={t_val:.2f}s",
            **base_layout
        )
    )
    st.plotly_chart(fig_c, use_container_width=True)

col3, col4 = st.columns(2)

# Liquid potential (phi_l)
with col3:
    fig_phi_l = go.Figure(
        data=go.Heatmap(
            x=X[:,0], y=Y[0,:], z=phi_l_preds[time_idx],
            colorscale='Inferno', zmin=phi_l_min, zmax=phi_l_max,
            colorbar=dict(title="φ_l (V)")
        ),
        layout=go.Layout(
            title=f"Liquid Potential (φ_l) at t={t_val:.2f}s",
            **base_layout
        )
    )
    st.plotly_chart(fig_phi_l, use_container_width=True)

# Template (psi)
with col4:
    fig_psi = go.Figure(
        data=go.Heatmap(
            x=X[:,0], y=Y[0,:], z=psi,
            colorscale='Greys', zmin=0, zmax=1,
            colorbar=dict(title="ψ")
        ),
        layout=go.Layout(
            title="Template (ψ)",
            **base_layout
        )
    )
    st.plotly_chart(fig_psi, use_container_width=True)

# Flux vector plot
st.header("Ion Flux (J)")
J_x, J_y = J_preds[time_idx]
scale = 0.1  # Adjust arrow length
fig_flux = go.Figure(
    data=go.Cone(
        x=X.flatten(), y=Y.flatten(), z=np.zeros_like(X.flatten()),
        u=J_x.flatten(), v=J_y.flatten(), w=np.zeros_like(J_x.flatten()),
        sizemode="scaled", sizeref=scale,
        colorscale='Blues', showscale=True,
        colorbar=dict(title="|J|")
    ),
    layout=go.Layout(
        title=f"Ion Flux (J) at t={t_val:.2f}s",
        xaxis=dict(title="x (cm)", range=[0, Lx]),
        yaxis=dict(title="y (cm)", range=[0, Ly], autorange=True),
        width=600,
        height=600
    )
)
st.plotly_chart(fig_flux, use_container_width=True)

# Boundary checks
st.header("Boundary Conditions Check")
if time_idx == 0:  # Only check at t=0 for initial conditions
    c_anode = c_preds[time_idx][:,0]  # y=0
    c_cathode = c_preds[time_idx][:,-1]  # y=Ly
    phi_anode = phi_preds[time_idx][:,0]
    phi_cathode = phi_preds[time_idx][:,-1]
    phi_l_anode = phi_l_preds[time_idx][:,0]
    phi_l_cathode = phi_l_preds[time_idx][:,-1]
    
    st.write("**At t=0:**")
    st.write(f"- Anode (y=0): c_mean={c_anode.mean():.2e}, expected=0")
    st.write(f"- Cathode (y=Ly): c_mean={c_cathode.mean():.2e}, expected={c_bulk:.2e}")
    st.write(f"- Anode (y=0): φ_mean={phi_anode.mean():.2f}, expected=1")
    st.write(f"- Cathode (y=Ly): φ_mean={phi_cathode.mean():.2f}, expected=0")
    st.write(f"- Anode (y=0): φ_l_mean={phi_l_anode.mean():.2f}, expected={phi_anode:.2f}")
    st.write(f"- Cathode (y=Ly): φ_l_mean={phi_l_cathode.mean():.2f}, expected=0")

# Clear cache button
if st.button("Clear Cache"):
    st.cache_data.clear()
    get_solution_data.cache_clear()
    st.success("Cache cleared! Please reload the page.")
