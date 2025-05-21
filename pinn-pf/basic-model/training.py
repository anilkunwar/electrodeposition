import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class SmoothSigmoid(nn.Module):  # \text{Custom activation for } \phi \in [0, 1]
    def __init__(self, slope=1.0):
        super().__init__()
        self.k = slope
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return self.scale * 1 / (1 + torch.exp(-self.k * x))  # \phi = s \cdot \frac{1}{1 + e^{-kx}}

class PhaseFieldPINN(nn.Module):  # \text{PINN to approximate } (\phi, c, \phi_l)
    def __init__(self, Lx, Ly, t_max, phi_anode):
        super().__init__()
        self.Lx = Lx  # \text{Domain width: } L_x = 0.01 \, \text{cm}
        self.Ly = Ly  # \text{Domain height: } L_y \in \{0.003, 0.006, 0.012\} \, \text{cm}
        self.t_max = t_max  # \text{Time: } t_{\text{max}} = 20 \, \text{s}
        self.phi_anode = phi_anode  # \text{Anode potential: } \phi_{\text{anode}} \in \{0.5, 0.6, 0.7, 0.8\} \, \text{V}
        self.D = 5.3e-6  # D = 5.3 \times 10^{-6} \, \text{cm}^2/\text{s}
        self.z = 2  # z = 2
        self.F = 96485 / 1e5  # F = 964.85 \, \text{C}/\text{mol}
        self.R = 8.314  # R = 8.314 \, \text{J}/(\text{mol·K})
        self.T = 298  # T = 298 \, \text{K}
        self.epsilon = 1e-4  # \varepsilon = 10^{-4} \, \text{cm}
        self.M = 1e-4  # M = 10^{-4}
        self.c_bulk = 2.5e-3  # c_{\text{bulk}} = 2.5 \times 10^{-3} \, \text{mol}/\text{cm}^3
        self.c_ref = 1e-3  # c_{\text{ref}} = 10^{-3} \, \text{mol}/\text{cm}^3
        self.i0 = 1e-3  # i_0 = 10^{-3} \, \text{A}/\text{cm}^2
        self.M_Cu = 63.546  # M_{\text{Cu}} = 63.546 \, \text{g}/\text{mol}
        self.rho_Cu = 8.96  # \rho_{\text{Cu}} = 8.96 \, \text{g}/\text{cm}^3
        self.beta = 1.0  # \beta = 1.0
        self.alpha = 1.0  # \alpha = 1.0
        self.a_index = 0.5  # a_{\text{index}} = 0.5
        self.y0 = 0.0005  # y_0 = 0.0005 \, \text{cm}
        
        # Normalize phi_anode
        self.phi_anode_norm = (phi_anode - 0.5) / (0.8 - 0.5)  # \phi_{\text{anode, norm}} = \frac{\phi_{\text{anode}} - 0.5}{0.8 - 0.5}
        
        self.shared_net = nn.Sequential(
            nn.Linear(4, 128), nn.Tanh(),  # \text{Inputs: } (x_{\text{norm}}, y_{\text{norm}}, t_{\text{norm}}, \phi_{\text{anode, norm}})
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh()
        )
        
        self.phi_head = nn.Sequential(
            nn.Linear(128, 1),
            SmoothSigmoid(slope=0.5),  # \text{Constrain } \phi \in [0, 1]
            nn.Linear(1, 1, bias=False),
        )
        self.c_head = nn.Sequential(
            nn.Linear(128, 1),
            nn.ReLU(),  # \text{Ensure } c \geq 0
            nn.Linear(1, 1, bias=False),
        )
        self.phi_l_head = nn.Sequential(
            nn.Linear(128, 1),
            nn.ReLU(),  # \text{Ensure } \phi_l \geq 0
            nn.Linear(1, 1, bias=False),
        )
        
        # Initialize output weights
        self.phi_head[2].weight.data.fill_(1.0)  # \text{Scale } \phi \text{ to } [0, 1]
        self.c_head[2].weight.data.fill_(self.c_bulk)  # \text{Scale } c \text{ to } c_{\text{bulk}}
        self.phi_l_head[2].weight.data.fill_(phi_anode)  # \text{Scale } \phi_l \text{ to } \phi_{\text{anode}}

    def forward(self, x, y, t):
        x_norm = x / self.Lx  # x_{\text{norm}} = \frac{x}{L_x}
        y_norm = y / self.Ly  # y_{\text{norm}} = \frac{y}{L_y}
        t_norm = t / self.t_max  # t_{\text{norm}} = \frac{t}{t_{\text{max}}}
        phi_anode_input = torch.full_like(x, self.phi_anode_norm)
        
        inputs = torch.cat([x_norm, y_norm, t_norm, phi_anode_input], dim=1)
        features = self.shared_net(inputs)
        phi = self.phi_head(features)
        c = self.c_head(features)
        phi_l = self.phi_l_head(features)
        return torch.cat([phi, c, phi_l], dim=1)  # \text{Output: } (\phi, c, \phi_l)

def laplacian(u, x, y):  # \text{Compute } \nabla^2 u
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True, retain_graph=True)[0]
    return u_xx + u_yy

def gradient(u, x):  # \text{Compute } \nabla u
    return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]

def psi_template(Lx, Ly, x, y, r=0.0005):  # \psi(x, y) \text{ for semicircle template}
    center_x, center_y = Lx / 2, 0
    distances = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
    psi = torch.where(distances <= r, torch.ones_like(x), torch.zeros_like(x))
    return psi

def physics_loss(model, x, y, t):  # \text{Enforce PDEs: Cahn-Hilliard with electrochemical source, ion transport}
    outputs = model(x, y, t)
    phi, c, phi_l = outputs[:, 0:1], outputs[:, 1:2], outputs[:, 2:3]
    
    # Compute derivatives
    phi_t = gradient(phi, t)  # \frac{\partial \phi}{\partial t}
    c_t = gradient(c, t)  # \frac{\partial c}{\partial t}
    phi_lap = laplacian(phi, x, y)  # \nabla^2 \phi
    c_lap = laplacian(c, x, y)  # \nabla^2 c
    phi_l_lap = laplacian(phi_l, x, y)  # \nabla^2 \phi_l
    
    # Template
    psi = psi_template(model.Lx, model.Ly, x, y).reshape_as(phi)  # \psi(x, y)
    
    # Interface delta function
    grad_phi_x = gradient(phi, x)
    grad_phi_y = gradient(phi, y)
    grad_phi_norm = torch.sqrt(grad_phi_x**2 + grad_phi_y**2 + 1e-10)
    delta_int = 6 * phi * (1 - phi) * (1 - psi) * grad_phi_norm  # \delta_{\text{int}} = 6 \phi (1 - \phi) (1 - \psi) \|\nabla \phi\|
    
    # Chemical potential
    f_prime_electrodeposit = model.beta * 2 * phi * (1 - phi) * (1 - 2 * phi)  # f'_{\text{electrodeposit}}
    f_prime_template = model.beta * 2 * (phi - 0.5)  # f'_{\text{template}}, h=0.5
    df_dphi = (1 + model.a_index) / 8 * (1 - psi) * f_prime_electrodeposit + (1 - model.a_index) / 8 * psi * f_prime_template  # \frac{\delta f}{\delta \phi}
    mu = -model.epsilon**2 * phi_lap + df_dphi - model.alpha * c  # \mu = -\varepsilon^2 \nabla^2 \phi + \frac{\delta f}{\delta \phi} - \alpha c
    mu_lap = laplacian(mu, x, y)  # \nabla^2 \mu
    
    # Electrochemical kinetics
    eta = -phi_l  # \eta = -\phi_l
    i_loc = model.i0 * (torch.exp(1.5 * model.F * eta / (model.R * model.T)) * c / model.c_ref - 
                        torch.exp(-0.5 * model.F * eta / (model.R * model.T)))  # i_{\text{loc}} = i_0 \left[ \exp\left( \frac{1.5 F \eta}{RT} \right) \frac{c}{c_{\text{ref}}} - \exp\left( -\frac{0.5 F \eta}{RT} \right) \right]
    i_loc = i_loc * delta_int  # \text{Localize at interface}
    
    # Deposition velocity
    u = -i_loc / (2 * model.F) * model.M_Cu / model.rho_Cu * 1e-2  # u = -\frac{i_{\text{loc}}}{2F} \cdot \frac{M_{\text{Cu}}}{\rho_{\text{Cu}}} \cdot 10^{-2}
    
    # Electrochemical source term
    Vm = model.M_Cu / model.rho_Cu  # V_m = \frac{M_{\text{Cu}}}{\rho_{\text{Cu}}}
    source_term = Vm / model.F * i_loc  # S = \frac{V_m}{F} i_{\text{loc}}
    
    # Cahn-Hilliard with advection and source
    residual_phi = phi_t - model.M * mu_lap + u * grad_phi_y * (1 - psi) + source_term  # \frac{\partial \phi}{\partial t} = M \nabla^2 \mu - u \frac{\partial \phi}{\partial y} (1 - \psi) + \frac{V_m}{F} i_{\text{loc}}
    
    # Ion transport
    c_eff = (1 - phi) * (1 - psi) * c  # c_{\text{eff}} = (1 - \phi)(1 - \psi)c
    c_eff_lap = laplacian(c_eff, x, y)  # \nabla^2 c_{\text{eff}}
    v_mig_x = model.z * model.F * model.D / (model.R * model.T) * grad_phi_x  # v_{\text{mig}, x}
    v_mig_y = model.z * model.F * model.D / (model.R * model.T) * grad_phi_y  # v_mig_y
    v_mig_grad_c_eff = v_mig_x * gradient(c_eff, x) + v_mig_y * gradient(c_eff, y)  # \mathbf{v}_{\text{mig}} \cdot \nabla c_{\text{eff}}
    S = -i_loc * delta_int / (2 * model.F * 1e6)  # S = -\frac{i_{\text{loc}} \delta_{\text{int}}}{2 F \cdot 10^6}
    residual_c = c_t - model.D * c_eff_lap - v_mig_grad_c_eff - S  # \frac{\partial c}{\partial t} = D \nabla^2 c_{\text{eff}} + \mathbf{v}_{\text{mig}} \cdot \nabla c_{\text{eff}} + S
    
    # Potential (simplified Poisson)
    residual_phi_l = phi_l_lap  # \nabla^2 \phi_l = 0
    
    return torch.mean(residual_phi**2 + residual_c**2 + residual_phi_l**2)

def boundary_loss_anode(model):  # \text{Anode BC: } y = 0
    num = 100
    x = torch.rand(num, 1) * model.Lx
    y = torch.zeros(num, 1)
    t = torch.rand(num, 1) * model.t_max
    outputs = model(x, y, t)
    phi, c, phi_l = outputs[:, 0:1], outputs[:, 1:2], outputs[:, 2:3]
    psi = psi_template(model.Lx, model.Ly, x, y).reshape_as(phi)
    return torch.mean(((phi - 1) * (1 - psi))**2) + torch.mean((c * (1 - psi))**2) + torch.mean((phi_l - model.phi_anode)**2)

def boundary_loss_cathode(model):  # \text{Cathode BC: } y = L_y
    num = 100
    x = torch.rand(num, 1) * model.Lx
    y = torch.full((num, 1), model.Ly)
    t = torch.rand(num, 1) * model.t_max
    outputs = model(x, y, t)
    phi, c, phi_l = outputs[:, 0:1], outputs[:, 1:2], outputs[:, 2:3]
    psi = psi_template(model.Lx, model.Ly, x, y).reshape_as(phi)
    return torch.mean((phi * (1 - psi))**2) + torch.mean(((c - model.c_bulk) * (1 - psi))**2) + torch.mean(phi_l**2)

def boundary_loss_sides(model):  # \text{Side BCs: } x = 0, L_x
    num = 100
    x_left = torch.zeros(num, 1, requires_grad=True)
    y = torch.rand(num, 1) * model.Ly
    t = torch.rand(num, 1) * model.t_max
    outputs_left = model(x_left, y, t)
    phi_left, c_left, phi_l_left = outputs_left[:, 0:1], outputs_left[:, 1:2], outputs_left[:, 2:3]
    
    x_right = torch.full((num, 1), model.Lx, requires_grad=True)
    outputs_right = model(x_right, y, t)
    phi_right, c_right, phi_l_right = outputs_right[:, 0:1], outputs_right[:, 1:2], outputs_right[:, 2:3]
    
    grad_phi_left = gradient(phi_left, x_left)
    grad_c_left = gradient(c_left, x_left)
    grad_phi_l_left = gradient(phi_l_left, x_left)
    grad_phi_right = gradient(phi_right, x_right)
    grad_c_right = gradient(c_right, x_right)
    grad_phi_l_right = gradient(phi_l_right, x_right)
    
    return (torch.mean(grad_phi_left**2 + grad_c_left**2 + grad_phi_l_left**2) +
            torch.mean(grad_phi_right**2 + grad_c_right**2 + grad_phi_l_right**2))

def initial_loss(model):  # \text{Initial conditions: } t = 0
    num = 500
    x = torch.rand(num, 1) * model.Lx
    y = torch.rand(num, 1) * model.Ly
    t = torch.zeros(num, 1)
    outputs = model(x, y, t)
    phi, c, phi_l = outputs[:, 0:1], outputs[:, 1:2], outputs[:, 2:3]
    psi = psi_template(model.Lx, model.Ly, x, y).reshape_as(phi)
    phi_init = (1 - psi) / 2 * (1 - torch.tanh((y - model.y0) / model.epsilon))  # \phi(x, y, 0) = \frac{1 - \psi}{2} \left[ 1 - \tanh\left( \frac{y - y_0}{\varepsilon} \right) \right]
    c_init = model.c_bulk * y / model.Ly * (1 - phi_init) * (1 - psi)  # c(x, y, 0) = c_{\text{bulk}} \cdot \frac{y}{L_y} \cdot (1 - \phi)(1 - \psi)
    return torch.mean((phi - phi_init)**2 + (c - c_init)**2 + phi_l**2)

def plot_losses(loss_history, Ly, phi_anode, output_dir):  # \text{Plot loss components}
    epochs = np.array(loss_history['epochs'])
    total_loss = np.array(loss_history['total'])
    physics_loss = np.array(loss_history['physics'])
    anode_loss = np.array(loss_history['anode'])
    cathode_loss = np.array(loss_history['cathode'])
    sides_loss = np.array(loss_history['sides'])
    initial_loss = np.array(loss_history['initial'])
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, total_loss, label='Total Loss', linewidth=2, color='black')
    plt.plot(epochs, physics_loss, label='Physics Loss', linewidth=1.5, linestyle='--', color='blue')
    plt.plot(epochs, anode_loss, label='Anode Boundary Loss', linewidth=1.5, linestyle='-.', color='red')
    plt.plot(epochs, cathode_loss, label='Cathode Boundary Loss', linewidth=1.5, linestyle=':', color='green')
    plt.plot(epochs, sides_loss, label='Sides Boundary Loss', linewidth=1.5, linestyle='-', color='purple')
    plt.plot(epochs, initial_loss, label='Initial Condition Loss', linewidth=1.5, linestyle='--', color='orange')
    
    plt.yscale('log')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title(f'Training Loss for Ly = {Ly*1e4:.0f} μm, φ_anode = {phi_anode:.1f} V', fontsize=16)
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend(fontsize=12, loc='upper right')
    plt.tight_layout()
    
    plot_filename = os.path.join(output_dir, f'loss_plot_ly_{Ly:.4f}_phi_{phi_anode:.1f}.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved loss plot to {plot_filename}")

def compute_flux(model, X, Y, t_val):  # \text{Compute } \mathbf{J}_{\text{Cu}}
    X_torch = torch.tensor(X, dtype=torch.float32, requires_grad=True).reshape(-1, 1)
    Y_torch = torch.tensor(Y, dtype=torch.float32, requires_grad=True).reshape(-1, 1)
    t = torch.full((X_torch.numel(), 1), t_val, dtype=torch.float32, requires_grad=True)
    
    outputs = model(X_torch, Y_torch, t)
    phi, c, phi_l = outputs[:, 0:1], outputs[:, 1:2], outputs[:, 2:3]
    psi = psi_template(model.Lx, model.Ly, X_torch, Y_torch).reshape_as(phi)
    
    c_eff = (1 - phi) * (1 - psi) * c
    grad_c_eff_x = gradient(c_eff, X_torch)
    grad_c_eff_y = gradient(c_eff, Y_torch)
    grad_phi_l_x = gradient(phi_l, X_torch)
    grad_phi_l_y = gradient(phi_l, Y_torch)
    
    v_mig_x = model.z * model.F * model.D / (model.R * model.T) * grad_phi_l_x
    v_mig_y = model.z * model.F * model.D / (model.R * model.T) * grad_phi_l_y
    
    J_x = -model.D * grad_c_eff_x - v_mig_x * c_eff
    J_y = -model.D * grad_c_eff_y - v_mig_y * c_eff
    
    return (J_x.detach().numpy().reshape(X.shape), J_y.detach().numpy().reshape(X.shape))

def evaluate_model(model, times, Lx, Ly):  # \text{Evaluate on 30x30 grid}
    x = torch.linspace(0, Lx, 30)
    y = torch.linspace(0, Ly, 30)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    phi_preds, c_preds, phi_l_preds, J_preds = [], [], [], []
    for t_val in times:
        t = torch.full((X.numel(), 1), t_val)
        outputs = model(X.reshape(-1,1), Y.reshape(-1,1), t)
        phi = outputs[:,0].detach().numpy().reshape(30,30).T  # \text{Transpose: rows = } y, \text{cols = } x
        c = outputs[:,1].detach().numpy().reshape(30,30).T
        phi_l = outputs[:,2].detach().numpy().reshape(30,30).T
        phi_preds.append(phi)
        c_preds.append(c)
        phi_l_preds.append(phi_l)
        
        J_x, J_y = compute_flux(model, X.numpy(), Y.numpy(), t_val)
        J_preds.append((J_x, J_y))
    
    x = torch.linspace(0, Lx, 30)
    y = torch.linspace(0, Ly, 30)
    X_psi, Y_psi = torch.meshgrid(x, y, indexing='ij')
    psi = psi_template(Lx, Ly, X_psi.reshape(-1,1), Y_psi.reshape(-1,1)).detach().numpy().reshape(30,30).T
    
    return X.numpy(), Y.numpy(), phi_preds, c_preds, phi_l_preds, psi, J_preds

def generate_parameter_sets(Lx, t_max, epochs):  # \text{Generate models for } L_y, \phi_{\text{anode}}
    Ly_range = [0.003, 0.006, 0.012]  # \text{cm}
    phi_anode_range = [0.5, 0.6, 0.7, 0.8]  # \text{V}
    
    params = []
    for Ly in Ly_range:
        for phi_anode in phi_anode_range:
            param_set = {
                'Lx': Lx,
                'Ly': float(Ly),
                't_max': t_max,
                'phi_anode': float(phi_anode),
                'epochs': epochs
            }
            params.append(param_set)
    return params

def train_PINN(Lx, Ly, t_max, phi_anode, epochs, output_dir):  # \text{Train PINN}
    model = PhaseFieldPINN(Lx, Ly, t_max, phi_anode)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200)
    
    x_pde = torch.rand(1000, 1, requires_grad=True) * Lx
    y_pde = torch.rand(1000, 1, requires_grad=True) * Ly
    t_pde = torch.rand(1000, 1, requires_grad=True) * t_max
    
    loss_history = {
        'epochs': [],
        'total': [],
        'physics': [],
        'anode': [],
        'cathode': [],
        'sides': [],
        'initial': []
    }
    
    last_lr = optimizer.param_groups[0]['lr']
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        phys_loss = physics_loss(model, x_pde, y_pde, t_pde)
        anode_loss = boundary_loss_anode(model)
        cathode_loss = boundary_loss_cathode(model)
        side_loss = boundary_loss_sides(model)
        init_loss = initial_loss(model)
        
        loss = 10 * phys_loss + 100 * anode_loss + 100 * cathode_loss + 50 * side_loss + 100 * init_loss  # \mathcal{L}
        
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # \text{Clip gradients}
        optimizer.step()
        
        scheduler.step(loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != last_lr:
            print(f"Epoch {epoch + 1}: Learning rate reduced to {current_lr}")
            last_lr = current_lr
        
        if (epoch + 1) % 100 == 0:
            loss_history['epochs'].append(epoch + 1)
            loss_history['total'].append(loss.item())
            loss_history['physics'].append(10 * phys_loss.item())
            loss_history['anode'].append(100 * anode_loss.item())
            loss_history['cathode'].append(100 * cathode_loss.item())
            loss_history['sides'].append(50 * side_loss.item())
            loss_history['initial'].append(100 * init_loss.item())
            
            if (epoch + 1) % 500 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Total Loss: {loss.item():.6f}, "
                      f"Physics: {10 * phys_loss.item():.6f}, "
                      f"Anode: {100 * anode_loss.item():.6f}, "
                      f"Cathode: {100 * cathode_loss.item():.6f}, "
                      f"Sides: {50 * side_loss.item():.6f}, "
                      f"Initial: {100 * init_loss.item():.6f}")
    
    plot_losses(loss_history, Ly, phi_anode, output_dir)
    
    return model, loss_history

def train_and_save_solutions(Lx, t_max, epochs, output_dir="pinn_solutions"):  # \text{Train and save all models}
    os.makedirs(output_dir, exist_ok=True)
    params = generate_parameter_sets(Lx, t_max, epochs)
    times = np.linspace(0, t_max, 50)
    
    for idx, param_set in enumerate(params):
        print(f"Training model {idx + 1}/{len(params)} for Ly={param_set['Ly']*1e4:.0f} μm, "
              f"Phi_anode={param_set['phi_anode']:.1f} V...")
        model, loss_history = train_PINN(
            param_set['Lx'], param_set['Ly'], param_set['t_max'],
            param_set['phi_anode'], param_set['epochs'], output_dir
        )
        
        X, Y, phi_preds, c_preds, phi_l_preds, psi, J_preds = evaluate_model(
            model, times, param_set['Lx'], param_set['Ly']
        )
        
        solution = {
            'params': param_set,
            'X': X,
            'Y': Y,
            'phi_preds': phi_preds,
            'c_preds': c_preds,
            'phi_l_preds': phi_l_preds,
            'psi': psi,
            'times': times,
            'J_preds': J_preds,
            'loss_history': loss_history,
            'orientation_note': 'phi_preds, c_preds, phi_l_preds, psi are arrays of shape (30,30) where rows (i) correspond to y-coordinates and columns (j) correspond to x-coordinates due to transpose.'
        }
        solution_filename = os.path.join(output_dir, 
            f"solution_ly_{param_set['Ly']:.4f}_phi_{param_set['phi_anode']:.1f}.pkl")
        with open(solution_filename, 'wb') as f:
            pickle.dump(solution, f)
        
        print(f"Saved solution {idx + 1} to {solution_filename}")
    
    return len(params)

if __name__ == "__main__":
    Lx = 0.01  # \text{cm}
    t_max = 20.0  # \text{s}
    epochs = 5000
    
    num_saved = train_and_save_solutions(Lx, t_max, epochs)
    print(f"Saved {num_saved} solutions to pinn_solutions/")
