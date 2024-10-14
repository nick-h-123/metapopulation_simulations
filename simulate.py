import numpy as np
import warnings
import matplotlib.pyplot as plt
import tqdm
from tqdm import tqdm
import os
import pandas as pd

def gaussian_initial(L, N, N0, bc='mixed'):
    # Set and plot initial value
    L_center = L / 2  # center point
    width = L / 25  # standard deviation, controls the width of the Gaussian
    
    # Set the initial condition using a Gaussian function
    x = np.linspace(0, L, N)
    u_0 = N0 * np.exp(-((x - L_center) ** 2) / (2 * width ** 2))
    
    # Apply boundary conditions
    if bc == 'mixed':
        u_0[-2] = u_0[-1]
        u_0[0] = 0
        u_0[1] = 0
    elif bc == 'neumann':
        u_0[1]  = u_0[0]
        u_0[-2] = u_0[-1]
    elif bc == 'dirichlet':
        u_0[0]  = 0
        u_0[-1] = 0
    return u_0


def sigmoid_initial(L, N, N0, growing=True, bc='mixed'):
    # Set and plot initial value
    L_center = L / 2  # center point
    width = L / 25  # controls the steepness of the sigmoid
    # Set the initial condition using a sigmoid function
    u_0 = N0 / (1 + np.exp(- (np.linspace(0, L, N) - L_center) / width))
    if bc == 'mixed':
        u_0[-2] = u_0[-1]
        u_0[0] = 0
        u_0[1] = 0
    elif bc == 'neumann':
        u_0[1] = u_0[0]
        u_0[-2] = u_0[-1]
    if not growing:
        u_0 = np.flip(u_0)
    return u_0

def calculate_num(u, dx):
    if u.ndim == 1:
        return np.sum(u*dx)
    elif u.ndim == 2:
        return np.sum(u*dx, axis=1)
    else:
        print("Invalid input u")

def get_filename(D,v,a,wind,coff,N,dx, bc='mixed'):
    base = f"D_{D:.2g}_v_{v:.5g}_a_{a:.2g}_wind_{wind:.2g}_coff_{coff:.2g}_N_{N}_dx_{dx:.2g}.npy"
    if bc=='mixed':
        return "solution_"+base
    elif bc=='neumann':
        return "solution_n_"+base
    elif bc=='dirichlet':
        return "solution_d_"+base

def save_g_solution(sol_array, D, V, a, wind, coff, N, dx, bc='mixed', save=True):
    if save:
        filename = f"g_sols/"+get_filename(D,V,a,wind,coff,N,dx, bc)
        data_to_save = {
            'params': [D, V, a, wind, coff, N, dx],
            'g_solution': sol_array[-1]
        }
        np.save(filename, data_to_save)
        print(f"Solution saved to {filename}")

class u_simulation:
    def __init__(self, u_0, bc, D, V, a, wind, coff, N, dx):
        self.u_0 = u_0          # Initial condition
        self.bc = bc            # Boundary condition ('dirichlet', 'neumann', 'mixed')
        self.D = D              # Diffusion coefficient
        self.V = V              # Velocity/(Fisher Velocity)
        self.a = a              # Growth parameter
        self.wind = wind        # Wind nonlinearity strength
        self.coff = coff        # Standard nonlinearity strength
        self.N = N              # Number of spatial points
        self.dx = dx            # Spatial step size
        
        self.u = u_0
        self.u_list = np.array([u_0.copy()])
        
    def simulate(self, M, dt, growth_bound=4, debug=False, save_skip=1):
        # debug: boundary conditions
        # save_skip: only save ever save_skip generations for memory reasons
        # Initialize the first time step
        D = self.D * np.ones(self.N)
        #D[:100] /= 100
        #D[-100:] /= 100
        wind = self.wind * np.ones(self.N)
        #wind[:100] /= 1000
        #wind[-100:] /= 1000
        a = self.a
        v = self.V * (2 * (D[0]*a)**0.5)
        coff = self.coff
        N = self.N
        dx = self.dx
        
        num_diffs = [1.0]
        
        aa = 1
        b = N-1
        k = 1e5  # Adjust this parameter to control the steepness of the smooth Heaviside
        x = k * (np.arange(aa, b) - N // growth_bound)
        theta = np.zeros_like(x)
        theta[x >= 0] = 1# / (1 + np.exp(-x[x >= 0]))  # for non-negative x
        theta[x < 0] = 0#np.exp(x[x < 0]) / (1 + np.exp(x[x < 0]))  # for negative x

        for k in tqdm(range(M)):
            # Enforce zero boundary condition on the right

            warnings.simplefilter("error", RuntimeWarning)
            try:
                # Update the interior points
                u = self.u
                u_next = self.u.copy()
                u_center = u[aa:b]
                diff = 0.5 * (u[aa+1:b+1] - u[aa-1:b-1]) / dx
                u_next[aa:b] = u_center + dt * (
                    D[aa:b] * (u[aa+1:b+1] - 2 * u_center + u[aa-1:b-1]) / dx**2 
                    - v * diff
                    + a * theta * u_center
                    - 2 * wind[aa:b] * diff**2
                    - 2 * coff * u_center**2
                )
                # Apply boundary conditions separately
                if self.bc == 'neumann':
                    # Neumann boundary conditions for the first and last points
                    u_next[0] = u[0] + dt * (
                        D[0] * (u[2] - 2 * u[1] + u[0]) / dx**2 
                        - 0.5 * v * (u[2] - u[0]) / dx
                        - 2 * 0.5**2 * wind[0] * (u[2] - u[0])**2 / dx**2
                        - 2 * coff * u[1]**2
                    )
                    u_next[-1] = u[-1] + dt * (
                        D[-1] * (u[-3] - 2 * u[-2] + u[-1]) / dx**2 
                        - 0.5 * v * (u[-1] - u[3]) / dx
                        + a * u[-2]
                        - 2 * 0.5**2 *  wind[-1] * (u[-1] - u[-3])**2 / dx**2
                        - 2* coff * u[-2]**2
                    )
                    u_next[1] = u_next[0]
                    u_next[-2] = u_next[-1]
                    if debug:
                        print(f"u_next[1]-u_next[0]: {u_next[1]-u_next[0]}")
                        print(f"u_next[-1]-u_next[-2]: {u_next[-1]-u_next[-2]}")

                elif self.bc == 'dirichlet':
                    # Dirichlet boundary conditions for the first and last points
                    # u_next = u_next
                    u_next[0] = 0
                    u_next[-1] = a/(2*coff)
                elif self.bc == 'mixed':
                    u_next -= u_next[0]
                    u_next[-1] = u[-1] + dt * (
                        D[-1] * (u[-3] - 2 * u[-2] + u[-1]) / dx**2 
                        - 0.5 * v * (u[-1] - u[-3]) / dx
                        + a * u[-2]
                        - 2 * 0.5**2 * wind[-1] * (u[-1] - u[-3])**2 / dx**2
                        - 2 * coff * u[-2]**2
                    )
                    u_next[-2] = u_next[-1]
                else:
                    raise ValueError(f"Use 'neumann' or 'dirichlet' or 'mixed'.")              

                # Append the updated u_next to the list

                #u_next = np.maximum(u_next, 0) # enforce non-negativity

                if (k+1) % save_skip == 0 or k == M-2:
                    self.u_list = np.append(self.u_list, [u_next.copy()], axis=0)
                    num_diff = calculate_num(u_next, dx)/calculate_num(u, dx)-1
                    num_diffs.append(num_diff)
                self.u = u_next

            except RuntimeWarning as e:
                # Print the time of the error
                print(f"RuntimeWarning occurred at time step {k} and time {k*dt}: {e}")
                break
        print(f"Simulation complete for v = {self.V}*v_f")
        print("Final relative number difference:", num_diffs[-1])
        # Convert the list of arrays into a 2D NumPy array
        filename = "u_sols/"+get_filename(D[0],self.V,a,wind[0],coff,N,dx,bc=self.bc)
        np.save(filename, self.u_list)
        print(f"Solution saved to {filename}")

def plot_time_slices(u, L, dt, N, M, D, V, a, coff, eps_over_delta, 
                     save_skip=1, fractions=[0.2, 0.4, 0.6, 0.8, 1.0], log_scale=False, 
                     xlim=None, ylim=None, psi_ylim=None, gamma=False, psi=True):  
    x = np.linspace(0, 1, N, dtype='f')
    colors = plt.cm.viridis(np.linspace(0, 1, len(fractions)))
    
    v = V * 2 * (D*a)**0.5
    # Determine the number of subplots
    ncols = 3 if psi else 1
    fig, axes = plt.subplots(1, ncols, figsize=(10 * ncols, 8))
    
    if coff!=0:
        asym=a/(2*coff)
    else:
        asym=u[-1][-1]

    for idx, fraction in enumerate(fractions):
        time_index = int(len(u) * fraction) if fraction not in [0, 1] else (0 if fraction == 0 else len(u) - 1)
        t_slice = time_index * dt * save_skip
        u_slice = u[time_index, 1:]
        X_slice = x[1:]
        u_slice = u_slice#-u_slice[0]
        color = colors[idx]

        # Plot on the first subplot (original or Γ)
        axes[0].plot(X_slice, u_slice, label=f't={t_slice:.2g}', color=color)
        if log_scale:
            axes[0].set_yscale("log")
        axes[0].set_title('Original' if not gamma else 'Γ')
        
        if idx == 0: norm = 1e10#u_slice[int((L/dx)//4)] * np.exp(-(0.5 * v * L/4 / D))
        if psi:
            # Plot the transformed version on the second subplot
            n = 500
            factor = (-X_slice*L+L/4) * v / D
            c_slice = np.exp(np.log(np.maximum(u_slice,0)+1e-100) + factor)
            psi_slice = np.exp(np.log(np.maximum(u_slice,0)+1e-100) + factor/2)
            #psi_slice /= norm
            axes[1].plot(X_slice, psi_slice, label=f't={t_slice:.2g}', color=color)
            axes[2].plot(X_slice, c_slice, label=f't={t_slice:.2g}', color=color)
            #axes[1].plot(X_slice, factor, color="pink")
            if log_scale:
                axes[1].set_yscale("log")
            axes[1].set_title(r'$\psi = u e^{-vx/2D}$')
    
    # Adjust settings for each axis
    axes[0].axvline(x=1/2, color='grey', linestyle='--')
    axes[0].axhline(y=asym, color='lightseagreen', linestyle='--')
    #axes[0].text(0, a/coff-0.05, 'a/δ', color='lightseagreen')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('u' if not gamma else 'Γ')
    axes[0].legend()

    if psi:
        axes[1].axvline(x=1/2, color='grey', linestyle='--')
        axes[1].set_xlabel('x')
        #axes[1].yaxis.tick_right()
        axes[1].yaxis.set_label_position("right")
        axes[1].set_ylabel('ψ')
        if psi_ylim is not None: axes[1].set_ylim(psi_ylim)
        axes[1].legend()
        axes[2].axvline(x=1/2, color='grey', linestyle='--')
        axes[2].set_title(r'$u e^{-v x/D}$')

    if xlim is not None:
        axes[0].set_xlim(xlim)
        axes[1].set_xlim(xlim)
        axes[2].set_xlim(xlim)
    if ylim is not None:
        axes[0].set_ylim(ylim)

    # Adjust layout to make subplots flush vertically
    plt.subplots_adjust(wspace=0.05)
    
    plt.suptitle(f'Time Slices, vf = {V:.3f}v, ε/δ = {eps_over_delta:.3f}')
    plt.show()
    
def simulate_g(g_0, u_0, bc, D, V, a, wind, coff, N, M, dx, dt, growth_bound=4, debug=False, save_skip=1, save=False):
    # u_0: np.array of size M
    # debug: boundary conditions
    # save_skip: only save ever save_skip generations for memory reasons
    # Initialize the first time step
    v = V * (2 * (D*a)**0.5)
    g = g_0.copy()
    u = u_0#/np.sum(g_0 * u_0) * dx
    # Store each time step in a list
    g_list = [g.copy()]
    for k in tqdm(range(M - 1)):
        # Enforce zero boundary condition on the right
        b = N-1
        warnings.simplefilter("error", RuntimeWarning)
        try:
            # Update the interior points
            g_next = g.copy()
            g_next[1:b] = g[1:b] + dt * (
                D * (g[0:b-1] - 2 * g[1:b] + g[2:b+1]) / dx**2 
                + 0.5 * v * (g[2:b+1] - g[0:b-1]) / dx
                + a * np.heaviside(np.arange(1, b) - N // growth_bound, 0) * g[1:b]
                + 2 * wind * g[1:b] * (u[:b-1] - 2 * u[1:b] + u[2:b+1]) / dx**2
                + 2 * 0.25 * wind * (g[2:b+1] - g[0:b-1]) / dx * (u[2:b+1] - u[0:b-1]) / dx
                - 2 * coff * u[1:b] * g[1:b]
            )
            # Apply boundary conditions separately
            if bc == 'neumann':
                # Neumann boundary conditions for the first and last points
                g_next[0] = g[0] + dt * (
                    D * (g[2] - 2 * g[1] + g[0]) / dx**2 
                    + 0.5 * v * (g[2] - g[0]) / dx
                    + 2 * wind * g[1] * (u[2] - 2 * u[1] + u[0]) / dx**2
                    + 2 * 0.25 * wind * (g[2] - g[0]) / dx * (u[2] - u[0]) / dx
                    - 2 * coff * u[1] * g[1]
                )
                g_next[-1] = g[-1] + dt * (
                    D * (g[-3] - 2 * g[-2] + g[-1]) / dx**2 
                    + 0.5 * v * (g[-1] - g[-3]) / dx
                    + a * g[-2]
                    + 2 * wind * g[-2] * (u[-3] - 2 * u[-2] + u[-1]) / dx**2
                    + 2 * 0.25 * wind * (g[-1] - g[-3]) / dx * (u[-1] - u[-3]) / dx
                    - 2 * coff * u[-2] * g[-2]
                )
                g_next[-2] = g_next[-1]
                g_next[1] = g_next[0]
                g_next -= g_next[-1]
                #if debug:
                #    print(f"u_next[1]-u_next[0]: {_next[1]-u_next[0]}")
                #    print(f"u_next[-1]-u_next[-2]: {u_next[-1]-u_next[-2]}")
            elif bc == 'mixed':
                g_next[0] = g[0] + dt * (
                    D * (g[2] - 2 * g[1] + g[0]) / dx**2 
                    + 0.5 * v * (g[2] - g[0]) / dx
                    + 2 * wind * g[1] * (u[2] - 2 * u[1] + u[0]) / dx**2
                    + 2 * 0.25 * wind * (g[2] - g[0]) / dx * (u[2] - u[0]) / dx
                    - 2 * coff * u[1] * g[1]
                )
                g_next[1] = g_next[0]
                g_next -= g_next[-1]
            elif bc == 'dirichlet':
                # Dirichlet boundary conditions for the first and last points
                g_next[0] = 0
                g_next[-1] = 0  
                
            g_next = np.maximum(g_next, 0)

            if k == M-2:
                final_num_diff = calculate_num(g_next, dx)/calculate_num(g, dx)-1
                
            g = g_next/np.sum(g_next * u * dx)
            g_save = g_next/np.sum(g_next * u * dx) 
            # Append the updated u_next to the list
            if (k+1) % save_skip == 0 or k == M-2: 
                g_list.append(g_save)         
        
        except RuntimeWarning as e:
            # Print the time of the error
            print(f"RuntimeWarning occurred at time step {k} and time {k*dt}: {e}")
            final_num_diff="N/A"
            break
        
    print("Final relative number difference:", final_num_diff)
    # Convert the list of arrays into a 2D NumPy array
    g_array = np.array(g_list)
    
    save_g_solution(g_array, D, V, a, wind, coff, N, dx, bc=bc, save=save)
    
    return g_array

def plot_g_time_slices(u, c, L, dt, N, M, V, D, a, eps_over_delta, wind=0,
                     save_skip=1, fractions=[0.2, 0.4, 0.6, 0.8, 1.0], log_scale=False, 
                     xlim=None, ylim=None, psi_xlim=None, psi_ylim=None, gamma=False, psi=True, exact=False):  
    x = np.linspace(0, L, N+1, dtype='f')
    colors = plt.cm.viridis(np.linspace(0, 1, len(fractions)))
    v = V * 2 * (D*a)**0.5
    
    # Determine the number of subplots
    if psi:
        ncols = 3
        fig, axes = plt.subplots(1, ncols, figsize=(10 * ncols, 8))
    
        if psi_xlim is None: psi_xlim = (0,1.)
        psi_max = 1e-100
        psi_min = 1e100

        for idx, fraction in enumerate(fractions):
            time_index = int(len(c) * fraction) if fraction not in [0, 1] else (0 if fraction == 0 else len(c) - 1)
            t_slice = time_index * dt * save_skip
            c_slice = c[time_index,:]
            c_slice /= np.sum(u*c_slice*L/N)
            X_slice = x[1:]
            #c_slice = np.maximum(c_slice-c_slice[0], 0)
            color = colors[idx]
            #print(c_slice[:10])
            # Plot on the first subplot (original or Γ)
            axes[0].plot(X_slice/L, c_slice, label=f't={t_slice:.2g}', color=color)
            if log_scale:
                axes[0].set_yscale("log")
            axes[0].set_title('c(x)' if not gamma else 'Γ')
            
            # Plot the transformed version on the second subplot
            n = 500
            factor =  (X_slice-L/4) * v / (2*D)#-(L/4-X_slice) * v / (2 * D)
            psi_slice = np.exp(np.log(np.maximum(c_slice,0)+1e-100) + factor)
            #u_slice = np.exp(np.log(np.maximum(c_slice,0)+1e-100) + 2*factor)
            axes[1].plot(X_slice/L, psi_slice, label=f't={t_slice:.2g}', color=color)
            #axes[1].plot(X_slice/L, u_slice, label=f't={t_slice:.2g}', color=color)
            #axes[1].plot(X_slice, factor, color="pink")
            if log_scale:
                axes[1].set_yscale("log")
            axes[1].set_title(r'$\psi = c(x,t) e^{vx/2D}$')
            mask = (X_slice >= psi_xlim[0]*L) & (X_slice <= psi_xlim[1]*L)
            psi_min = np.minimum(psi_min, np.min(psi_slice[mask]))
            psi_max = np.maximum(psi_max, np.max(psi_slice[mask]))
            
            axes[2].plot(X_slice/L, u*c_slice, label=f't={t_slice:.2g}', color=color)

        if psi_ylim is None: psi_ylim = (0.9*psi_min, 1.5*psi_max)

        if exact:
            #np.log(np.maximum(u,0)+1e-100)
            c_SS = u * np.exp(- v*x[1:]/D - 2 * wind * u / D)
            norm = np.sum(u*c_SS*L/N)
            c_SS /= norm
            psi_SS = np.exp(np.log(np.maximum(c_SS,0)+1e-100) + (x[1:]-L/4) * v / (2*D))
            axes[0].plot(X_slice/L, c_SS, linestyle="--", label="SS")  
            axes[1].plot(X_slice/L, psi_SS, linestyle="--", label="SS")  
            axes[2].plot(X_slice/L, u**2 * np.exp(-v*x[1:]/D-2*wind*u/D)/norm, linestyle="--", label="SS")
        # Adjust settings for each axis
        axes[0].set_xlabel('x/L')
        axes[0].set_ylabel('c')

        axes[1].yaxis.tick_right()
        axes[1].yaxis.set_label_position("right")
        #axes[1].set_ylabel('ψ(c)')
        #axes[1].set_ylim(min(psi_slice[:int(len(psi_slice)*0.8)]), max(psi_slice[:int(len(psi_slice)*0.8)]))
        axes[1].set_xlim((0,0.8))
        
        axes[2].yaxis.tick_right()
        axes[2].yaxis.set_label_position("right")
        axes[2].set_ylabel('u g')
        axes[2].set_title('u(x) * c(x,t)')
        #axes[1].set_ylim(psi_ylim)
        #axes[1].set_xlim(psi_xlim)

        for ax in axes: 
            ax.set_xlim(xlim)
            ax.set_xlabel('x/L')
            ax.axvline(x=1/4, color='grey', linestyle='--')
            ax.legend()
        #axes[2].set_xlim((0,1))
            
        #if xlim is not None:
            #axes[0].set_xlim(xlim)
            #if not psi: axes[1].set_xlim(xlim)
        if ylim is not None:
            axes[0].set_ylim(ylim)
        
        # Adjust layout to make subplots flush vertically
        plt.subplots_adjust(wspace=0.05)

        plt.suptitle(f'Time Slices, vf = {V:.3g}v, ε/δ = {eps_over_delta:.3f}, wind = {wind:.3f}')        
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        for idx, fraction in enumerate(fractions):
            time_index = int(len(u) * fraction) if fraction not in [0, 1] else (0 if fraction == 0 else len(u) - 1)
            t_slice = time_index * dt * save_skip
            u_slice = u[time_index, 1:]
            X_slice = x[1:]
            u_slice = u_slice - u_slice[0]
            color = colors[idx]

            # Plot on the single subplot (original or Γ)
            ax.plot(X_slice/L, u_slice, label=f't={t_slice:.2g}', color=color)
            if log_scale:
                ax.set_yscale("log")
            #ax.set_title('Original' if not gamma else 'Γ')

        # Add vertical and horizontal lines for reference
        ax.axvline(x=1/2, color='grey', linestyle='--')
        ax.axhline(y=a/coff, color='lightseagreen', linestyle='--')
        #ax.text(0, a/coff-0.05, 'a/δ', color='lightseagreen')

        # Adjust axis labels and legend
        ax.set_xlabel('x/L')
        ax.set_ylabel('u' if not gamma else 'Γ')
        ax.legend()

        # Set x and y limits if provided
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        
        plt.title(f'Time Slices, vf = {V:.3f}v, ε/δ = {eps_over_delta:.3g}')    
    
    plt.show()
    
def g_sweep(simulation_params, M, dt, bc='mixed', save_skip=1000, load_g=False, plot_func=None):
    # Sweep over parameters
    g_sols = []
    
    for i, params in enumerate(simulation_params):
        
        D, V, a, wind, coff, N, dx, dt = params
        L = int((N-1)*dx)
        #u_filename = f"u_sols/solution_D_{D}_v_{v}_a_{a}_wind_{wind}_coff_{coff}_N_{N}_dx_{dx}.npy"
        u_filename = "u_sols/" + get_filename(D,V,a,wind,coff,N,dx, bc=bc)
        # Check if the previous simulation of u exists
        if os.path.exists(u_filename):
            # Load the u_sols data from the file
            u_sol = np.load(u_filename)
            print(f"Loaded u_sols for D={D}, V={V}, a={a}, wind={wind}, coff={coff}, N={N}, dx={dx}")
        else:
            print("Missing file for u(x):", u_filename)
        sigma0 = L / 20  # controls the steepness of the sigmoid
        N0 = 10  # asymptote for the sigmoid
        if load_g:
            g_filename = "g_sols/" + get_filename(D,V,a,wind,coff,N,dx,bc=bc)
            if os.path.exists(g_filename):
                print(f"Loaded pre-simulated c(x)")
                data = np.load(g_filename, allow_pickle=True).item()
                g_0 = data['g_solution']
            else: 
                print(f"File '{g_filename}' NOT found. Using sigmoidal initial condition.")
                g_0 = N0/(sigma0 * np.sqrt(2 * np.pi)) * np.exp( - (np.linspace(0, L, N) - L // 2)**2 / (2 * sigma0**2))
        else:
            g_0 = N0/(sigma0 * np.sqrt(2 * np.pi)) * np.exp( - (np.linspace(0, L, N) - L // 2)**2 / (2 * sigma0**2))
        #print(u_sol[-1])
        vf = 2*(D*a)**0.5
        g_sols = simulate_g(g_0, u_sol[-1], "neumann", D, V*vf, a, wind, coff, N, M, dx, dt, debug=False, save_skip=save_skip, save=True)
        
        if i == plot_func:
            plot_g_time_slices(u_sol[-1][:-1], g_sols, L, dt, N, M, V, D, a, a/coff, xlim=(0,0.5),
                     save_skip=save_skip, fractions=[0.2, 0.4, 0.6, 0.8, 1.0], log_scale=False)
            

def plot_number_of_islands(data_filename, params, power=1, xlim=None, one_minus=True, log=True, ylog=False):
    # Load the CSV data into a pandas DataFrame
    df = pd.read_csv(data_filename)
    
    plt.figure(figsize=(8, 6))
    for param in params:
        D, a, wind, coff, N, dx = param
        # Filter the DataFrame for the specific parameters
        filtered_df = df[
            (df['D'] == D) &
            (df['a'] == a) &
            (df['wind'] == wind) &
            (df['coff'] == coff) &
            (df['N'] == N) &
            (df['dx'] == dx)
        ]

        # Plot N_I vs. v
        filtered_df = filtered_df.sort_values(by='N_I')

        # Plot N_I vs. v
        if one_minus:
            plt.plot(filtered_df['N_I'], 1 - filtered_df['v'], marker='o', linestyle='--', 
                 label=f"D={D}, a={a}, wind={wind}, coff={coff}")
            plt.ylabel('1-v/vf')
            plt.legend(loc='upper right')  # Legend moved to bottom right
        else:
            plt.plot(filtered_df['N_I'], filtered_df['v'], marker='o', linestyle='--', 
                 label=f"D={D}, a={a}, wind={wind}, coff={coff}")
            if power==1: plt.ylabel(r'v/v_f')
            else: plt.ylabel(r'$(v/v_f)^3$')
            plt.legend(loc='lower right')  # Legend moved to bottom right
        if log:
            plt.xscale('log')
        if ylog:
            plt.yscale('log')
        plt.xlabel('N_I')
        plt.grid(True)
    
    if xlim: plt.xlim(xlim)

    plt.show()
    
def plot_nonlin_vs_NI(data_filename, params, power=1, log=True, ylim_c=None, ylim_w=None, ylog=False, loc='upper right'):
    # Load the CSV data into a pandas DataFrame
    df = pd.read_csv(data_filename)
    
    # Set up the figure and two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # Sharing y-axis for consistent scaling

    for param in params:
        D, a, wind, coff, N, dx = param
        
        # Filter the DataFrame for the specific parameters
        filtered_df = df[
            (df['D'] == D) &
            (df['a'] == a) &
            (df['wind'] == wind) &
            (df['coff'] == coff) &
            (df['N'] == N) &
            (df['dx'] == dx)
        ]
        
        # Sort by N_I for proper plotting
        filtered_df = filtered_df.sort_values(by='N_I')

        # Plot c_nonlin vs. N_I on the first subplot (axes[0])
        axes[0].plot(filtered_df['N_I'], filtered_df['c_nonlin']**power, marker='o', linestyle='--', 
                     label=f"D={D}, a={a}, wind={wind}, coff={coff}")
        #print(filtered_df['c_nonlin'])
        axes[0].set_xlabel('N_I')
        #axes[0].set_title('c_nonlin vs N_I')
        axes[0].grid(True)

        # Plot w_nonlin vs. N_I on the second subplot (axes[1])
        axes[1].plot(filtered_df['N_I'], filtered_df['w_nonlin']**power, marker='o', linestyle='--', 
                     label=f"D={D}, a={a}, wind={wind}, coff={coff}")
        axes[1].set_xlabel('N_I')
        #axes[1].set_title('w_nonlin vs N_I')
        axes[1].grid(True)

        if log:
            axes[0].set_xscale('log')
            axes[1].set_xscale('log')
        if ylog:
            axes[0].set_yscale('log')
            axes[1].set_yscale('log')

    # Set y-label only on the left subplot (shared y-axis)
    if power == 1:
        axes[0].set_ylabel('∫u^2 c')
        axes[1].set_ylabel('∫u\'^2 c')
    else:
        axes[0].set_ylabel(f'(∫u^2 c)^{power:3g}')
        axes[1].set_ylabel(f'(∫u\'^2 c)^{power:3g}')
    if ylim_c is not None: axes[0].set_ylim(ylim_c)
    if ylim_w is not None: axes[1].set_ylim(ylim_w)
    # Add legends to both subplots
    axes[0].legend(loc=loc)
    axes[1].legend(loc=loc)
    
    plt.suptitle("Conventional and Wind Nonlinearities")
    plt.tight_layout()  # Adjust layout to avoid overlap
    plt.show()

    
def process_gs(directory='g_sols'):
    # Initialize an empty list to collect data
    data_list = []
    # Iterate over each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".npy"):  # Only process .npy files
            filepath = os.path.join(directory, filename)
            
            # Load the file (assuming it contains a dictionary with params and g_sol)
            data = np.load(filepath, allow_pickle=True).item()
            # Unpack the parameters and solution
            D, v, a, wind, coff, N, dx = data['params']  # Unpacking the parameters
            g_sol = data['g_solution']  # The solution array
            u_file = "u_sols/"+get_filename(D,v,a,wind,coff,N,dx,bc=bc)
            u_data = np.load(u_file)
            u_sol = u_data[-1]
            # Calculate the sum over g_sol * dx
            N_I = np.sum(g_sol * dx)
            conv_nonlin = np.sum(u_sol**2 * g_sol * dx)
            wind_nonlin = np.sum((u_sol[1:]-u_sol[:-1])**2 * g_sol[:-1] / dx)
            # Append the parameters and result to the data_list
            data_list.append([D, v, a, wind, coff, N, dx, N_I, conv_nonlin, wind_nonlin])
            
    # Convert the list of data to a pandas DataFrame
    df = pd.DataFrame(data_list, columns=['D', 'v', 'a', 'wind', 'coff', 'N', 'dx', 'N_I', "c_nonlin", "w_nonlin"])

    # Save the DataFrame to a CSV file
    output_filepath = os.path.join(directory, 'g_data.csv')
    df.to_csv(output_filepath, index=False)

    print(f"Processed data saved to {output_filepath}")    