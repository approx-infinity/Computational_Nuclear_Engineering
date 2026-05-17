import torch 
import numpy as np
import matplotlib.pyplot as plt

def main():
    Lx = 100.0    # thickness in x direction
    Ly = 100.0    # thickness in y direction
    N = 1000       # number of grid points in x & y direction
    h = Lx / N    # size of the grid cells

    D = 1.0            # Diffusion Coefficenit (cm)
    S = 100.0          # Source term
    sigma_a1 = 0.02    # macroscopic cross section for 0-40cm and 60-100cm
    sigma_a2 = 0.1     # Absorption macroscopic cross section for 40-60cm (control rod region)

    r = D / h**2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define grids
    x = torch.linspace(0, Lx, N, dtype=torch.float64, device=device)
    y = torch.linspace(0, Ly, N, dtype=torch.float64, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    # define material region
    # control rod region: 40 <= x <=60 and 40 <= y <= 60
    control_rod_mask = (X >= 40.0) & (X <= 60.0) & (Y >= 40.0) & (Y <= 60.0)

    # initialize denominator matrix for every grid point
    denominator = torch.full((N, N), 4*r + sigma_a1, dtype=torch.float64, device=device)
    denominator[control_rod_mask] = 4*r + sigma_a2

    phi_current = torch.zeros((N,N), dtype=torch.float64, device=device)
    phi_new = torch.zeros((N,N), dtype=torch.float64, device=device)

    tolerance = 10e-7

    while True:
        # BC: phi[0] = phi[N] = 0
        
        # 2D vectorized Stencil Update:
        # phi_current[2:, 1:-1]  -> phi_current[i+1, j]
        # phi_current[:-2, 1:-1] -> phi_current[i-1, j]
        # phi_current[1:-1, 2:]  -> phi_current[i, j+1]
        # phi_current[1:-1, :-2] -> phi_current[i, j-1]
        phi_new[1:-1, 1:-1] = (S + r * (
            phi_current[2:, 1:-1] +
            phi_current[:-2, 1:-1] +
            phi_current[1:-1, 2:] +
            phi_current[1:-1, :-2]
        )) / denominator[1:-1, 1:-1]

        max_diff = torch.max(torch.abs(phi_current - phi_new)).item()

        if max_diff < tolerance:
            break

        phi_current = phi_new.clone()

    phi_final = phi_current.cpu().numpy()
    X_cpu = X.cpu().numpy()
    Y_cpu = Y.cpu().numpy()

    with open("2D_Diffusion_result.csv", "w") as f:
        f.write("x(cm) y(cm) flux(n/cm2)\n")
        for i in range(N):
            for j in range(N):
                f.write(f"{X_cpu[i, j]:.4f} {Y_cpu[i, j]:.4f} {phi_final[i, j]:.6f}\n")

    
    plt.figure(figsize=(8, 6))
    heatmap = plt.pcolormesh(X_cpu, Y_cpu, phi_final, cmap="viridis")
    plt.colorbar(heatmap, label="Neutron Flux ($\phi$)")

    plt.xlabel("X position (cm)")
    plt.ylabel("Y position (cm)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

