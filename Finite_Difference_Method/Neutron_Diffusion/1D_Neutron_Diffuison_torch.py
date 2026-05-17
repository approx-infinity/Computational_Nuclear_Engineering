import torch 
import matplotlib.pyplot as plt
import numpy as np

def main():
    L = 100.0  # thickness of the slab (cm)
    Nx = 1000  # number of grid points
    dx = L / Nx

    S = 100.0       # constant source term
    D = 1.0         # Diffusion coefficient (cm)
    sigma_a = 0.02  # absorption macroscopic cross section (1/cm)

    r = D / dx**2


    # select device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    phi_current = torch.zeros(Nx + 1, dtype=torch.float64, device=device)
    phi_new = torch.zeros(Nx + 1, dtype=torch.float64, device=device)

    tolerance = 1e-8

    while True:
        # BC: phi[0] = phi[Nx] = 0
        phi_new[0] = 0.0
        phi_new[Nx] = 0.0

        # Vectorized update for interior points (1 to Nx-1)
        # phi_current[2:] represents phi[i+1]
        # phi_current[:-2] represents phi[i-1]

        phi_new[1:-1] = (r* (phi_current[2:] + phi_current[:-2]) + S) / (2*r + sigma_a)

        max_diff = torch.max(torch.abs(phi_current - phi_new)).item()

        if max_diff < tolerance:
            break

        phi_current = phi_new.clone()

    
    phi_final = phi_current.cpu().numpy()   # move tensor back to cpu to save it in a file

    with open("Diffusion_result_py.csv", "w") as f:
        # Write spatial grid points
        grid_points = [str(i * dx) for i in range(Nx+1)]
        f.write(" ".join(grid_points) + "\n")

        #Write calculated flux values
        flux_values = [str(val) for val in phi_final]
        f.write(" ".join(flux_values) + "\n")
    
    x_coords = np.linspace(0, L, Nx+1)

    plt.plot(x_coords, phi_final)
    plt.xlabel("Position(x)")
    plt.ylabel("flux")
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    main()
