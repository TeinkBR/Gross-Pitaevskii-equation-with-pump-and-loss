"""PRB 106, 245304, deterministic Eqs. (1)-(2); periodic 2D FFT grid.
Units: micrometres, ps, meV; |psi|² and n_X in micrometres**-2.
P is static; the imaginary wall is an added absorber, not paper physics.
"""
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

class GPE_solver_class:
    def __init__(self, P=None, points=256, half_length=48.0,
                 alpha=0.00016, G=0.0008, gamma=1/5.5, Gamma_X=1/5.5,
                 R=0.002/0.6582119569, eta=2.0, hbar=0.6582119569, m=0.28):
        assert eta == 0 or Gamma_X > 0, 'Nonzero eta requires Gamma_X > 0.'
        self.points = points
        self.hbar, self.m = hbar, m
        assert hbar > 0 and m > 0 and min(alpha, G, gamma, Gamma_X, R, eta) >= 0
        self.half_length = half_length
        self.alpha, self.G = alpha, G
        self.gamma, self.Gamma_X = gamma, Gamma_X
        self.R, self.eta = R, eta
        self.P = P
        self.t = 0.0
        self.method = 'strang'
    def grid(self, points=None, half_length=None):
        if points is not None:
            self.points = points
        if half_length is not None:
            self.half_length = half_length
        self.dx = 2 * self.half_length / self.points
        self.x = -self.half_length + self.dx * torch.arange(self.points, dtype=torch.float64)
        self.X, self.Y = torch.meshgrid(self.x, self.x, indexing='ij')
        self.potential = torch.zeros_like(self.X, dtype=torch.complex128)
        if self.P is None:
            self.P = torch.zeros_like(self.X)
        else:
            self.P = torch.zeros_like(self.X) + torch.as_tensor(self.P, dtype=self.X.dtype, device=self.X.device)
        return self.x
    def fft(self):
        self.k = 2 * torch.pi * torch.fft.fftfreq(self.points, d=self.dx, dtype=self.x.dtype, device=self.x.device)
        self.kinetic_energy = self.hbar**2 / (2*self.m) * (self.k[:, None] ** 2 + self.k[None, :] ** 2)
        return self.k
    def initial_state(self, psi, n_X=0.0):
        self.psi = torch.as_tensor(psi, dtype=self.potential.dtype, device=self.X.device).clone()
        self.n_X = torch.zeros_like(self.X) + torch.as_tensor(n_X, dtype=self.X.dtype, device=self.X.device)
        assert self.psi.shape == self.X.shape and torch.all(self.n_X >= 0)
        self.t = 0.0
        return (self.psi, self.n_X)
    def gaussian_wave(self, envelope_alpha=-0.125, x0=0.0, k0=0.0, y0=0.0, ky0=0.0):
        radius_squared = (self.X - x0) ** 2 + (self.Y - y0) ** 2
        phase = k0 * (self.X - x0) + ky0 * (self.Y - y0)
        return torch.exp(envelope_alpha * radius_squared) * torch.exp(1j * phase)
    def vortex_wave(self, envelope_alpha=-0.125, charge=1, x0=0.0, y0=0.0):
        X, Y = (self.X, self.Y)
        envelope = self.gaussian_wave(envelope_alpha=envelope_alpha, x0=x0, y0=y0)
        core = X - x0 + 1j * (1 if charge > 0 else -1 if charge < 0 else 0) * (Y - y0)
        return core ** abs(charge) * envelope
    def gaussian_pump(self, mu=(0.0, 0.0), P0=0.0, sigma=3.0):
        X, Y = (self.X, self.Y)
        radius_squared = (X - mu[0]) ** 2
        radius_squared = radius_squared + (Y - mu[1]) ** 2
        return P0 * torch.exp(-radius_squared / (2 * sigma ** 2))
    def rotating_pump(self, t=0.0, strength=0.6, radius=5.0, rotation=0.15):
        X, Y = (self.X, self.Y)
        angle = X.new_tensor(rotation * t)
        radius_squared = X ** 2 + Y ** 2
        radial_factor = strength * torch.exp(1 - radius_squared / radius ** 2) / radius ** 2
        along_lobe = X * torch.cos(angle) + Y * torch.sin(angle)
        return 4 * radial_factor * along_lobe ** 2
    def tanh_wall(self, wall_position=35.0, wall_width=2.5, wall_strength=40.0, geometry="square"):
        edge = torch.maximum(abs(self.X), abs(self.Y)) 
        #if geometry == "square" else torch.hypot(self.X, self.Y)
        return -1j * wall_strength * 0.5 * (1.0 + torch.tanh((edge - wall_position) / wall_width))
    def infinite_wall(self, wall_position=30, wall_width=2.0, wall_height=1000.0):  # Legacy name: absorbing strip.
        edge = torch.abs(self.X)
        edge = torch.maximum(edge, torch.abs(self.Y))
        inside_wall = (edge >= wall_position) & (edge < wall_position + wall_width)
        return -1j * wall_height * inside_wall
    def Potential_V(self, boundary_type=None):
        walls = () if boundary_type is None else (boundary_type,) if callable(boundary_type) else boundary_type
        for wall in walls:
            self.potential += wall()
        return self.potential

    def time_derivatives(self, t, psi, n_X, include_kinetic=True):
        """Paper Eqs. (1)-(2): joint stage fields, static P; V is an added energy."""
        density = abs(psi) ** 2
        potential = self.potential + self.alpha * density + self.reservoir_energy(n_X)
        dpsi = (-1j * potential / self.hbar + 0.5 * (self.R * n_X - self.gamma)) * psi
        dn_X = self.P - (self.Gamma_X + self.R * density) * n_X
        if include_kinetic:
            kinetic = torch.fft.ifftn(self.kinetic_energy * torch.fft.fftn(psi))
            dpsi = dpsi - 1j * kinetic / self.hbar
        return (dpsi, dn_X)
    def reservoir_energy(self, n_X, P=None):
        if P is None:
            P = self.P
        background = self.eta * P / self.Gamma_X if self.eta else 0.0
        return self.G * (n_X + background)
    @torch.no_grad()
    def rk4_run(self, dt):
       # if dt * self.kinetic_energy.max() / self.hbar > 8**0.5:
        #    raise ValueError("RK4 free-kinetic stability exceeded: reduce dt; local terms may require more.")
        psi, n, t = (self.psi, self.n_X, self.t)
        k1, q1 = self.time_derivatives(t, psi, n)
        k2, q2 = self.time_derivatives(t + dt / 2, psi + dt * k1 / 2, n + dt * q1 / 2)
        k3, q3 = self.time_derivatives(t + dt / 2, psi + dt * k2 / 2, n + dt * q2 / 2)
        k4, q4 = self.time_derivatives(t + dt, psi + dt * k3, n + dt * q3)
        self.psi = psi + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        self.n_X = n + dt * (q1 + 2 * q2 + 2 * q3 + q4) / 6
        self.t += dt
        self.method = 'rk4'
        return (self.psi, self.n_X)
    @torch.no_grad()
    def strang_step(self, dt):
        """Local midpoint exponential half / exact kinetic full / local half; order 2."""
        psi, n = (self.psi, self.n_X)
        half_dt = dt / 2
        for half in range(2):
            #safe_rate = torch.where(rate == 0, torch.ones_like(rate), rate)
          #  source = -torch.expm1(-rate * half_dt / 2) / safe_rate
           # source = torch.where(rate == 0, half_dt / 2, source)
            density = psi.real * psi.real + psi.imag * psi.imag
            density_rate = self.R * n - self.gamma + 2 * self.potential.imag / self.hbar
            density_mid = density * torch.exp(density_rate * half_dt / 2)
            #energy_mid = self.potential + self.alpha * density_mid + self.reservoir_energy(n_mid)

            #density = psi.real * psi.real + psi.imag * psi.imag
            psi = psi * torch.exp(-1j*(half_dt/self.hbar) *(self.alpha * density_mid + self.G *(n + (self.eta / self.Gamma_X) * self.P) + self.potential) + 0.5*self.hbar*(self.R * n - self.gamma))#(-1j * energy_mid / self.hbar + 0.5 * (self.R * n_mid - self.gamma)) * half_dt)
            
            #density = psi.real * psi.real + psi.imag * psi.imag
            rate = self.Gamma_X + self.R * density_mid
            n = n * torch.exp(-rate * half_dt) + self.P * half_dt

            # rate = self.Gamma_X + self.R * density_mid
          #  safe_rate = torch.where(rate == 0, torch.ones_like(rate), rate)
           # source = -torch.expm1(-rate * half_dt) / safe_rate
           # source = torch.where(rate == 0, half_dt, source)
            #n = n * torch.exp(-rate * half_dt) + self.P * half_dt
            if half == 0:
                modes = torch.fft.fftn(psi)
                psi = torch.fft.ifftn(modes * torch.exp(-1j * self.kinetic_energy * dt / self.hbar))
        self.psi, self.n_X = (psi, n)
        self.t += dt
        self.method = 'strang'
        return (self.psi, self.n_X)
    def particle_number_rate(self):
        density = abs(self.psi) ** 2
        rate = self.P - self.gamma * density - self.Gamma_X * self.n_X
        rate = rate + 2 * self.potential.imag / self.hbar * density
        return self.dx ** 2 * rate.sum()
    def record_state(self):
        self.time_records.append(self.t)
        self.field_records.append(self.psi.clone())
        self.reservoir_records.append(self.n_X.clone())
        self.P_records.append(self.P.clone())
        self.source_records.append(self.source_integral.clone())
    @torch.no_grad()
    def time_evolution(self, dt, t_final, record_every=40, live=True, plot_every=10, step_method=None):
        assert dt > 0 and t_final >= self.t
        if step_method is None:
            step_method = self.strang_step
        if live:
            plt.ion()
            self.figure, self.ax = plt.subplots(2)

            self.visualise()
        self.time_records, self.field_records, self.reservoir_records = ([], [], [])
        self.P_records, self.source_records = ([], [])
        self.source_integral = self.X.new_zeros(())
        self.record_state()
        step_number = 0
        while self.t < t_final:
            step_size = min(dt, t_final - self.t)
            before = self.particle_number_rate()
            step_method(step_size)
            if not (torch.isfinite(self.psi).all() and torch.isfinite(self.n_X).all() and (self.n_X >= 0).all()):
                raise FloatingPointError("Nonfinite field or negative reservoir: refine dt / inspect inputs.")
            after = self.particle_number_rate()
           # self.source_integral += step_size * (before + after) / 2
            step_number += 1
            final = self.t >= t_final
            
            if live and (step_number % plot_every == 0 or final):
                self.visualise()
                plt.pause(0.001)
            if step_number % record_every == 0 or final:
                self.record_state()
        self.dt = dt
        self.times = self.X.new_tensor(self.time_records)
        self.fields = torch.stack(self.field_records)
        self.reservoir_history = torch.stack(self.reservoir_records)
        self.P_history = torch.stack(self.P_records)
        self.source_history = torch.stack(self.source_records)
        self.reservoir_energy_history = self.reservoir_energy(self.reservoir_history, self.P_history)
        self.result = (self.x, self.times, self.fields, self.reservoir_history, self.source_history, self.P_history, self.reservoir_energy_history)
        return self.result
    def visualise(self):
        self.ax[0].clear()
        self.ax[1].clear()
        x = self.x.detach().cpu().numpy()
        density = (abs(self.psi) ** 2).detach().cpu().numpy()
        mesh = self.ax[0].pcolormesh(x, x, density.T, shading='auto', vmin=0)
        exciton_number = self.ax[1].imshow(self.n_X, extent= (-50,50, -50,50), origin = "lower")
        if not hasattr(self, 'colorbar'):
            self.colorbar = self.figure.colorbar(mesh, ax=self.ax[0], label='Density |psi|² (µm⁻²)', extend='max')
            self.colorbar2 = self.figure.colorbar(exciton_number,ax=self.ax[1], label= 'exicton number', extend= 'max')
        self.colorbar2.update_normal(exciton_number)
        self.colorbar.update_normal(mesh)
        self.ax[0].set(title=f'Condensate density |psi|², t={self.t:.3f} ps', xlabel='x (µm)', ylabel='y (µm)', aspect='equal')
        self.ax[1].set(title=f'Exicton number n_X, t={self.t:.3f} ', xlabel='x (µm)', ylabel='y (µm)', aspect='equal')

    def angular_momentum(self, psi=None):
        psi = self.psi if psi is None else psi
        field_k = torch.fft.fftn(psi)
        derivative_x = torch.fft.ifftn(1j * self.k[:, None] * field_k)
        derivative_y = torch.fft.ifftn(1j * self.k[None, :] * field_k)
        action = -1j * (self.X * derivative_y - self.Y * derivative_x)
        return torch.real(torch.sum(torch.conj(psi) * action)) / torch.sum(torch.abs(psi) ** 2)
if __name__ == "__main__":
    # Paper Sec. II: g=1 µeV µm², |chi|²=0.4; energies below are meV.
    points = 256
    half_length = 50.0  # Domain [-L, L) µm along both axes.
    hbar, m, g, chi_squared = 0.6582119569, 0.28, 0.001, 0.4
    GPE_run = GPE_solver_class(
        points=points, half_length=half_length,
        alpha=g*chi_squared**2, G=2*g*chi_squared, hbar=hbar, m=m,
        gamma=1/5.5, Gamma_X=1/5.5,
        R=2*g/hbar, eta=2.0,
    )
    P0 = 16.0  # µm^-2 ps^-1; retained example value, not a paper fit.
    sigma = 1.6
    seed_amplitude = 1e-6  # Unnormalized random complex field, as before.
    initial_reservoir = 0.0
    wall_position, wall_width, wall_strength = 45.0, 2.5, 20.0
    dt, t_final = 0.02, 120.0
    record_every, plot_every = 40, 10
    GPE_run.grid()
    GPE_run.fft()
    GPE_run.P = GPE_run.gaussian_pump(P0=P0, sigma=sigma)
    # Legacy positive-biased box-wide seed; run_tuned.py uses correlated, localized noise.
    psi = seed_amplitude*(torch.rand_like(GPE_run.X)+1j*torch.rand_like(GPE_run.X))
    GPE_run.initial_state(psi, n_X=initial_reservoir)
    GPE_run.potential = GPE_run.tanh_wall(wall_position, wall_width, wall_strength, geometry="square")
    step_method = GPE_run.strang_step
    # step_method = GPE_run.rk4_run  # Reduce dt as required by its stability check; test accuracy.
    GPE_run.time_evolution(dt, t_final, record_every, live=True,
                           plot_every=plot_every, step_method=step_method)
    output = Path(__file__).resolve().parent
    def numpy(array):
        return array.detach().cpu().numpy()
    np.savez_compressed(
        output / "run.npz", x=numpy(GPE_run.x), times=numpy(GPE_run.times),
        psi=numpy(GPE_run.fields), n_X=numpy(GPE_run.reservoir_history),
        P=numpy(GPE_run.P_history), source_integral=numpy(GPE_run.source_history),
        reservoir_energy=numpy(GPE_run.reservoir_energy_history),
        potential=numpy(GPE_run.potential), dt=GPE_run.dt, method=GPE_run.method,
        units="um, ps, meV; densities um^-2; P um^-2 ps^-1",
        hbar=hbar, m=m, alpha=GPE_run.alpha, G=GPE_run.G, R=GPE_run.R,
        gamma=GPE_run.gamma, Gamma_X=GPE_run.Gamma_X, eta=GPE_run.eta, P0=P0, sigma=sigma,
    )
    GPE_run.figure.savefig(output / "run_density.png", dpi=160)
    plt.close(GPE_run.figure)
