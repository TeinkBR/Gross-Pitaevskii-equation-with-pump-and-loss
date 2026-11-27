r"""Time-delay polaritonics, Supplementary Note 1, Eqs. (S1)-(S3).
Source: https://doi.org/10.1038/s42005-019-0271-0
Periodic 2D FFT grid with an added absorbing wall; static P and potential.
Units: um, ps, meV; |psi|², n_A, n_I: um^-2; alpha, G: meV um²;
gamma, Gamma_A, Gamma_I, W: ps^-1; R: um² ps^-1; P: um^-2 ps^-1.
"""
from dataclasses import dataclass, field
from functools import partial
from math import ceil
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
HBAR = 0.6582119569
# Spatial PDE source equations are cited inline below: main text Eqs. (1)-(4), (12)-(15)
# and Supplementary Note 1 Eqs. (S1)-(S3), pp. 4-5. LaTeX is kept in comments
# so each model/observable can be traced to its paper definition. This spatial
# solver has ballistic propagation between pump spots, but no explicit c_j(t-tau) term.

def squared_density(psi):
    real, imag = psi.real, psi.imag
    return real*real + imag*imag


def tensor_to_numpy(value):
    """Move one tensor to CPU memory and convert it to a NumPy array."""
    detached = value.detach()
    cpu_value = detached.cpu()
    return cpu_value.numpy()


@dataclass
class Config:
    grid_points: int = 512
    box_half_width: float = 60.
    dt: float = .1
    t_final: float = 700.
    settling_time: float = 0.
    record_every: int = 1
    live: bool = False
    frame_every: int = 1000
    average_images: bool = False
    pump_separation: object = 12.7  # None selects one pump at x=0.
    # Supplementary Fig. 1(c-h) states $P_0=1.5P_{\rm thr}^{(1)}$; this numeric
    # P0 uses our local threshold estimate, not a pump-power conversion from experiment.
    P0: float = 2.79365
    pump_sigma: float = .8493218
    seed: int = 42
    device: object = None
    dtype: object = None
    absorber_position: object = None
    absorber_width: float = 3.
    absorber_strength: float = 5.
    steady_reservoirs: bool = False
    save_outputs: bool = False
    output: object = None
    compress: bool = False

    def __post_init__(self):
        if self.record_every <= 0 or self.frame_every < 0:
            raise ValueError('Require record_every > 0 and frame_every >= 0.')

class GPESolver:
    def __init__(self, points=256, half_length=48.0,
                 alpha=0.00165, G=0.033, gamma=1/5.5,
                 Gamma_A=0.05, Gamma_I=0.002, W=0.1,
                 R=0.129/HBAR, hbar=HBAR, m=0.28):
        # Supplementary Note 1 parameter choices: $m=0.28\,\mathrm{meV\,ps^2\,\mu m^{-2}}$,
        # $\gamma=1/5.5\,\mathrm{ps}^{-1}$, $\Gamma_A=1/20\,\mathrm{ps}^{-1}$,
        # $\Gamma_I=1/500\,\mathrm{ps}^{-1}$, and $W=0.1\,\mathrm{ps}^{-1}$.
        # Eq. (S1): $\tilde{\alpha}=3.3\,\mu\mathrm{eV}\,\mu\mathrm{m}^2$
        # and $\tilde g=20\tilde{\alpha}$. In the code's energy-unit convention,
        # $\alpha=\tilde{\alpha}/2$ and $G=\tilde g/2$ (meV um^2).
        # R assumes the supplement's printed um^-2 is a typo for um^2:
        # $R_{\rm code}=(0.129\,\mathrm{meV}\,\mu\mathrm{m}^2)/\hbar$.
        assert hbar > 0 and m > 0
        assert min(alpha, G, gamma, Gamma_A, Gamma_I, W, R) >= 0
        self.points, self.half_length = points, half_length
        self.hbar, self.m = hbar, m
        self.alpha, self.G, self.gamma = alpha, G, gamma
        self.Gamma_A, self.Gamma_I, self.W = Gamma_A, Gamma_I, W
        self.R, self.t = R, 0.0

    def grid(self, points=None, half_length=None, device="mps", dtype=None, P=0.):
        device = torch.device(device)
        dtype = dtype if dtype is not None else torch.float32 if device.type == "mps" else torch.float64
        if device.type == "mps" and dtype == torch.float64:
            raise ValueError("MPS cannot use float64/complex128; use float32/complex64 or device='cpu'.")
        self.points = self.points if points is None else points
        self.half_length = self.half_length if half_length is None else half_length
        assert self.points >= 2 and self.half_length > 0
        assert dtype in (torch.float32, torch.float64)
        self.dx = 2 * self.half_length / self.points
        self.x = -self.half_length+self.dx*torch.arange(self.points,dtype=dtype,device=device)
        self.X, self.Y = torch.meshgrid(self.x, self.x, indexing="ij")
        complex_dtype = torch.complex64 
        self.set_absorber(torch.zeros_like(self.X, dtype=complex_dtype))
        self.P = torch.zeros_like(self.X)+torch.as_tensor(P,dtype=dtype,device=device)
        self.psi, self.n_A, self.n_I = torch.zeros_like(self.absorber),torch.zeros_like(self.X),torch.zeros_like(self.X)
        self.t = 0.0
        return self.x

    def set_absorber(self, absorber):
        """Set the static complex absorbing wall; never edit it in place."""
        self.absorber = absorber

    def fft(self, dt):
        """Prepare the spatial Fourier grid and kinetic propagator for stepping."""
        assert dt > 0
        # Fourier collocation (Supplementary Note 1, spectral space method):
        # $\mathcal{T}\psi=\mathcal{F}^{-1}[\hbar^2|\mathbf{k}|^2\mathcal{F}[\psi]/(2m)]$,
        # with $|\mathbf{k}|^2=k_x^2+k_y^2$.
        self.k = 2*torch.pi*torch.fft.fftfreq(self.points,d=self.dx,dtype=self.x.dtype,device=self.x.device)
        self.kinetic_energy = self.hbar**2/(2*self.m)*(self.k[:,None]**2+self.k[None,:]**2)
        self.kinetic_dt = dt
        self.kinetic_propagator = torch.exp(-1j * self.kinetic_energy * dt / self.hbar)
        # This necessary RK4 free-kinetic bound is prepared outside the step loop.
        self.rk4_kinetic_limit = (8**0.5 * self.hbar / self.kinetic_energy.max()).item()
        self._step_constants = {}
        self._constants(dt)
        return self.k, self.kinetic_propagator

    def tanh_wall(self, absorber_position=35.0, absorber_width=2.5, absorber_strength=40.0, geometry="square"):
        # Numerical boundary absorber (not a paper interaction term):
        # $V_{\rm abs}(\mathbf r)=-i\,V_w[1+\tanh((r_{\rm edge}-r_w)/w)]/2$.
        assert geometry in ("square", "circle")
        edge = torch.maximum(abs(self.X),abs(self.Y)) if geometry == "square" else torch.hypot(self.X,self.Y)
        return -1j * absorber_strength * 0.5 * (1.0 + torch.tanh((edge - absorber_position) / absorber_width))

    def gaussian_pump(self, centre=(0.,0.), P0=0., sigma=3.):
        # Supplementary Note 1, p. 5: $P(\mathbf r)=P_0\exp[-|\mathbf r-\mathbf r_j|^2/(2\sigma^2)]$;
        # $\mathrm{FWHM}=2\sqrt{2\ln 2}\,\sigma$ ($\sigma=0.8493\,\mu\mathrm m$ gives 2 um).
        return P0*torch.exp(-((self.X-centre[0])**2+(self.Y-centre[1])**2)/(2*sigma**2))

    def time_derivatives(self,psi, n_A, n_I):
                # Supplementary Note 1, Eqs. (S1)-(S3), in energy units,
                # with $\alpha=\tilde{\alpha}/2$ and $G=\tilde g/2$:
                # \[
                # i\hbar\partial_t\psi=[-\hbar^2\nabla^2/(2m)+V+\alpha|\psi|^2
                #   +G(n_A+n_I)+i\hbar(Rn_A-\gamma)/2]\psi,
                # \quad \partial_t n_A=Wn_I-(\Gamma_A+R|\psi|^2)n_A,
                # \quad \partial_t n_I=P-(\Gamma_I+W)n_I.
                # \]
                # $V$ here is the solver's optional complex absorber.
        density = squared_density(psi)
        local_energy = self.absorber + self.alpha*density + self.G*(n_A+n_I)
        dpsi = (-1j*local_energy/self.hbar + 0.5*(self.R*n_A-self.gamma))*psi
        # The last term is the spectral kinetic part: -i T psi / hbar.
        dpsi = dpsi-1j*torch.fft.ifftn(self.kinetic_energy*torch.fft.fftn(psi))/self.hbar
        dn_A = self.W*n_I-(self.Gamma_A+self.R*density)*n_A   # active reservoir
        dn_I = self.P-(self.Gamma_I+self.W)*n_I                # inactive reservoir
                
        return dpsi, dn_A, dn_I

    @torch.inference_mode()
    def rk4_run(self, dt):
        # Classical coupled RK4 applied to the same S1-S3 right-hand side.
        # The supplement reports a linear-multistep time integrator; this RK4
        # option is a local alternative, not the paper's reported integrator.
        if dt <= 0 or dt > self.rk4_kinetic_limit:
            raise ValueError("RK4 free-kinetic stability limit exceeded; reduce dt.")

       

        state = self.psi, self.n_A, self.n_I
        slope1 = self.time_derivatives(*state)
        slope2 = self.time_derivatives(*(quantity+dt*slope/2 for quantity,slope in zip(state,slope1)))
        slope3 = self.time_derivatives(*(quantity+dt*slope/2 for quantity,slope in zip(state,slope2)))
        slope4 = self.time_derivatives(*(quantity+dt*slope for quantity,slope in zip(state,slope3)))
        self.psi, self.n_A, self.n_I = tuple(quantity+dt*(s1+2*s2+2*s3+s4)/6
                                             for quantity,s1,s2,s3,s4 in zip(state,slope1,slope2,slope3,slope4))
        return self.psi, self.n_A, self.n_I
    def _constants(self, dt):
        key = (dt, self.Gamma_I, self.W)
        if key not in self._step_constants:
            half_dt, inactive_loss_rate = dt/2, self.Gamma_I+self.W
            propagator = self.kinetic_propagator if dt == self.kinetic_dt else torch.exp(-1j*self.kinetic_energy*dt/self.hbar)
            exponent = self.x.new_tensor(-inactive_loss_rate*half_dt)
            inactive_decay = torch.exp(exponent)
            inactive_source_factor = -torch.expm1(exponent)/inactive_loss_rate if inactive_loss_rate else half_dt
            self._step_constants[key] = half_dt,propagator,inactive_decay,inactive_source_factor
        return self._step_constants[key]

    @torch.inference_mode()
    def strang_step(self, dt):
        """Local half / kinetic full / local half; frozen-coefficient split of S1-S3."""
        assert dt > 0
        half_dt, propagator, inactive_decay, inactive_source_factor = self._constants(dt)
        for half in range(2):
            psi, n_A, n_I = self.psi,self.n_A,self.n_I  # values at the substep start
            density = squared_density(psi)
            active_loss_rate = self.Gamma_A+self.R*density
            exponent = -active_loss_rate*half_dt
            zero_loss = active_loss_rate == 0
            safe_loss_rate = torch.where(zero_loss,1.,active_loss_rate)
            active_source_factor = torch.where(zero_loss,half_dt,-torch.expm1(exponent)/safe_loss_rate)
            # Numerical split of the paper RHS, not an additional paper equation.
            # During each local interval h=dt/2, rho and the coefficients are
            # frozen. The n_A update holds n_I at its substep-start value;
            # n_I itself advances with its static pump source. The reservoir substeps are
            # \[
            # n_A(t+h)=n_A(t)e^{-ah}+Wn_I(t)(1-e^{-ah})/a,\quad
            # n_I(t+h)=n_I(t)e^{-bh}+P(1-e^{-bh})/b,
            # \]
            # with $a=\Gamma_A+R\rho$, $b=\Gamma_I+W$ (use the $a\to0$ or
            # $b\to0$ limit). The field substep is
            # \[
            # \psi(t+h)=\psi(t)\exp\{h[-iE_{\rm loc}/\hbar+(Rn_A-\gamma)/2]\},
            # \quad E_{\rm loc}=V+\alpha\rho+G(n_A+n_I),
            # \]
            # with $\rho$, $n_A$, and $n_I$ frozen at the substep start.
            # This local exponential approximation plus the Fourier kinetic step
            # is the code's chosen integrator; the paper instead reports linear multistep.
            local_energy = self.absorber+self.alpha*density+self.G*(n_A+n_I) 
            self.psi = psi*torch.exp((-1j*local_energy/self.hbar+0.5*(self.R*n_A-self.gamma))*half_dt)
            ## active n_A and inactive n_I reservoir
            self.n_A = n_A*torch.exp(exponent)+self.W*n_I*active_source_factor
            self.n_I = n_I*inactive_decay+self.P*inactive_source_factor
            if half == 0:
                # Fourier kinetic substep from S1:
                # $\widehat\psi(\mathbf k,t+\Delta t)=e^{-i\hbar|\mathbf k|^2\Delta t/(2m)}\widehat\psi(\mathbf k,t)$.
                self.psi = torch.fft.ifftn(propagator*torch.fft.fftn(self.psi))
        return self.psi,self.n_A,self.n_I

    @property
    def k_axis(self):
        return torch.fft.fftshift(self.k)

def energy_and_delay_axes(n_records, record_interval, hbar=HBAR, max_delay=50.):
    """Build energy and delay labels for the recorded time samples.

    This function labels the temporal axes; it does not transform data. The
    spatial Fourier grid built by ``GPESolver.fft`` has its own spacing.
    """
    # Uniform-time Fourier bins: $E=\hbar\omega=2\pi\hbar f$; delays are
    # integer multiples of the complex-signal sampling interval.
    energy = torch.fft.fftshift(2*torch.pi*hbar*torch.fft.fftfreq(n_records,d=record_interval,dtype=torch.float64)) if n_records else torch.empty(0,dtype=torch.float64)
    max_lag_steps = min(int(max_delay/record_interval), max(0, n_records-2)) ## make sure that the lag always integer and never negative or exceeds the length of your signal.

    delays = torch.arange(-max_lag_steps, max_lag_steps+1, dtype=torch.float64)*record_interval ## Builds a symmetric 1D tensor of integer steps from -max_lag_steps to +max_lag_steps, then scales them by the record interval
    # which is dt*record_every
    return energy, delays

@torch.inference_mode()
def spectra(psi_ky0, condensate_amplitudes, count=None):
    # Main-text Eqs. (1)-(2) motivate resolving the two condensate signals into
    # even/odd parity. With centre amplitudes $c_1,c_2$,
    # $c_e=(c_1+c_2)/\sqrt{2}$ and $c_o=(c_1-c_2)/\sqrt{2}$; spectra are $|\mathcal F_t[c]|^2$.
    # With $E_m=2\pi\hbar f_m$, PyTorch's inverse time FFT is
    # $A_j(E_m)=N^{-1}\sum_n c_j(t_n)e^{+iE_mt_n/\hbar}$; the centre intensity is $|A_j|^2$.
    # The y integral selects $k_y=0$; the x FFT then gives the paper's Fig. 2e-f
    # observable $I(k_x,E)\propto|\mathcal F_x\mathcal F_t^{(+)}[\int dy\,\psi(x,y,t)]|^2$.
    count = condensate_amplitudes.shape[0] if count is None else count
    if count < 2: ## need at least two time samples (energy bins) for a spectrum
        raise ValueError("At least two planned observations are required.")
    spectral_amplitudes = torch.fft.fftshift(torch.fft.ifft(condensate_amplitudes, n=count, dim=0), dim=0)
    # Spatial FFT uses the negative-exponential convention; scale factors in x
    # and y are omitted from the displayed intensity (arbitrary normalization).
    psi_kx_E = torch.fft.fft(psi_ky0, dim=1)
    psi_kx_E = torch.fft.ifft(psi_kx_E, n=count, dim=0)
    observables = dict(centre_spectrum=spectral_amplitudes.abs().square(),
                       dispersion=torch.fft.fftshift(psi_kx_E.abs().square(), dim=(0, 1)))
  ## return reslut as
    if condensate_amplitudes.shape[1] == 2: ## confirming the two pump spots (columns of condensate_amplitudes)
        # Paper Eqs. (1)-(2): even/odd amplitudes; Supplementary Fig. 1(h):
        # $z=(\rho_1-\rho_2)/(\rho_1+\rho_2)$ and $\phi=\arg(c_1c_2^*)$.
        # Here $\rho_j=|c_j|^2$ uses the interpolated value at pump centre j,
        # rather than a spatial integral over a finite condensate-sized region.
        parity_amplitudes = torch.stack((spectral_amplitudes.sum(dim=1), spectral_amplitudes[:, 0]-spectral_amplitudes[:, 1]), dim=1)/2**.5
        ## parity is calculated from taking the sum of the
        spot_density = condensate_amplitudes.abs().square()
        total_density = spot_density.sum(dim=1)
        relative_phase = torch.angle(condensate_amplitudes[:, 0]*condensate_amplitudes[:, 1].conj())
        observables.update(parity_spectrum=parity_amplitudes.abs().square(),
            density_imbalance=torch.where(total_density > 0, (spot_density[:, 0]-spot_density[:, 1])/total_density, float("nan")),
            relative_phase=relative_phase.masked_fill((spot_density == 0).any(dim=1), float("nan")))
    return observables  ## dictionary updated with the parity

@torch.inference_mode()
def fringe_visibility(condensate_amplitudes, delays, record_interval):
    # Main-text Eqs. (3)-(4) give the ideal equal-weight parity limit
    # $V_{11}(\tau)=|\cos(\Delta_\mu\tau/(2\hbar))|$ and
    # $V_{12}(\tau)=|\sin(\Delta_\mu\tau/(2\hbar))|$,
    # $\Delta_\mu=|\mu_e-\mu_o|$.
    # This routine computes the more general finite-record fringe contrast
    # \[
    # V_{1j}(\tau)=\frac{2|\sum_{n\in\mathrm{overlap}}c_1[n]^*c_j[n+\ell]|}
    # {\sum_{n\in\mathrm{overlap}}|c_1[n]|^2+\sum_{n\in\mathrm{overlap}}|c_j[n+\ell]|^2},
    # \quad \ell=\tau/\Delta t.
    # \]
    # corresponding to the paper's interferometric visibility. Channel 0 is
    # c_1; channel 1 (when present) is c_2. Zero padding makes the FFT
    # correlation linear rather than circular; the normalisation uses only overlap.
    condensate_amplitudes = condensate_amplitudes.to(device="cpu")
    n_samples, n_spots = condensate_amplitudes.shape
    lag_steps = torch.round(delays.cpu()/record_interval).long() ## round each delay to the nearest whole number of samples
    fft_length = 1 << (2*n_samples-1).bit_length()
    amplitude_fft = torch.fft.fft(condensate_amplitudes, n=fft_length, dim=0)
    cross_correlation = torch.fft.ifft(amplitude_fft[:, :1].conj()*amplitude_fft, dim=0)[lag_steps.remainder(fft_length)]
    spot_density = condensate_amplitudes.abs().square().to(torch.float64)
    cumulative_density = torch.cat((spot_density.new_zeros((1, n_spots)), spot_density.cumsum(dim=0)))
    ## window start of the reference arm (c_1) and of the delayed arm (c_j):
    reference_start, delayed_start = (-lag_steps).clamp(0, n_samples), lag_steps.clamp(0, n_samples) ## clamp the window starts to the record
    overlap = (n_samples-lag_steps.abs()).clamp_min(0)
    reference_arm_intensity = cumulative_density[reference_start+overlap, 0]-cumulative_density[reference_start, 0]
    delayed_arm_intensity = cumulative_density[delayed_start+overlap]-cumulative_density[delayed_start]
    total_arm_intensity = reference_arm_intensity[:, None]+delayed_arm_intensity
    valid = (n_samples >= 2) & (overlap > 0)[:, None] & (total_arm_intensity > 0) ## True where the visibility is defined
    return torch.where(valid, 2*cross_correlation.abs()/total_arm_intensity, float("nan")) ## NaN where it is undefined

@dataclass
class Result:
    solver: GPESolver
    n_records: int
    record_interval: float = 0.
    n_recorded: int = 0
    final: bool = False
    record_times: object = None
    frame_times: object = None
    energy_axis: object = None
    coherence_delays: object = None
    psi_ky0: object = None
    condensate_amplitudes: object = None
    psi_frames: object = None
    n_A_frames: object = None
    n_I_frames: object = None
    observables: dict = field(default_factory=dict)
    averages: dict = field(default_factory=dict)
    figure: object = None

    @property
    def min_samples(self):
        return min(8, max(2, self.n_records))

def analyse(result, final=False):
    result.final = final
    if result.n_recorded >= 2:
        psi_ky0 = result.psi_ky0[:result.n_recorded]
        condensate_amplitudes = result.condensate_amplitudes[:result.n_recorded]
        if final:
            psi_ky0, condensate_amplitudes = (tensor.to(device='cpu', dtype=torch.complex64)
                                              for tensor in (psi_ky0, condensate_amplitudes))
        result.observables = spectra(psi_ky0, condensate_amplitudes, count=result.n_records)
        result.observables['fringe_visibility'] = fringe_visibility(
            condensate_amplitudes, result.coherence_delays, result.record_interval)

@torch.inference_mode()
def evolve(solver, config, pump_centres=(0.,), stepper=None):

    start = solver.t
    if config.dt <= 0 or config.t_final < start:
        raise ValueError('Require dt > 0 and t_final >= start.')
    n_steps = round((config.t_final-start)/config.dt)
    record_steps = [step for step in range(0, n_steps+1, config.record_every)
                    if start+step*config.dt >= config.settling_time]
    frame_steps = sorted({*range(0, n_steps+1, config.frame_every), n_steps}) if config.frame_every else []
    record_step_set, frame_step_set = set(record_steps), set(frame_steps)

    result = Result(solver=solver, n_records=len(record_steps), record_interval=config.record_every*config.dt) ## read in the length of the records
    result.record_times = torch.tensor([start+step*config.dt for step in record_steps], dtype=torch.float64)
    result.frame_times = torch.tensor([start+step*config.dt for step in frame_steps], dtype=torch.float64)
    result.energy_axis, result.coherence_delays = energy_and_delay_axes(
        result.n_records,
        result.record_interval,
        solver.hbar,
    )
    #allocates an empty arrays for
    result.psi_ky0 = solver.psi.new_zeros((result.n_records, solver.points))
    result.condensate_amplitudes = solver.psi.new_zeros((result.n_records, len(pump_centres)))
    result.psi_frames = solver.psi.new_empty((len(frame_steps), solver.points, solver.points))
    result.n_A_frames = solver.n_A.new_empty(result.psi_frames.shape)
    result.n_I_frames = solver.n_I.new_empty(result.psi_frames.shape)

    centre_position = (solver.x.new_tensor(pump_centres)+solver.half_length)/solver.dx
    centre_cell = centre_position.floor()
    centre_index, centre_fraction = centre_cell.long(), centre_position-centre_cell
    if not ((centre_index >= 0) & (centre_index+1 < solver.points)).all():
        raise ValueError('Centres must lie inside the grid.')
    if config.average_images:
        summed_density = torch.zeros_like(solver.X)
        summed_k_density = torch.zeros_like(solver.X)

    advance = stepper or solver.strang_step
    frame_index, last_analysed_count, last_draw_time = 0, 0, -float('inf')
    for step in range(n_steps+1):
        solver.t = start+step*config.dt
        if step in frame_step_set:
            result.psi_frames[frame_index] = solver.psi
            result.n_A_frames[frame_index] = solver.n_A
            result.n_I_frames[frame_index] = solver.n_I
            frame_index += 1
        if step in record_step_set:
            record_index = result.n_recorded
            # Paper dispersion panel: sum over y samples approximates
            # $\psi_{k_y=0}(x,t)=\int dy\,\psi(x,y,t)$. At pump centre $j$ record
            # $c_j(t)=\psi(x_j,y=0,t)$, with linear interpolation in x below.
            result.psi_ky0[record_index] = solver.psi.sum(dim=1)*solver.dx
            result.condensate_amplitudes[record_index] = (
                (1-centre_fraction)*solver.psi[centre_index, solver.points//2]
                + centre_fraction*solver.psi[centre_index+1, solver.points//2])
            if config.average_images:
                summed_density += squared_density(solver.psi)
                summed_k_density += torch.fft.fftn(solver.psi).abs().square()
            result.n_recorded += 1
            if config.live and step < n_steps and perf_counter()-last_draw_time >= .5:
                if result.n_recorded >= max(result.min_samples, ceil(1.1*last_analysed_count)):
                    analyse(result)
                    last_analysed_count = result.n_recorded
                
                last_draw_time = perf_counter()
        if step < n_steps:
            advance(config.dt)

    if config.average_images and result.n_recorded:
        result.averages = dict(mean_density=summed_density/result.n_recorded,
                               mean_momentum_intensity=torch.fft.fftshift(summed_k_density/result.n_recorded))
    analyse(result, final=True)
   
    return result
