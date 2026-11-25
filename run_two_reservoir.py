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

@dataclass
class Config:
    points: int = 512
    half_length: float = 60.
    dt: float = .005
    t_final: float = 700.
    settling_time: float = 0.
    record_every: int = 10
    live: bool = False
    frame_every: int = 1000
    average_images: bool = False
    separations: tuple = (12.7,)
    # Supplementary Fig. 1(c-h) states $P_0=1.5P_{\rm thr}^{(1)}$; this numeric
    # P0 uses our local threshold estimate, not a pump-power conversion from experiment.
    P0: float = 2.79365
    sigma: float = .8493218
    seed: int = 42
    device: object = None
    dtype: object = None
    wall_position: object = None
    wall_width: float = 3.
    wall_strength: float = 5.
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
            raise ValueError("MPS cannot use float64/complex128; use float32 or device='cpu'.")
        self.points = self.points if points is None else points
        self.half_length = self.half_length if half_length is None else half_length
        assert self.points >= 2 and self.half_length > 0
        assert dtype in (torch.float32, torch.float64)
        self.dx = 2 * self.half_length / self.points
        self.x = -self.half_length+self.dx*torch.arange(self.points,dtype=dtype,device=device)
        self.X, self.Y = torch.meshgrid(self.x, self.x, indexing="ij")
        complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        self.set_potential(torch.zeros_like(self.X, dtype=complex_dtype))
        self.P = torch.zeros_like(self.X)+torch.as_tensor(P,dtype=dtype,device=device)
        self.psi, self.n_A, self.n_I = torch.zeros_like(self.potential),torch.zeros_like(self.X),torch.zeros_like(self.X)
        self.t = 0.0
        return self.x

    def set_potential(self, V):
        """Set static V and absorption together; never edit potential in place."""
        self.potential, self.absorption = V, 2*V.imag/self.hbar

    def fft(self, dt):
        assert dt > 0
        # Fourier collocation (Supplementary Note 1, spectral space method):
        # $\mathcal{T}\psi=\mathcal{F}^{-1}[\hbar^2|\mathbf{k}|^2\mathcal{F}[\psi]/(2m)]$,
        # with $|\mathbf{k}|^2=k_x^2+k_y^2$.
        self.k = 2*torch.pi*torch.fft.fftfreq(self.points,d=self.dx,dtype=self.x.dtype,device=self.x.device)
        self.kinetic_energy = self.hbar**2/(2*self.m)*(self.k[:,None]**2+self.k[None,:]**2)
        self.kinetic_dt = dt
        self.kinetic_phase = torch.exp(-1j * self.kinetic_energy * dt / self.hbar)
        # This necessary RK4 free-kinetic bound is prepared outside the step loop.
        self.rk4_kinetic_limit = (8**0.5 * self.hbar / self.kinetic_energy.max()).item()
        self._step_constants = {}
        self._constants(dt)
        return self.k, self.kinetic_phase

    def tanh_wall(self, wall_position=35.0, wall_width=2.5, wall_strength=40.0, geometry="square"):
        # Numerical boundary absorber (not a paper interaction term):
        # $V_{\rm abs}(\mathbf r)=-i\,V_w[1+\tanh((r_{\rm edge}-r_w)/w)]/2$.
        assert geometry in ("square", "circle")
        edge = torch.maximum(abs(self.X),abs(self.Y)) if geometry == "square" else torch.hypot(self.X,self.Y)
        return -1j * wall_strength * 0.5 * (1.0 + torch.tanh((edge - wall_position) / wall_width))

    def gaussian_pump(self, mu=(0.,0.), P0=0., sigma=3.):
        # Supplementary Note 1, p. 5: $P(\mathbf r)=P_0\exp[-|\mathbf r-\boldsymbol\mu|^2/(2\sigma^2)]$;
        # $\mathrm{FWHM}=2\sqrt{2\ln 2}\,\sigma$ ($\sigma=0.8493\,\mu\mathrm m$ gives 2 um).
        return P0*torch.exp(-((self.X-mu[0])**2+(self.Y-mu[1])**2)/(2*sigma**2))

    def time_derivatives(self, psi, n_A, n_I):
        # Supplementary Note 1, Eqs. (S1)-(S3), converted to explicit energy units
        # with $\alpha=\tilde{\alpha}/2$ and $G=\tilde g/2$:
        # \[
        # i\hbar\partial_t\psi=[-\hbar^2\nabla^2/(2m)+V+\alpha|\psi|^2
        #   +G(n_A+n_I)+i\hbar(Rn_A-\gamma)/2]\psi,
        # \quad \partial_t n_A=Wn_I-(\Gamma_A+R|\psi|^2)n_A,
        # \quad \partial_t n_I=P-(\Gamma_I+W)n_I.
        # \]
        # $V$ here is the solver's optional complex absorber; it is not a
        # prescribed term in Supplementary Note 1's S1-S3 model.
        density = squared_density(psi)
        total_energy = self.potential + self.alpha*density + self.G*(n_A+n_I)
        dpsi = (-1j*total_energy/self.hbar + 0.5*(self.R*n_A-self.gamma))*psi
        # The last term is the spectral kinetic part: -i T psi / hbar.
        dpsi = dpsi-1j*torch.fft.ifftn(self.kinetic_energy*torch.fft.fftn(psi))/self.hbar
        dn_A = self.W*n_I-(self.Gamma_A+self.R*density)*n_A ## splitting into active
        dn_I = self.P-(self.Gamma_I+self.W)*n_I
         # and inactive reservoir 
        return dpsi, dn_A, dn_I

    @torch.inference_mode()
    def rk4_run(self, dt):
        # Classical coupled RK4 applied to the same S1-S3 right-hand side.
        # The supplement reports a linear-multistep time integrator; this RK4
        # option is a local alternative, not the paper's reported integrator.
        if dt <= 0 or dt > self.rk4_kinetic_limit:
            raise ValueError("RK4 free-kinetic stability limit exceeded; reduce dt.")
        state = self.psi, self.n_A, self.n_I
        k1 = self.time_derivatives(*state)
        k2 = self.time_derivatives(*(y+dt*k/2 for y,k in zip(state,k1)))
        k3 = self.time_derivatives(*(y+dt*k/2 for y,k in zip(state,k2)))
        k4 = self.time_derivatives(*(y+dt*k for y,k in zip(state,k3)))
        self.psi, self.n_A, self.n_I = tuple(y+dt*(a+2*b+2*c+d)/6 for y,a,b,c,d in zip(state,k1,k2,k3,k4))
        return self.psi, self.n_A, self.n_I

    def _constants(self, dt):
        key = (dt, self.Gamma_I, self.W)
        if key not in self._step_constants:
            half_t, rate = dt/2, self.Gamma_I+self.W
            phase = self.kinetic_phase if dt == self.kinetic_dt else torch.exp(-1j*self.kinetic_energy*dt/self.hbar)
            exponent = self.x.new_tensor(-rate*half_t)
            decay = torch.exp(exponent)
            source = -torch.expm1(exponent)/rate if rate else half_t
            self._step_constants[key] = half_t,phase,decay,source
        return self._step_constants[key]

    @torch.inference_mode()
    def strang_step(self, dt):
        """Local half / kinetic full / local half; frozen-coefficient split of S1-S3."""
        assert dt > 0
        half_t, phase, decay, source_I = self._constants(dt)
        for half in range(2):
            psi, active, inactive = self.psi,self.n_A,self.n_I
            density = squared_density(psi)
            rate = self.Gamma_A+self.R*density
            exponent = -rate*half_t
            zero = rate == 0
            safe_rate = torch.where(zero,1.,rate)
            source = torch.where(zero,half_t,-torch.expm1(exponent)/safe_rate)
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
            # Previous midpoint predictors are disabled: no dt/4 updates run.
            # active_mid = active*torch.exp(exponent/2)+self.W*inactive*torch.where(zero,half_t/2,-torch.expm1(exponent/2)/safe_rate)
            # inactive_mid = inactive*torch.sqrt(decay)+self.P*source_I/(1+torch.sqrt(decay))
            # density_mid = density*torch.exp((self.R*active-self.gamma+self.absorption)*half_t/2)
            # energy_mid = self.potential+self.alpha*density_mid+self.G*(active_mid+inactive_mid)
            # E  = V+ \alpha * \psi^2 + G*(active+ inactive)
            energy = self.potential+self.alpha*density+self.G*(active+inactive) 
            self.psi = psi*torch.exp((-1j*energy/self.hbar+0.5*(self.R*active-self.gamma))*half_t)
            ## active n_A and inactive n_I reservoir
            self.n_A = active*torch.exp(exponent)+self.W*inactive*source
            self.n_I = inactive*decay+self.P*source_I
            if half == 0:
                # Fourier kinetic substep from S1:
                # $\widehat\psi(\mathbf k,t+\Delta t)=e^{-i\hbar|\mathbf k|^2\Delta t/(2m)}\widehat\psi(\mathbf k,t)$.
                self.psi = torch.fft.ifftn(phase*torch.fft.fftn(self.psi))
        return self.psi,self.n_A,self.n_I

    @property
    def momentum_axis(self):
        return torch.fft.fftshift(self.k)

    # def initial_state(self, psi, n_A=0., n_I=0.):
    #     self.psi = torch.as_tensor(psi,dtype=self.potential.dtype,device=self.x.device).clone()
    #     self.n_A, self.n_I = (torch.zeros_like(self.X)+torch.as_tensor(v,dtype=self.x.dtype,device=self.x.device) for v in (n_A,n_I))
    #     assert self.psi.shape == self.n_A.shape == self.n_I.shape == self.X.shape
    #     assert (self.n_A >= 0).all() and (self.n_I >= 0).all()
    #     self.t = 0.
    #     return self.psi, self.n_A, self.n_I
    # def gaussian_wave(self, envelope_alpha=-.125, x0=0., k0=0., y0=0., ky0=0.):
    #     return torch.exp(envelope_alpha*((self.X-x0)**2+(self.Y-y0)**2)+1j*(k0*(self.X-x0)+ky0*(self.Y-y0)))
    # def vortex_wave(self, envelope_alpha=-0.125, charge=1, x0=0., y0=0.):
    #     core = self.X-x0 + 1j*(1 if charge > 0 else -1 if charge < 0 else 0)*(self.Y-y0)
    #     return core**abs(charge)*self.gaussian_wave(envelope_alpha, x0, y0=y0)
    # def angular_momentum(self, psi=None):
    #     psi = self.psi if psi is None else psi
    #     field_k = torch.fft.fftn(psi)
    #     derivative_x, derivative_y = (torch.fft.ifftn(1j*k*field_k) for k in (self.k[:,None],self.k[None,:]))
    #     return torch.real(torch.sum(torch.conj(psi)*(-1j*(self.X*derivative_y-self.Y*derivative_x))))/torch.sum(torch.abs(psi)**2)

def axes(count, sample_interval, hbar=HBAR, max_delay=50.):
    # Uniform-time Fourier bins: $E=\hbar\omega=2\pi\hbar f$; delays are
    # integer multiples of the complex-signal sampling interval.
    energy = torch.fft.fftshift(2*torch.pi*hbar*torch.fft.fftfreq(count,d=sample_interval,dtype=torch.float64)) if count else torch.empty(0,dtype=torch.float64)
    lag = min(int(max_delay/sample_interval), max(0, count-2))
    delays = torch.arange(-lag, lag+1, dtype=torch.float64)*sample_interval
    return energy, delays

@torch.inference_mode()
def spectra(line_history, centre_history, count=None):
    # Main-text Eqs. (1)-(2) motivate resolving the two condensate signals into
    # even/odd parity. With centre amplitudes $c_1,c_2$,
    # $c_e=(c_1+c_2)/\sqrt{2}$ and $c_o=(c_1-c_2)/\sqrt{2}$; spectra are $|\mathcal F_t[c]|^2$.
    # With $E_m=2\pi\hbar f_m$, PyTorch's inverse time FFT is
    # $A_j(E_m)=N^{-1}\sum_n c_j(t_n)e^{+iE_mt_n/\hbar}$; the centre intensity is $|A_j|^2$.
    # The y integral selects $k_y=0$; the x FFT then gives the paper's Fig. 2e-f
    # observable $I(k_x,E)\propto|\mathcal F_x\mathcal F_t^{(+)}[\int dy\,\psi(x,y,t)]|^2$.
    count = centre_history.shape[0] if count is None else count
    if count < 2:
        raise ValueError("At least two planned observations are required.")
    amplitude = torch.fft.fftshift(torch.fft.ifft(centre_history, n=count, dim=0), dim=0)
    # Spatial FFT uses the negative-exponential convention; scale factors in x
    # and y are omitted from the displayed intensity (arbitrary normalization).
    dispersion = torch.fft.fft(line_history, dim=1)
    dispersion = torch.fft.ifft(dispersion, n=count, dim=0)
    result = dict(centre_spectrum=amplitude.abs().square(),
                  dispersion=torch.fft.fftshift(dispersion.abs().square(), dim=(0, 1)))
    if centre_history.shape[1] == 2:
        # Paper Eqs. (1)-(2): even/odd amplitudes; Supplementary Fig. 1(h):
        # $z=(\rho_1-\rho_2)/(\rho_1+\rho_2)$ and $\phi=\arg(c_1c_2^*)$.
        # Here $\rho_j=|c_j|^2$ uses the interpolated value at pump centre j,
        # rather than a spatial integral over a finite condensate-sized region.
        parity = torch.stack((amplitude.sum(dim=1), amplitude[:, 0]-amplitude[:, 1]), dim=1)/2**.5
        density = centre_history.abs().square()
        total = density.sum(dim=1)
        phase = torch.angle(centre_history[:, 0]*centre_history[:, 1].conj())
        result.update(parity_spectrum=parity.abs().square(),
            density_imbalance=torch.where(total > 0, (density[:, 0]-density[:, 1])/total, float("nan")),
            relative_phase=phase.masked_fill((density == 0).any(dim=1), float("nan")))
    return result

@torch.inference_mode()
def coherence(fields, delays, sample_interval):
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
    fields = fields.to(device="cpu")
    count, channels = fields.shape
    lags = torch.round(delays.cpu()/sample_interval).long()
    size = 1 << (2*count-1).bit_length()
    transformed = torch.fft.fft(fields, n=size, dim=0)
    correlation = torch.fft.ifft(transformed[:, :1].conj()*transformed, dim=0)[lags.remainder(size)]
    powers = fields.abs().square().to(torch.float64)
    prefix = torch.cat((powers.new_zeros((1, channels)), powers.cumsum(dim=0)))
    first, second = (-lags).clamp(0, count), lags.clamp(0, count)
    overlap = (count-lags.abs()).clamp_min(0)
    first_power = prefix[first+overlap, 0]-prefix[first, 0]
    second_power = prefix[second+overlap]-prefix[second]
    power = first_power[:, None]+second_power
    valid = (count >= 2) & (overlap > 0)[:, None] & (power > 0)
    return torch.where(valid, 2*correlation.abs()/power, float("nan"))

@dataclass
class Result:
    solver: GPESolver
    planned: int
    sample_interval: float = 0.
    count: int = 0
    final: bool = False
    observation_times: object = None
    times: object = None
    energy_axis: object = None
    coherence_delays: object = None
    line_history: object = None
    centre_history: object = None
    fields: object = None
    active_history: object = None
    inactive_history: object = None
    observables: dict = field(default_factory=dict)
    averages: dict = field(default_factory=dict)
    plot: object = None

    @property
    def min_samples(self):
        return min(8, max(2, self.planned))

def analyse(r, final=False):
    r.final = final
    if r.count >= 2:
        lines, centres = r.line_history[:r.count], r.centre_history[:r.count]
        if final:
            lines, centres = (v.to(device='cpu', dtype=torch.complex128) for v in (lines, centres))
        r.observables = spectra(lines, centres, count=r.planned)
        r.observables['fringe_visibility'] = coherence(centres, r.coherence_delays, r.sample_interval)

@torch.inference_mode()
def evolve(s, c, pump_centres=(0.,), stepper=None):
    start = s.t
    if c.dt <= 0 or c.t_final < start:
        raise ValueError('Require dt > 0 and t_final >= start.')
    n_steps = round((c.t_final-start)/c.dt)
    records = [k for k in range(0, n_steps+1, c.record_every)
               if start+k*c.dt >= c.settling_time]
    frames = sorted({*range(0, n_steps+1, c.frame_every), n_steps}) if c.frame_every else []
    record_set, frame_set = set(records), set(frames)

    r = Result(solver=s, planned=len(records), sample_interval=c.record_every*c.dt)
    r.observation_times = torch.tensor([start+k*c.dt for k in records], dtype=torch.float64)
    r.times = torch.tensor([start+k*c.dt for k in frames], dtype=torch.float64)
    r.energy_axis, r.coherence_delays = axes(r.planned, r.sample_interval, s.hbar)
    r.line_history = s.psi.new_zeros((r.planned, s.points))
    r.centre_history = s.psi.new_zeros((r.planned, len(pump_centres)))
    r.fields = s.psi.new_empty((len(frames), s.points, s.points))
    r.active_history = s.n_A.new_empty(r.fields.shape)
    r.inactive_history = s.n_I.new_empty(r.fields.shape)

    pos = (s.x.new_tensor(pump_centres)+s.half_length)/s.dx
    base = pos.floor()
    index, fraction = base.long(), pos-base
    if not ((index >= 0) & (index+1 < s.points)).all():
        raise ValueError('Centres must lie inside the grid.')
    if c.average_images:
        density_sum = torch.zeros_like(s.X)
        momentum_sum = torch.zeros_like(s.X)

    step = stepper or s.strang_step
    stored, last_count, last_draw = 0, 0, -float('inf')
    for k in range(n_steps+1):
        s.t = start+k*c.dt
        if k in frame_set:
            r.fields[stored] = s.psi
            r.active_history[stored] = s.n_A
            r.inactive_history[stored] = s.n_I
            stored += 1
        if k in record_set:
            i = r.count
            # Paper dispersion panel: sum over y samples approximates
            # $\psi_{k_y=0}(x,t)=\int dy\,\psi(x,y,t)$. At pump centre $j$ record
            # $c_j(t)=\psi(x_j,y=0,t)$, with linear interpolation in x below.
            r.line_history[i] = s.psi.sum(dim=1)*s.dx
            r.centre_history[i] = ((1-fraction)*s.psi[index, s.points//2]
                                   + fraction*s.psi[index+1, s.points//2])
            if c.average_images:
                density_sum += squared_density(s.psi)
                momentum_sum += torch.fft.fftn(s.psi).abs().square()
            r.count += 1
            if c.live and k < n_steps and perf_counter()-last_draw >= .5:
                if r.count >= max(r.min_samples, ceil(1.1*last_count)):
                    analyse(r)
                    last_count = r.count
                plot(r)
                last_draw = perf_counter()
        if k < n_steps:
            step(c.dt)

    if c.average_images and r.count:
        r.averages = dict(mean_density=density_sum/r.count,
                          mean_momentum_intensity=torch.fft.fftshift(momentum_sum/r.count))
    analyse(r, final=True)
    if c.live:
        plot(r)
    return r

def run(c, stepper=None):
    if not c.separations or (None in c.separations and len(c.separations) > 1):
        raise ValueError('separations must be nonempty; None cannot be mixed with other values.')
    device = c.device or ('mps' if torch.backends.mps.is_available() else 'cpu')
    wall = .83*c.half_length if c.wall_position is None else c.wall_position
    output = Path(c.output or Path(__file__).parent)
    for separation in c.separations:
        s = GPESolver(points=c.points, half_length=c.half_length)
        s.grid(device=device, dtype=c.dtype)
        s.fft(c.dt)
        s.set_potential(s.tanh_wall(wall, c.wall_width, c.wall_strength))
        centres = (0.,) if separation is None else (-separation/2, separation/2)
        # The dyad pump is the sum of identical CW Gaussians centered at
        # $\mathbf r_j=(\pm d/2,0)$: $P(\mathbf r)=\sum_jP_0\exp[-|\mathbf r-\mathbf r_j|^2/(2\sigma^2)]$.
        for centre in centres:
            s.P = s.P+s.gaussian_pump((centre, 0.), c.P0, c.sigma)
        torch.manual_seed(c.seed)
        # Main text Eq. (13) and Supplementary Note 1 describe white-noise
        # initial conditions. This implementation uses independent centered
        # uniform real/imaginary samples (not Gaussian):
        # $\psi_0=10^{-6}[(U_1-1/2)+i(U_2-1/2)]$, $U_j\sim\mathrm{Uniform}(0,1)$.
        s.psi = 1e-6*(torch.rand_like(s.X)-.5+1j*(torch.rand_like(s.X)-.5))
        if c.steady_reservoirs:
            # Zero-field steady state derived from S2-S3 at fixed P:
            # $n_I=P/(\Gamma_I+W)$, $n_A=Wn_I/\Gamma_A$. This is an optional
            # initialization; default zero reservoirs reproduce a pump turn-on.
            s.n_I = s.P/(s.Gamma_I+s.W)
            s.n_A = s.W*s.n_I/s.Gamma_A
        step = None if stepper is None else partial(stepper, s)
        result = evolve(s, c, centres, step)
        if c.save_outputs:
            name = 'run' if len(c.separations) == 1 else f'run_d{separation:g}'
            save(result, output/name, c.compress)
    return result

def save(r, path, compress=False):
    path, s = Path(path), r.solver
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(x=s.x, P=s.P, potential=s.potential, n_A=s.n_A, n_I=s.n_I,
                  momentum_axis=s.momentum_axis, psi=r.fields, times=r.times,
                  observation_times=r.observation_times, line_history=r.line_history,
                  centre_history=r.centre_history, energy_axis=r.energy_axis,
                  coherence_delays=r.coherence_delays, active_history=r.active_history,
                  inactive_history=r.inactive_history, **r.observables, **r.averages)
    data = {key: value.detach().cpu().numpy() for key, value in arrays.items()}
    writer = np.savez_compressed if compress else np.savez
    writer(path.parent/(path.name+'.npz'), **data)
    if r.plot is not None:
        r.plot['figure'].savefig(path.parent/(path.name+'_live.png'), dpi=150)

def plot(r):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    s, n = r.solver, r.count
    stride = max(1, (n+999)//1000)
    maps = [v.detach().cpu().numpy().T for v in
            (squared_density(s.psi), torch.angle(s.psi), s.n_I, s.n_A)]
    x = s.x.cpu().numpy()
    energy = r.energy_axis.numpy()
    delays = r.coherence_delays.numpy()
    times = r.observation_times[:n:stride].numpy()
    density = r.centre_history[:n:stride].abs().square().cpu().numpy()
    history = r.line_history[:n:stride].abs().square().cpu().numpy()

    if r.plot is None:
        plt.ion()
        figure, grid = plt.subplots(3, 3, figsize=(16, 12), constrained_layout=True)
        ax = grid.ravel()
        images = [a.imshow(m, origin='lower', extent=(x[0], x[-1], x[0], x[-1]),
                           cmap='twilight' if i == 1 else 'viridis')
                  for i, (a, m) in enumerate(zip(ax, maps))]
        for i, image in enumerate(images):
            figure.colorbar(image, ax=ax[i], label='Phase (rad)' if i == 1 else 'Density (um^-2)')
            ax[i].set(xlabel='x (um)', ylabel='y (um)')
        traces = [ax[i].plot([], [], '-', [], [], '--') for i in (4, 5, 7)]
        k = s.momentum_axis.cpu().numpy()
        energy_limits = (energy[0], energy[-1]) if len(energy) > 1 else (0, 4)
        delay_limit = delays[-1] if len(delays) > 1 else r.sample_interval/2
        dispersion = ax[6].imshow(np.full((2, 2), 1e-4), origin='lower', aspect='auto',
                                  extent=(k[0], k[-1], *energy_limits),
                                  cmap='magma', norm=LogNorm(vmin=1e-4, vmax=1))
        figure.colorbar(dispersion, ax=ax[6], label='Intensity / maximum')
        history_image = ax[8].imshow(np.zeros((1, s.points)), origin='lower', aspect='auto',
                                     extent=(x[0], x[-1], 0, r.sample_interval))
        ax[4].set(xlabel='Energy (meV)', ylabel='Centre spectral intensity', xlim=(0, 4))
        ax[5].set(xlabel='Delay (ps)', ylabel='Fringe visibility',
                  xlim=(-delay_limit, delay_limit), ylim=(0, 1.05))
        ax[6].set(xlabel='kx (um^-1)', ylabel='Energy (meV)', xlim=(-3, 3), ylim=(0, 4))
        ax[7].set(xlabel='Time (ps)', ylabel='Centre density')
        ax[8].set(xlabel='x (um)', ylabel='Time (ps)')
        r.plot = dict(figure=figure, ax=ax, images=images, traces=traces,
                      dispersion=dispersion, history=history_image)
        plt.show(block=False)

    p = r.plot
    for i, (image, values) in enumerate(zip(p['images'], maps)):
        image.set_data(values)
        image.set_clim((-np.pi, np.pi) if i == 1 else (0, float(values.max()) or 1.))
        p['ax'][i].set_title(f"{('Density', 'Phase', 'Inactive', 'Active')[i]}, t={s.t:.2f} ps")

    curves = []
    title = f'Waiting for {r.min_samples} samples ({r.count}/{r.planned})'
    if r.observables:
        title = 'Final' if r.final else 'Zero-padded preview (approximate)'
        obs = {key: r.observables[key].detach().cpu().numpy()
               for key in ('dispersion', 'centre_spectrum', 'fringe_visibility')}
        scale = obs['dispersion'].max()
        if scale > 0:
            p['dispersion'].set_data(obs['dispersion']/scale)
        curves += [(p['traces'][0], energy, obs['centre_spectrum']),
                   (p['traces'][1], delays, obs['fringe_visibility'])]
    for panel in (4, 5, 6):
        p['ax'][panel].set_title(title)
    if n:
        curves.append((p['traces'][2], times, density))
        half = stride*r.sample_interval/2
        p['history'].set_data(history)
        p['history'].set_extent((x[0], x[-1], times[0]-half, times[-1]+half))
        p['history'].set_clim(0, float(history.max()) or 1.)
    for lines, xs, ys in curves:
        for i, line in enumerate(lines):
            line.set_data(*((xs, ys[:, i]) if i < ys.shape[1] else ([], [])))
    for panel in (4, 5, 7):
        p['ax'][panel].relim()
        p['ax'][panel].autoscale_view(scalex=panel == 7)
    # p['figure'].canvas.draw_idle()
    # p['figure'].canvas.flush_events()
    plt.pause(0.001)
    if r.final:
        plt.ioff()
        plt.show()

if __name__ == '__main__':
    run(Config(live=True))
