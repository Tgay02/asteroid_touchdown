"""
2D powered-descent lander on a small asteroid: box body, two legs.

Frame : local flat surface frame, x horizontal, z up (anti-gravity). 
    Sloped terrain z_s = tan(phi)*x.
Angle : theta = rotation about +Y_B (body z axis -> world x for theta > 0),
     same sign as the book's [C].
Model : - translation: point mass, constant effective gravity, main engine along body z
        - rotation + flexible/slosh modes: Eqs. 10.5.2, 10.5.3, 10.5.7 
          (augmented with contact torque)
        - contact: spring-damper normal force + regularized Coulomb friction at 6 points
          (2 feet + 4 hull corners; any hull-corner load counts as a body strike);
          normal force is Hunt-Crossley by default (Kelvin-Voigt selectable), 
          with damping calibrated to a
          requested restitution (see restitution_to_chi)
        - guidance/control: vertical velocity-vs-altitude profile + 
          lateral velocity damping + PD attitude;
          optional post-touchdown attitude hold, through a reaction wheel (momentum-limited)
          or thruster couples (propellant-limited), 
          plus an optional hold-down thruster pressing the lander onto the ground
State : s = 
[x, z, vx, vz, theta, omega, eta(m), sigma(n), eta_dot(m), sigma_dot(n), h_wheel, rcs_impulse_used]

Requires: numpy, scipy, matplotlib

SOURCES for default parameters (details in the Params comments):
 [B19]  Barnouin et al. 2019, Nature Geoscience 12, 247-252 (Bennu shape, mass, spin, 
                                                             surface gravity, slopes)
 [S19]  Scholten et al. 2019, A&A 632, L3 (MASCOT descent and bouncing path on Ryugu)
 [Bi17] Biele et al. 2017, arXiv:1705.00701 (structural coefficient of restitution of MASCOT)
 [Bi15] Biele et al. 2015, Science 349, aaa9816 (Philae landings, surface strength)
 [W22]  Walsh et al. 2022, Science Advances (Bennu near-zero cohesion, TAGSAM contact at 10 cm/s)
 [Ba21] Ballouz et al. 2021, MNRAS 507 (friction angle / cohesion effects on TAGSAM penetration)
 [H19]  Hayabusa2 touchdown dynamics report, IAC-19.A3.4B.1 
         (8.5 m final descent, -7.4 cm/s, slope < 30 deg)
 [Re20] "Regolith behavior under asteroid-level gravity conditions: Low-velocity impacts into mm- 
         and cm-sized grain targets", arXiv:2008.08720 
         (MASCOT contact-by-contact restitution on Ryugu)
 [Ro]   Philae hold-down thruster: arXiv:2208.08476 (Past and Future Comet Missions); 
         ESA Rosetta blog, Nov 2014
 [HC75] Hunt & Crossley 1975, J. Appl. Mech. 42, 440-445 (damping and restitution; 
        basis of the Hunt-Crossley force law)
"""
from dataclasses import dataclass, field, replace
from functools import lru_cache
from collections import Counter
import itertools
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


# ============================================================================ contact calibration
def impact_1d(chi, k_n, n, m_eff, v0, hc):
    """One-dimensional normal impact of a contact point on the terrain, using the same force law 
    as the simulation. Returns the normal restitution |v_out| / v0.
    hc=True : Hunt-Crossley  F = k_n x^n (1 + chi * xdot)   
        (zero force at first contact and at release) [HC75]
    hc=False: Kelvin-Voigt   F = k_n x + k_n * chi * xdot   
        (n must be 1; force jumps at first contact)
    Both are clipped at F >= 0 (the ground can push but not pull)."""
    def f(t, y):
        x, v = y                                   # penetration depth, penetration rate
        if x <= 0.0:
            return [v, 0.0]
        F = k_n * x**n * (1.0 + chi * v) if hc else k_n * x + k_n * chi * v
        return [v, -max(F, 0.0) / m_eff]

    def sep(t, y): return y[0]
    sep.terminal, sep.direction = True, -1
    sol = solve_ivp(f, (0.0, 120.0), [0.0, v0], events=[sep], rtol=1e-9, atol=1e-12)
    if sol.t_events[0].size == 0:
        raise RuntimeError(
            "Contact did not separate within 120 s; check chi, k_n, and m_eff."
        )
    v_out = sol.y_events[0][0][1]   # velocity at the separation event
    return abs(v_out) / v0

@lru_cache(maxsize=2048)
def restitution_to_chi(e, k_n, n, m_eff, v_ref, hc):
    """Damping coefficient chi [s/m] that gives normal restitution e for an impact at speed v_ref.
    Solved numerically: the textbook formula e = exp(-pi*zeta/sqrt(1-zeta^2)) 
    is wrong once the force is clipped at zero. For Hunt-Crossley the restitution 
    depends on impact speed, so e applies at v_ref only."""
    e = float(np.clip(e, 0.003, 0.999))
    g = lambda lc: impact_1d(10.0**lc, k_n, n, m_eff, v_ref, hc) - e
    return 10.0 ** brentq(g, -4.0, 4.0, xtol=1e-4)


# ============================================================================ parameters
@dataclass
class Params:
    # --- asteroid (Bennu) and terrain
    GM: float = 4.891                  # m^3/s^2 = G * 7.329e10 kg  [B19 mass]
    R_ast: float = 244.0               # m, equivalent-volume radius [B19 table, as retrieved]
    spin_period: float = 4.2976 * 3600  # s  [B19]
    slope_deg: float = 0.0 # terrain slope; Bennu mean 17 +/- 2 deg, range 0-92 [B19]; H19 limited
                           # to < 30
    mu: float = float(np.tan(np.deg2rad(30.0)))  
    # 0.577: regolith friction angle ~28-30 deg in TAGSAM models [Ba21].
    
    #   CAUTION: that is an internal friction angle of the regolith, 
    #   not a foot-pad/regolith interface value.

    # --- lander box + legs
    mass: float = 500.0                # kg; same scale as real landers/spacecraft: MASCOT 10, 
                                       # Philae ~100, OSIRIS-REx 1300 [Ba21]
    width: float = 1.5                 # m
    height: float = 1.0                # m
    leg_x: float = 1.0                 # m, foot half-span from the CG
    leg_len: float = 0.5               # m, foot below the bottom of the box

    # --- contact (normal force)
    contact_model: str = "hunt_crossley"   # "kelvin_voigt" reproduces the earlier linear spring-damper
    k_c: float = 2.0e4                 # N/m, secant stiffness at delta_ref. PLACEHOLDER (no regolith-stiffness source found)
    hc_n: float = 1.5                  # force exponent: 3/2 is the Hertz value for a sphere on a plate [HC75]; 1 = linear
    delta_ref: float = 5.0e-3          # m, penetration at which the secant stiffness equals k_c
    restitution: float = 0.4           # normal restitution; MASCOT structure on a hard wall: median ~0.4, max ~0.6,
    #   range ~0-0.8 [Bi17]; Philae soft-site rebound speed ~1/3 of impact [Bi15 via Ricochets paper].
    #   Hunt-Crossley: this is the value AT restitution_v_ref; it falls as impact speed rises.
    restitution_v_ref: float = 0.10    # m/s; MASCOT hard-wall tests used 0.1-0.2 m/s [Bi17]; TAGSAM contact 0.1 m/s [W22]
    eps_v: float = 1.0e-4              # m/s, friction regularization
    F_leg_max: float = 1500.0          # N, leg overload limit

    # --- actuators
    T_max: float = 20.0                # N
    tau_max: float = 5.0               # N m
    thrust_scale: float = 1.0          # thrust magnitude error (Monte Carlo)
    thrust_angle_err: float = 0.0      # rad, thrust misalignment (Monte Carlo)

    # --- post-touchdown attitude hold (all numeric/boolean so every one can be a Monte Carlo variable)
    post_td_control: bool = True      # master switch; off reproduces the earlier passive behavior
    post_td_tau_max: float = 5.0       # N m, torque authority after touchdown (0 = no control). PLACEHOLDER
    post_td_wn: float = 0.3            # rad/s, hold-loop bandwidth (same as the descent loop)
    post_td_zeta: float = 0.7
    post_td_align: float = 1.0         # 1 = align with the terrain slope, 0 = hold local vertical
    post_td_delay: float = 0.0         # s, latency from first contact until the hold engages. PLACEHOLDER
    post_td_duration: float = np.inf   # s, how long the hold stays active after first contact
    h_max: float = np.inf              # N m s, reaction-wheel momentum capacity (inf = unlimited). PLACEHOLDER

    # --- attitude actuator. "wheel": reaction wheel, limited by momentum capacity h_max.
    #     "rcs": thruster couples, limited by torque rcs_tau and total angular impulse rcs_budget (propellant).
    attitude_actuator: str = "wheel"
    rcs_tau: float = 2.0               # N m, thruster-couple torque (e.g. 2 x 1 N at a 1 m arm). PLACEHOLDER
    rcs_budget: float = np.inf         # N m s, total angular impulse available (inf = unlimited). PLACEHOLDER
    rcs_u_db: float = 0.01             # N m, soft deadband on the commanded torque (no firing for tiny commands)

    # --- hold-down thruster: pushes the lander toward the ground along body -z after first contact.
    #     Philae carried one for this purpose [Ro]; no thrust level was found, so the numbers are PLACEHOLDERS.
    hold_down: bool = True            # master switch
    hold_force: float = 2.0            # N (maximum thrust)
    hold_mode: str = "constant"        # "constant" = full thrust for the window; "rebound" = proportional to the
    #                                    CG velocity away from the terrain, capped at hold_force
    hold_gain: float = 0.5             # 1/s, rebound mode: F = mass * gain * (outward normal speed)
    hold_delay: float = 0.0            # s after first contact before the thruster starts
    hold_duration: float = 120.0       # s the thruster stays enabled once started
    hold_angle_err: float = 0.0        # rad, misalignment from body -z (pushes sideways if nonzero)

    # --- guidance and control
    k_h: float = 0.05                  # 1/s, descent profile slope
    v_td: float = 0.074                # m/s, touchdown speed: Hayabusa2 final descent 7.4 cm/s [H19]; TAGSAM 10 cm/s [W22]
    v_max: float = 0.10                # m/s
    kv: float = 0.4                    # 1/s, vertical velocity loop
    kx: float = 0.15                   # 1/s, lateral velocity loop
    h_fade: float = 4.0                # m, lateral control fades out below this altitude
    a_floor: float = 0.002             # m/s^2, avoids a degenerate tilt command
    th_max_deg: float = 35.0
    att_wn: float = 0.3                # rad/s
    att_zeta: float = 0.7

    # --- flexible and slosh modes (book model); b = pitch row of (K C)^T, d = pitch row of D
    use_modes: bool = True
    K: np.ndarray = field(default_factory=lambda: np.array([[0, 0.8, 0], [0, 0.3, 0.0]]))
    Dc: np.ndarray = field(default_factory=lambda: np.array([[0.0], [0.5], [0.0]]))
    alpha_deg: float = 30.0
    zeta_eta: np.ndarray = field(default_factory=lambda: np.array([0.005, 0.005]))
    Om_eta: np.ndarray = field(default_factory=lambda: np.array([2.0, 5.0]))
    zeta_sig: np.ndarray = field(default_factory=lambda: np.array([0.01]))
    Om_sig: np.ndarray = field(default_factory=lambda: np.array([1.5]))

    # --- run limits and classification
    t_max: float = 21600.0             # s (6 h): Hunt-Crossley hopping decays slowly at low speed
    chunk: float = 60.0                # s, settle check interval
    v_rest: float = 2.0e-3             # m/s
    h_rest: float = 0.10               # m: 'at rest' also needs the lowest point within this height of the terrain
    h_lost: float = 30.0               # m, rebound above this counts as escaped

    @property
    def g(self):                       # equatorial effective gravity (gravity minus centrifugal)
        Om = 2 * np.pi / self.spin_period
        return self.GM / self.R_ast**2 - Om**2 * self.R_ast

    @property
    def Iyy(self):                     # box, 2D
        return self.mass * (self.width**2 + self.height**2) / 12.0


# ============================================================================ lander model
class Lander:
    def __init__(self, P: Params):
        self.P = P
        phi = np.deg2rad(P.slope_deg)
        self.phi, self.tphi, self.sphi, self.cphi = phi, np.tan(phi), np.sin(phi), np.cos(phi)
        self.lz = P.height / 2 + P.leg_len
        # normal-force law, damping calibrated to the requested restitution for a centered two-foot impact
        # (each foot sees half the mass)
        self.hc = P.contact_model == "hunt_crossley"
        if not self.hc and P.contact_model != "kelvin_voigt":
            raise ValueError("contact_model must be 'hunt_crossley' or 'kelvin_voigt'")
        self.hc_n = P.hc_n if self.hc else 1.0          # NOTE: self.n is the slosh-mode count
        self.k_n = P.k_c * P.delta_ref ** (1.0 - self.hc_n)
        self.chi = restitution_to_chi(float(P.restitution), float(self.k_n), float(self.hc_n),
                                      float(P.mass / 2.0), float(P.restitution_v_ref), self.hc)
        hw, hh = P.width / 2, P.height / 2
        self.pts = np.array([[-P.leg_x, -self.lz], [P.leg_x, -self.lz],      # 0,1 feet
                             [-hw, -hh], [hw, -hh], [-hw, hh], [hw, hh]])    # 2..5 hull corners

        # modes
        if P.use_modes:
            a = np.deg2rad(P.alpha_deg)
            C = np.array([[np.cos(a), 0, -np.sin(a)], [0, 1, 0], [np.sin(a), 0, np.cos(a)]])
            self.b = (P.K @ C).T[1, :]                 # 1 x m
            self.d = P.Dc[1, :]                        # 1 x n
            self.zeta_eta, self.Om_eta = P.zeta_eta, P.Om_eta
            self.zeta_sig, self.Om_sig = P.zeta_sig, P.Om_sig
        else:
            self.b, self.d = np.zeros(0), np.zeros(0)
            self.zeta_eta = self.Om_eta = self.zeta_sig = self.Om_sig = np.zeros(0)
        self.m, self.n = len(self.b), len(self.d)
        m, n = self.m, self.n
        Mm = np.block([[np.array([[P.Iyy]]), self.b[None, :], self.d[None, :]],
                       [-self.b[:, None], np.eye(m), np.zeros((m, n))],
                       [-self.d[:, None], np.zeros((n, m)), np.eye(n)]])
        self.Minv = np.linalg.inv(Mm)
        self.cd_eta, self.k_eta = 2 * self.zeta_eta * self.Om_eta, self.Om_eta**2
        self.cd_sig, self.k_sig = 2 * self.zeta_sig * self.Om_sig, self.Om_sig**2
        self.Kp = P.Iyy * P.att_wn**2
        self.Kd = 2 * P.att_zeta * P.att_wn * P.Iyy
        self.Kp2 = P.Iyy * P.post_td_wn**2
        self.Kd2 = 2 * P.post_td_zeta * P.post_td_wn * P.Iyy
        self.t_td = np.inf                                   # time of first contact (set during simulate)
        if P.attitude_actuator not in ("wheel", "rcs"):
            raise ValueError("attitude_actuator must be 'wheel' or 'rcs'")
        if P.hold_mode not in ("constant", "rebound"):
            raise ValueError("hold_mode must be 'constant' or 'rebound'")
        rcs = P.attitude_actuator == "rcs"
        self.tau_desc_lim = P.rcs_tau if rcs else P.tau_max             # torque authority during descent
        self.tau_post_lim = P.rcs_tau if rcs else P.post_td_tau_max     # torque authority after touchdown
        self.ns = 6 + 2 * m + 2 * n + 2                      # last two states: wheel momentum, RCS impulse used
        self.i_hw, self.i_rcs = self.ns - 2, self.ns - 1

    # ---- geometry helpers
    def foot_altitude(self, s):
        """Vertical height of the feet above the terrain."""
        return (s[1] - self.tphi * s[0]) - self.lz

    # ---- contact
    def contact(self, s):
        P = self.P
        x, z, vx, vz, th, w = s[:6]
        c, sn = np.cos(th), np.sin(th)
        bx, bz = self.pts[:, 0], self.pts[:, 1]
        rx, rz = c * bx + sn * bz, -sn * bx + c * bz          # world-frame offsets from the CG
        px, pz = x + rx, z + rz
        vpx, vpz = vx + w * rz, vz - w * rx
        d = (pz - self.tphi * px) * self.cphi                  # signed distance to terrain
        pen = np.maximum(-d, 0.0)
        vn = -self.sphi * vpx + self.cphi * vpz
        vt = self.cphi * vpx + self.sphi * vpz
        spring = self.k_n * pen**self.hc_n
        F = spring * (1.0 - self.chi * vn) if self.hc else spring - self.k_n * self.chi * vn   # penetration rate = -vn
        Fn = np.where(pen > 0, np.maximum(F, 0.0), 0.0)
        Ft = -P.mu * Fn * np.tanh(vt / P.eps_v)
        Fx = -Fn * self.sphi + Ft * self.cphi
        Fz = Fn * self.cphi + Ft * self.sphi
        return Fx.sum(), Fz.sum(), np.sum(rz * Fx - rx * Fz), d, Fn

    # ---- guidance + control (truth-state feedback for now)
    def control(self, s):
        P = self.P
        vx, vz, th, w = s[2], s[3], s[4], s[5]
        h = self.foot_altitude(s)
        vz_cmd = -np.clip(P.k_h * h, P.v_td, P.v_max)
        az = P.g + P.kv * (vz_cmd - vz)
        ax = -P.kx * vx * np.clip(h / P.h_fade, 0.0, 1.0)
        th_max = np.deg2rad(P.th_max_deg)
        # tilt toward the commanded acceleration, but only as far as thrust is actually demanded: with no vertical
        # thrust demand (az <= 0) tilting does nothing useful and only wastes wheel momentum
        th_cmd = np.arctan2(ax, max(az, P.a_floor)) * np.clip(az / P.a_floor, 0.0, 1.0)
        th_cmd = np.clip(th_cmd, -th_max, th_max)
        T = P.mass * max(az, 0.0) / max(np.cos(th), 0.5)
        T = np.clip(T, 0.0, P.T_max) * P.thrust_scale
        tau = np.clip(self.Kp * (th_cmd - th) - self.Kd * w, -self.tau_desc_lim, self.tau_desc_lim)
        return T, tau

    # ---- post-touchdown attitude hold and reaction-wheel limit
    def post_td_torque(self, t, s):
        """Torque-limited PD toward (align * terrain slope), active from first contact + delay for `duration`."""
        P = self.P
        if (not P.post_td_control) or t < self.t_td + P.post_td_delay or t > self.t_td + P.post_td_duration:
            return 0.0
        u = self.Kp2 * (P.post_td_align * self.phi - s[4]) - self.Kd2 * s[5]
        return float(np.clip(u, -self.tau_post_lim, self.tau_post_lim))

    def wheel(self, u_cmd, h_w):
        """Reaction-wheel momentum limit. The wheel reacts against the body (dh_w/dt = -u), so the torque tapers
        to zero as |h_w| approaches h_max (linear taper over the last 2% of capacity)."""
        hm = max(self.P.h_max, 0.0)
        if (not np.isfinite(hm)) or u_cmd == 0.0:
            return u_cmd
        headroom = (h_w + hm) if u_cmd > 0 else (hm - h_w)
        return u_cmd * float(np.clip(headroom / max(0.02 * hm, 1e-9), 0.0, 1.0))

    def actuate(self, u_cmd, s):
        """Maps a commanded body torque to (applied torque, d(wheel momentum)/dt, d(RCS angular impulse used)/dt)."""
        P = self.P
        if P.attitude_actuator == "rcs":
            u = np.sign(u_cmd) * max(abs(u_cmd) - P.rcs_u_db, 0.0)           # soft deadband, saves propellant
            if np.isfinite(P.rcs_budget):                                     # propellant runs out -> taper to zero
                u *= float(np.clip((P.rcs_budget - s[self.i_rcs]) / max(0.02 * P.rcs_budget, 1e-9), 0.0, 1.0))
            return u, 0.0, abs(u)
        u = self.wheel(u_cmd, s[self.i_hw])
        return u, -u, 0.0

    def hold_force_vec(self, t, s):
        """World-frame force of the hold-down thruster. Fires along body -z (a body-fixed thruster, so it points
        wherever the lander points), inside the window [t_td + delay, t_td + delay + duration]."""
        P = self.P
        t0 = self.t_td + P.hold_delay
        if (not P.hold_down) or P.hold_force <= 0.0 or t < t0 or t > t0 + P.hold_duration:
            return 0.0, 0.0
        if P.hold_mode == "rebound":
            v_out = -self.sphi * s[2] + self.cphi * s[3]                      # CG velocity away from the terrain
            F = float(np.clip(P.mass * P.hold_gain * max(v_out, 0.0), 0.0, P.hold_force))
        else:
            F = P.hold_force
        ang = s[4] + P.hold_angle_err
        return -F * np.sin(ang), -F * np.cos(ang)

    # ---- equations of motion
    def rhs(self, t, s, powered):
        P, m, n = self.P, self.m, self.n
        x, z, vx, vz, th, w = s[:6]
        eta, sig = s[6:6 + m], s[6 + m:6 + m + n]
        etad, sigd = s[6 + m + n:6 + 2 * m + n], s[6 + 2 * m + n:6 + 2 * m + 2 * n]
        if powered:
            T, u_cmd = self.control(s)                         # descent: thrust + attitude torque
        else:
            T, u_cmd = 0.0, self.post_td_torque(t, s)          # after touchdown: thrust off, optional attitude hold
        tau_c, dh_w, d_rcs = self.actuate(u_cmd, s)
        Fx, Fz, tau_k, _, _ = self.contact(s)
        Fhx, Fhz = self.hold_force_vec(t, s)
        a = T / P.mass
        ang = th + P.thrust_angle_err
        ax = a * np.sin(ang) + (Fx + Fhx) / P.mass
        az = -P.g + a * np.cos(ang) + (Fz + Fhz) / P.mass
        rhs_q = np.concatenate(([tau_c + tau_k],
                                -self.cd_eta * etad - self.k_eta * eta,
                                -self.cd_sig * sigd - self.k_sig * sig))
        qdd = self.Minv @ rhs_q                               # [w_dot; eta_ddot; sigma_ddot]
        ds = np.empty_like(s)
        ds[0:6] = [vx, vz, ax, az, w, qdd[0]]
        ds[6:6 + m] = etad
        ds[6 + m:6 + m + n] = sigd
        ds[6 + m + n:6 + 2 * m + n] = qdd[1:1 + m]
        ds[6 + 2 * m + n:6 + 2 * m + 2 * n] = qdd[1 + m:]
        ds[self.i_hw] = dh_w
        ds[self.i_rcs] = d_rcs
        return ds

    def _hold_impulse(self, t, Y):
        """Total impulse of the hold-down thruster over the run (N s). Exact for constant mode, trapezoid otherwise."""
        P = self.P
        if (not P.hold_down) or not np.isfinite(self.t_td):
            return 0.0
        if P.hold_mode == "constant":
            t0 = self.t_td + P.hold_delay
            return float(P.hold_force * max(0.0, min(t[-1], t0 + P.hold_duration) - t0))
        F = np.array([np.hypot(*self.hold_force_vec(t[k], Y[:, k])) for k in range(len(t))])
        return float(np.sum(0.5 * (F[1:] + F[:-1]) * np.diff(t)))

    # ---- simulation
    def simulate(self, ic=None, guided=True):
        """ic: dict(x0, h0, vx0, vz0, th0, w0), h0 = initial foot altitude. Returns a result dict."""
        P = self.P
        self.t_td = np.inf
        ic0 = dict(x0=0.0, h0=15.0, vx0=0.02, vz0=-0.2, th0=0.0, w0=0.0)
        ic0.update(ic or {})
        s0 = np.zeros(self.ns)
        s0[0] = ic0["x0"]
        s0[1] = self.tphi * ic0["x0"] + self.lz + ic0["h0"]
        s0[2:6] = [ic0["vx0"], ic0["vz0"], ic0["th0"], ic0["w0"]]
        # shift up/down so the lowest contact point starts exactly h0 above the terrain
        s0[1] += (ic0["h0"] * self.cphi - self.contact(s0)[3].min()) / self.cphi

        def touchdown(t, y): return self.contact(y)[3].min()
        touchdown.terminal, touchdown.direction = True, -1

        def escape(t, y): return P.h_lost - (y[1] - self.tphi * y[0])
        escape.terminal, escape.direction = True, -1

        # phase 1: powered descent until first contact
        s1 = solve_ivp(lambda t, y: self.rhs(t, y, guided), (0, P.t_max), s0, method="DOP853",
                       rtol=1e-8, atol=1e-10, events=[touchdown], max_step=2.0)
        T_list, Y_list = [s1.t], [s1.y]
        res = dict(outcome=None, t_td=np.nan, td=None)
        if s1.status != 1:
            res["outcome"] = "no_touchdown"
        else:
            res["t_td"] = s1.t[-1]
            self.t_td = s1.t[-1]
            y = s1.y[:, -1]
            vn_in = -(-self.sphi * y[2] + self.cphi * y[3])     # positive = approaching terrain
            vt = self.cphi * y[2] + self.sphi * y[3]
            res["td"] = dict(vn=vn_in, vt=vt, th_rel=y[4] - self.phi, w=y[5], x=y[0])

            # phase 2: thrust off, integrate in chunks until settled / escaped / timeout
            t, settled, escaped = s1.t[-1], False, False
            while t < P.t_max and not (settled or escaped):
                sol = solve_ivp(lambda tt, yy: self.rhs(tt, yy, False), (t, min(t + P.chunk, P.t_max)),
                                y, method="LSODA", rtol=1e-7, atol=1e-10, events=[escape])
                T_list.append(sol.t[1:]); Y_list.append(sol.y[:, 1:])
                t, y = sol.t[-1], sol.y[:, -1]
                escaped = sol.status == 1
                quiet = max(np.hypot(sol.y[2], sol.y[3]).max(), np.abs(sol.y[5]).max()) < P.v_rest
                settled = quiet and self.contact(y)[3].min() < P.h_rest   # quiet and near the surface (not mid-flight)
            res["outcome"] = "escaped" if escaped else ("settled" if settled else "not_settled")

        t_all, Y = np.concatenate(T_list), np.hstack(Y_list)
        # diagnostics over the whole history
        Fn_foot, Fn_hull = 0.0, 0.0
        for k in range(Y.shape[1]):
            Fn = self.contact(Y[:, k])[4]
            Fn_foot, Fn_hull = max(Fn_foot, Fn[:2].max()), max(Fn_hull, Fn[2:].max())
        res.update(t=t_all, Y=Y, peak_Fn_foot=Fn_foot, peak_Fn_hull=Fn_hull, peak_wheel_h=float(np.abs(Y[self.i_hw]).max()), rcs_impulse=float(Y[self.i_rcs, -1]),
                   hold_impulse=self._hold_impulse(t_all, Y),
                   th_final_rel=Y[4, -1] - self.phi, x_final=Y[0, -1])
        # final classification
        if res["outcome"] == "settled":
            if Fn_hull > 0:
                res["outcome"] = "body_strike"
            elif Fn_foot > P.F_leg_max:
                res["outcome"] = "leg_overload"
            else:
                res["outcome"] = "success"
        elif res["outcome"] == "escaped":
            pass
        return res



def restitution_vs_speed(P, speeds):
    """Normal restitution of the configured contact law for centered two-foot impacts at the given speeds."""
    L = Lander(P)
    return np.array([impact_1d(L.chi, L.k_n, L.hc_n, P.mass / 2.0, float(v), L.hc) for v in speeds])


# ============================================================================ plotting
def plot_run(res, P):
    import matplotlib.pyplot as plt
    L = Lander(P)
    t, Y = res["t"], res["Y"]
    fig, ax = plt.subplots(4, 2, figsize=(11, 13))
    xs = np.linspace(Y[0].min() - 2, Y[0].max() + 2, 2)
    ax[0, 0].plot(xs, np.tan(L.phi) * xs, "k-", lw=2, label="terrain")
    ax[0, 0].plot(Y[0], Y[1] - L.lz, label="foot-level path")
    ax[0, 0].set_xlabel("x (m)"); ax[0, 0].set_ylabel("z (m)"); ax[0, 0].legend(); ax[0, 0].grid(True)
    ax[0, 0].set_title(f"outcome: {res['outcome']}")
    td = res["t_td"]
    ax[0, 1].plot(t, Y[2], label="vx"); ax[0, 1].plot(t, Y[3], label="vz")
    ax[0, 1].set_xlim(0, (td if np.isfinite(td) else t[-1]) + 120)
    ax[0, 1].set_xlabel("t (s)"); ax[0, 1].set_ylabel("m/s"); ax[0, 1].legend(); ax[0, 1].grid(True)
    ax[1, 0].plot(t, np.rad2deg(Y[4])); ax[1, 0].set_ylabel("theta (deg)"); ax[1, 0].set_xlabel("t (s)")
    ax[1, 0].set_xlim(0, (td if np.isfinite(td) else t[-1]) + 120); ax[1, 0].grid(True)
    if L.m:
        for i in range(L.m):
            ax[1, 1].plot(t, Y[6 + i], label=f"eta{i + 1}")
    if L.n:
        ax[1, 1].plot(t, Y[6 + L.m], label="sigma1")
    ax[1, 1].set_xlim(0, (td if np.isfinite(td) else t[-1]) + 120)
    ax[1, 1].set_xlabel("t (s)"); ax[1, 1].set_ylabel("mode coordinates"); ax[1, 1].grid(True)
    if L.m or L.n:
        ax[1, 1].legend()
    ax[2, 0].plot(t, Y[5]); ax[2, 0].set_ylabel("omega (rad/s)"); ax[2, 0].set_xlabel("t (s)"); ax[2, 0].grid(True)
    ax[2, 0].set_xlim(0, (td if np.isfinite(td) else t[-1]) + 120)
    xl = (0, (td if np.isfinite(td) else t[-1]) + 120)
    ax[2, 1].plot(t, Y[L.i_hw], label="wheel momentum")
    ax[2, 1].plot(t, Y[L.i_rcs], label="RCS impulse used")
    ax[2, 1].set_ylabel("N m s"); ax[2, 1].set_xlabel("t (s)"); ax[2, 1].set_xlim(*xl); ax[2, 1].grid(True)
    ax[2, 1].legend()
    if P.attitude_actuator == "wheel" and np.isfinite(P.h_max):
        ax[2, 1].axhline(P.h_max, color="r", ls="--", lw=0.8); ax[2, 1].axhline(-P.h_max, color="r", ls="--", lw=0.8)
    if P.attitude_actuator == "rcs" and np.isfinite(P.rcs_budget):
        ax[2, 1].axhline(P.rcs_budget, color="r", ls="--", lw=0.8)
    Fh = np.array([np.hypot(*L.hold_force_vec(t[k], Y[:, k])) for k in range(len(t))])
    ax[3, 0].plot(t, Fh); ax[3, 0].set_ylabel("hold-down thrust (N)"); ax[3, 0].set_xlabel("t (s)")
    ax[3, 0].set_xlim(*xl); ax[3, 0].grid(True)
    alt = np.array([L.contact(Y[:, k])[3].min() for k in range(len(t))])
    ax[3, 1].plot(t, alt); ax[3, 1].set_ylabel("lowest point above terrain (m)"); ax[3, 1].set_xlabel("t (s)")
    ax[3, 1].set_xlim(*xl); ax[3, 1].set_ylim(-0.05, min(max(alt[t <= xl[1]].max(), 0.1), 20)); ax[3, 1].grid(True)
    fig.tight_layout()
    return fig


# ============================================================================ Monte Carlo
PARAM_FIELDS = set(Params.__dataclass_fields__) # pylint: disable=E1101



def build_case(sample, base_P, base_ic):
    """sample: dict of name -> value. Names in Params go to the parameters; the rest go to the initial condition.
    The touchdown shortcuts vn, vt, th, w set the initial condition 5 cm above the terrain with thrust off."""
    pv = {k: v for k, v in sample.items() if k in PARAM_FIELDS}
    P = replace(base_P, **pv)
    ic = dict(base_ic)
    ic.update({k: v for k, v in sample.items() if k not in PARAM_FIELDS})
    if "vn" in ic or "vt" in ic:
        phi = np.deg2rad(P.slope_deg)
        vn, vt = ic.pop("vn", 0.0), ic.pop("vt", 0.0)
        ic["vx0"] = vt * np.cos(phi) + vn * np.sin(phi)
        ic["vz0"] = vt * np.sin(phi) - vn * np.cos(phi)
        ic["th0"] = ic.pop("th", 0.0) + phi
        ic["w0"] = ic.pop("w", 0.0)
        ic["h0"] = 0.05
    return P, ic


def _run_case(args):
    sample, base_P, base_ic, guided = args
    P, ic = build_case(sample, base_P, base_ic)
    r = Lander(P).simulate(ic, guided)
    out = dict(sample)
    out.update(outcome=r["outcome"], peak_Fn_foot=r["peak_Fn_foot"], peak_Fn_hull=r["peak_Fn_hull"],
               peak_wheel_h=r["peak_wheel_h"], rcs_impulse=r["rcs_impulse"], hold_impulse=r["hold_impulse"],
               th_final_rel=r["th_final_rel"], x_final=r["x_final"])
    if r["td"]:
        out.update({f"td_{k}": v for k, v in r["td"].items()})
    return out


def sample_space(space, log2_n, seed=0):
    """Sobol design over `space`. A (lo, hi) tuple is a continuous uniform range. A list is a discrete choice made
    with equal probability, and may hold booleans or strings (e.g. post_td_control=[False, True],
    contact_model=["hunt_crossley", "kelvin_voigt"]). Returns a list of {name: value} dicts.
    NOTE: small designs are correlated across variables (10 variables: pairwise corr up to ~0.4 at 32 points,
    ~0.05 at 256). Use >= 256 points before slicing results by one variable, or use the `paired` option of
    monte_carlo for switches such as post_td_control."""
    from scipy.stats import qmc
    names = list(space)
    U = qmc.Sobol(d=len(names), seed=seed).random_base2(log2_n)
    rows = []
    for u in U:
        row = {}
        for k, uk in zip(names, u):
            spec = space[k]
            if isinstance(spec, list):
                row[k] = spec[min(int(uk * len(spec)), len(spec) - 1)]
            else:
                lo, hi = spec
                row[k] = float(lo + uk * (hi - lo))
        rows.append(row)
    return rows


def monte_carlo(space, log2_n=6, guided=False, base_P=None, base_ic=None, seed=0, workers=1,
                paired=None, scenarios=None):
    """Sobol sampling of `space` = {name: (lo, hi) | [choices]}. Returns a list of result dicts (one per run).
    Any Params field can be a variable, including every actuator setting. Other names go to the initial
    condition. guided=False -> touchdown envelope (thrust off, start 5 cm up); guided=True -> full descent.

    Comparing options on identical sampled cases (each result carries `sample_id` and a `scenario` label;
    use scenario_summary / paired_comparison afterwards):
      paired    = {name: [levels]}  -> every sample is run for the full cross product of the levels
                  (good for independent on/off switches).
      scenarios = {label: {Params overrides}}  -> every sample is run once per named scenario
                  (good for alternatives, e.g. 'RCS hold' vs 'wheel hold' vs 'hold-down').
    Sample the switch inside `space` only for large designs (see sample_space note)."""
    if paired and scenarios:
        raise ValueError("use either paired or scenarios, not both")
    base_P, base_ic = base_P or Params(), base_ic or {}
    if scenarios:
        variants = [(lab, dict(ov)) for lab, ov in scenarios.items()]
    else:
        factors = list((paired or {}).items())
        variants = [(",".join(f"{k}={v}" for k, v in combo), dict(combo))
                    for combo in itertools.product(*[[(k, v) for v in vals] for k, vals in factors])]
    jobs = []
    for i, row in enumerate(sample_space(space, log2_n, seed)):
        for lab, ov in variants:
            case = dict(row, **ov, sample_id=i)
            if paired or scenarios:
                case["scenario"] = lab
            jobs.append((case, base_P, base_ic, guided))
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers) as ex:
            return list(ex.map(_run_case, jobs))
    return [_run_case(j) for j in jobs]


def paired_comparison(results, key, off, on, show=True):
    """For results made with monte_carlo(paired={key: [...]}), count outcome transitions case by case
    from level `off` to level `on`. Returns a Counter of (outcome_off, outcome_on)."""
    by = {}
    for r in results:
        by.setdefault(r["sample_id"], {})[r[key]] = r["outcome"]
    trans = Counter((d[off], d[on]) for d in by.values() if off in d and on in d)
    if show:
        n = sum(trans.values())
        print(f"{key}: {off} -> {on}  ({n} paired cases)")
        for (a, b), c in trans.most_common():
            print(f"  {a:>12s} -> {b:<12s} {c:4d}")
        succ_off = sum(c for (a, _), c in trans.items() if a == "success")
        succ_on = sum(c for (_, b), c in trans.items() if b == "success")
        print(f"  successes: {succ_off} -> {succ_on}")
    return trans


def scenario_summary(results, show=True):
    """One row per scenario of a paired run: outcome counts plus the mean propellant-like cost of each option
    (hold-down impulse in N s, RCS angular impulse in N m s) and mean peak wheel momentum."""
    outcomes = sorted({r["outcome"] for r in results})
    rows = {}
    for lab in dict.fromkeys(r.get("scenario", "") for r in results):
        sub = [r for r in results if r.get("scenario", "") == lab]
        rows[lab] = dict(n=len(sub), counts={o: sum(r["outcome"] == o for r in sub) for o in outcomes},
                         hold_Ns=float(np.mean([r["hold_impulse"] for r in sub])),
                         rcs_Nms=float(np.mean([r["rcs_impulse"] for r in sub])),
                         wheel_Nms=float(np.mean([r["peak_wheel_h"] for r in sub])))
    if show:
        w = max(len(k) for k in rows)
        print(f"{'scenario':<{w}} | n  | " + " ".join(f"{o:>11s}" for o in outcomes) + " | hold N s | RCS N m s | wheel N m s")
        for lab, v in rows.items():
            print(f"{lab:<{w}} | {v['n']:2d} | " + " ".join(f"{v['counts'][o]:11d}" for o in outcomes) +
                  f" | {v['hold_Ns']:8.1f} | {v['rcs_Nms']:9.2f} | {v['wheel_Nms']:11.2f}")
    return rows


def outcome_table(results, key, bins=None, show=True):
    """Outcome counts grouped by a discrete variable `key`, or by `bins` equal-width bins of a continuous one.
    Returns {group_label: {outcome: count}} and prints it."""
    vals = [r[key] for r in results]
    if bins:
        edges = np.linspace(min(vals), max(vals), bins + 1)
        order = [f"{edges[i]:.3g}..{edges[i + 1]:.3g}" for i in range(bins)]
        grp = [order[min(int(np.searchsorted(edges, v, side="right")) - 1, bins - 1)] for v in vals]
    else:
        order = sorted({str(v) for v in vals})
        grp = [str(v) for v in vals]
    outcomes = sorted({r["outcome"] for r in results})
    table = {g: {o: sum(1 for r, gg in zip(results, grp) if gg == g and r["outcome"] == o) for o in outcomes}
             for g in order if g in grp}
    if show:
        print(f"{key:>26s} | n  | " + " ".join(f"{o:>12s}" for o in outcomes))
        for g, row in table.items():
            print(f"{g:>26s} | {sum(row.values()):2d} | " + " ".join(f"{row[o]:12d}" for o in outcomes))
    return table


def plot_envelope(results, xkey, ykey):
    import matplotlib.pyplot as plt
    colors = dict(success="tab:green", body_strike="tab:red", leg_overload="tab:orange",
                  escaped="tab:purple", not_settled="tab:gray", no_touchdown="k")
    fig, ax = plt.subplots()
    for o in sorted({r["outcome"] for r in results}):
        pts = [(r[xkey], r[ykey]) for r in results if r["outcome"] == o]
        ax.scatter(*zip(*pts), c=colors.get(o, "b"), label=f"{o} ({len(pts)})", s=25)
    ax.set_xlabel(xkey); ax.set_ylabel(ykey); ax.grid(True); ax.legend()
    return fig


# ============================================================================ demo
if __name__ == "__main__":
    import os, time
    import matplotlib.pyplot as plt

    P = Params(use_modes=True)                  # modes off for speed; set use_modes=True to include flex/slosh
    print(f"g_eff = {P.g:.3e} m/s^2,  Iyy = {P.Iyy:.1f} kg m^2,  "
          f"static tip limit = {np.rad2deg(np.arctan(P.leg_x / (P.height / 2 + P.leg_len))):.1f} deg")

    # named scenarios = alternatives to compare on identical sampled cases. Edit freely: any Params field works.
    scenarios = {
        "passive": {},
        "wheel hold": dict(post_td_control=True),
        "RCS hold": dict(post_td_control=True, attitude_actuator="rcs"),
        "hold-down (rebound)": dict(hold_down=True, hold_mode="rebound"),
        "hold-down (rebound) + wheel": dict(hold_down=True, hold_mode="rebound", post_td_control=True),
        "hold-down (rebound) + RCS": dict(hold_down=True, hold_mode="rebound", post_td_control=True,
                                          attitude_actuator="rcs"),
    }

    # 1) nominal guided descent (Hayabusa2-like final descent [H19]) under each scenario
    ic = dict(h0=8.5, vx0=0.01, vz0=-0.074, th0=0.0)
    for name, ov in scenarios.items():
        Pk = replace(P, **ov)
        t0 = time.time()
        res = Lander(Pk).simulate(ic, guided=True)
        print(f"nominal descent, {name:28s}: {res['outcome']:12s} hold {res['hold_impulse']:6.1f} N s, "
              f"RCS {res['rcs_impulse']:5.2f} N m s, wheel peak {res['peak_wheel_h']:4.2f} N m s ({time.time() - t0:.1f} s)")
    plot_run(Lander(replace(P, **scenarios["hold-down (rebound) + RCS"])).simulate(ic, guided=True),
             replace(P, **scenarios["hold-down (rebound) + RCS"]))

    # 2) Monte Carlo: touchdown envelope, every sampled case run under every scenario.
    #    Variables not used by a scenario are simply ignored by it. Use log2_n >= 8 for real studies.
    space = dict(vn=(0.02, 0.15), vt=(0.0, 0.10), th=(-0.3, 0.3), w=(-0.05, 0.05), slope_deg=(0.0, 30.0),
                 restitution=(0.05, 0.8),
                 post_td_tau_max=(0.5, 5.0), h_max=(0.5, 10.0),                    # reaction wheel
                 rcs_tau=(0.5, 3.0), rcs_budget=(1.0, 20.0),                       # thruster couples
                 hold_force=(0.2, 3.0), hold_duration=(10.0, 200.0), hold_gain=(0.2, 2.0),   # hold-down
                 post_td_delay=(0.0, 60.0), hold_delay=(0.0, 30.0))
    t0 = time.time()
    mc = monte_carlo(space, log2_n=5, guided=False, base_P=P, workers=os.cpu_count(), scenarios=scenarios)
    print(f"\nMonte Carlo: {len(mc)} runs in {time.time() - t0:.0f} s\n")
    scenario_summary(mc)
    print()
    paired_comparison(mc, "scenario", "passive", "hold-down (rebound) + wheel")
    plot_envelope([r for r in mc if r["scenario"] == "hold-down (rebound) + wheel"], "vn", "vt")
    plt.show()


# I want to plot the probability of success for each variable change over said variables whole
# range