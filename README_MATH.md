# Mathematical specification of the Furuta controller

This guide specifies the simulator, learning problem, and evaluation for the
current 200 Hz SAC workflow. Angles use radians, angular velocities use rad/s,
and current uses amperes. The workflow trains one symmetric neural controller
on the nominal simulated plant. The saved controller generates current directly
from observations at either evaluation rate.

## State and continuous-time plant

Let $q=(\theta_1,\theta_2)^\top$ denote the arm and pendulum angles and let
$\omega=(\omega_1,\omega_2)^\top=\dot q$. The physical state is
$x=(\theta_1,\theta_2,\omega_1,\omega_2)^\top$. The pendulum hangs at
$\theta_2=0$ and is upright at $\theta_2=\pi \pmod{2\pi}$. The arm target
is $\theta_1=0 \pmod{2\pi}$. Define the signed periodic errors by

$$
\begin{aligned}
\mathrm{wrap}(z)&=(z+\pi)\bmod(2\pi)-\pi,\\
e_1&=\mathrm{wrap}(\theta_1),\qquad
e_2=\mathrm{wrap}(\theta_2-\pi).
\end{aligned}
$$

The simulation retains the unwrapped arm angle and wraps the pendulum after
each plant step. The plant takes current $I$ as an ideal input. Current
produces torque $k_m I$ on the arm, and the pendulum joint is passive.
The model assumes rigid bodies, viscous damping, and periodic joint travel.

| Symbol    |                                                              Value | Meaning                              |
| --------- | -----------------------------------------------------------------: | ------------------------------------ |
| $g$       |                                  $9.80665\thinspace\mathrm{m/s^2}$ | Gravitational acceleration           |
| $k_m$     |                         $0.0975\thinspace\mathrm{N\thinspace m/A}$ | Motor torque constant                |
| $m_1$     |                                      $0.1222\thinspace\mathrm{kg}$ | Arm mass                             |
| $J_{1zz}$ |         $1.127084\times10^{-5}\thinspace\mathrm{kg\thinspace m^2}$ | Arm inertia about its center of mass |
| $l_1$     |                                     $0.027248\thinspace\mathrm{m}$ | Arm axis to arm center of mass       |
| $L_1$     |                                       $0.0855\thinspace\mathrm{m}$ | Arm axis to pendulum pivot           |
| $b_1$     |          $10^{-6}\thinspace\mathrm{N\thinspace m\thinspace s/rad}$ | Arm damping                          |
| $m_2$     |                                       $0.028\thinspace\mathrm{kg}$ | Pendulum mass                        |
| $r_2$     |                                       $0.0045\thinspace\mathrm{m}$ | Pendulum radius                      |
| $L_2$     |                                        $0.161\thinspace\mathrm{m}$ | Pendulum length                      |
| $l_2$     |                                                            $L_2/2$ | Pendulum pivot to center of mass     |
| $b_2$     | $4.4\times10^{-5}\thinspace\mathrm{N\thinspace m\thinspace s/rad}$ | Pendulum damping                     |

The pendulum inertia and the difference that appears in the coupling terms are

$$
J_{2xx}=\tfrac12 m_2r_2^2,\qquad
J_{2yy}=J_{2zz}=\tfrac14 m_2r_2^2+\tfrac1{12}m_2L_2^2,\qquad
D=J_{2yy}+m_2l_2^2-J_{2xx}.
$$

Let $s=\sin\theta_2$, $c=\cos\theta_2$ and $s_d=\sin(2\theta_2)$.
The equation of motion and mass matrix are

$$
\dot q=\omega,\qquad
M(q)\dot\omega=
\begin{bmatrix}
k_m I\\
0
\end{bmatrix}
-C(q,\omega)\omega-G(q)-B\omega,
$$

$$
\begin{aligned}
M(q)&=\begin{bmatrix}
M_{11}&M_{12}\\
M_{12}&M_{22}
\end{bmatrix},\\
M_{11}&=J_{1zz}+m_1l_1^2+m_2L_1^2+(J_{2yy}+m_2l_2^2)s^2+J_{2xx}c^2,\\
M_{12}&=m_2L_1l_2c,\qquad M_{22}=J_{2zz}+m_2l_2^2.
\end{aligned}
$$

$$
C(q,\omega)=
\begin{bmatrix}
D\omega_2s_d & -m_2L_1l_2s\omega_2\\
-\tfrac12D\omega_1s_d & 0
\end{bmatrix},
\qquad
G(q)=\begin{bmatrix}
0\\
m_2gl_2s
\end{bmatrix},
\qquad B=\mathrm{diag}(b_1,b_2).
$$

The total mechanical energy gives a sign and units check. With
$E_{\mathrm{mech}}=\tfrac12\omega^\top M\omega+m_2gl_2(1-\cos\theta_2)$,

$$
\dot E_{\mathrm{mech}}=k_mI\omega_1-b_1\omega_1^2-b_2\omega_2^2.
$$

The simulator solves the two-dimensional linear system for $\dot\omega$ at
each stage of classical RK4. With $f(x,I)=(\omega^\top,\dot\omega^\top)^\top$
and plant step $h=0.001\thinspace\mathrm{s}$,

$$
\begin{aligned}
k_1&=f(x,I),& k_2&=f(x+\tfrac h2 k_1,I),\\
k_3&=f(x+\tfrac h2 k_2,I),& k_4&=f(x+h k_3,I),\\
x^+&=x+\tfrac h6(k_1+2k_2+2k_3+k_4).
\end{aligned}
$$

The plant uses float64. A control decision supplies a normalized command
$a_k\in[-1,1]$ and holds $I_k=I_{\max}a_k$, with $I_{\max}=1\thinspace\mathrm A$.
Training and checkpoint validation hold it for five plant steps
($T=5\thinspace\mathrm{ms}$). Final evaluation repeats the same policy at five and
two steps ($T=5\thinspace\mathrm{ms}$ and $T=2\thinspace\mathrm{ms}$). The command is held
across all four RK4 stages of each plant step. [Plant implementation](src/furuta_model.py)

## Observations, starts, and episode outcomes

The actor and critics see the seven float32 features in this order.

$$
o_k =
\begin{bmatrix}
\sin\theta_1 & \cos\theta_1 & \mathrm{clip}(\omega_1/30) &
\sin e_2 & \cos e_2 & \mathrm{clip}(\omega_2/30) &
\mathrm{clip}(a_{k-1})
\end{bmatrix}^\top.
$$

Here $\mathrm{clip}$ limits its input to $[-1,1]$. The previous
normalized command makes the current-change cost observable. At ordinary
reset, $a_{-1}=0$. Draw $\theta_1,\theta_2$ independently from
$\mathcal U[-\pi,\pi)$ and $\omega_1,\omega_2$ independently from
$\mathcal U[-6,6)$ in state order. The training environment uses the seeded
Gymnasium random generator. A five-second episode contains at most 5,000
plant steps or 1,000 training decisions.

| Region or guard      | Condition                                                                                                                |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Safe simulated state | All state components finite and $\lvert\omega_1\rvert,\lvert\omega_2\rvert\leq30$                                        |
| Mechanical capture   | $\lvert e_2\rvert\leq\pi/18$, $\lvert\omega_1\rvert,\lvert\omega_2\rvert\leq2$                                           |
| LQR teaching box     | $\lvert(e_1,e_2,\omega_1,\omega_2)\rvert\leq(0.20,0.10,0.50,1.00)$ componentwise                                         |
| Balance box          | $\lvert e_1\rvert\leq0.25$, $\lvert e_2\rvert\leq\pi/36$, $\lvert\omega_1\rvert\leq0.50$, $\lvert\omega_2\rvert\leq1.00$ |

Mechanical capture records when the pendulum is close to upright and moving
slowly. The arm angle is unrestricted for this diagnostic. Safety and
balance are checked after every plant step. The balance counter resets when
a step leaves the balance box and succeeds after 1,000 consecutive valid
steps. Training and checkpoint validation terminate on success or a safety
violation and truncate at the horizon. If integration fails, terminate from
the last finite state and report a violation. The final evaluation uses the
success rules specified below. [Task implementation](src/furuta_env.py)

## Reward and energy potential

Let $x_{k+1}$ be the state after the held command and $F_k$ the indicator
of a valid, constraint-satisfying transition. Evaluate the state costs on
$x_{k+1}$ and charge the command costs once per control decision. With
$\Delta a_k=a_k-a_{k-1}$, the base reward is

$$
\begin{aligned}
c_k &= 1.5e_1^2+e_2^2+0.015\omega_1^2+0.010\omega_2^2+
       2.0a_k^2+15.0(\Delta a_k)^2,\\
r_k^0&=F_k-0.1c_k+100\thinspace\mathbf{1}[\text{successful termination}].
\end{aligned}
$$

$F_k=0$ after a constraint violation or integration failure. The effective
action and action-change weights are $0.20$ and $1.50$. With $I_{\max}=1$ A
they also multiply the numerical squared current and squared current change
in A. The discrete change term uses $\Delta a_k$ directly. Evaluation
reports the rate-normalized current slew separately.

Define the nominal pendulum-only mechanical energy with zero potential at
the hanging equilibrium and a target at stationary upright.

$$
E_p(x)=\tfrac12 M_{22}\omega_2^2+m_2gl_2(1-\cos\theta_2),
\qquad E_\star=2m_2gl_2,\qquad d(x)=\frac{E_p(x)-E_\star}{E_\star}.
$$

The potential is bounded and uses the same discount as SAC.

$$
\Phi(x)=-0.25\thinspace\frac{d(x)^2}{1+d(x)^2},
\qquad
r_k=r_k^0+\gamma\Phi^+(x_{k+1})-\Phi(x_k),
\qquad \gamma=0.99.
$$

Use $\Phi^+=0$ for a true termination and $\Phi^+=\Phi$ for continuing
steps and time-limit truncations. SAC bootstraps through time limits.
This energy term shapes the reward. The controller still chooses the
current, and the energy calculation uses pendulum motion alone.
Checkpoint validation uses $r_k^0$. [Energy implementation](src/guided_sac.py)

## Exact reflection symmetry

The observation reflection and action reflection are

$$
R=\mathrm{diag}(-1,1,-1,-1,1,-1,-1),
\qquad (o,a)\mapsto(Ro,-a).
$$

The actor uses a fully connected ReLU trunk with two hidden layers.
The current configuration has widths $(16,16)$. Let $m(o)$ be its affine
mean head and $\ell(o)$ its affine log-standard-deviation head, clipped
to $[-20,2]$ as in SAC. Apply the same trunk and heads to the original
and reflected observations, then compute

$$
\mu(o)=\tfrac12[m(o)-m(Ro)],
\qquad
\log\sigma(o)=\tfrac12[\ell(o)+\ell(Ro)].
$$

Inside the LQR teaching box use $\sigma=0.02$ in the pre-squash action
space. The stochastic action is $\tanh(\mu+\sigma\varepsilon)$ for
$\varepsilon\sim\mathcal N(0,1)$. Deterministic evaluation and export use
$a(o)=\tanh\mu(o)$. As $R^2=\mathrm{Id}$ and $\tanh$ is odd,
$a(Ro)=-a(o)$. The two passes share their parameters.

The two critics are separate ReLU networks with input $(o,a)$, widths
$(64,64)$, and one scalar output each. For each critic, including its
target copy, use the same raw network $q_i$ twice.

$$
Q_i(o,a)=\tfrac12[q_i(o,a)+q_i(Ro,-a)].
$$

This gives $Q_i(Ro,-a)=Q_i(o,a)$. The deterministic export contains the
actor trunk and mean head. The $7\to8\to8\to1$ version has 128 weights
and 17 stored biases, for 145 parameters. The $7\to16\to16\to1$ version
has 417. The reflection is a fixed buffer. [Policy implementation](src/guided_sac.py)

## Local LQR teaching

The teacher uses the wrapped error state $z=(e_1,e_2,\omega_1,\omega_2)^\top$
and the saved gain $K$. Its target is independent of the previous command.

$$
K=\begin{bmatrix}-0.4472&3.4376&-0.1555&0.3031\end{bmatrix},
\qquad
a_{\mathrm{LQR}}(z)=
\frac{\mathrm{clip}(-Kz,-I_{\max},I_{\max})}{I_{\max}}.
$$

Generate 192 starting error states uniformly inside the teaching box with
the training seed, offset the pendulum coordinate by $\pi$, and simulate
each for at most 120 ms (24 decisions). Before each step, stop the example
trajectory if it has left the teaching box. Save the observation and
normalized LQR target. Assign each saved observation an independent
$\mathcal U[-1,1)$ previous command. This randomized feature is for the
training example. The plant advances under the teacher's held current.

The supervised loss uses 128 examples sampled with replacement and ten
fixed probes to constrain local directional gains. For probe axis $i$ in
$(e_1,e_2,\omega_1,\omega_2,a_{k-1})$, use displacement
$\delta b_i$ with $b=(0.10,0.05,0.25,0.50,0.25)$ and
$\delta\in\lbrace 0.10,0.002\rbrace$. Let $p_{\delta,i}$ denote its encoded
observation and $t_{\delta,i}$ its normalized LQR command. Then

$$
\begin{aligned}
L_{\mathrm{examples}} &=
\frac1{128}\sum_{j=1}^{128}
  (a(o_j)-a_{\mathrm{LQR}}(z_j))^2,\\
L_{\mathrm{probes}} &=
\frac12\sum_{\delta\in\lbrace 0.10,0.002\rbrace}\frac1{5\delta^2}
  \sum_{i=1}^{5}(a(p_{\delta,i})-t_{\delta,i})^2,\\
L_{\mathrm{teach}} &= L_{\mathrm{examples}}+L_{\mathrm{probes}}.
\end{aligned}
$$

Run 500 actor Adam updates on $L_{\mathrm{teach}}$ before collecting
environment transitions. Reuse that optimizer and add
$5\nabla L_{\mathrm{teach}}$ to each SAC actor update. For replay
observations in the LQR box, detach the action and log probability on
the SAC actor gradient path. Those samples still contribute forward
values, critic learning, and temperature learning. The box test uses
cosine thresholds for the angles and absolute normalized speed
thresholds. The learned policy supplies the rollout commands after
random replay warm-up, including when the system is near upright.

## SAC updates and checkpoint choice

The implementation uses Stable-Baselines3 SAC with two critics and
their target copies. For a replay sample $(o,a,r,o',d)$, let $d=1$ on
true termination and $d=0$ otherwise, including time limits. With
$\tilde a'\sim\pi(\cdot\mid o')$, the one-step target is

$$
y=r+(1-d)\gamma
\left(\min_{i=1,2}\overline Q_i(o',\tilde a')
      -\alpha\log\pi(\tilde a'\mid o')\right).
$$

Use the following critic and actor losses, with $\tilde a\sim\pi(\cdot\mid o)$.
SAC evaluates the squashed action density using the tanh change-of-variables
correction. Use the local gradient rule above in the actor loss.

$$
L_Q=\tfrac12\sum_{i=1}^2\mathbb E[(Q_i(o,a)-y)^2],
\qquad
L_{\pi}=\mathbb E\left[
\alpha\log\pi(\tilde a\mid o)-\min_{i=1,2}Q_i(o,\tilde a)
\right].
$$

The temperature starts at $\alpha=0.1$. Write $\beta=\log\alpha$
and let $\mathrm{detach}$ preserve a forward value while stopping
its gradient. The temperature loss uses target entropy $\mathcal H_\star=-1$.

$$
L_\alpha=-\mathbb E[
\beta\thinspace\mathrm{detach}(\log\pi(\tilde a\mid o)+\mathcal H_\star)
].
$$

The target critic parameters are updated by $\overline w\gets(1-\tau)\overline w+\tau w$ after each SAC
update, with $\tau=0.005$. [SAC paper](https://arxiv.org/abs/1812.05905)
· [SB3 2.9.0 SAC implementation](https://stable-baselines3.readthedocs.io/en/v2.9.0/_modules/stable_baselines3/sac/sac.html)

| Setting                         | Current value                                                |
| ------------------------------- | ------------------------------------------------------------ |
| Training seed and budget        | 0 and 300,000 decisions                                      |
| Replay warm-up and capacity     | 50,000 uniform-action decisions and 1,000,000 transitions    |
| Optimizer                       | Adam, learning rate $2\times10^{-4}$, replay batch 256       |
| Update cadence                  | One SAC gradient step per control decision after warm-up     |
| Energy, discount, target update | $0.25$, $\gamma=0.99$, $\tau=0.005$                          |
| Validation                      | Every 25,000 decisions, 50 deterministic starts, seed 10,000 |

Before each checkpoint validation, reset the validation RNG to the same
seed. Run 50 consecutive episodes for each checkpoint with the deterministic
actor. Rank checkpoints lexicographically by balance rate, rate of balance
after visiting the teaching box, and mean undiscounted base reward. Region
visits are checked at reset and at decision endpoints in validation.
Keep the earlier checkpoint on an exact tie. Export the selected
checkpoint after training. The run's saved task and training JSON give the
exact settings for every run. Matching the mathematical model defines the
same experiment. Exact parameter reproduction also depends on the same
software, RNG sequence, and numerical execution. [Configuration](src/config.py)

## Benchmarks and local diagnostics

The **nominal** panel runs 100 starts at each rate. For seed $s$ from
80,000 to 80,099, create a NumPy seed sequence, spawn two children, and
use the first child to draw the four initial coordinates in order. Safety
failure stops the run. Success requires the last one second of the
five-second episode to remain in the balance box. Mechanical capture is
recorded at the initial state or first qualifying plant step.

The **near-upright** panel starts at the 81 points of the Cartesian
product $\lbrace -0.85,0,0.85\rbrace^4$ scaled by the teaching bounds and offset to
upright. Cycle the previous command through $-0.75,0,0.75$ in product
order. Score the deterministic actor and the LQR separately using the
training episode rule, so a one-second hold anywhere in the horizon
counts. For current agreement, use a separate grid
$\lbrace -1,-0.6,0,0.6,1\rbrace^4$ scaled by the same bounds and cross it with
previous commands $-0.75,-0.25,0.25,0.75$. Report mean, 95th percentile,
and maximum absolute current error across the 2,500 probes.

The **post-capture** panel uses the 30 CSV states, including the held
command just before the handoff and the time remaining for each start.
It applies the nominal final-hold test over that remaining duration.
Both rates use the same file. [Post-capture fixture](models/reference/capture_starts.csv)

For a successful nominal or post-capture rollout, completion time is the
start of the final uninterrupted balance hold plus one second. The
current-change statistic uses command differences at control decisions
and includes the first change from the saved previous command.
With $N$ changes during an episode, calculate

$$
\Delta I_{\mathrm{RMS}} =
\sqrt{\frac{1}{N}\sum_{k=1}^{N}(I_k-I_{k-1})^2},
\qquad
S_{\mathrm{RMS}}=\frac{\Delta I_{\mathrm{RMS}}}{T}.
$$

The summaries average the per-episode values. This reports discrete
command slew in A/s. It models the change in command and is independent
of any inner current-control dynamics on hardware. [Evaluator](src/evaluate.py)

The **local diagnostics** linearize the nonlinear held-current closed loop
at the upright equilibrium. In local coordinates
$\xi=(e_1,e_2,\omega_1,\omega_2,a_{k-1})$, define the one-decision map
$F_{\mathrm{cl}}$ by applying the deterministic actor and advancing the
plant for one control period, then storing the new command as the fifth
state. With the $i$-th coordinate basis vector $u_i$ and $\epsilon=10^{-4}$,

$$
[A_{\mathrm{cl}}]_{:i}\approx
\frac{F_{\mathrm{cl}}(\epsilon u_i)-F_{\mathrm{cl}}(-\epsilon u_i)}{2\epsilon}.
$$

The reported poles are the eigenvalues of $A_{\mathrm{cl}}$. Its last
row multiplied by $I_{\max}$ gives the actor's current Jacobian. The
teacher's unsaturated current Jacobian in the same coordinates is
$[-K,0]$. These derivatives characterize the equilibrium. The benchmark
counts measure finite-amplitude recovery on the named panels.
