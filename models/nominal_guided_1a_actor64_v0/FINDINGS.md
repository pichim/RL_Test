# Nominal SAC Result

Current limit: ±1 A. Normalized current and action-change weights: 4, 12.

Successes: 100/100. Unsafe episodes: 0. Captures: 100.

Training wall time: 2112.4 s. Final evaluation wall time: 88.4 s.

Fixed nominal plant and direct current. LQR supplied training targets. The standalone actor is saved to training/actor.pt2. Final evaluation uses the learned neural actor. The policy is selected using training validation, not evaluation seeds.

Success requires the final second within 5 degrees of upright, 0.25 rad absolute arm position, 0.5 rad/s arm speed, and 1 rad/s pendulum speed. Initial states still vary across the existing broad reset distribution.

See evaluation/summary.txt and evaluation/sac/training_nominal for metrics, PNGs, and traces. One seed and a five-second test do not establish long-duration or hardware reliability.
