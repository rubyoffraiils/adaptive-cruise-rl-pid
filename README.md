# RL vs PID for Adaptive Cruise Control

After getting rear-ended, I was curious abt how adaptive cruise control actually works, so I built this project!🚗💥🚗

I created a 1D driving simulator and compared a Bayesian-tuned PID controller with a PPO reinforcement learning policy. I also tested different reward functions, random seeds, and three driving styles: aggressive, safe, and smooth.

**Final results:** PID achieved **0.927 m** mean tracking error, while the best PPO policy achieved **1.024 m**.

📄 **[Read my full report here!!!](RL_vs_PID_for_Adaptive_Cruise_Control.pdf)**

## Video Demos
- [PID Demo](final/videos/demo_PID.mp4)
- [RL Demo](final/videos/demo_RL.mp4)

## Run Locally

```bash
git clone <repository-url>
cd adaptive-cruise-rl-pid
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

```bash
# Run the PID controller
python src/run_pid.py

# Train a new PPO policy
python src/train_rl.py

# Seed sweep (5 seeds)
python src/train_seeds.py

# Driving style variants (aggressive/safe/smooth)
python src/train_styles.py

# Generate comparison plots
python src/plot_rl_comparison.py
```

## Stack

Python, Gymnasium, Stable-Baselines3, Optuna, NumPy, Matplotlib
