# RL vs PID for Adaptive Cruise Control 🚗💥🚗 

after getting rear-ended, i was curious abt how adaptive cruise control actually works so i worked on this! i built a 1D driving simulator and compared a bayesian-tuned PID controller with a PPO reinforcement learning policy. i also tested different reward functions, random seeds, and 3 driving styles (aggressive, safe, and smooth).

**final results:** PID achieved **0.927 m** mean tracking error and the best PPO policy achieved **1.024 m**.

i always learn best by writing things down, so i put together this report as a way of documenting everything i tried, built and learned.
**[read it here !! :)](RL_vs_PID_for_Adaptive_Cruise_Control.pdf)**

## Video Demos
- [PID Demo](final/videos/demo_PID.mp4)
- [RL Demo](final/videos/demo_RL.mp4)
