# Learned Optimizer

A learned optimizer implementation for gradient descent optimization.

---

Improvements - 

Yes, your intuition is completely correct. The "perfect AGC" problem is actually deeply embedded in both of your core channel generator files. 

Because your training loops (`l2o_basic.py`, `l2o_mlp.py`, etc.) pull data directly from these generators, **your entire dataset is currently simulating an idealized, perfectly equalized environment regarding signal amplitude.**

Here are the specific, concerning normalizations you need to address:

### 1. `wireline_channel.py` (The Standard Generator)
There are **two** forced normalizations happening sequentially in this file, which completely flattens any insertion loss variations:

**A. Energy Normalization (Line 29)**
In `generate_batch()`:
```python
h = h / torch.norm(h)  # Normalize energy
```
* **The Issue:** Just like in the synthetic generator, this forces the mathematical "vector length" of every channel to 1.0, destroying the physics of signal attenuation over distance.

**B. The Explicit "Ideal AGC" (Line 50)**
In `generate_received_signal()`:
```python
# Prompt 2: Main Cursor Normalization / Ideal AGC
# Normalize so the main cursor (peak amplitude) is exactly 1.0
peak_vals, _ = torch.max(torch.abs(h_batch), dim=1, keepdim=True)
h_batch = h_batch / (peak_vals + 1e-8)
```
* **The Issue:** The comment literally says "Ideal AGC"! This ensures that the highest voltage the receiver ever sees is always exactly 1.0. If you are training your MLP or RNN to drive a real piece of hardware, it will fail because a real receiver might see a main cursor peak of 0.05V, not 1.0V.

### 2. `advanced_channel_gen.py` (The Advanced Generator)
This file is meant to be your physically realistic, "hard" dataset, but it is heavily normalized, defeating the purpose of the advanced physics.

**A. Base Impulse Response Normalization (Lines 101-102)**
In `_compute_base_impulse_response()`:
```python
# h = h / max_vals
h = h / torch.norm(h, dim=1, keepdim=True)
```
* **The Issue:** Same as before. A severe channel (high $\tau$, high $\gamma$) should result in a tiny signal. This scales it back up to a massive signal.

**B. Time-Varying Drift Normalization (Lines 159-160)**
In `_generate_time_varying_channel()`:
```python
# Re-normalize at each time step to maintain peak amplitude of 1.0
max_vals = h_time_varying.max(dim=2, keepdim=True)[0]
# h_time_varying = h_time_varying / max_vals
h_time_varying = h_time_varying / torch.norm(h_time_varying, dim=2, keepdim=True)
```
* **The Issue:** **This is the most concerning one.** You are simulating a channel that drifts sinusoidally over time (e.g., due to heating/cooling). The physics of this drift dictates that the signal *amplitude* should swell and shrink. By normalizing it at *every single time step*, you are actively fighting the physics you just simulated. You are forcing the drifting channel to maintain a constant volume, destroying the amplitude modulation (AM) effect of the drift before your Neural Network ever gets to see it.

### How to Fix This

To expose your Learned Optimizer to the brutal reality of actual hardware, you should:

1.  **Comment out/Remove** all four of the `torch.norm()` and `max_vals` divisions shown above.
2.  Let the impulse responses be tiny ($O(10^{-2})$ or $O(10^{-3})$).
3.  **The Result:** When you run your training scripts, your `norm_sq` and `e_t` inputs to the neural network will drop from comfortable $\sim 1.0$ values down to microscopic numbers. 
4. **The Next Step:** Your neural network will likely crash or fail to converge initially. This is expected! It means you now need to either:
    * Scale the inputs to the neural network mathematically (e.g., using Layer Normalization on the input features).
    * Implement a simple, differentiable **VGA (Variable Gain Amplifier)** block *before* your CTLE that the network must also learn to control to boost the signal back to $\pm 1.0$.