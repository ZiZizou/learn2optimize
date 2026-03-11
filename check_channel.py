import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class WirelineChannelGenerator:
    def __init__(self, num_taps=50):
        self.num_taps = num_taps

    def generate_batch(self, batch_size):
        t = torch.linspace(0, 5, self.num_taps)
        channels = []
        for _ in range(batch_size):
            tau = 1.0 
            h = torch.exp(-t / tau) * torch.sin(t + 1e-3) 
            h = h / torch.norm(h) 
            channels.append(h)
        return torch.stack(channels)

gen = WirelineChannelGenerator()
h = gen.generate_batch(1)[0]
plt.plot(h.numpy(), 'o-')
plt.title("Channel Impulse Response")
plt.xlabel("Tap Index")
plt.ylabel("Amplitude")
plt.grid(True)
plt.savefig("channel_ir.png")
print(f"Main cursor index (max amplitude): {torch.argmax(torch.abs(h)).item()}")
print(f"Impulse response: {h.numpy()}")
