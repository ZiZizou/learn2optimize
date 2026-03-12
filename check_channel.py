import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from wireline_channel import WirelineChannelGenerator

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
