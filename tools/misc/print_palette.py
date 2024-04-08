import random

num_colors = 178

PALETTE = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(num_colors)]

print("PALETTE =", PALETTE)