import argparse
import re
import matplotlib.pyplot as plt
import os


parser = argparse.ArgumentParser(
    description="Plot training vs. validation loss from a .out log"
)
parser.add_argument(
    "--filename",
    type=str,
    default="lth_train_392733.out",
    help="Name of the log file inside ./train_out/"
)
args = parser.parse_args()


filename = args.filename
LOG_FILE_PATH = os.path.join(os.getcwd(), "train_out", filename)

iter_pattern = re.compile(r"^iter\s+(\d+):\s+loss\s+([\d\.]+)")
step_pattern = re.compile(
    r"^step\s+(\d+):\s+train loss\s+([\d\.]+),\s+val loss\s+([\d\.]+)"
)

iter_indices = []
iter_losses = []

step_indices = []
val_losses = []

with open(LOG_FILE_PATH, "r") as f:
    for line in f:
        line = line.strip()
        m_iter = iter_pattern.match(line)
        if m_iter:
            iter_indices.append(int(m_iter.group(1)))
            iter_losses.append(float(m_iter.group(2)))
            continue

        m_step = step_pattern.match(line)
        if m_step:
            step_indices.append(int(m_step.group(1)))
            val_losses.append(float(m_step.group(3)))
            continue

plt.figure(figsize=(10, 6))

plt.plot(
    iter_indices,
    iter_losses,
    label="Iter Loss",
    color="C0",
    linewidth=1,
    alpha=0.7,
)

plt.plot(
    step_indices,
    val_losses,
    label="Validation Loss (every step)",
    color="C1",
    marker="o",
    linestyle="--",
    linewidth=1.5,
)

plt.title("Training Curves: Iteration Loss vs. Validation Loss")
plt.xlabel("Iteration / Step Index")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()


base = os.path.splitext(filename)[0]
plt.savefig(f"loss_{base}_train.png")
