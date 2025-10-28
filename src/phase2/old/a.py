vals = {
    "learning_rate": "1, 2",
    "batch_size": "256, 512",
    "optimizer": "adam, adamw",
    "weight_decay": "0, 1e-5, 1e-3",
    "epochs": "1"
}
counts = [len([v.strip() for v in s.split(',') if v.strip()]) for s in vals.values()]
total = 1
for c in counts:
    total *= c
print("Per-model runs:", total)