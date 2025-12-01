from src.data.dataloader import create_dataloader
from pathlib import Path

root = Path("/Users/pc/Desktop/CODING/Others/fairvoice")
dataset, _ = create_dataloader(root, batch_size=4)

print("\n=== FIRST 20 LABELS ===")
for i in range(20):
    item = dataset.items[i]
    print(i, item["file"], item["label"])
