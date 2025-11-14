#!/usr/bin/env python3
"""
Rebalance dataset after cleaning corrupted videos

Current situation:
- train/violent: 7,040 (was 8,216, lost 1,176)
- train/nonviolent: 8,190 (was 8,203, lost 13)
- Imbalance: 1,150 videos

Solution: Remove 1,150 random nonviolent videos to match violent count
"""

import random
import shutil
from pathlib import Path

TRAINING_DIR = Path("/workspace/Training")

print("=" * 80)
print("REBALANCE DATASET AFTER CLEANUP")
print("=" * 80)
print()

# Count current videos after cleanup
train_violent = list((TRAINING_DIR / "train" / "Violent").glob("*.*"))
train_nonviolent = list((TRAINING_DIR / "train" / "NonViolent").glob("*.*"))
val_violent = list((TRAINING_DIR / "val" / "Violent").glob("*.*"))
val_nonviolent = list((TRAINING_DIR / "val" / "NonViolent").glob("*.*"))
test_violent = list((TRAINING_DIR / "test" / "Violent").glob("*.*"))
test_nonviolent = list((TRAINING_DIR / "test" / "NonViolent").glob("*.*"))

print("Current counts (after corruption cleanup):")
print(f"  train/Violent:     {len(train_violent):,}")
print(f"  train/NonViolent:  {len(train_nonviolent):,}")
print(f"  val/Violent:       {len(val_violent):,}")
print(f"  val/NonViolent:    {len(val_nonviolent):,}")
print(f"  test/Violent:      {len(test_violent):,}")
print(f"  test/NonViolent:   {len(test_nonviolent):,}")
print()

# Calculate imbalances
train_diff = len(train_nonviolent) - len(train_violent)
val_diff = len(val_nonviolent) - len(val_violent)
test_diff = len(test_nonviolent) - len(test_violent)

print("Imbalances:")
print(f"  train: {train_diff:,} excess nonviolent")
print(f"  val:   {val_diff:,} excess nonviolent")
print(f"  test:  {test_diff:,} excess nonviolent")
print()

total_to_remove = train_diff + val_diff + test_diff

if total_to_remove <= 0:
    print("✅ Dataset is already balanced!")
    exit(0)

print("=" * 80)
print("REBALANCING PLAN")
print("=" * 80)
print()
print(f"Will remove {total_to_remove:,} nonviolent videos:")
print(f"  - train: {train_diff:,} videos")
print(f"  - val:   {val_diff:,} videos")
print(f"  - test:  {test_diff:,} videos")
print()

print("After rebalancing:")
print(f"  train: {len(train_violent):,} violent = {len(train_violent):,} nonviolent")
print(f"  val:   {len(val_violent):,} violent = {len(val_violent):,} nonviolent")
print(f"  test:  {len(test_violent):,} violent = {len(test_violent):,} nonviolent")
print()

response = input("Proceed with rebalancing? Type 'YES' to continue: ")
if response.strip().upper() != "YES":
    print("Cancelled.")
    exit(0)

print()
print("=" * 80)
print("REMOVING EXCESS NONVIOLENT VIDEOS")
print("=" * 80)
print()

# Create backup directory
backup_dir = Path("/workspace/excess_nonviolent_videos")
backup_dir.mkdir(exist_ok=True)

# Remove from train
if train_diff > 0:
    print(f"Removing {train_diff:,} from train/NonViolent...")
    random.seed(42)
    random.shuffle(train_nonviolent)
    to_remove = train_nonviolent[:train_diff]

    for video in to_remove:
        target = backup_dir / f"train_{video.name}"
        shutil.move(str(video), str(target))

    print(f"  ✓ Moved {len(to_remove):,} videos")

# Remove from val
if val_diff > 0:
    print(f"Removing {val_diff:,} from val/NonViolent...")
    random.seed(42)
    random.shuffle(val_nonviolent)
    to_remove = val_nonviolent[:val_diff]

    for video in to_remove:
        target = backup_dir / f"val_{video.name}"
        shutil.move(str(video), str(target))

    print(f"  ✓ Moved {len(to_remove):,} videos")

# Remove from test
if test_diff > 0:
    print(f"Removing {test_diff:,} from test/NonViolent...")
    random.seed(42)
    random.shuffle(test_nonviolent)
    to_remove = test_nonviolent[:test_diff]

    for video in to_remove:
        target = backup_dir / f"test_{video.name}"
        shutil.move(str(video), str(target))

    print(f"  ✓ Moved {len(to_remove):,} videos")

print()
print(f"✓ Moved {total_to_remove:,} videos to: {backup_dir}")

print()
print("=" * 80)
print("REBALANCING COMPLETE")
print("=" * 80)
print()

# Final counts
final_train_violent = len(list((TRAINING_DIR / "train" / "Violent").glob("*.*")))
final_train_nonviolent = len(list((TRAINING_DIR / "train" / "NonViolent").glob("*.*")))
final_val_violent = len(list((TRAINING_DIR / "val" / "Violent").glob("*.*")))
final_val_nonviolent = len(list((TRAINING_DIR / "val" / "NonViolent").glob("*.*")))
final_test_violent = len(list((TRAINING_DIR / "test" / "Violent").glob("*.*")))
final_test_nonviolent = len(list((TRAINING_DIR / "test" / "NonViolent").glob("*.*")))

print("Final balanced dataset:")
print(f"  train: {final_train_violent:,} violent = {final_train_nonviolent:,} nonviolent")
print(f"  val:   {final_val_violent:,} violent = {final_val_nonviolent:,} nonviolent")
print(f"  test:  {final_test_violent:,} violent = {final_test_nonviolent:,} nonviolent")
print()

total_violent = final_train_violent + final_val_violent + final_test_violent
total_nonviolent = final_train_nonviolent + final_val_nonviolent + final_test_nonviolent

print(f"Total: {total_violent:,} violent = {total_nonviolent:,} nonviolent")
print()

if total_violent == total_nonviolent:
    print("✅ PERFECT BALANCE ACHIEVED!")
else:
    diff = abs(total_violent - total_nonviolent)
    print(f"⚠️  Difference: {diff} videos ({diff/(total_violent+total_nonviolent)*100:.2f}%)")

print()
print("=" * 80)
print("READY FOR TRAINING")
print("=" * 80)
print()
print("Run training with:")
print("  python3 train_OPTIMIZED_192CPU.py")
print()
