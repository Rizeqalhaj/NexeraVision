# Debug Slow Videos - Find Exactly Which Videos Are Causing Slowdown

## What Changed

I've added detailed debugging to `train_rtx5000_dual_IMPROVED.py` that will:

1. **Show every video being processed** with index and filename
2. **Time each video** and show processing time
3. **Flag slow videos** (>5 seconds) in real-time
4. **Log slow videos** to file: `/workspace/checkpoints_improved/train_slow_videos.txt`
5. **Show summary** at end with top 10 slowest videos

## Output You'll See

```
[15200/21839] Processing: violent_009265.mp4
  ⚠️  SLOW VIDEO: violent_009265.mp4 took 45.23s

[15201/21839] Processing: violent_009266.mp4
  ✓ OK: 0.85s

[15202/21839] Processing: violent_009267.mp4
  ⚠️  SLOW VIDEO: violent_009267.mp4 took 38.12s
```

## How to Use

1. **Start training normally:**
   ```bash
   cd /workspace/violence_detection_mvp
   python3 train_rtx5000_dual_IMPROVED.py --dataset-path /workspace/organized_dataset
   ```

2. **Watch the output** - you'll see EXACTLY which videos are slow

3. **When it slows down around video 15,204:**
   - You'll see which specific video is taking >5 seconds
   - It will be logged to `/workspace/checkpoints_improved/train_slow_videos.txt`

4. **After extraction completes (or you stop it):**
   - Check the slow videos log: `cat /workspace/checkpoints_improved/train_slow_videos.txt`
   - You'll get a summary of the top 10 slowest videos

## Next Steps

Once we identify the slow videos:

1. **Option A: Remove them manually**
   ```bash
   # Example if violent_009265.mp4 is slow
   rm /workspace/organized_dataset/train/violent/violent_009265.mp4
   ```

2. **Option B: I create a cleanup script**
   - Based on the slow_videos.txt log
   - Automatically removes all videos that took >5s

3. **Option C: Skip them during training**
   - I can modify the script to skip slow videos automatically
   - Continue processing without them

## What We'll Learn

This will tell us:
- Is it the SAME videos causing slowdown every time?
- How many slow videos are there? (10? 100? 1000?)
- Are they corrupted files or just complex videos?
- Is the slowdown gradual or sudden at specific videos?

**Now run training and let's see exactly what videos are the problem!**
