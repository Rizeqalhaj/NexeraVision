#!/bin/bash
# Paste this ENTIRE script into Jupyter terminal to start downloads

cd /workspace && \
pip3 install --quiet kaggle && \
mkdir -p ~/.kaggle && \
echo '{"username":"issadalu","key":"5aabafacbfdefea1bf4f2171d98cc52b"}' > ~/.kaggle/kaggle.json && \
chmod 600 ~/.kaggle/kaggle.json && \
python3 << 'EOF'
import subprocess, os, json, sys
from pathlib import Path
from datetime import datetime

datasets = [
    ('vulamnguyen/rwf2000', 'tier1/RWF2000', 'RWF-2000', 2000, 1.5),
    ('odins0n/ucf-crime-dataset', 'tier1/UCF_Crime', 'UCF-Crime', 1900, 12.0),
    ('toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd', 'tier1/SCVD', 'SmartCity-CCTV', 4000, 3.5),
    ('mohamedmustafa/real-life-violence-situations-dataset', 'tier1/RealLife', 'Real-Life Violence', 2000, 2.0),
    ('arnab91/eavdd-violence', 'tier1/EAVDD', 'EAVDD', 1500, 1.8),
]

print("="*80)
print("NexaraVision Dataset Download - Starting")
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
sys.stdout.flush()

results = []
for i, (kid, path, name, vids, size) in enumerate(datasets, 1):
    print(f"\n{'='*80}")
    print(f"[{i}/{len(datasets)}] {name} ({vids:,} videos, ~{size}GB)")
    print(f"{'='*80}")
    sys.stdout.flush()

    out = Path(f"/workspace/datasets/{path}")
    out.mkdir(parents=True, exist_ok=True)

    try:
        start = datetime.now()
        print(f"⏳ Downloading from Kaggle...")
        sys.stdout.flush()

        result = subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', kid, '-p', str(out), '--unzip'],
            capture_output=True,
            text=True,
            timeout=3600
        )

        if result.returncode == 0:
            vcnt = sum(len(list(out.rglob(f'*.{ext}'))) for ext in ['mp4','avi','mkv','webm','mov','MP4','AVI','MKV'])
            sz = sum(f.stat().st_size for f in out.rglob('*') if f.is_file()) / (1024**3)
            elapsed = (datetime.now() - start).total_seconds()

            print(f"✅ SUCCESS!")
            print(f"   Videos: {vcnt:,}")
            print(f"   Size: {sz:.2f} GB")
            print(f"   Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
            sys.stdout.flush()

            results.append({
                'name': name,
                'status': 'success',
                'videos': vcnt,
                'size_gb': round(sz, 2),
                'time_s': round(elapsed, 1)
            })
        else:
            print(f"❌ FAILED: {result.stderr[:300]}")
            sys.stdout.flush()
            results.append({'name': name, 'status': 'failed', 'error': result.stderr[:200]})

    except Exception as e:
        print(f"❌ ERROR: {str(e)[:300]}")
        sys.stdout.flush()
        results.append({'name': name, 'status': 'error', 'error': str(e)[:200]})

with open('/workspace/datasets/download_results.json', 'w') as f:
    json.dump({'timestamp': datetime.now().isoformat(), 'results': results}, f, indent=2)

successful = [r for r in results if r['status'] == 'success']
failed = [r for r in results if r['status'] != 'success']

print(f"\n{'='*80}")
print(f"📊 DOWNLOAD SUMMARY")
print(f"{'='*80}")
print(f"\n✅ Successful: {len(successful)}/{len(results)} datasets")
print(f"📹 Total Videos: {sum(r.get('videos',0) for r in successful):,}")
print(f"💾 Total Size: {sum(r.get('size_gb',0) for r in successful):.2f} GB")
print(f"⏱️  Total Time: {sum(r.get('time_s',0) for r in successful)/60:.1f} minutes")

if failed:
    print(f"\n❌ Failed: {len(failed)} datasets")
    for r in failed:
        print(f"   - {r['name']}: {r.get('error', 'Unknown')[:100]}")

print(f"\n📄 Results saved to: /workspace/datasets/download_results.json")
print(f"\n⏰ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

print(f"\n💾 Final Disk Usage:")
os.system("du -sh /workspace/datasets/* 2>/dev/null")
os.system("df -h /workspace | tail -1")
EOF
