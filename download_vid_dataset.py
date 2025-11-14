#!/usr/bin/env python3
"""
Harvard Dataverse VID Dataset Downloader
Downloads the VID violence detection dataset (3,000 videos)
Uses Dataverse API - NO registration required!
"""

import requests
from pathlib import Path
from tqdm import tqdm

# VID Dataset files from Harvard Dataverse API
VID_FILES = [
    # Non-violence videos (Blurred)
    {'id': 10123371, 'name': 'nv_b_01.rar', 'size': 713547207},
    {'id': 10123372, 'name': 'nv_b_02.rar', 'size': 618719884},
    {'id': 10123400, 'name': 'nv_b_03.rar', 'size': 771826413},
    {'id': 10123399, 'name': 'nv_b_04.rar', 'size': 699320769},
    {'id': 10123403, 'name': 'nv_b_05.rar', 'size': 707663172},
    {'id': 10123402, 'name': 'nv_b_06.rar', 'size': 474233176},
    {'id': 10384240, 'name': 'nv_b_07.rar', 'size': 540773067},

    # Non-violence videos (Masked)
    {'id': 10123407, 'name': 'nv_m_01.rar', 'size': 2493783537},
    {'id': 10123405, 'name': 'nv_m_02.rar', 'size': 2297863972},
    {'id': 10123406, 'name': 'nv_m_03.rar', 'size': 2516229876},
    {'id': 10123408, 'name': 'nv_m_04.rar', 'size': 2625209968},
    {'id': 10123409, 'name': 'nv_m_05.rar', 'size': 2457551069},
    {'id': 10123411, 'name': 'nv_m_06.rar', 'size': 2076341611},
    {'id': 10384241, 'name': 'nv_m_07.rar', 'size': 1641801116},

    # Violence videos - Classroom (Blurred)
    {'id': 10123417, 'name': 'v_c_b_01.rar', 'size': 594209838},
    {'id': 10123423, 'name': 'v_c_b_02.rar', 'size': 663506245},
    {'id': 10123429, 'name': 'v_c_b_03.rar', 'size': 633255474},

    # Violence videos - Classroom (Masked)
    {'id': 10123416, 'name': 'v_c_m_01.rar', 'size': 2447977176},
    {'id': 10123425, 'name': 'v_c_m_02.rar', 'size': 2595037382},
    {'id': 10123420, 'name': 'v_c_m_03.rar', 'size': 2343257157},

    # Violence videos - Domestic
    {'id': 10123419, 'name': 'v_d_b_01.rar', 'size': 836969248},
    {'id': 10123414, 'name': 'v_d_m_01.rar', 'size': 2529472087},

    # Violence videos - Others
    {'id': 10384242, 'name': 'v_o_b_01.rar', 'size': 194766468},
    {'id': 10384243, 'name': 'v_o_m_01.rar', 'size': 751024919},

    # Violence videos - Street (Blurred)
    {'id': 10123421, 'name': 'v_s_b_01.rar', 'size': 780351471},
    {'id': 10123427, 'name': 'v_s_b_02.rar', 'size': 778282988},
    {'id': 10123430, 'name': 'v_s_b_03.rar', 'size': 749062933},
    {'id': 10123422, 'name': 'v_s_b_04.rar', 'size': 212804984},

    # Violence videos - Street (Masked)
    {'id': 10123418, 'name': 'v_s_m_01.rar', 'size': 2644890304},
    {'id': 10123415, 'name': 'v_s_m_02.rar', 'size': 2599523033},
    {'id': 10123428, 'name': 'v_s_m_03.rar', 'size': 2627082310},
    {'id': 10123426, 'name': 'v_s_m_04.rar', 'size': 864634056},
]

def download_file(file_id, filename, filesize, output_dir):
    """Download file from Harvard Dataverse"""
    url = f"https://dataverse.harvard.edu/api/access/datafile/{file_id}"
    output_path = output_dir / filename

    # Skip if already downloaded
    if output_path.exists():
        if output_path.stat().st_size == filesize:
            print(f"  ✓ {filename} already downloaded")
            return True
        else:
            print(f"  ⚠️  {filename} incomplete, re-downloading...")

    print(f"  📥 Downloading {filename} ({filesize/(1024**3):.2f} GB)...")

    try:
        response = requests.get(url, stream=True, timeout=300)

        if response.status_code != 200:
            print(f"  ❌ HTTP {response.status_code} for {filename}")
            return False

        with open(output_path, 'wb') as f:
            with tqdm(total=filesize, unit='B', unit_scale=True, desc=f"    {filename}") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

        print(f"  ✅ Downloaded {filename}")
        return True

    except Exception as e:
        print(f"  ❌ Error downloading {filename}: {e}")
        return False

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Harvard VID Dataset Downloader')
    parser.add_argument('--output-dir', default='/workspace/datasets/phase1/academic/vid',
                       help='Output directory')
    parser.add_argument('--blurred-only', action='store_true',
                       help='Download only blurred versions (smaller files)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("HARVARD VID DATASET DOWNLOADER")
    print("="*80)
    print(f"Output: {output_dir}")
    print("")
    print("📊 Dataset Info:")
    print("  • 3,000 videos (1,500 violent, 1,500 non-violent)")
    print("  • Categories: Street, Classroom, Domestic, Others violence")
    print("  • Both blurred and masked versions available")
    print("  • Total size: ~45 GB (blurred) or ~90 GB (all)")
    print("")

    files_to_download = VID_FILES
    if args.blurred_only:
        files_to_download = [f for f in VID_FILES if '_b_' in f['name']]
        print(f"💡 Downloading BLURRED versions only ({len(files_to_download)} files)")
    else:
        print(f"📥 Downloading ALL versions ({len(files_to_download)} files)")

    print("")

    total_size = sum(f['size'] for f in files_to_download)
    print(f"Total download size: {total_size/(1024**3):.2f} GB")
    print("")

    downloaded = 0
    failed = 0

    for i, file_info in enumerate(files_to_download, 1):
        print(f"\n[{i}/{len(files_to_download)}] {file_info['name']}")

        if download_file(file_info['id'], file_info['name'], file_info['size'], output_dir):
            downloaded += 1
        else:
            failed += 1

    print("\n" + "="*80)
    print("DOWNLOAD COMPLETE")
    print("="*80)
    print(f"✅ Downloaded: {downloaded}/{len(files_to_download)} files")
    if failed > 0:
        print(f"❌ Failed: {failed} files")
    print(f"📁 Location: {output_dir}")
    print("")
    print("🔄 NEXT STEPS:")
    print("1. Extract RAR files:")
    print(f"   cd {output_dir} && unrar x '*.rar'")
    print("")
    print("2. Organize videos:")
    print("   Violent videos: 1,500")
    print("   Non-violent videos: 1,500")
    print("   Total: 3,000 videos ready for training!")
    print("")

if __name__ == "__main__":
    main()
