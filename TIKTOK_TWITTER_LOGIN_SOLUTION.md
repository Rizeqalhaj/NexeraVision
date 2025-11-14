# TikTok & Twitter Login Solution Guide

Both TikTok and Twitter now require login to view content. Here are **3 working solutions** ranked by ease of implementation.

---

## Solution 1: Browser Cookies Export (EASIEST - Recommended)

### Why This Works
- Uses your existing logged-in browser session
- No need to hardcode passwords
- Works for both TikTok and Twitter
- Most reliable and safe

### Step-by-Step Instructions

#### For TikTok:

**Step 1: Install Browser Extension**
```
Chrome: "Get cookies.txt LOCALLY" extension
Firefox: "cookies.txt" extension
```

**Step 2: Login to TikTok**
```
1. Open Chrome/Firefox
2. Go to https://www.tiktok.com
3. Login with your account
4. Browse a hashtag to verify you're logged in
```

**Step 3: Export Cookies**
```
1. Click the extension icon
2. Click "Export" or "Download"
3. Save as: cookies_tiktok.txt
4. Move to: /home/admin/Desktop/NexaraVision/cookies_tiktok.txt
```

**Step 4: Update TikTok Scraper**
```bash
# I'll create an updated version that uses cookies
```

#### For Twitter:

**Step 1: Same Extension (Already Installed)**

**Step 2: Login to Twitter**
```
1. Open Chrome/Firefox
2. Go to https://twitter.com
3. Login with your account
4. Search for "fight" to verify you're logged in
```

**Step 3: Export Cookies**
```
1. Click the extension icon
2. Click "Export" or "Download"
3. Save as: cookies_twitter.txt
4. Move to: /home/admin/Desktop/NexaraVision/cookies_twitter.txt
```

---

## Solution 2: Direct URL Collection (NO SCRAPING NEEDED)

### Alternative Approach: Skip Scraping, Use Direct Download

Both platforms have **massive existing collections** of video URLs available:

#### TikTok Video URLs from Public Sources

**Option A: Use TikTok Trending API (Unofficial)**
```python
# No login needed - uses public trending data
# Can get 1000s of video URLs without authentication
```

**Option B: Use Pre-Collected TikTok Fight Video Lists**
```
Many GitHub repos have fight video URL collections
Search: "tiktok fight videos dataset github"
```

#### Twitter Video URLs from Public Sources

**Option A: Use Twitter API v2 (If You Have API Key)**
```python
# Twitter Developer Account (free tier)
# Can search and download videos programmatically
```

**Option B: Use Public Twitter Archives**
```
Many datasets exist with Twitter fight video URLs
Search: "twitter fight videos dataset"
```

---

## Solution 3: Use Gallery-dl (Works Without Playwright)

### gallery-dl: Alternative Scraper That Handles Auth Better

**Install**:
```bash
pip install --break-system-packages gallery-dl
```

**For TikTok**:
```bash
# Create config with cookies
gallery-dl --cookies cookies_tiktok.txt "https://www.tiktok.com/tag/cctv"
```

**For Twitter**:
```bash
# Create config with cookies
gallery-dl --cookies cookies_twitter.txt "https://twitter.com/search?q=cctv%20fight"
```

---

## Solution 4: Use Existing Video Datasets (FASTEST)

### Why Scrape When Datasets Already Exist?

#### Kaggle Datasets
```bash
# Install Kaggle CLI
pip install --break-system-packages kaggle

# Search for fight video datasets
kaggle datasets list -s "fight videos"
kaggle datasets list -s "violence detection"
kaggle datasets list -s "cctv fights"

# Download dataset
kaggle datasets download -d <dataset-id>
```

#### Academic Datasets
- **UCF Crime Dataset**: 1,900+ violent videos
- **RWF-2000**: 2,000 fight videos
- **Violent Flows**: 246 fight videos
- **Hockey Fight Dataset**: 1,000 videos

---

## RECOMMENDED SOLUTION: Use yt-dlp with Cookie Files

### Why This Is Best

1. **No Playwright Needed**: yt-dlp handles authentication
2. **Works for Both**: TikTok and Twitter supported
3. **No Scraping Required**: Just provide video URLs
4. **Faster**: Direct download without browser automation
5. **More Reliable**: yt-dlp maintained by large community

### Implementation

I'll create a new approach:
1. Use yt-dlp to download directly from hashtag/search URLs
2. Use cookie files for authentication
3. yt-dlp will handle extraction automatically

---

## Quick Start: Cookie-Based Solution

### Step 1: Get Your Cookies

**Using Browser Developer Tools (Manual)**:
```
1. Open Chrome
2. Login to TikTok/Twitter
3. Press F12 (Developer Tools)
4. Go to Application → Cookies
5. Copy all cookies
6. Save to file
```

**Using Extension (Easier)**:
```
1. Install "EditThisCookie" or "Get cookies.txt"
2. Login to TikTok/Twitter
3. Click extension icon
4. Export cookies
5. Save as cookies_tiktok.txt / cookies_twitter.txt
```

### Step 2: Place Cookie Files

```bash
/home/admin/Desktop/NexaraVision/cookies_tiktok.txt
/home/admin/Desktop/NexaraVision/cookies_twitter.txt
```

### Step 3: Run Updated Scrapers

```bash
# I'll create updated versions that use these cookie files
python3 scrape_tiktok_with_cookies.py
python3 scrape_twitter_with_cookies.py
```

---

## Alternative: Use yt-dlp Archive Feature

### TikTok Direct Download (No Scraping)

```bash
# Download all videos from a TikTok user
yt-dlp --cookies cookies_tiktok.txt "https://www.tiktok.com/@username"

# Download from hashtag
yt-dlp --cookies cookies_tiktok.txt "https://www.tiktok.com/tag/cctv"

# Download multiple hashtags
for tag in cctv fight security; do
    yt-dlp --cookies cookies_tiktok.txt "https://www.tiktok.com/tag/$tag"
done
```

### Twitter Direct Download (No Scraping)

```bash
# Download from Twitter search
yt-dlp --cookies cookies_twitter.txt "https://twitter.com/search?q=cctv%20fight%20filter:videos"

# Download multiple searches
for query in "cctv fight" "security camera"; do
    yt-dlp --cookies cookies_twitter.txt "https://twitter.com/search?q=$query%20filter:videos"
done
```

---

## What I'll Build For You

### Option A: Cookie-Based Playwright Scrapers (Most Control)
- Updated TikTok scraper with cookie authentication
- Updated Twitter scraper with cookie authentication
- You provide cookie files, scrapers use them

### Option B: yt-dlp Direct Download (Fastest)
- No Playwright needed
- Direct download from hashtags/searches
- Uses yt-dlp's built-in extraction
- Just provide cookie files

### Option C: Hybrid Approach (Best of Both)
- Use Playwright to collect URLs (with cookies)
- Use yt-dlp to download videos
- Combines control + reliability

---

## Which Solution Do You Want?

**Choice 1**: I'll update the scrapers to use cookie files with Playwright
- Pros: Same approach as Reddit/YouTube, you control scraping logic
- Cons: You need to export cookies manually

**Choice 2**: I'll create yt-dlp direct download scripts
- Pros: Faster, simpler, no Playwright needed
- Cons: Less control over which videos are collected

**Choice 3**: I'll create hybrid scripts
- Pros: Best reliability, most videos collected
- Cons: Slightly more complex setup

---

## Immediate Next Step

**Tell me which option you prefer, OR:**

**Quick Test**: Let's see if yt-dlp can download from TikTok/Twitter with cookies:

```bash
# Test TikTok
yt-dlp --cookies cookies_tiktok.txt "https://www.tiktok.com/tag/cctv"

# Test Twitter
yt-dlp --cookies cookies_twitter.txt "https://twitter.com/search?q=fight"
```

**If you have cookies available, I can test this right now!**

Otherwise, I'll create the cookie-based Playwright scrapers (Option A) which gives you the most control.

What's your preference?
