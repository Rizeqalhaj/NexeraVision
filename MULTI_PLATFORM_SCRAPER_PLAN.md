# Multi-Platform Non-Violence Video Scraper - Sequential Plan

## Goal
Collect 6,000+ non-violence videos from multiple platforms (Reddit, YouTube, Vimeo, etc.) with anti-detection, rate-limit avoidance, and smart recovery.

## Platform Strategy

### 1. Reddit (Primary - Best Source)
**API**: PRAW (Python Reddit API Wrapper)
**Rate Limits**: 60 requests/minute (OAuth authenticated)
**Anti-Detection**: Official API, no scraping needed
**Target Subreddits**:
- r/PublicFreakout (filter: non-violent)
- r/MadeMeSmile
- r/HumansBeingBros
- r/sports
- r/concerts
- r/festivals
- r/streetphotography
- r/cctv (normal activities)
- r/peoplewatching

**Expected Yield**: 2,000-3,000 videos

### 2. YouTube (Secondary - Curated Content)
**API**: yt-dlp search functionality
**Rate Limits**: No official API, use delays (5-10 sec between searches)
**Anti-Detection**: yt-dlp built-in evasion
**Search Queries**: Same 70 non-violence keywords from scrape_nonviolence_keywords.py
**Expected Yield**: 1,500-2,500 videos

### 3. Vimeo (Tertiary - High Quality)
**API**: Playwright with stealth plugins
**Rate Limits**: 60 requests/minute
**Anti-Detection**: Randomized user agents, delays
**Search Queries**: Non-violence keywords
**Expected Yield**: 500-1,000 videos

### 4. Archive.org (Bonus - Public Domain)
**API**: Official Internet Archive API
**Rate Limits**: Unlimited (public)
**Anti-Detection**: Official API
**Search**: "surveillance normal", "CCTV normal", "public space"
**Expected Yield**: 500-1,000 videos

## Anti-Detection Techniques

### Layer 1: Request Randomization
- **User Agent Rotation**: 50+ real browser user agents
- **Random Delays**: 2-10 seconds between requests (exponential backoff on errors)
- **Session Management**: New session every 100 requests
- **Request Headers**: Full browser-like headers (Accept, Accept-Language, etc.)

### Layer 2: Behavioral Mimicry
- **Human-like Patterns**: Variable scroll speeds, random pauses
- **Cookie Persistence**: Maintain cookies across requests
- **Referer Chains**: Proper referer header progression
- **Timezone/Locale**: Match headers to realistic timezone

### Layer 3: Stealth Automation
- **Playwright Stealth**: playwright-stealth plugin
- **WebDriver Detection Bypass**: Remove navigator.webdriver flag
- **Canvas Fingerprint Randomization**: Prevent fingerprinting
- **WebRTC Leak Prevention**: Disable WebRTC

## Rate Limit Strategy

### Per-Platform Rate Limiters
```python
PLATFORM_LIMITS = {
    'reddit': {'requests_per_minute': 55, 'backoff_multiplier': 2},
    'youtube': {'requests_per_minute': 10, 'backoff_multiplier': 3},
    'vimeo': {'requests_per_minute': 50, 'backoff_multiplier': 2},
    'archive': {'requests_per_minute': 100, 'backoff_multiplier': 1.5}
}
```

### Exponential Backoff
- Initial delay: 5 seconds
- On rate limit: delay × backoff_multiplier
- Max delay: 300 seconds (5 minutes)
- Max retries: 5 attempts

### Request Queuing
- Thread-safe queue per platform
- Priority queue: Reddit > YouTube > Vimeo > Archive
- Parallel platform processing with independent queues

## Smart Features

### 1. Auto-Resume on Failure
- **Checkpoint System**: Save state every 100 URLs
- **Resume File**: `scraper_checkpoint_{platform}_{timestamp}.json`
- **State Includes**: Last processed keyword, collected URLs, failed attempts
- **Recovery**: Auto-detect checkpoint on restart

### 2. Platform Health Monitoring
- **Success Rate**: Track successful vs failed requests
- **Response Times**: Monitor average response time
- **Auto-Throttle**: Reduce rate if success < 80%
- **Platform Skip**: Disable platform if success < 50% after 50 requests

### 3. Duplicate Detection
- **URL Deduplication**: Hash-based set for O(1) lookup
- **Cross-Platform**: Detect same video on multiple platforms
- **Video Hash**: Compare video content hash if available

### 4. Progress Reporting
- **Real-time Stats**: URLs/minute, success rate, ETA
- **Platform Breakdown**: Per-platform statistics
- **Live Dashboard**: Terminal-based progress display

## Implementation Architecture

### Modular Platform Adapters
```
scrape_multiplatform.py
├── RedditAdapter(BasePlatform)
│   ├── authenticate()
│   ├── search(keyword)
│   └── get_video_urls()
├── YouTubeAdapter(BasePlatform)
│   ├── search(keyword)
│   └── extract_urls()
├── VimeoAdapter(BasePlatform)
│   ├── stealth_search(keyword)
│   └── extract_urls()
└── ArchiveAdapter(BasePlatform)
    ├── api_search(keyword)
    └── get_urls()
```

### Shared Components
- **RateLimiter**: Token bucket algorithm per platform
- **SessionManager**: Browser session pooling
- **CheckpointManager**: State persistence and recovery
- **HealthMonitor**: Platform health tracking

## Execution Flow

1. **Initialization**
   - Load checkpoint if exists
   - Initialize platform adapters
   - Setup rate limiters
   - Configure stealth settings

2. **Parallel Platform Scraping**
   - Thread per platform (4 threads total)
   - Each thread processes keywords independently
   - Shared URL collection (thread-safe)

3. **Keyword Processing**
   - For each keyword:
     - Check rate limit
     - Apply random delay
     - Execute search
     - Extract URLs
     - Deduplicate
     - Checkpoint every 100 URLs

4. **Error Handling**
   - Rate limit: Exponential backoff
   - Connection error: Retry with new session
   - Platform down: Skip and continue
   - Persistent failures: Disable platform

5. **Finalization**
   - Merge all URLs
   - Remove duplicates
   - Save to file
   - Generate statistics report

## Expected Timeline

- **Setup**: 2 minutes (install dependencies, authenticate)
- **Reddit**: 30-45 minutes (2,000-3,000 URLs)
- **YouTube**: 60-90 minutes (1,500-2,500 URLs)
- **Vimeo**: 30-45 minutes (500-1,000 URLs)
- **Archive**: 20-30 minutes (500-1,000 URLs)

**Total**: 2-3 hours for 5,000-7,500 non-violence video URLs

## Dependencies
```
praw                 # Reddit API
yt-dlp              # YouTube search
playwright          # Browser automation
playwright-stealth  # Anti-detection
requests            # HTTP requests
fake-useragent      # User agent rotation
```

## Output Files
- `nonviolence_urls_reddit.txt` - Reddit URLs
- `nonviolence_urls_youtube.txt` - YouTube URLs
- `nonviolence_urls_vimeo.txt` - Vimeo URLs
- `nonviolence_urls_archive.txt` - Archive.org URLs
- `nonviolence_urls_ALL_PLATFORMS.txt` - Deduplicated combined URLs
- `scraper_stats.json` - Statistics report
- `checkpoint_*.json` - Recovery checkpoints

## Success Criteria
- Collect 6,000+ unique non-violence video URLs
- Success rate > 85% across all platforms
- No platform bans or rate limit lockouts
- Full checkpoint/resume capability
- Complete execution in < 4 hours
