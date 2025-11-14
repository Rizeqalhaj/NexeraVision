# 🎬 Nexara Vision Prototype - Visual Preview

## Interface Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           🌐 Browser - vision.nexaratech.io             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│                       ╔═══════════════════════════════╗                 │
│                       ║    NEXARA VISION              ║                 │
│                       ║ Real-time AI-Powered Violence ║                 │
│                       ║        Detection              ║                 │
│                       ║    [PROTOTYPE v1.0]           ║                 │
│                       ╚═══════════════════════════════╝                 │
│                                                                         │
│  ┌──────────────────────────────┐  ┌─────────────────────────────────┐│
│  │ 📹 Video Upload              │  │ 📊 Real-time Analysis           ││
│  ├──────────────────────────────┤  ├─────────────────────────────────┤│
│  │                              │  │                                 ││
│  │   ╔════════════════════╗     │  │  ● Connected                    ││
│  │   ║                    ║     │  │  ┌───────────────────────────┐  ││
│  │   ║       🎬           ║     │  │  │ Analyzing frame 15/20...  │  ││
│  │   ║ Drag & Drop Video  ║     │  │  └───────────────────────────┘  ││
│  │   ║       Here         ║     │  │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░ 75%       ││
│  │   ║                    ║     │  │                                 ││
│  │   ║  or click to browse║     │  │  ┌─────────┐  ┌──────────┐     ││
│  │   ╚════════════════════╝     │  │  │ Violence│  │Non-Viol. │     ││
│  │                              │  │  │  23.4%  │  │  76.6%   │     ││
│  │   Supported: MP4, AVI, MOV   │  │  └─────────┘  └──────────┘     ││
│  │   Max: 100MB                 │  │                                 ││
│  │                              │  │  📈 LIVE CHART                   ││
│  │   ┌────────────────────────┐ │  │  ┌───────────────────────────┐ ││
│  │   │ 📄 sample_video.mp4    │ │  │  │ 100%─┐                    │ ││
│  │   │ Size: 5.2 MB           │ │  │  │      │╲  Violence (Red)   │ ││
│  │   └────────────────────────┘ │  │  │  50%─┤ ╲                  │ ││
│  │                              │  │  │      │  ╲                 │ ││
│  │   ┌────────────────────────┐ │  │  │   0%─┴───╲──Non-Viol─────│ ││
│  │   │  🚀 Start Real-time    │ │  │  │      F1 F5 F10 F15 F20  │ ││
│  │   │      Analysis          │ │  │  └───────────────────────────┘ ││
│  │   └────────────────────────┘ │  │                                 ││
│  │                              │  │                                 ││
│  └──────────────────────────────┘  └─────────────────────────────────┘│
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                     ✅ FINAL RESULTS                             │  │
│  ├─────────────────────────────────────────────────────────────────┤  │
│  │                              ✅                                  │  │
│  │                         NON-VIOLENT                              │  │
│  │                      Confidence: 87.3%                           │  │
│  │                                                                  │  │
│  │   ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐      │  │
│  │   │Frames: 20    │  │Time: 3.2s    │  │Confidence: 87%  │      │  │
│  │   └──────────────┘  └──────────────┘  └─────────────────┘      │  │
│  │                                                                  │  │
│  │   [📹 Analyze Another Video]                                    │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│               Powered by NexaraTech | AI Research & Development        │
└─────────────────────────────────────────────────────────────────────────┘
```

## Key Visual Elements

### 1. Header Section
- **Large Gradient Logo**: "NEXARA VISION" with blue-purple-pink gradient
- **Subtitle**: "Real-time AI-Powered Violence Detection"
- **Version Badge**: Semi-transparent "PROTOTYPE v1.0" pill

### 2. Left Panel - Upload
- **Drag & Drop Zone**: Dashed purple border, animated on hover
- **Large Upload Icon**: Animated 🎬 emoji (pulses)
- **Clear Instructions**: "Drag & Drop" or "click to browse"
- **File Info Card**: Shows filename and size after selection
- **CTA Button**: Purple gradient "Start Real-time Analysis" button

### 3. Right Panel - Real-time Analysis
- **Connection Status**: Green "● Connected" indicator (top-right)
- **Status Box**: Gray box with blinking dot + status message
- **Progress Bar**: Purple-pink gradient, smooth animation
- **Live Metrics**: Two large cards showing probabilities
  - Red card: Violence percentage (updates live)
  - Green card: Non-violence percentage (updates live)
- **Live Chart**: Chart.js line graph
  - Red line: Violence probability across frames
  - Green line: Non-violence probability
  - X-axis: Frame numbers (F1-F20)
  - Y-axis: Percentage (0-100%)
  - Smooth curves with fill

### 4. Results Card (appears when complete)
- **Large Icon**: ⚠️ for violence or ✅ for safe
- **Bold Title**: "VIOLENCE DETECTED" or "NON-VIOLENT"
- **Confidence Score**: Large percentage
- **Statistics Grid**: Three boxes showing:
  - Frames Analyzed
  - Processing Time
  - Confidence Score
- **Reset Button**: Gray "Analyze Another Video"
- **Color Coded**:
  - Violence: Red border, pink background
  - Safe: Green border, light green background

### 5. Footer
- **Company Attribution**: "Powered by NexaraTech"
- **Tagline**: "AI Research & Development"
- **White Text**: Semi-transparent on gradient background

## Color Palette

```css
/* Primary Gradient Background */
background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #7e22ce 100%);

/* Logo Gradient */
gradient(135deg, #60a5fa, #a78bfa, #ec4899);

/* CTA Button */
gradient(135deg, #7c3aed, #6d28d9);

/* Progress Bar */
gradient(90deg, #7c3aed, #ec4899);

/* Violence Theme */
background: #fef2f2;
border: #ef4444;
text: #dc2626;

/* Safe Theme */
background: #f0fdf4;
border: #22c55e;
text: #16a34a;

/* Cards */
background: white;
shadow: rgba(0,0,0,0.3);
```

## Animations

1. **Page Load**: Fade-in animations for header and cards
2. **Upload Icon**: Continuous pulse (scale 1.0 ↔ 1.05)
3. **Status Dot**: Blinking opacity (1.0 ↔ 0.3)
4. **Progress Bar**: Smooth width transitions
5. **Chart Updates**: 300ms smooth line animations
6. **Hover Effects**: Buttons lift 2px on hover
7. **Drag Over**: Upload area scales to 1.02

## Responsive Behavior

### Desktop (>968px)
- Two-column layout (upload | analysis)
- Full chart width
- All metrics visible

### Tablet/Mobile (<968px)
- Single column layout
- Upload panel on top
- Analysis panel below
- Chart maintains aspect ratio
- Metrics stack vertically

## Real-time Update Flow

```
User clicks "Start Analysis"
         ↓
WebSocket connects → "● Connected"
         ↓
Frame 1 extracted → Progress: 5%
         ↓
Frame 5 extracted → Progress: 25% → Chart point F5 added
         ↓
VGG19 features extracted → Progress: 70%
         ↓
Frame 10 analyzed → Metrics update → Chart updates → Progress: 85%
         ↓
Frame 20 analyzed → Final metrics → Chart complete → Progress: 100%
         ↓
Results card appears → Smooth scroll → "Analyze Another" button
```

## User Experience Highlights

✅ **No page refreshes** - All updates via WebSocket
✅ **Instant feedback** - Every action has visual response
✅ **Clear progress** - User always knows what's happening
✅ **Professional design** - Enterprise-quality aesthetics
✅ **Smooth animations** - Modern, fluid transitions
✅ **Color-coded info** - Red=danger, Green=safe, intuitive
✅ **Large touch targets** - Mobile-friendly interactions
✅ **Accessible** - Clear labels, high contrast

## Browser Compatibility

- ✅ Chrome/Edge 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Mobile browsers (iOS Safari, Chrome Mobile)
- ⚠️ IE11 not supported (requires modern JS features)
