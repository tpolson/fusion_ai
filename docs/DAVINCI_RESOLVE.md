# Fusion AI in DaVinci Resolve

## Yes, It Works in DaVinci Resolve! ✅

Fusion AI works with **both**:
- **Fusion Studio** (standalone application)
- **DaVinci Resolve Fusion page** (integrated Fusion)

They use the **same Fusion engine** and **same fuse plugin system**.

## Key Differences

### Fusion Studio vs. Resolve Fusion Page

| Feature | Fusion Studio | Resolve Fusion Page |
|---------|---------------|---------------------|
| Fusion Engine | ✅ Same | ✅ Same |
| Fuse Plugins | ✅ Works | ✅ Works |
| Python Integration | ✅ Works | ✅ Works |
| AI Nodes Available | ✅ All 13 nodes | ✅ All 13 nodes |
| Workflow | Standalone comp | Integrated timeline |

### Fuses Directory Locations

The fuses can be installed in **either or both** locations:

**Fusion Studio Fuses:**
- Windows: `%APPDATA%\Blackmagic Design\Fusion\Fuses\`
- macOS: `~/Library/Application Support/Blackmagic Design/Fusion/Fuses/`
- Linux: `~/.fusion/BlackmagicDesign/Fusion/Fuses/`

**DaVinci Resolve Fuses:**
- Windows: `%APPDATA%\Blackmagic Design\DaVinci Resolve\Fusion\Fuses\`
- macOS: `~/Library/Application Support/Blackmagic Design/DaVinci Resolve/Fusion/Fuses/`
- Linux: `~/.local/share/DaVinci Resolve/Fusion/Fuses/`

**Pro Tip:** Install in **both** locations to use AI nodes in both applications!

## Installation for DaVinci Resolve

### Quick Install

```bash
# 1. Setup venv (same as before)
cd ~/projects/fusion_ai
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e ".[all]"

# 2. Install fuses (auto-detects Resolve)
python scripts/configure_fuses.py --resolve

# OR install to both Fusion and Resolve
python scripts/configure_fuses.py --both
```

### Manual Install for Resolve

**Windows:**
```powershell
# Copy fuses to Resolve
xcopy fusion_ai\fuses\*.fuse "%APPDATA%\Blackmagic Design\DaVinci Resolve\Fusion\Fuses\" /Y
```

**macOS:**
```bash
# Copy fuses to Resolve
cp fusion_ai/fuses/*.fuse ~/Library/Application\ Support/Blackmagic\ Design/DaVinci\ Resolve/Fusion/Fuses/
```

**Linux:**
```bash
# Copy fuses to Resolve
mkdir -p ~/.local/share/DaVinci\ Resolve/Fusion/Fuses/
cp fusion_ai/fuses/*.fuse ~/.local/share/DaVinci\ Resolve/Fusion/Fuses/
```

## Using AI Nodes in DaVinci Resolve

### Workflow 1: Fusion Page (Recommended)

1. **Switch to Fusion page** in DaVinci Resolve
2. **Add AI nodes** from Effects Library → "AI Tools"
3. **Connect nodes** in node graph
4. **Render** - works exactly like Fusion Studio

**Example Timeline:**
```
Timeline Clip → Fusion Page
├── MediaIn1 (timeline video)
├── DepthAnythingV2 (generate depth)
├── AIWANInpaint (remove object)
└── MediaOut1 (back to timeline)
```

### Workflow 2: Fusion Clip on Timeline

1. **Right-click clip** on timeline → "New Fusion Clip"
2. **Opens in Fusion page** with MediaIn/MediaOut
3. **Add AI nodes** between MediaIn and MediaOut
4. **Close Fusion** - changes appear on timeline

### Workflow 3: Fusion Composition (Standalone)

1. **Media Pool** → Right-click → "Create Fusion Composition"
2. **Opens blank Fusion comp**
3. **Add Loader** → AI nodes → Saver
4. **Render composition**
5. **Use in timeline** like any media

## Common Resolve Workflows

### Remove Object from Timeline Clip

```
Timeline Clip → Fusion Page:
MediaIn1 → AIWANInpaint ← Mask (paint out object) → MediaOut1
```

**Steps:**
1. Right-click clip → "Open in Fusion page"
2. Add **AIWANInpaint** node
3. Add **Paint** node, draw mask on object
4. Connect: MediaIn → AIWANInpaint (Input), Paint → AIWANInpaint (Mask)
5. Set temporal consistency to 0.85
6. Enable occlusion detection
7. Render - object removed from entire clip!

### Add Depth-Based Effects

```
Timeline Clip → Fusion Page:
MediaIn1 → DepthAnythingV2 → (use for fog/DOF/relighting) → MediaOut1
```

**Steps:**
1. Open clip in Fusion page
2. Add **DepthAnythingV2** node
3. Use depth map for:
   - Fog (Fog node with depth control)
   - Depth of field (blur based on depth)
   - Relighting (light falloff by depth)

### Character Animation from Video

```
Timeline Clip → Fusion Page:
MediaIn1 → AIControlNet (OpenPose) → MediaOut1
```

**Steps:**
1. Open video clip with person
2. Add **AIControlNet** node
3. Set Type: OpenPose
4. Enable Auto Preprocess
5. Enter prompt: "anime character, detailed, colorful"
6. Render - person becomes animated character!

### AI Upscaling in Timeline

```
Timeline Clip (1080p) → Fusion Page:
MediaIn1 → AIUpscale → Scale (to timeline res) → MediaOut1
```

**Steps:**
1. Open low-res clip in Fusion page
2. Add **AIUpscale** node
3. Select method: Real-ESRGAN x4
4. Render - clip is now 4K quality!

### Style Transfer on Timeline

```
Timeline Clip → Fusion Page:
MediaIn1 → AIStyleTransfer ← StyleLoader → MediaOut1
```

**Steps:**
1. Open clip in Fusion page
2. Add **Loader** node with style image
3. Add **AIStyleTransfer** node
4. Connect MediaIn → Content, StyleLoader → Style
5. Adjust style strength
6. Render - entire clip styled!

## Performance Tips for Resolve

### Timeline Rendering

- **Cache Smart**: Enable "Render Cache" → "Smart" for AI nodes
- **Proxy Mode**: Work in timeline resolution, render at full quality
- **Pre-render Comps**: Render Fusion clips separately, then edit

### Memory Management

- **Close Other Apps**: AI models use VRAM
- **Render One Clip at a Time**: Don't stack multiple AI nodes on same frame
- **Use FP16**: Select FP16 in AI node inspector for 2x speed

### Timeline Organization

```
Timeline Structure:
├── V1: Original footage
├── V2: Fusion clips with AI processing
└── V3: Overlays and final comp
```

## Resolve-Specific Features

### Cache AI Results

```python
# In Fusion page, right-click AI node
→ "Cache" → "Cache to Disk (Background)"
```

This pre-renders AI processing so timeline playback is smooth.

### Use Resolve's Color Management

AI nodes output in **Rec.709** by default. Resolve will automatically convert to your timeline color space.

For best results:
1. **Timeline Color Space**: Match your camera/delivery
2. **AI Processing**: Let Resolve handle color conversion
3. **Grade After AI**: Apply color grading after AI nodes

### Integration with Other Resolve Pages

**Edit Page:**
- See AI-processed clips in timeline
- Trim, cut, arrange normally

**Color Page:**
- Grade after AI processing
- Use depth maps for selective grading
- Masks from AI inpainting

**Fairlight Page:**
- Audio works normally
- AI processing doesn't affect audio

**Deliver Page:**
- Render final output with AI enhancements
- All AI processing baked in

## Troubleshooting in Resolve

### AI Nodes Don't Appear

**Problem:** Can't find "AI Tools" in Effects Library

**Solution:**
1. Check fuses are in correct directory (Resolve, not just Fusion)
2. Restart DaVinci Resolve
3. Check: `%APPDATA%\Blackmagic Design\DaVinci Resolve\Fusion\Fuses\`

### Slow Rendering in Timeline

**Problem:** Timeline playback stutters with AI nodes

**Solution:**
1. Enable Smart Render Cache
2. Pre-render Fusion clips: Right-click → "Render in Place"
3. Lower timeline resolution during editing

### Python Not Found in Resolve

**Problem:** AI nodes show error

**Solution:**
- Ensure Python path in fuses points to venv
- Run: `python scripts/configure_fuses.py --resolve`
- Restart Resolve after configuration

### Different Results in Fusion Page vs Render

**Problem:** Preview looks different from rendered output

**Solution:**
- Check timeline resolution vs render resolution
- Ensure "Use Render Settings" is enabled for AI nodes
- Verify color space consistency

## Best Practices for Resolve

### 1. Edit First, AI Second

```
Workflow:
1. Edit entire timeline (rough cut)
2. Lock picture
3. Apply AI processing to specific clips
4. Finalize edit
```

### 2. Use Fusion Clips Strategically

Only create Fusion clips for sections needing AI:
- Object removal shots
- VFX shots requiring depth
- Style transfer sections
- Character animation moments

### 3. Optimize Render Order

```
Render order:
1. Pre-render heavy AI clips (inpainting, ControlNet)
2. Cache to disk
3. Continue editing with cached versions
4. Final render at delivery quality
```

### 4. Leverage Resolve's Project Management

```
Project Organization:
├── Master Project
│   ├── Bins
│   │   ├── Original Footage
│   │   ├── AI Processed Clips
│   │   └── Fusion Compositions
│   └── Timelines
│       ├── Rough Cut
│       ├── VFX Timeline (with AI)
│       └── Final Master
```

## Version Compatibility

**Tested with:**
- ✅ DaVinci Resolve 18.x
- ✅ DaVinci Resolve 19.x
- ✅ DaVinci Resolve Studio (paid version)
- ✅ DaVinci Resolve Free

**Note:** Free version has same Fusion capabilities as Studio for AI nodes!

## Resolve vs Fusion Studio - Which to Use?

### Use **DaVinci Resolve** when:
- ✅ Working with timeline/footage
- ✅ Need integrated editing, color, audio
- ✅ Delivering final video projects
- ✅ Object removal in editorial context

### Use **Fusion Studio** when:
- ✅ Pure compositing work
- ✅ Complex node trees
- ✅ Standalone VFX shots
- ✅ Rendering image sequences

**Best approach:** Install fuses in **both** locations, use whichever fits your workflow!

## Summary

**Key Points:**
- ✅ Works identically in DaVinci Resolve and Fusion Studio
- ✅ Install fuses in Resolve's Fuses directory
- ✅ All 13 AI nodes available in Fusion page
- ✅ Integrates with Resolve's timeline, color, delivery
- ✅ Use `--resolve` flag with configure script
- ✅ Cache AI processing for smooth timeline playback

**Fusion AI brings professional AI capabilities to your DaVinci Resolve workflow!** 🎬
