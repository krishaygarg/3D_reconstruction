/* ==============================================================================
   Group 1: 360 Turntable Viewer & Quality Inspector Script
   ============================================================================== */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const container = document.getElementById("turntable-container");
  const frameImg = document.getElementById("active-frame-img");
  const scrubber = document.getElementById("frame-scrubber");
  const hudAngle = document.getElementById("hud-angle");
  const hudFrame = document.getElementById("hud-frame");
  const hudSharpness = document.getElementById("hud-sharpness");
  const dragIndicator = document.getElementById("drag-indicator");
  const emptyState = document.getElementById("empty-state");
  const fileInput = document.getElementById("file-input");
  const videoInput = document.getElementById("video-input");
  const loadFolderBtn = document.getElementById("load-folder-btn");
  const loadVideoBtn = document.getElementById("load-video-btn");
  const downloadZipBtn = document.getElementById("download-zip-btn");
  const demoLoadBtn = document.getElementById("demo-load-btn");
  const demoConvertBtn = document.getElementById("demo-convert-btn");

  // Video processing modal elements
  const modal = document.getElementById("video-processing-modal");
  const modalHeading = document.getElementById("modal-heading");
  const modalStatusText = document.getElementById("modal-status-text");
  const modalProgressFill = document.getElementById("modal-progress-fill");
  const modalProgressLabel = document.getElementById("modal-progress-label");
  const modalFrameCounter = document.getElementById("modal-frame-counter");

  const btnPrev = document.getElementById("btn-prev");
  const btnNext = document.getElementById("btn-next");
  const btnPlay = document.getElementById("btn-play");
  const playIcon = document.getElementById("play-icon");
  const btnSpeed = document.getElementById("btn-speed");
  const btnFullscreen = document.getElementById("btn-fullscreen");
  const btnToggleHud = document.getElementById("btn-toggle-hud");

  const chartCanvas = document.getElementById("sharpness-chart");
  const chartCtx = chartCanvas ? chartCanvas.getContext("2d") : null;

  // State
  let frames = [];
  let currentIndex = 0;
  let isPlaying = false;
  let playInterval = null;
  let playbackSpeed = 1.0;
  let isDragging = false;
  let startX = 0;
  let startIndex = 0;
  let showHud = true;

  // Initialize with demo test frames if available
  loadDemoFrames();

  // Load Demo Data
  async function loadDemoFrames() {
    try {
      const response = await fetch("test_artifacts/test_frames/dataset_metadata.json");
      if (!response.ok) throw new Error("Metadata not found");
      const data = await response.json();
      
      frames = data.frames.map(f => ({
        url: `test_artifacts/test_frames/${f.filename}`,
        sharpness: f.sharpness,
        timestamp: f.timestamp_sec,
        replaced: f.replaced
      }));

      // Update metrics UI
      if (data.stats) {
        document.getElementById("val-frame-count").textContent = data.stats.final_extracted_count;
        document.getElementById("val-resolution").textContent = data.stats.final_resolution;
        document.getElementById("val-mean-sharpness").textContent = data.stats.mean_sharpness;
        document.getElementById("val-blur-details").textContent = 
          `${data.stats.blurry_detected} blurry detected • ${data.stats.smart_replaced} replaced`;
      }

      if (downloadZipBtn) downloadZipBtn.style.display = "inline-flex";
      initViewer();
    } catch (err) {
      console.log("No default test frames loaded via fetch (could be local file:// protocol). Showing selector.");
      // Check if we can still render placeholder turntable
      generateFallbackCanvasFrames();
    }
  }

  // Generate synthetic procedural turntable frames in memory if opened directly via file://
  function generateFallbackCanvasFrames() {
    const total = 100;
    frames = [];
    const offCanvas = document.createElement("canvas");
    offCanvas.width = 1280;
    offCanvas.height = 720;
    const ctx = offCanvas.getContext("2d");

    for (let i = 0; i < total; i++) {
      const angle = (i / total) * Math.PI * 2;
      const deg = Math.round((i / total) * 360);
      
      // Draw simulated 360 view
      ctx.fillStyle = "#0f172a";
      ctx.fillRect(0, 0, 1280, 720);

      // Floor grid
      ctx.strokeStyle = "#1e293b";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(0, 520);
      ctx.lineTo(1280, 520);
      ctx.stroke();

      // Object shadow
      ctx.fillStyle = "rgba(0,0,0,0.6)";
      ctx.beginPath();
      ctx.ellipse(640, 520, 200, 30, 0, 0, Math.PI * 2);
      ctx.fill();

      // Revolving textured cylinder
      const grad = ctx.createLinearGradient(480, 200, 800, 200);
      grad.addColorStop(0, "#1e1b4b");
      grad.addColorStop(0.5, "#4338ca");
      grad.addColorStop(1, "#312e81");
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.roundRect(500, 220, 280, 280, 20);
      ctx.fill();

      // Text marker rotating
      ctx.fillStyle = "#38bdf8";
      ctx.font = "bold 28px 'JetBrains Mono', monospace";
      ctx.textAlign = "center";
      const shiftX = Math.sin(angle) * 80;
      ctx.fillText(`360° ORBIT: ${deg}°`, 640 + shiftX, 360);
      ctx.font = "18px 'Plus Jakarta Sans', sans-serif";
      ctx.fillStyle = "#94a3b8";
      ctx.fillText(`Frame ${i + 1} / ${total} • 1280x720`, 640, 410);

      const dataUrl = offCanvas.toDataURL("image/jpeg", 0.9);
      frames.push({
        url: dataUrl,
        sharpness: Number((175 + Math.sin(i * 0.2) * 15).toFixed(1)),
        timestamp: (i * 0.13).toFixed(2),
        replaced: (i === 53)
      });
    }

    if (downloadZipBtn) downloadZipBtn.style.display = "inline-flex";
    initViewer();
  }

  function initViewer() {
    if (frames.length === 0) return;
    emptyState.style.display = "none";
    scrubber.max = frames.length - 1;
    scrubber.value = 0;
    currentIndex = 0;
    resizeChartCanvas();
    updateFrame(0);
  }

  function updateFrame(index) {
    if (frames.length === 0) return;
    currentIndex = (index + frames.length) % frames.length;
    const f = frames[currentIndex];

    frameImg.src = f.url;
    scrubber.value = currentIndex;

    const angleDeg = Math.round((currentIndex / frames.length) * 360);
    hudAngle.innerHTML = `Angle: <strong>${angleDeg}&deg;</strong>`;
    hudFrame.innerHTML = `Frame: <strong>${currentIndex + 1} / ${frames.length}</strong>`;
    hudSharpness.innerHTML = `Sharpness: <strong>${f.sharpness}</strong>`;

    renderSharpnessChart();
  }

  // Playback Controls
  function togglePlay() {
    if (isPlaying) {
      pause();
    } else {
      play();
    }
  }

  function play() {
    isPlaying = true;
    playIcon.innerHTML = '<rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect>';
    dragIndicator.style.opacity = "0";
    
    const intervalMs = Math.round(1000 / (24 * playbackSpeed));
    clearInterval(playInterval);
    playInterval = setInterval(() => {
      updateFrame(currentIndex + 1);
    }, intervalMs);
  }

  function pause() {
    isPlaying = false;
    playIcon.innerHTML = '<polygon points="5 3 19 12 5 21 5 3"></polygon>';
    clearInterval(playInterval);
    dragIndicator.style.opacity = "0.85";
  }

  btnPlay.addEventListener("click", togglePlay);

  btnPrev.addEventListener("click", () => {
    pause();
    updateFrame(currentIndex - 1);
  });

  btnNext.addEventListener("click", () => {
    pause();
    updateFrame(currentIndex + 1);
  });

  scrubber.addEventListener("input", (e) => {
    pause();
    updateFrame(parseInt(e.target.value, 10));
  });

  // Playback speed cycle: 0.5x -> 1x -> 2x
  btnSpeed.addEventListener("click", () => {
    if (playbackSpeed === 1.0) playbackSpeed = 2.0;
    else if (playbackSpeed === 2.0) playbackSpeed = 0.5;
    else playbackSpeed = 1.0;

    btnSpeed.textContent = `Speed: ${playbackSpeed}x`;
    if (isPlaying) play(); // Restart with new speed
  });

  // Drag-to-Rotate Interaction
  container.addEventListener("mousedown", (e) => {
    pause();
    isDragging = true;
    startX = e.clientX;
    startIndex = currentIndex;
    dragIndicator.style.opacity = "0";
  });

  window.addEventListener("mousemove", (e) => {
    if (!isDragging || frames.length === 0) return;
    const deltaX = e.clientX - startX;
    // Every 8 pixels dragged steps one frame
    const frameDelta = Math.floor(deltaX / 8);
    const newIdx = (startIndex + frameDelta) % frames.length;
    updateFrame(newIdx < 0 ? newIdx + frames.length : newIdx);
  });

  window.addEventListener("mouseup", () => {
    if (isDragging) {
      isDragging = false;
      if (!isPlaying) dragIndicator.style.opacity = "0.85";
    }
  });

  // Touch Support for Mobile / Trackpad
  container.addEventListener("touchstart", (e) => {
    pause();
    isDragging = true;
    startX = e.touches[0].clientX;
    startIndex = currentIndex;
    dragIndicator.style.opacity = "0";
  }, { passive: true });

  window.addEventListener("touchmove", (e) => {
    if (!isDragging || frames.length === 0) return;
    const deltaX = e.touches[0].clientX - startX;
    const frameDelta = Math.floor(deltaX / 10);
    const newIdx = (startIndex + frameDelta) % frames.length;
    updateFrame(newIdx < 0 ? newIdx + frames.length : newIdx);
  }, { passive: true });

  window.addEventListener("touchend", () => {
    isDragging = false;
    if (!isPlaying) dragIndicator.style.opacity = "0.85";
  });

  // Keyboard Shortcuts
  window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.target.isContentEditable) return;
    if (e.code === "Space") {
      e.preventDefault();
      togglePlay();
    } else if (e.code === "ArrowLeft") {
      pause();
      updateFrame(currentIndex - 1);
    } else if (e.code === "ArrowRight") {
      pause();
      updateFrame(currentIndex + 1);
    } else if (e.code === "Home") {
      pause();
      updateFrame(0);
    } else if (e.code === "End") {
      pause();
      updateFrame(frames.length - 1);
    }
  });

  // Fullscreen
  btnFullscreen.addEventListener("click", () => {
    if (!document.fullscreenElement) {
      container.requestFullscreen().catch(err => console.log(err));
    } else {
      document.exitFullscreen();
    }
  });

  // Info overlay toggle
  btnToggleHud.addEventListener("click", () => {
    showHud = !showHud;
    document.querySelector(".viewer-hud").style.display = showHud ? "flex" : "none";
  });

  // User Local File Loader
  loadFolderBtn.addEventListener("click", () => fileInput.click());
  if (demoLoadBtn) {
    demoLoadBtn.addEventListener("click", () => loadDemoFrames());
  }

  fileInput.addEventListener("change", (e) => {
    const files = Array.from(e.target.files).filter(f => f.type.startsWith("image/"));
    if (files.length === 0) return;

    // Natural sort by filename
    files.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: 'base' }));

    frames = files.map((file, i) => ({
      url: URL.createObjectURL(file),
      sharpness: 180,
      timestamp: (i * 0.15).toFixed(2),
      replaced: false
    }));

    document.getElementById("val-frame-count").textContent = frames.length;
    if (downloadZipBtn) downloadZipBtn.style.display = "inline-flex";
    initViewer();
  });

  // User Video Loaders
  if (loadVideoBtn) loadVideoBtn.addEventListener("click", () => videoInput.click());
  if (demoConvertBtn) demoConvertBtn.addEventListener("click", () => videoInput.click());
  if (videoInput) {
    videoInput.addEventListener("change", (e) => {
      const file = e.target.files[0];
      if (file) convertVideoToImages(file);
      videoInput.value = "";
    });
  }

  // Calculate Laplacian variance sharpness score from canvas
  function calculateCanvasSharpness(ctx, width, height) {
    const sampleW = Math.min(width, 640);
    const sampleH = Math.round(height * (sampleW / width));

    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = sampleW;
    tempCanvas.height = sampleH;
    const tempCtx = tempCanvas.getContext("2d", { willReadFrequently: true });
    tempCtx.drawImage(ctx.canvas, 0, 0, sampleW, sampleH);

    const imgData = tempCtx.getImageData(0, 0, sampleW, sampleH);
    const data = imgData.data;

    const gray = new Float32Array(sampleW * sampleH);
    for (let i = 0, j = 0; i < data.length; i += 4, j++) {
      gray[j] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    }

    let sum = 0;
    let sumSq = 0;
    let count = 0;

    for (let y = 1; y < sampleH - 1; y++) {
      const rowOffset = y * sampleW;
      for (let x = 1; x < sampleW - 1; x++) {
        const idx = rowOffset + x;
        const lap = gray[idx - sampleW] + gray[idx - 1] + gray[idx + 1] + gray[idx + sampleW] - 4 * gray[idx];
        sum += lap;
        sumSq += lap * lap;
        count++;
      }
    }

    if (count === 0) return 0;
    const mean = sum / count;
    const variance = (sumSq / count) - (mean * mean);
    return Math.round(variance * 10) / 10;
  }

  // Client-Side Video-to-Images Converter
  async function convertVideoToImages(videoFile) {
    if (!videoFile) return;

    modal.style.display = "flex";
    modalHeading.textContent = `Converting ${videoFile.name}`;
    modalStatusText.textContent = "Loading video metadata...";
    modalProgressFill.style.width = "5%";
    modalProgressLabel.textContent = "5%";
    modalFrameCounter.textContent = "Probing video container...";

    const videoUrl = URL.createObjectURL(videoFile);
    const video = document.createElement("video");
    video.preload = "auto";
    video.src = videoUrl;
    video.muted = true;
    video.playsInline = true;

    try {
      await new Promise((resolve, reject) => {
        video.onloadedmetadata = resolve;
        video.onerror = () => reject(new Error("Unable to decode this video file format in browser."));
      });

      const duration = video.duration || 10;
      const origWidth = video.videoWidth || 1920;
      const origHeight = video.videoHeight || 1080;

      // Downscale longest edge to 1280px preserving aspect ratio
      const maxEdge = 1280;
      let targetWidth = origWidth;
      let targetHeight = origHeight;
      const longest = Math.max(origWidth, origHeight);
      if (longest > maxEdge) {
        const scale = maxEdge / longest;
        targetWidth = Math.round(origWidth * scale);
        targetHeight = Math.round(origHeight * scale);
      }
      if (targetWidth % 2 !== 0) targetWidth--;
      if (targetHeight % 2 !== 0) targetHeight--;

      // Shake trimming calculation
      let trimStart = 1.5;
      let trimEnd = 1.5;
      if (duration <= 3.0) {
        if (duration > 1.0) {
          trimStart = Math.min(0.3, duration * 0.1);
          trimEnd = Math.min(0.3, duration * 0.1);
        } else {
          trimStart = 0;
          trimEnd = 0;
        }
      }

      const usableStart = trimStart;
      const usableEnd = Math.max(usableStart + 0.1, duration - trimEnd);
      const usableDuration = usableEnd - usableStart;

      // Target frame count (80-130 target or scaled)
      let frameCount = 100;
      if (usableDuration < 4.0) {
        frameCount = Math.max(20, Math.min(100, Math.round(usableDuration * 25)));
      }

      const timestamps = [];
      const step = usableDuration / frameCount;
      for (let i = 0; i < frameCount; i++) {
        timestamps.push(usableStart + (i + 0.5) * step);
      }

      const offscreenCanvas = document.createElement("canvas");
      offscreenCanvas.width = targetWidth;
      offscreenCanvas.height = targetHeight;
      const ctx = offscreenCanvas.getContext("2d", { willReadFrequently: true });

      const extracted = [];
      modalStatusText.textContent = `Extracting & normalizing ${frameCount} frames...`;

      for (let i = 0; i < timestamps.length; i++) {
        const t = timestamps[i];
        video.currentTime = t;

        await new Promise(res => {
          let tid;
          const onSeeked = () => {
            clearTimeout(tid);
            video.removeEventListener("seeked", onSeeked);
            res();
          };
          tid = setTimeout(onSeeked, 1500);
          video.addEventListener("seeked", onSeeked);
        });

        ctx.drawImage(video, 0, 0, targetWidth, targetHeight);
        const sharpness = calculateCanvasSharpness(ctx, targetWidth, targetHeight);
        const dataUrl = offscreenCanvas.toDataURL("image/jpeg", 0.95);

        extracted.push({
          url: dataUrl,
          sharpness: sharpness,
          timestamp: Number(t.toFixed(2)),
          replaced: false,
          blurry: false,
          width: targetWidth,
          height: targetHeight
        });

        const percent = Math.round(((i + 1) / timestamps.length) * 85);
        modalProgressFill.style.width = `${percent}%`;
        modalProgressLabel.textContent = `${percent}%`;
        modalFrameCounter.textContent = `${i + 1} / ${timestamps.length} frames normalized`;
      }

      modalStatusText.textContent = "Analyzing blur thresholds & searching neighbor frames...";
      modalProgressFill.style.width = "90%";
      modalProgressLabel.textContent = "90%";

      const scores = extracted.map(f => f.sharpness);
      const sorted = [...scores].sort((a, b) => a - b);
      const medianScore = sorted[Math.floor(sorted.length / 2)] || 150;
      const meanScore = scores.reduce((a, b) => a + b, 0) / scores.length;
      const blurThreshold = Math.max(20, medianScore * 0.55);

      let blurryDetected = 0;
      let replacedCount = 0;

      for (let i = 0; i < extracted.length; i++) {
        if (extracted[i].sharpness < blurThreshold) {
          blurryDetected++;
          extracted[i].blurry = true;

          // Probe small neighbor offsets +/- 0.08s
          const baseT = extracted[i].timestamp;
          let bestNeighbor = extracted[i];
          const neighborOffsets = [-0.08, -0.04, 0.04, 0.08];

          for (const dt of neighborOffsets) {
            const probeT = Math.max(usableStart, Math.min(usableEnd, baseT + dt));
            video.currentTime = probeT;
            await new Promise(res => {
              let tid = setTimeout(res, 800);
              const onSeeked = () => { clearTimeout(tid); video.removeEventListener("seeked", onSeeked); res(); };
              video.addEventListener("seeked", onSeeked);
            });
            ctx.drawImage(video, 0, 0, targetWidth, targetHeight);
            const candSharpness = calculateCanvasSharpness(ctx, targetWidth, targetHeight);
            if (candSharpness > bestNeighbor.sharpness) {
              bestNeighbor = {
                url: offscreenCanvas.toDataURL("image/jpeg", 0.95),
                sharpness: candSharpness,
                timestamp: Number(probeT.toFixed(2)),
                replaced: true,
                blurry: false,
                width: targetWidth,
                height: targetHeight
              };
            }
          }

          if (bestNeighbor.replaced) {
            extracted[i] = bestNeighbor;
            replacedCount++;
          }
        }
      }

      frames = extracted;

      // Update metrics
      document.getElementById("val-frame-count").textContent = frames.length;
      document.getElementById("val-resolution").textContent = `${targetWidth} × ${targetHeight}`;
      document.getElementById("val-mean-sharpness").textContent = meanScore.toFixed(1);
      document.getElementById("val-blur-details").textContent = 
        blurryDetected > 0 ? `${blurryDetected} blurry detected • ${replacedCount} smart replaced` : "All candidate frames sharp";

      const trimCard = document.querySelector("#metrics-section .metric-card:nth-child(4) .metric-value");
      if (trimCard) trimCard.textContent = (trimStart + trimEnd).toFixed(1);

      const statusBadge = document.getElementById("pipeline-status-badge");
      if (statusBadge) {
        statusBadge.className = "status-pill status-ready";
        statusBadge.innerHTML = '<span class="status-dot"></span> Video Converted';
      }

      if (downloadZipBtn) downloadZipBtn.style.display = "inline-flex";

      initViewer();

      modalProgressFill.style.width = "100%";
      modalProgressLabel.textContent = "100%";
      modalStatusText.textContent = "Extraction Complete!";
      setTimeout(() => {
        modal.style.display = "none";
      }, 500);

    } catch (err) {
      console.error("Video conversion error:", err);
      alert(`Could not process video: ${err.message}`);
      modal.style.display = "none";
    } finally {
      URL.revokeObjectURL(videoUrl);
    }
  }

  // Drag and Drop support for any video file or image sequence
  window.addEventListener("dragover", (e) => {
    e.preventDefault();
  });

  window.addEventListener("drop", (e) => {
    e.preventDefault();
    if (!e.dataTransfer || !e.dataTransfer.files || e.dataTransfer.files.length === 0) return;
    const fileList = Array.from(e.dataTransfer.files);

    const videoFile = fileList.find(f => f.type.startsWith("video/") || /\.(mp4|mov|webm|avi|mkv|m4v)$/i.test(f.name));
    if (videoFile) {
      convertVideoToImages(videoFile);
      return;
    }

    const imageFiles = fileList.filter(f => f.type.startsWith("image/") || /\.(jpe?g|png|webp)$/i.test(f.name));
    if (imageFiles.length > 0) {
      imageFiles.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: 'base' }));
      frames = imageFiles.map((file, i) => ({
        url: URL.createObjectURL(file),
        sharpness: 180,
        timestamp: (i * 0.15).toFixed(2),
        replaced: false
      }));
      document.getElementById("val-frame-count").textContent = frames.length;
      if (downloadZipBtn) downloadZipBtn.style.display = "inline-flex";
      initViewer();
    }
  });

  // Download Extracted Images as ZIP
  if (downloadZipBtn) {
    downloadZipBtn.addEventListener("click", async () => {
      if (frames.length === 0) return;
      if (typeof JSZip === "undefined") {
        alert("JSZip library is still loading. Please try again in a moment.");
        return;
      }

      const origText = downloadZipBtn.innerHTML;
      downloadZipBtn.innerHTML = "Zipping...";
      downloadZipBtn.disabled = true;

      try {
        const zip = new JSZip();
        const total = frames.length;
        const meta = {
          source_type: "client_web_export",
          pipeline: "web_dashboard_converter",
          stats: {
            final_extracted_count: total,
            target_range: [80, 130],
            target_met: (total >= 80 && total <= 130),
            final_resolution: document.getElementById("val-resolution").textContent,
            mean_sharpness: Number(document.getElementById("val-mean-sharpness").textContent) || 0
          },
          frames: frames.map((f, idx) => ({
            filename: `frame_${String(idx + 1).padStart(4, '0')}.jpg`,
            timestamp_sec: f.timestamp,
            sharpness: f.sharpness,
            replaced: !!f.replaced
          }))
        };
        zip.file("dataset_metadata.json", JSON.stringify(meta, null, 2));

        for (let i = 0; i < frames.length; i++) {
          const f = frames[i];
          const filename = `frame_${String(i + 1).padStart(4, '0')}.jpg`;
          if (f.url.startsWith("data:")) {
            const base64Data = f.url.split(",")[1];
            zip.file(filename, base64Data, { base64: true });
          } else {
            const resp = await fetch(f.url);
            const blob = await resp.blob();
            zip.file(filename, blob);
          }
        }

        const zipBlob = await zip.generateAsync({ type: "blob" });
        const downloadUrl = URL.createObjectURL(zipBlob);
        const a = document.createElement("a");
        a.href = downloadUrl;
        a.download = "group2_handoff.zip";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(downloadUrl);
      } catch (err) {
        console.error("ZIP creation error:", err);
        alert(`Error creating zip: ${err.message}`);
      } finally {
        downloadZipBtn.innerHTML = origText;
        downloadZipBtn.disabled = false;
      }
    });
  }

  // Handle Chart Canvas Dimensions with DevicePixelRatio
  function resizeChartCanvas() {
    if (!chartCanvas) return;
    const rect = chartCanvas.getBoundingClientRect();
    if (rect.width > 0) {
      const dpr = window.devicePixelRatio || 1;
      chartCanvas.width = Math.round(rect.width * dpr);
      chartCanvas.height = Math.round(rect.height * dpr);
      renderSharpnessChart();
    }
  }
  window.addEventListener("resize", resizeChartCanvas);

  // Render Interactive Sharpness Chart
  function renderSharpnessChart() {
    if (!chartCtx || frames.length === 0) return;
    const width = chartCanvas.width;
    const height = chartCanvas.height;
    chartCtx.clearRect(0, 0, width, height);

    const barWidth = width / frames.length;
    const maxScore = Math.max(220, frames.reduce((m, f) => Math.max(m, f.sharpness || 0), 0));

    // Draw baseline
    chartCtx.strokeStyle = "rgba(255, 255, 255, 0.1)";
    chartCtx.lineWidth = 1;
    chartCtx.beginPath();
    chartCtx.moveTo(0, height - 10);
    chartCtx.lineTo(width, height - 10);
    chartCtx.stroke();

    frames.forEach((f, i) => {
      const barHeight = ((f.sharpness / maxScore) * (height - 30));
      const x = i * barWidth;
      const y = height - 10 - barHeight;

      if (i === currentIndex) {
        // Highlight active position
        chartCtx.fillStyle = "#ffffff";
        chartCtx.fillRect(x, 4, Math.max(barWidth, 3), height - 14);

        // Marker indicator
        chartCtx.fillStyle = "#38bdf8";
        chartCtx.beginPath();
        chartCtx.arc(x + barWidth / 2, 6, 3, 0, Math.PI * 2);
        chartCtx.fill();
      } else if (f.replaced) {
        chartCtx.fillStyle = "#f59e0b"; // Replaced frame amber
        chartCtx.fillRect(x, y, Math.max(barWidth - 1, 1), barHeight);
      } else if (f.blurry) {
        chartCtx.fillStyle = "#f43f5e"; // Blurry flagged frame rose
        chartCtx.fillRect(x, y, Math.max(barWidth - 1, 1), barHeight);
      } else {
        chartCtx.fillStyle = "rgba(56, 189, 248, 0.6)"; // Regular cyan
        chartCtx.fillRect(x, y, Math.max(barWidth - 1, 1), barHeight);
      }
    });
  }

  // Chart Click and Drag-Scrubbing
  let isChartScrubbing = false;
  function scrubFromChart(e) {
    if (!chartCanvas || frames.length === 0) return;
    const rect = chartCanvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const ratio = Math.max(0, Math.min(1, clickX / rect.width));
    const targetIdx = Math.min(Math.floor(ratio * frames.length), frames.length - 1);
    pause();
    updateFrame(targetIdx);
  }

  if (chartCanvas) {
    chartCanvas.addEventListener("mousedown", (e) => {
      isChartScrubbing = true;
      scrubFromChart(e);
    });
    window.addEventListener("mousemove", (e) => {
      if (isChartScrubbing) scrubFromChart(e);
    });
    window.addEventListener("mouseup", () => {
      isChartScrubbing = false;
    });
  }

  // CLI Tabs Switching
  const tabs = document.querySelectorAll(".cli-tab");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));

      tab.classList.add("active");
      const targetId = tab.getAttribute("data-tab");
      document.getElementById(targetId).classList.add("active");
    });
  });

  // Copy Buttons
  document.querySelectorAll(".copy-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const text = btn.getAttribute("data-copy");
      navigator.clipboard.writeText(text).then(() => {
        const origSvg = btn.innerHTML;
        btn.innerHTML = '<span style="font-size:10px; font-weight:700; color:#000;">DONE</span>';
        btn.style.background = "#10b981";
        setTimeout(() => {
          btn.innerHTML = origSvg;
          btn.style.background = "";
        }, 1500);
      });
    });
  });
});
