(() => {
  document.addEventListener("DOMContentLoaded", () => {
    const playerConfig = window.livenessPlayerConfig || {};
    const videoFilename = playerConfig.videoFilename;
    const frameEndpoint = playerConfig.frameEndpoint;
    const analysisEndpoint = playerConfig.analysisEndpoint;
    const videoEl = document.getElementById("uploadedVideo");
    const faceImg = document.getElementById("facePreview");
    const analyzeBtn = document.getElementById("analyzeBtn");
    const progressEl = document.getElementById("analysisProgress");
    const progressBar = document.getElementById("analysisProgressBar");
    const resultsEl = document.getElementById("analysisResults");

    if (!videoEl || !faceImg || !videoFilename || !frameEndpoint) {
      console.warn("Player configuration missing required elements.");
      return;
    }

    let isFetching = false;
    let lastUrl = null;

    const fetchFrame = async (force = false) => {
      if (isFetching && !force) {
        return;
      }

      const currentTime = videoEl.currentTime || 0;
      isFetching = true;

      const url = `${frameEndpoint}?video=${encodeURIComponent(
        videoFilename
      )}&t=${currentTime}`;

      try {
        const response = await fetch(url, { cache: "no-store" });
        if (!response.ok) {
          throw new Error("Failed to fetch frame.");
        }
        const blob = await response.blob();
        const objectUrl = URL.createObjectURL(blob);
        faceImg.src = objectUrl;
        if (lastUrl) {
          URL.revokeObjectURL(lastUrl);
        }
        lastUrl = objectUrl;
      } catch (error) {
        console.error(error);
      } finally {
        isFetching = false;
      }
    };

    // Poll the backend regularly while the video is playing.
    setInterval(() => {
      if (!videoEl.paused && !videoEl.ended) {
        fetchFrame();
      }
    }, 250);

    ["seeked", "pause", "loadeddata"].forEach((eventName) => {
      videoEl.addEventListener(eventName, () => fetchFrame(true));
    });

    if (analyzeBtn && analysisEndpoint) {
      analyzeBtn.addEventListener("click", () => runAnalysis());
    }

    function setProgress(value) {
      if (!progressBar) return;
      progressBar.style.width = `${value}%`;
    }

    async function runAnalysis() {
      if (!analysisEndpoint) {
        return;
      }

      analyzeBtn.disabled = true;
      resultsEl?.classList.add("hidden");
      progressEl?.classList.remove("hidden");
      setProgress(5);

      let progressValue = 5;
      const progressInterval = setInterval(() => {
        progressValue = Math.min(progressValue + Math.random() * 10, 90);
        setProgress(progressValue);
      }, 400);

      try {
        const response = await fetch(analysisEndpoint, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ video: videoFilename }),
        });

        if (!response.ok) {
          throw new Error("Analysis request failed.");
        }

        const data = await response.json();
        setProgress(100);
        renderAnalysis(data);
      } catch (error) {
        console.error(error);
        if (resultsEl) {
          resultsEl.classList.remove("hidden");
          resultsEl.innerHTML = `<div class="message error">${error.message}</div>`;
        }
      } finally {
        clearInterval(progressInterval);
        setTimeout(() => {
          progressEl?.classList.add("hidden");
          setProgress(0);
        }, 600);
        analyzeBtn.disabled = false;
      }
    }

    function renderAnalysis(data) {
      if (!resultsEl) return;
      const frames = data?.frames || [];
      if (!frames.length) {
        resultsEl.classList.remove("hidden");
        resultsEl.innerHTML =
          "<p>No facial landmarks detected across the video.</p>";
        return;
      }

      const chartId = "landmarkChart";
      const samples = data.df_samples || [];
      window.hsiBandData = {};
      window.gradcamSources = {};
      samples.forEach((sample) => {
        if (sample?.bands?.length && sample.id) {
          window.hsiBandData[sample.id] = sample.bands;
        }
        if (sample?.gradcam_source && sample?.image_file && sample.id) {
          window.gradcamSources[sample.id] = {
            hsi: sample.gradcam_source,
            face: sample.image_file,
          };
        }
      });

      const tableRows = frames
        .map((frame) => {
          const eyeLeft = formatPoint(frame.eyes?.left);
          const eyeRight = formatPoint(frame.eyes?.right);
          const nose = formatPoint(frame.nose);
          const mouth = formatPoint(frame.mouth);
          return `<tr>
            <td>${frame.time.toFixed(2)}s</td>
            <td>${eyeLeft}</td>
            <td>${eyeRight}</td>
            <td>${nose}</td>
            <td>${mouth}</td>
          </tr>`;
        })
        .join("");

      const diameterRows = frames
        .map((frame) => {
          const left = formatDiameter(frame.eyes?.left);
          const right = formatDiameter(frame.eyes?.right);
          return `<tr>
            <td>${frame.time.toFixed(2)}s</td>
            <td>${left}</td>
            <td>${right}</td>
          </tr>`;
        })
        .join("");
        
      // --- ADDED: Prepare data for the new relationship analysis table ---
      const relationshipRows = frames
        .map((frame) => {
          const eyeDist = frame.eye_dist != null ? frame.eye_dist.toFixed(2) : "—";
          const ratio = frame.eye_mouth_ratio != null ? frame.eye_mouth_ratio.toFixed(3) : "—";
          return `<tr>
            <td>${frame.time.toFixed(2)}s</td>
            <td>${eyeDist}</td>
            <td>${ratio}</td>
          </tr>`;
        })
        .join("");
      // --- END of ADDED section ---

      resultsEl.classList.remove("hidden");
      const sampleTabs = buildDetectorTabs(samples);

      resultsEl.innerHTML = `
        <section class="analysis-section">
          <h3>Analysis 1: Landmark Trajectories</h3>
          <p>Frames analyzed: ${data.frames_analyzed || frames.length} &bull; Duration: ${
        data.duration?.toFixed ? data.duration.toFixed(2) : "—"
      }s</p>
          <canvas id="${chartId}" width="640" height="240"></canvas>
          <button class="analysis-collapse" id="analysisToggle1">
            <span>▶</span>
            <strong>View tabular coordinates</strong>
          </button>
          <div class="analysis-table hidden" id="analysisTable1">
            <table>
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Eye (Left)</th>
                  <th>Eye (Right)</th>
                  <th>Nose</th>
                  <th>Mouth</th>
                </tr>
              </thead>
              <tbody>
                ${tableRows}
              </tbody>
            </table>
          </div>
        </section>
        <section class="analysis-section">
          <h3>Analysis 2: Eye Diameter Stability</h3>
          <p>Eye circle diameters approximated from Haar detections. Values fluctuate with blinking or lighting changes.</p>
          <canvas id="eyeDiameterChart" width="640" height="240"></canvas>
          <button class="analysis-collapse" id="analysisToggle2">
            <span>▶</span>
            <strong>View diameter table</strong>
          </button>
          <div class="analysis-table hidden" id="analysisTable2">
            <table>
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Diameter (Left)</th>
                  <th>Diameter (Right)</th>
                </tr>
              </thead>
              <tbody>
                ${diameterRows}
              </tbody>
            </table>
          </div>
        </section>
        <section class="analysis-section">
          <h3>Analysis 3: Sampled Frame Classification</h3>
          <p>Randomly sampled ${samples.length || 0} frames were saved and evaluated.</p>
          ${sampleTabs}
        </section>
        <section class="analysis-section">
          <h3>Analysis 4: Facial Landmark Relationship Stability</h3>
          <p>Tracks the geometric proportions of the face. The ratio should remain stable in real videos, even as the head moves.</p>
          <canvas id="relationshipChart" width="640" height="240"></canvas>
          <button class="analysis-collapse" id="analysisToggle4">
            <span>▶</span>
            <strong>View relationship data</strong>
          </button>
          <div class="analysis-table hidden" id="analysisTable4">
            <table>
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Eye-to-Eye Distance</th>
                  <th>Eye-Mouth Ratio</th>
                </tr>
              </thead>
              <tbody>
                ${relationshipRows}
              </tbody>
            </table>
          </div>
        </section>
        <div id="bandModal" class="band-modal hidden">
          <div class="band-modal-content">
            <button class="band-modal-close" aria-label="Close bands view">&times;</button>
            <h4>Hyperspectral Bands</h4>
            <div id="bandModalGrid" class="band-grid"></div>
          </div>
        </div>
        <div id="gradcamModal" class="gradcam-modal hidden">
          <div class="gradcam-modal-content">
            <button class="gradcam-modal-close" aria-label="Close Grad-CAM view">&times;</button>
            <h4>Grad-CAM Analysis</h4>
            <p id="gradcamStatus" class="gradcam-status"></p>
            <div class="gradcam-visual hidden" id="gradcamVisual">
              <img id="gradcamImage" class="gradcam-image" alt="Grad-CAM visualization" />
              <div class="gradcam-legend">
                <span>High Activation</span>
                <div class="gradcam-legend-bar"></div>
                <span>Low Activation</span>
              </div>
            </div>
          </div>
        </div>
      `;

      attachTableToggle("analysisToggle1", "analysisTable1");
      attachTableToggle("analysisToggle2", "analysisTable2");
      // --- ADDED: Attach toggle for the new table ---
      attachTableToggle("analysisToggle4", "analysisTable4");

      const chartCanvas = document.getElementById(chartId);
      if (chartCanvas) {
        drawLineChart(chartCanvas, [
          {
            label: "Nose X",
            color: "#fbbf24",
            values: frames.map((f) => ({
              time: f.time,
              value: f.nose ? f.nose.x : null,
            })),
          },
          {
            label: "Nose Y",
            color: "#fb7185",
            values: frames.map((f) => ({
              time: f.time,
              value: f.nose ? f.nose.y : null,
            })),
          },
          {
            label: "Mouth Y",
            color: "#60a5fa",
            values: frames.map((f) => ({
              time: f.time,
              value: f.mouth ? f.mouth.y : null,
            })),
          },
        ]);
      }

      const diameterChart = document.getElementById("eyeDiameterChart");
      if (diameterChart) {
        drawLineChart(
          diameterChart,
          [
            {
              label: "Left Eye Diameter",
              color: "#a78bfa",
              values: frames.map((f) => ({
                time: f.time,
                value: f.eyes?.left ? f.eyes.left.diameter : null,
              })),
            },
            {
              label: "Right Eye Diameter",
              color: "#fb923c",
              values: frames.map((f) => ({
                time: f.time,
                value: f.eyes?.right ? f.eyes.right.diameter : null,
              })),
            },
          ],
          { maxValue: 200 }
        );
      }
      
      const relationshipChart = document.getElementById("relationshipChart");
      if (relationshipChart) {
        drawLineChart(
          relationshipChart,
          [
            {
              label: "Eye-to-Eye Distance",
              color: "#34d399", // Green
              values: frames.map((f) => ({
                time: f.time,
                value: f.eye_dist,
              })),
            },
            {
              label: "Eye-Mouth Ratio",
              color: "#f472b6", // Pink
              // We scale the ratio by 100 to make it visible on the same chart as the distance
              values: frames.map((f) => ({
                time: f.time,
                value: f.eye_mouth_ratio ? f.eye_mouth_ratio * 100 : null,
              })),
            },
          ],
          { maxValue: 150 } // A reasonable max value for pixel distances in a 256x256 crop
        );
      }
      setupAnalysisTabs(resultsEl);
      setupBandModal(resultsEl);
      setupGradcamButtons(resultsEl);
    }

    function attachTableToggle(buttonId, tableId) {
      const toggleButton = document.getElementById(buttonId);
      const tableWrapper = document.getElementById(tableId);
      if (!toggleButton || !tableWrapper) return;
      toggleButton.addEventListener("click", () => {
        tableWrapper.classList.toggle("hidden");
        const icon = toggleButton.querySelector("span");
        if (icon) {
          icon.textContent = tableWrapper.classList.contains("hidden")
            ? "▶"
            : "▼";
        }
      });
    }

    function formatPoint(point) {
      if (!point) {
        return "—";
      }
      return `x:${point.x?.toFixed(1) ?? point.x}, y:${point.y?.toFixed(
        1
      ) ?? point.y}`;
    }

    function formatDiameter(point) {
      if (!point || point.diameter == null) {
        return "—";
      }
      return point.diameter.toFixed(2);
    }

    function formatPercent(value) {
      if (value == null || Number.isNaN(value)) {
        return "—";
      }
      return `${(value * 100).toFixed(1)}%`;
    }

    function buildDetectorTabs(samples) {
      const detectorTabs = [
        { key: "hyperspectral", label: "Hyperspectral DF" },
        { key: "xception", label: "Xception" },
        { key: "vit_b16", label: "ViT/B-16" },
        { key: "swin_v2_b", label: "SwinV2-B" },
      ];

      if (!samples.length) {
        return "<p>No frames could be analyzed.</p>";
      }

      const buttons = detectorTabs
        .map(
          (tab, index) =>
            `<button class="tab-button ${
              index === 0 ? "active" : ""
            }" data-tab="${tab.key}">${tab.label}</button>`
        )
        .join("");

      const contents = detectorTabs
        .map(
          (tab, index) =>
            `<div class="tab-content ${
              index === 0 ? "active" : ""
            }" data-tab="${tab.key}">
              ${buildDetectorGallery(samples, tab.key)}
            </div>`
        )
        .join("");

      return `<div class="analysis-tabs">
        <div class="tab-buttons">${buttons}</div>
        ${contents}
      </div>`;
    }

    function buildDetectorGallery(samples, detectorKey) {
      if (!samples.length) {
        return "<p>No frames could be analyzed.</p>";
      }
      const cards = samples
        .map((sample, idx) => {
          const detection = sample.detectors?.[detectorKey];
          const label = detection?.label || "unknown";
          const tagClass =
            label.toLowerCase() === "fake" ? "fake" : "real";
          const errorMsg = detection?.error;

          const probReal = formatPercent(detection?.prob_real);
          const probFake = formatPercent(detection?.prob_fake);
          const showBands =
            detectorKey === "hyperspectral" && sample.bands?.length;
          const bandButton = showBands
            ? `<button class="secondary-button view-bands-btn" data-sample="${sample.id}">View Multi Bands</button>`
            : "";
          const gradcamButtons =
            detectorKey === "hyperspectral" && sample.gradcam_source
              ? `<div class="gradcam-controls">
                  <button class="secondary-button gradcam-btn" data-class="0" data-sample="${sample.id}">
                    <span>(Class: Real)</span>
                    <span>Grad-CAM Analysis</span>
                  </button>
                  <button class="secondary-button gradcam-btn" data-class="1" data-sample="${sample.id}">
                    <span>(Class: Fake)</span>
                    <span>Grad-CAM Analysis</span>
                  </button>
                </div>`
              : "";

          return `<div class="analysis-card">
            <img src="${sample.image_url}" alt="Sample frame ${idx + 1}" loading="lazy" />
            <div><strong>Timestamp:</strong> ${sample.time.toFixed(2)}s</div>
            ${
              errorMsg
                ? `<div class="message error">${errorMsg}</div>`
                : `<div><span class="tag ${tagClass}">${label}</span></div>
                  <div>Real probability: ${probReal}</div>
                  <div>Fake probability: ${probFake}</div>
                  ${bandButton}
                  ${gradcamButtons}`
            }
          </div>`;
        })
        .join("");
      return `<div class="analysis-gallery">${cards}</div>`;
    }

    function setupAnalysisTabs(container) {
      if (!container) return;
      const tabContainers = container.querySelectorAll(".analysis-tabs");
      tabContainers.forEach((tabsRoot) => {
        const buttons = tabsRoot.querySelectorAll(".tab-button");
        const contents = tabsRoot.querySelectorAll(".tab-content");
        buttons.forEach((btn) => {
          btn.addEventListener("click", () => {
            const target = btn.dataset.tab;
            buttons.forEach((b) => b.classList.remove("active"));
            contents.forEach((content) =>
              content.classList.remove("active")
            );
            btn.classList.add("active");
            const activeContent = tabsRoot.querySelector(
              `.tab-content[data-tab="${target}"]`
            );
            if (activeContent) {
              activeContent.classList.add("active");
            }
          });
        });
      });
    }

    function setupBandModal(container) {
      const modal = document.getElementById("bandModal");
      const grid = document.getElementById("bandModalGrid");
      if (!modal || !grid) return;

      const closeBtn = modal.querySelector(".band-modal-close");
      const closeModal = () => {
        modal.classList.add("hidden");
        grid.innerHTML = "";
        document.body.style.overflow = "";
      };

      container.querySelectorAll(".view-bands-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
          const sampleId = btn.dataset.sample;
          openBandModal(sampleId, modal, grid);
        });
      });

      closeBtn?.addEventListener("click", closeModal);
      modal.addEventListener("click", (event) => {
        if (event.target === modal) {
          closeModal();
        }
      });
      if (!setupBandModal.bound) {
        window.addEventListener("keydown", (event) => {
          if (!modal.classList.contains("hidden") && event.key === "Escape") {
            closeModal();
          }
        });
        setupBandModal.bound = true;
      }
    }

    function openBandModal(sampleId, modal, grid) {
      const bandData = window.hsiBandData?.[sampleId];
      if (!bandData || !bandData.length) {
        grid.innerHTML = "<p>No band data available for this sample.</p>";
      } else {
        const content = bandData
          .map(
            (band) => `<div class="band-cell">
              <img src="${band.image_url}" alt="Band ${band.band_index}" loading="lazy" />
              <span>Band ${band.band_index + 1}${
                band.wavelength ? ` &middot; ${band.wavelength} nm` : ""
              }</span>
            </div>`
          )
          .join("");
        grid.innerHTML = content;
      }
      modal.classList.remove("hidden");
      document.body.style.overflow = "hidden";
    }

    setupBandModal.bound = false;

    function setupGradcamButtons(container) {
      const modal = document.getElementById("gradcamModal");
      const imgEl = document.getElementById("gradcamImage");
      const statusEl = document.getElementById("gradcamStatus");
      const visual = document.getElementById("gradcamVisual");
      if (!modal || !imgEl || !statusEl || !visual) return;

      const closeModal = () => {
        modal.classList.add("hidden");
        document.body.style.overflow = "";
        imgEl.src = "";
        visual.classList.add("hidden");
        statusEl.textContent = "";
      };

      const closeBtn = modal.querySelector(".gradcam-modal-close");
      closeBtn?.addEventListener("click", closeModal);
      modal.addEventListener("click", (event) => {
        if (event.target === modal) {
          closeModal();
        }
      });
      if (!setupGradcamButtons.bound) {
        window.addEventListener("keydown", (event) => {
          if (!modal.classList.contains("hidden") && event.key === "Escape") {
            closeModal();
          }
        });
        setupGradcamButtons.bound = true;
      }

      container.querySelectorAll(".gradcam-btn").forEach((btn) => {
        btn.addEventListener("click", async () => {
          const sampleId = btn.dataset.sample;
          const classId = Number(btn.dataset.class);
          const source = window.gradcamSources?.[sampleId];
          if (!source) {
            return;
          }

          modal.classList.remove("hidden");
          document.body.style.overflow = "hidden";
          visual.classList.add("hidden");
          statusEl.textContent = "Generating Grad-CAM...";

          try {
            const response = await fetch("/gradcam", {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify({
                class_id: classId,
                hsi_source: source.hsi,
                face_source: source.face,
              }),
            });
            const data = await response.json();
            if (!response.ok || !data.image_url) {
              throw new Error(data.description || "Grad-CAM generation failed.");
            }
            imgEl.src = `${data.image_url}?t=${Date.now()}`;
            statusEl.textContent = `Class ${classId} focus`;
            visual.classList.remove("hidden");
          } catch (error) {
            statusEl.textContent = error.message;
          }
        });
      });
    }
    setupGradcamButtons.bound = false;

    function drawLineChart(canvas, series, options = {}) {
      const ctx = canvas.getContext("2d");
      const width = canvas.width;
      const height = canvas.height;
      ctx.clearRect(0, 0, width, height);

      const padding = { left: 50, right: 20, top: 20, bottom: 30 };
      ctx.fillStyle = "#0d1117";
      ctx.fillRect(0, 0, width, height);

      ctx.strokeStyle = "#30363d";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padding.left, padding.top);
      ctx.lineTo(padding.left, height - padding.bottom);
      ctx.lineTo(width - padding.right, height - padding.bottom);
      ctx.stroke();

      const maxTime =
        Math.max(
          ...series.flatMap((s) => s.values.map((v) => v.time || 0)),
          0
        ) || 1;
      const maxValue = options.maxValue ?? 256;

      series.forEach((line) => {
        ctx.beginPath();
        ctx.strokeStyle = line.color;
        ctx.lineWidth = 2;
        let started = false;
        line.values.forEach((point) => {
          if (point.value == null) {
            started = false;
            return;
          }
          const x =
            padding.left +
            (point.time / maxTime) * (width - padding.left - padding.right);
          const y =
            height -
            padding.bottom -
            (point.value / maxValue) * (height - padding.top - padding.bottom);
          if (!started) {
            ctx.moveTo(x, y);
            started = true;
          } else {
            ctx.lineTo(x, y);
          }
        });
        ctx.stroke();
      });

      // Legend
      const legendX = padding.left + 10;
      let legendY = padding.top;
      ctx.font = "12px Arial";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      series.forEach((line, index) => {
        ctx.fillStyle = line.color;
        ctx.fillRect(legendX, legendY + (index * 20) - 4, 15, 8);
        ctx.fillStyle = "#c9d1d9";
        ctx.fillText(line.label, legendX + 20, legendY + (index * 20));
      });
    }
  });
})();
