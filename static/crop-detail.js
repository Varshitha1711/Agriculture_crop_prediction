/**
 * Crop Detail Page Handler
 * 
 * Displays crop-specific guidance and yield time-series from APY data.
 * Features:
 * - Static crop metadata (overview, irrigation guidance)
 * - Automatic yield chart rendering when GPS location available
 * - Fallback manual state/district entry for yield lookup
 * - District spelling suggestions from APY data
 */

const params = new URLSearchParams(window.location.search);
const cropKey = (params.get("crop") || "").trim().toLowerCase();
const STORAGE_KEY = "cropRec_lastPrediction";

/**
 * Display placeholder message when yield data unavailable.
 * Hides chart and shows user-friendly error message.
 * 
 * @param {string} mainMsg - Main message (unused; always shows generic message)
 */
function buildYieldPlaceholder(mainMsg) {
    const el = document.getElementById("yield-placeholder");
    el.style.display = "none";
    document.getElementById("chart-panel").style.display = "none";
    // Always keep it short to avoid showing raw loader/tooling messages to end users.
    const msg =
        "No yield chart data available for this district and crop in your APY file.";
    // Intentionally short: remove installer/error-instruction text from the UI.
    el.innerHTML = "";
}

/**
 * Hide the yield placeholder element.
 */
function hidePlaceholder() {
    const el = document.getElementById("yield-placeholder");
    el.style.display = "none";
    el.innerHTML = "";
}

/**
 * Show the manual region fallback (state/district manual entry form).
 * 
 * @param {string} message - Error message to display above the form
 */
function showRegionFallback(message) {
    const panel = document.getElementById("region-fallback");
    const errEl = document.getElementById("region-error");
    if (message) errEl.textContent = message;
    panel.style.display = "block";
}

/**
 * Hide the manual region fallback.
 */
function hideRegionFallback() {
    const panel = document.getElementById("region-fallback");
    const errEl = document.getElementById("region-error");
    if (errEl) errEl.textContent = "";
    panel.style.display = "none";
}

/**
 * Fetch yield time-series by state and district from API.
 * 
 * @param {string} state - State name
 * @param {string} district - District name
 * @returns {Promise<Object>} API response with {series, note, configured, location}
 * @throws {Error} If API request fails
 */
async function fetchYieldSeriesByStateDistrict(state, district) {
    const res = await fetch(
        `/api/yield-series?crop=${encodeURIComponent(cropKey)}&state=${encodeURIComponent(state)}&district=${encodeURIComponent(district)}`
    );
    const j = await res.json().catch(() => ({}));
    if (!res.ok) {
        throw new Error(j.detail || "Yield request failed.");
    }
    return j;
}

/**
 * Fetch yield time-series by latitude/longitude from API.
 * Server performs reverse geocoding and APY lookup automatically.
 * 
 * @param {number} lat - Latitude
 * @param {number} lon - Longitude
 * @returns {Promise<Object>} API response with {series, note, configured, location}
 * @throws {Error} If API request fails
 */
async function fetchYieldSeriesByLatLon(lat, lon) {
    const res = await fetch(
        `/api/yield-series?crop=${encodeURIComponent(cropKey)}&lat=${encodeURIComponent(lat)}&lon=${encodeURIComponent(lon)}`
    );
    const j = await res.json().catch(() => ({}));
    if (!res.ok) {
        throw new Error(j.detail || "Yield request failed.");
    }
    return j;
}

/**
 * Render yield time-series as Chart.js line chart.
 * Falls back to placeholder if no data points available.
 * 
 * @param {Object} data - Yield series response {series, note, message, location}
 * @returns {Promise<boolean>} True if chart rendered, false if placeholder shown
 */
async function renderYieldChartFromData(data) {
    const ctxEl = document.getElementById("yield-context");
    const chartPanel = document.getElementById("chart-panel");
    const series = data.series || [];
    const points = series.filter((p) => p.yield != null && !Number.isNaN(p.yield));

    if (!points.length) {
        chartPanel.style.display = "none";
        buildYieldPlaceholder(data.note || data.message || "No yield points found.");
        return false;
    }

    hidePlaceholder();
    chartPanel.style.display = "block";

    ctxEl.textContent = [
        data.location && (data.location.district || data.location.state)
            ? `Matched area: ${data.location.district || ""}${data.location.district && data.location.state ? ", " : ""}${data.location.state || ""}.`
            : "",
        data.note || "",
    ]
        .filter(Boolean)
        .join(" ");

    // Wait for Chart.js to load (with 8s timeout)
    await new Promise((r) => {
        if (typeof Chart !== "undefined") return r();
        const t0 = Date.now();
        const id = setInterval(() => {
            if (typeof Chart !== "undefined") {
                clearInterval(id);
                r();
            } else if (Date.now() - t0 > 8000) {
                clearInterval(id);
                r();
            }
        }, 50);
    });

    if (typeof Chart === "undefined") {
        ctxEl.textContent += " Chart.js did not load.";
        buildYieldPlaceholder(data.message || "Chart library unavailable.");
        return false;
    }

    const labels = points.map((p) => String(p.year));
    const values = points.map((p) => p.yield);
    const canvas = document.getElementById("yield-chart");
    const ctx = canvas.getContext("2d");
    new Chart(ctx, {
        type: "line",
        data: {
            labels,
            datasets: [
                {
                    label: "Yield (units from your APY source)",
                    data: values,
                    borderColor: "#2d6a4f",
                    backgroundColor: "rgba(45,106,79,0.1)",
                    tension: 0.2,
                    fill: true,
                },
            ],
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: "Official APY values from your local dataset",
                },
            },
            scales: { y: { beginAtZero: false } },
        },
    });

    return true;
}

/**
 * Load and display static crop metadata from crops.json.
 * Updates page title, overview, irrigation guidance.
 */
async function loadStaticCrop() {
    const titleEl = document.getElementById("crop-title");
    const subEl = document.getElementById("crop-sub");
    const overviewEl = document.getElementById("overview-text");
    const irrigEl = document.getElementById("irrigation-text");
    const waterNote = document.getElementById("water-note");

    if (!cropKey) {
        titleEl.textContent = "No crop selected";
        overviewEl.textContent = "Open this page from the recommendations list.";
        return;
    }

    let cropData;
    try {
        const res = await fetch("data/crops.json");
        cropData = await res.json();
    } catch {
        titleEl.textContent = "Could not load crop data";
        return;
    }

    const entry = cropData && cropData[cropKey];
    if (entry) {
        titleEl.textContent = entry.name || capitalize(cropKey);
        subEl.textContent = "Guidance notes (general; follow local extension).";
        overviewEl.textContent = entry.overview || "";
        irrigEl.textContent = entry.irrigation || "";
        waterNote.textContent = entry.water_note || "";
    } else {
        titleEl.textContent = capitalize(cropKey);
        subEl.textContent = "";
        overviewEl.textContent =
            "No static profile in data/crops.json for this label; see model ranking only.";
        irrigEl.textContent = "";
        waterNote.textContent = "";
    }
}

/**
 * Load and render yield time-series for the crop.
 * Attempts automatic GPS-based lookup; falls back to manual state/district entry.
 */
async function loadYieldSeries() {
    const ctxEl = document.getElementById("yield-context");
    const chartPanel = document.getElementById("chart-panel");

    if (!cropKey) {
        buildYieldPlaceholder("");
        return;
    }

    const raw = sessionStorage.getItem(STORAGE_KEY);
    if (!raw) {
        ctxEl.textContent = "";
        buildYieldPlaceholder("No prediction session found.");
        return;
    }

    let payload;
    try {
        payload = JSON.parse(raw);
    } catch {
        ctxEl.textContent = "";
        buildYieldPlaceholder("Could not read saved prediction.");
        return;
    }

    hideRegionFallback();
    chartPanel.style.display = "none";
    ctxEl.textContent = "";

    const geo = payload.geo || null;
    if (!geo || geo.lat == null || geo.lon == null) {
        showRegionFallback(
            "Automatic GPS match needs `Use my location` on Predict so GPS is saved. You can also enter state & district manually below."
        );
        buildYieldPlaceholder("");
        return;
    }

    ctxEl.textContent = "";

    let data = null;
    try {
        data = await fetchYieldSeriesByLatLon(geo.lat, geo.lon);
    } catch (e) {
        showRegionFallback(e.message || "Yield request failed. Enter state & district manually.");
        buildYieldPlaceholder("Could not load yield automatically. Use region inputs below.");
        return;
    }

    if (data.configured === false) {
        const msg = data.message || "";
        const lower = msg.toLowerCase();
        if (lower.includes("geocoding failed") || lower.includes("please enter state and district")) {
            showRegionFallback(msg);
            buildYieldPlaceholder("");
            return;
        }
        ctxEl.textContent = msg || "APY file not configured.";
        buildYieldPlaceholder(msg || "");
        return;
    }

    const loaded = await renderYieldChartFromData(data);
    if (!loaded) {
        // If no points for matched district, always allow the user to pick
        // a district spelling that exists in the APY export.
        showRegionFallback(data.message || "No yield points found. Try another district spelling.");
        const st = (data.location && data.location.state) ? data.location.state : (document.getElementById("region-state")?.value || "");
        if (st) {
            const panel = document.getElementById("region-error");
            try {
                const res = await fetch(`/api/apy-districts?state=${encodeURIComponent(st)}&limit=15`);
                const j = await res.json().catch(() => ({}));
                if (j && j.configured && Array.isArray(j.districts) && j.districts.length) {
                    panel.textContent =
                        (panel.textContent ? panel.textContent + " " : "") +
                        `Available districts include: ${j.districts.slice(0, 6).join(", ")}.`;
                }
            } catch {
                // ignore suggestions failure
            }
        }
    }
}

/**
 * Initialize crop detail page: load static data and yield series.
 */
async function init() {
    await loadStaticCrop();
    await loadYieldSeries();
}

/**
 * Manual state/district form submission handler.
 * Fetches and renders yield series for user-entered state/district.
 */
document.getElementById("btn-load-region").addEventListener("click", async () => {
    const state = document.getElementById("region-state").value.trim();
    const district = document.getElementById("region-district").value.trim();
    const errEl = document.getElementById("region-error");
    errEl.textContent = "";

    if (!state || !district) {
        errEl.textContent = "Please enter both state and district.";
        return;
    }

    try {
        const data = await fetchYieldSeriesByStateDistrict(state, district);
        hideRegionFallback();
        const ok = await renderYieldChartFromData(data);
        if (!ok) {
            showRegionFallback(data.message || "No yield points found. Check district spelling.");
            const errEl = document.getElementById("region-error");
            errEl.textContent = data.message || "No yield points found. Check district spelling.";
            try {
                const res = await fetch(`/api/apy-districts?state=${encodeURIComponent(state)}&limit=15`);
                const j = await res.json().catch(() => ({}));
                if (j && j.configured && Array.isArray(j.districts) && j.districts.length) {
                    errEl.textContent =
                        (errEl.textContent ? errEl.textContent + " " : "") +
                        `Available districts include: ${j.districts.slice(0, 6).join(", ")}.`;
                }
            } catch {
                // ignore
            }
        }
    } catch (e) {
        errEl.textContent = e.message || "Failed to load yield series.";
    }
});

// Initialize page on load
init();
