document.addEventListener("DOMContentLoaded", async () => {
  const container = document.getElementById("alerts-container");
  const paginationEl = document.getElementById("alerts-pagination");

  const btnAll = document.getElementById("filter-all");
  const btnHigh = document.getElementById("filter-high");
  const btnMedium = document.getElementById("filter-medium");
  const btnLow = document.getElementById("filter-low");

  let allAlerts = [];
  let filteredAlerts = [];
  let currentFilter = "ALL";
  let currentPage = 1;
  const itemsPerPage = 15;

  try {
    const projects = await loadProjectsData();

    projects.forEach((p) => {
      if (Array.isArray(p.alerts) && p.alerts.length > 0) {
        p.alerts.forEach((alertMsg) => {
          allAlerts.push({
            work_id: p.work_id,
            project_name: p.project_name,
            location: `${p.district ? p.district + ', ' : ''}${p.state || 'N/A'}`,
            risk_level: p.risk_level || "LOW",
            risk_score: p.risk_score,
            message: alertMsg,
            recommendation: p.recommendation
          });
        });
      }
    });

    if (allAlerts.length === 0) {
      projects.forEach((p) => {
        allAlerts.push({
          work_id: p.work_id,
          project_name: p.project_name,
          location: `${p.district ? p.district + ', ' : ''}${p.state || 'N/A'}`,
          risk_level: p.risk_level || "LOW",
          risk_score: p.risk_score,
          message: `Project monitored under risk score evaluation (${p.risk_score !== null ? p.risk_score : 'N/A'}/100).`,
          recommendation: p.recommendation
        });
      });
    }

    updateFilterCounts();
    applyFilter("ALL");

  } catch (err) {
    if (container) {
      container.innerHTML = `<div style="text-align: center; color: #dc2626; padding: 2rem;">Failed to load risk alerts. Please check console.</div>`;
    }
  }

  function updateFilterCounts() {
    const highCount = allAlerts.filter(a => a.risk_level === "HIGH").length;
    const medCount = allAlerts.filter(a => a.risk_level === "MEDIUM").length;
    const lowCount = allAlerts.filter(a => a.risk_level === "LOW").length;

    if (btnAll) btnAll.textContent = `All Flags (${allAlerts.length.toLocaleString()})`;
    if (btnHigh) btnHigh.textContent = `High Risk Flags (${highCount.toLocaleString()})`;
    if (btnMedium) btnMedium.textContent = `Medium Risk Flags (${medCount.toLocaleString()})`;
    if (btnLow) btnLow.textContent = `Low Risk Flags (${lowCount.toLocaleString()})`;
  }

  function applyFilter(filter) {
    currentFilter = filter;
    currentPage = 1;

    document.querySelectorAll(".filter-btn").forEach(btn => btn.classList.remove("active"));
    if (filter === "ALL" && btnAll) btnAll.classList.add("active");
    if (filter === "HIGH" && btnHigh) btnHigh.classList.add("active");
    if (filter === "MEDIUM" && btnMedium) btnMedium.classList.add("active");
    if (filter === "LOW" && btnLow) btnLow.classList.add("active");

    if (filter === "ALL") {
      filteredAlerts = allAlerts;
    } else {
      filteredAlerts = allAlerts.filter(a => a.risk_level === filter);
    }

    renderAlerts();
  }

  function renderAlerts() {
    if (!container) return;
    container.innerHTML = "";

    if (filteredAlerts.length === 0) {
      container.innerHTML = `<div style="text-align: center; color: #64748b; padding: 2rem; background: #fff; border-radius: 8px;">No risk alerts found for this filter.</div>`;
      if (paginationEl) paginationEl.innerHTML = "";
      return;
    }

    const startIndex = (currentPage - 1) * itemsPerPage;
    const paginatedItems = filteredAlerts.slice(startIndex, startIndex + itemsPerPage);

    const fragment = document.createDocumentFragment();

    paginatedItems.forEach((item) => {
      const card = document.createElement("div");
      const riskClass = item.risk_level === "HIGH" ? "high-risk" : item.risk_level === "MEDIUM" ? "medium-risk" : "low-risk";
      const iconClass = item.risk_level === "HIGH" ? "ri-error-warning-line" : item.risk_level === "MEDIUM" ? "ri-alert-line" : "ri-checkbox-circle-line";
      const iconColor = item.risk_level === "HIGH" ? "#dc2626" : item.risk_level === "MEDIUM" ? "#d97706" : "#16a34a";
      const safeId = String(item.work_id).replace(/'/g, "\\'");

      card.className = `alert-card ${riskClass}`;
      card.innerHTML = `
        <div class="alert-header">
          <div class="alert-title">
            <i class="${iconClass}" style="color: ${iconColor}; font-size: 1.25rem;"></i>
            ${item.project_name}
          </div>
          <span class="alert-badge badge-${item.risk_level}">${item.risk_level} RISK (${item.risk_score !== null ? item.risk_score : 'N/A'})</span>
        </div>
        <div class="alert-meta">
          <strong>Work ID:</strong> ${item.work_id} &nbsp;|&nbsp; <strong>Location:</strong> ${item.location}
        </div>
        <div class="alert-body">
          ${item.message}
        </div>
        <div class="alert-actions">
          <button class="btn-inspect" onclick="inspectProject('${safeId}')">
            Inspect Project Details <i class="ri-arrow-right-line"></i>
          </button>
        </div>
      `;
      fragment.appendChild(card);
    });

    container.appendChild(fragment);
    renderPaginationControls();
  }

  function renderPaginationControls() {
    if (!paginationEl) return;

    const totalPages = Math.ceil(filteredAlerts.length / itemsPerPage) || 1;

    paginationEl.innerHTML = `
      <div style="color: #64748b;">Page <strong>${currentPage}</strong> of <strong>${totalPages}</strong> (${filteredAlerts.length.toLocaleString()} total alerts)</div>
      <div style="display: flex; gap: 0.5rem;">
        <button id="alert-prev" ${currentPage === 1 ? 'disabled' : ''} style="padding: 0.4rem 0.8rem; cursor: pointer; border-radius: 4px; border: 1px solid #cbd5e1; background: #fff;">Previous</button>
        <button id="alert-next" ${currentPage >= totalPages ? 'disabled' : ''} style="padding: 0.4rem 0.8rem; cursor: pointer; border-radius: 4px; border: 1px solid #cbd5e1; background: #fff;">Next</button>
      </div>
    `;

    document.getElementById("alert-prev")?.addEventListener("click", () => {
      if (currentPage > 1) {
        currentPage--;
        renderAlerts();
        window.scrollTo({ top: 0, behavior: 'smooth' });
      }
    });

    document.getElementById("alert-next")?.addEventListener("click", () => {
      if (currentPage < totalPages) {
        currentPage++;
        renderAlerts();
        window.scrollTo({ top: 0, behavior: 'smooth' });
      }
    });
  }

  btnAll?.addEventListener("click", () => applyFilter("ALL"));
  btnHigh?.addEventListener("click", () => applyFilter("HIGH"));
  btnMedium?.addEventListener("click", () => applyFilter("MEDIUM"));
  btnLow?.addEventListener("click", () => applyFilter("LOW"));
});

function inspectProject(workId) {
  if (!workId) return;
  window.location.href = `index.html?id=${encodeURIComponent(workId)}`;
}