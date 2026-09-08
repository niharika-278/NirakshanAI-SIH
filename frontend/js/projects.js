document.addEventListener("DOMContentLoaded", async () => {
  const tableBody = document.getElementById("projects-table-body");
  const searchInput = document.getElementById("search-input");
  const riskFilter = document.getElementById("risk-filter");
  const statusFilter = document.getElementById("status-filter");
  const projectCountEl = document.getElementById("project-count");

  let allProjects = [];
  let filteredProjects = [];
  let currentPage = 1;
  const rowsPerPage = 20;
  let debounceTimer = null;

  // Show Skeleton / Spinner while fetching
  if (tableBody) {
    tableBody.innerHTML = `
      <tr>
        <td colspan="7" style="text-align:center; padding:2rem; color:#2563eb;">
          <i class="ri-loader-4-line" style="font-size:1.8rem; display:inline-block; animation:spin 1s linear infinite;"></i>
          <div style="margin-top:0.5rem; font-weight:600;">Fetching Records...</div>
        </td>
      </tr>`;
  }

  try {
    // Non-blocking fetch
    allProjects = await loadProjectsData();
    filteredProjects = allProjects;
    
    // Fast First Render
    renderPage();
  } catch (err) {
    if (tableBody) {
      tableBody.innerHTML = `<tr><td colspan="7" style="text-align:center; color:#dc2626; padding:1.5rem;">Failed to load projects. Check console.</td></tr>`;
    }
  }

  function renderPage() {
    if (!tableBody) return;
    tableBody.innerHTML = "";

    if (projectCountEl) projectCountEl.textContent = filteredProjects.length.toLocaleString();

    if (filteredProjects.length === 0) {
      tableBody.innerHTML = `<tr><td colspan="7" style="text-align:center; color:#64748b; padding:1.5rem;">No matching projects found.</td></tr>`;
      renderPaginationControls(0);
      return;
    }

    const startIndex = (currentPage - 1) * rowsPerPage;
    const paginatedItems = filteredProjects.slice(startIndex, startIndex + rowsPerPage);

    const fragment = document.createDocumentFragment();

    paginatedItems.forEach((p) => {
      const tr = document.createElement("tr");

      const riskBadgeClass = p.risk_level === "HIGH" ? "badge-high" : p.risk_level === "MEDIUM" ? "badge-medium" : "badge-low";
      const sanctionFormatted = typeof p.sanction_amount === 'number' && p.sanction_amount > 0 ? `₹ ${(p.sanction_amount / 100000).toFixed(2)} L` : 'N/A';
      const safeId = String(p.work_id).replace(/'/g, "\\'");

      tr.innerHTML = `
        <td><strong>${p.work_id}</strong></td>
        <td>
          <div style="font-weight: 600; color: #0f172a;">${p.project_name}</div>
          <div style="font-size: 0.75rem; color: #64748b;">${p.category}</div>
        </td>
        <td>${p.district ? p.district + ', ' : ''}${p.state || ''}</td>
        <td>${sanctionFormatted}</td>
        <td><span class="badge ${riskBadgeClass}">${p.risk_level} (${p.risk_score !== null ? p.risk_score : 'N/A'})</span></td>
        <td>
          <button class="action-btn" onclick="openProjectInDashboard('${safeId}')">
            Inspect <i class="ri-arrow-right-line"></i>
          </button>
        </td>
      `;
      fragment.appendChild(tr);
    });

    tableBody.appendChild(fragment);
    renderPaginationControls(filteredProjects.length);
  }

  function renderPaginationControls(totalItems) {
    let paginationContainer = document.getElementById("pagination-container");
    if (!paginationContainer) {
      paginationContainer = document.createElement("div");
      paginationContainer.id = "pagination-container";
      paginationContainer.style.cssText = "display: flex; justify-content: space-between; align-items: center; margin-top: 1rem; font-size: 0.85rem;";
      tableBody.parentElement.after(paginationContainer);
    }

    const totalPages = Math.ceil(totalItems / rowsPerPage) || 1;

    paginationContainer.innerHTML = `
      <div style="color: #64748b;">Page <strong>${currentPage}</strong> of <strong>${totalPages}</strong></div>
      <div style="display: flex; gap: 0.5rem;">
        <button id="prev-page" ${currentPage === 1 ? 'disabled' : ''} style="padding: 0.4rem 0.8rem; cursor: pointer; border-radius: 4px; border: 1px solid #cbd5e1; background: #fff;">Previous</button>
        <button id="next-page" ${currentPage >= totalPages ? 'disabled' : ''} style="padding: 0.4rem 0.8rem; cursor: pointer; border-radius: 4px; border: 1px solid #cbd5e1; background: #fff;">Next</button>
      </div>
    `;

    document.getElementById("prev-page")?.addEventListener("click", () => {
      if (currentPage > 1) {
        currentPage--;
        renderPage();
      }
    });

    document.getElementById("next-page")?.addEventListener("click", () => {
      if (currentPage < totalPages) {
        currentPage++;
        renderPage();
      }
    });
  }

  function applyFilters() {
    const searchTerm = searchInput ? searchInput.value.trim().toLowerCase() : "";
    const selectedRisk = riskFilter ? riskFilter.value : "ALL";
    const selectedStatus = statusFilter ? statusFilter.value : "ALL";

    filteredProjects = allProjects.filter((p) => {
      // 1. Search Filter
      const matchesSearch = !searchTerm ||
        String(p.work_id || "").toLowerCase().includes(searchTerm) ||
        String(p.project_name || "").toLowerCase().includes(searchTerm) ||
        String(p.district || "").toLowerCase().includes(searchTerm) ||
        String(p.state || "").toLowerCase().includes(searchTerm);

      // 2. Risk Filter
      const matchesRisk = selectedRisk === "ALL" || p.risk_level === selectedRisk;

      // 3. Exact Status Match
      let matchesStatus = true;
      if (selectedStatus !== "ALL") {
        const rawStatus = String(
          p.status || p.raw_status || p.work_completion_status || p.completion_status || p.work_status || ""
        ).trim().toUpperCase();

        if (selectedStatus === "COMPLETED") {
          matchesStatus = rawStatus.includes("COMPLETED") && !rawStatus.includes("NOT");
        } else if (selectedStatus === "NOT COMPLETED") {
          matchesStatus = rawStatus.includes("NOT") || !rawStatus.includes("COMPLETED");
        }
      }

      return matchesSearch && matchesRisk && matchesStatus;
    });

    currentPage = 1;
    renderPage();
  }

  // Event Listeners Attached Properly
  if (searchInput) {
    searchInput.addEventListener("input", () => {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(applyFilters, 250);
    });
  }

  if (riskFilter) riskFilter.addEventListener("change", applyFilters);
  if (statusFilter) statusFilter.addEventListener("change", applyFilters);
});

// Global Function
function openProjectInDashboard(workId) {
  if (!workId) return;
  window.location.href = `index.html?id=${encodeURIComponent(workId)}`;
}