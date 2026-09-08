document.addEventListener("DOMContentLoaded", async () => {
  const searchBtn = document.getElementById("search-btn");
  const projectInput = document.getElementById("project-id-input");

  try {
    const projects = await loadProjectsData();

    if (!projects || projects.length === 0) {
      showErrorState("No projects found in data/results.json");
      return;
    }

    // Read ID from URL parameters
    const urlParams = new URLSearchParams(window.location.search);
    const rawUrlProjectId = urlParams.get("id");

    if (rawUrlProjectId) {
      const decodedId = decodeURIComponent(rawUrlProjectId).trim();
      if (projectInput) projectInput.value = decodedId;
      await renderProjectDetails(decodedId);
    } else {
      // Load initial project from JSON file
      const initialId = projects[0].work_id;
      if (projectInput) projectInput.value = initialId;
      await renderProjectDetails(initialId);
    }
  } catch (err) {
    console.error("Dashboard init error:", err);
    showErrorState("Error loading project data from data/results.json. Please check if file exists.");
  }

  // Search Button Click Event
  if (searchBtn) {
    searchBtn.addEventListener("click", () => {
      if (projectInput) {
        const inputVal = projectInput.value.trim();
        renderProjectDetails(inputVal);
      }
    });
  }

  // Search Input Enter Key Event
  if (projectInput) {
    projectInput.addEventListener("keypress", (e) => {
      if (e.key === "Enter") {
        const inputVal = projectInput.value.trim();
        renderProjectDetails(inputVal);
      }
    });
  }

  // Sidebar Toggle Handling & Icon Swap
  const toggleBtn = document.getElementById("sidebar-toggle");
  if (toggleBtn) {
    toggleBtn.addEventListener("click", () => {
      document.body.classList.toggle("sidebar-collapsed");
      const icon = toggleBtn.querySelector("i");
      if (icon) {
        icon.className = document.body.classList.contains("sidebar-collapsed")
          ? "ri-menu-unfold-line"
          : "ri-menu-fold-line";
      }
    });
  }
});

function selectDemoId(id) {
  const projectInput = document.getElementById("project-id-input");
  if (projectInput) projectInput.value = id;
  renderProjectDetails(id);
}

function navigateTo(pageUrl) {
  if (!pageUrl.startsWith('./') && !pageUrl.startsWith('http')) {
    window.location.href = './' + pageUrl;
  } else {
    window.location.href = pageUrl;
  }
}

async function renderProjectDetails(projectId) {
  const container = document.getElementById("project-details-container");
  if (!container) return;

  if (!projectId) {
    container.style.display = "none";
    return;
  }

  try {
    const projects = await loadProjectsData();
    const cleanSearchId = String(projectId).trim().toUpperCase();

    // Matching Work ID
    const project = projects.find(p => String(p.work_id).trim().toUpperCase() === cleanSearchId);

    if (!project) {
      container.style.display = "block";
      container.innerHTML = `
        <div style="text-align: center; color: #dc2626; padding: 2rem; background: #fff; border-radius: 8px; border: 1px solid #fee2e2;">
          <i class="ri-error-warning-line" style="font-size: 2.5rem; display: block; margin-bottom: 0.5rem;"></i>
          <h3>Project Record "${projectId}" Not Found</h3>
          <p style="color: #64748b; font-size: 0.85rem;">Please check the Work ID in data/results.json</p>
        </div>
      `;
      return;
    }

    // Dynamic Category Handling
    const categoryName = project.category || project.work_category || project.sector || project.head_of_development || "General";

    // Amount Formatter (Handles raw numbers vs pre-formatted string)
    const formatAmount = (amt) => {
      if (typeof amt === 'number') {
        // If amount is raw rupees (> 10000), divide by 100000. Else assume already in Lakhs.
        const lakhVal = amt > 10000 ? amt / 100000 : amt;
        return `₹ ${lakhVal.toFixed(2)} Lakhs`;
      }
      return amt || 'N/A';
    };

    const sanctionFormatted = formatAmount(project.sanction_amount);
    const disbursedFormatted = formatAmount(project.total_disbursed);

    // Dynamic Risk Badge Color
    const riskLevel = String(project.risk_level || "LOW").toUpperCase();
    let badgeBg = "#f0fdf4";
    let badgeColor = "#166534";
    
    if (riskLevel === "HIGH") {
      badgeBg = "#fef2f2";
      badgeColor = "#dc2626";
    } else if (riskLevel === "MEDIUM") {
      badgeBg = "#fffbeb";
      badgeColor = "#b45309";
    }

    container.style.display = "block";
    container.innerHTML = `
      <div style="background: #fff; padding: 1.5rem; border-radius: 8px; border: 1px solid #e2e8f0; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1rem; border-bottom: 1px solid #f1f5f9; padding-bottom: 1rem; flex-wrap: wrap; gap: 0.5rem;">
          <div>
            <h2 style="margin: 0 0 0.25rem 0; font-size: 1.25rem; color: #0f172a;">${project.project_name || 'Unnamed Project'}</h2>
            <div style="font-size: 0.85rem; color: #64748b;">
              ID: <strong>${project.work_id}</strong> | Location: ${project.district || 'N/A'}, ${project.state || 'N/A'} | Category: ${categoryName}
            </div>
          </div>
          <div class="badge-official" style="padding: 0.4rem 0.8rem; font-weight: 600; background: ${badgeBg}; color: ${badgeColor}; border-radius: 4px;">
            <i class="ri-shield-alert-line"></i> ${riskLevel} RISK (${project.risk_score !== null && project.risk_score !== undefined ? project.risk_score : 'N/A'})
          </div>
        </div>

        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; background: #f8fafc; padding: 1rem; border-radius: 6px;">
          <div>
            <span style="font-size: 0.75rem; color: #64748b; text-transform: uppercase; display: block;">Sanctioned / Expenditure</span>
            <strong style="font-size: 0.95rem; color: #0f172a;">${sanctionFormatted} / ${disbursedFormatted}</strong>
          </div>
          
          <div>
            <span style="font-size: 0.75rem; color: #64748b; text-transform: uppercase; display: block;">ML Anomaly / Rule Score</span>
            <strong style="font-size: 0.95rem; color: #0f172a;">ML: ${project.ml_anomaly_score !== null && project.ml_anomaly_score !== undefined ? project.ml_anomaly_score : 'N/A'} | Rule: ${project.rule_score !== null && project.rule_score !== undefined ? project.rule_score : 'N/A'}</strong>
          </div>
        </div>

        <div style="margin-bottom: 1.25rem;">
          <h4 style="color: #dc2626; margin: 0 0 0.5rem 0; font-size: 0.9rem;">
            <i class="ri-error-warning-line"></i> Detected Anomaly Signals
          </h4>
          <ul style="padding-left: 1.25rem; font-size: 0.85rem; color: #881337; display: flex; flex-direction: column; gap: 0.35rem; margin: 0;">
            ${
              project.alerts && project.alerts.length > 0 
                ? project.alerts.map(item => {
                    const text = typeof item === 'string' ? item : (item.reason || item.indicator || item.message || 'Anomaly flagged');
                    return `<li>${text}</li>`;
                  }).join('')
                : '<li style="color: #64748b;">No critical anomalies detected for this project.</li>'
            }
          </ul>
        </div>

        <div style="padding: 0.85rem; background: #eff6ff; border-left: 4px solid #1d4ed8; border-radius: 4px;">
          <h4 style="color: #1e40af; margin: 0 0 0.25rem 0; font-size: 0.9rem;">
            <i class="ri-lightbulb-line"></i> Decision Support Recommendation
          </h4>
          <p style="font-size: 0.85rem; color: #1e3a8a; margin: 0;">${project.recommendation || 'No specific action required.'}</p>
        </div>
      </div>
    `;

    container.scrollIntoView({ behavior: 'smooth', block: 'start' });

  } catch (err) {
    console.error("Error rendering details:", err);
    showErrorState("Error rendering project record details.");
  }
}

function showErrorState(message) {
  const container = document.getElementById("project-details-container");
  if (container) {
    container.style.display = "block";
    container.innerHTML = `
      <div style="padding: 1.5rem; background: #ffffff; border-radius: 8px; border: 1px solid #fee2e2; color: #dc2626; text-align: center;">
        <i class="ri-error-warning-line" style="font-size: 1.8rem; display: block; margin-bottom: 0.5rem;"></i>
        ${message}
      </div>
    `;
  }
}