document.addEventListener("DOMContentLoaded", async () => {
  try {
    const projects = await loadProjectsData();
    renderRiskDistributionChart(projects);
    renderStateBudgetChart(projects);
  } catch (err) {
    console.error("Error loading analytics data:", err);
  }
});

/**
 * Calculates High, Medium, and Low risk counts from actual dataset
 * and updates the Doughnut chart.
 */
function renderRiskDistributionChart(projects) {
  let highCount = 0;
  let mediumCount = 0;
  let lowCount = 0;

  projects.forEach((p) => {
    const level = String(p.risk_level || "").toUpperCase();
    if (level === "HIGH") highCount++;
    else if (level === "MEDIUM") mediumCount++;
    else lowCount++;
  });

  const ctx = document.getElementById("riskDistributionChart")?.getContext("2d");
  if (!ctx) return;

  new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["High Risk Anomaly", "Medium Priority Watch", "Low Risk / Normal"],
      datasets: [
        {
          data: [highCount, mediumCount, lowCount],
          backgroundColor: ["#ef4444", "#f59e0b", "#10b981"],
          borderWidth: 2,
          borderColor: "#ffffff"
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "bottom",
          labels: {
            usePointStyle: true,
            padding: 20,
            font: { size: 12, family: "sans-serif" }
          }
        },
        tooltip: {
          callbacks: {
            label: function (context) {
              const val = context.raw || 0;
              const total = context.dataset.data.reduce((a, b) => a + b, 0);
              const percentage = total > 0 ? ((val / total) * 100).toFixed(1) : 0;
              return ` ${context.label}: ${val.toLocaleString()} (${percentage}%)`;
            }
          }
        }
      },
      cutout: "65%"
    }
  });
}

/**
 * Groups total sanction amounts by State
 * and plots the Bar chart for top states with multiple colors.
 */
function renderStateBudgetChart(projects) {
  const stateTotals = {};

  projects.forEach((p) => {
    // Read state from project record
    let rawState = p.state || p.state_name || "Unspecified";
    
    // Clean string and formatting
    let cleanState = String(rawState).trim();
    if (!cleanState) cleanState = "Unspecified";

    cleanState = cleanState
      .toLowerCase()
      .split(" ")
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ");

    // Ensure amount calculation in Lakhs
    let amount = Number(p.sanction_amount) || 0;
    if (amount > 100000) {
      amount = amount / 100000;
    }

    if (!stateTotals[cleanState]) {
      stateTotals[cleanState] = 0;
    }
    stateTotals[cleanState] += amount;
  });

  // Sort states by highest budget allocation
  const sortedStates = Object.keys(stateTotals)
    .sort((a, b) => stateTotals[b] - stateTotals[a])
    .slice(0, 10); // Top 10 states

  const labels = sortedStates;
  const dataValues = sortedStates.map((st) => Math.round(stateTotals[st]));

  // Canvas targeting fallback logic (supports both stateBudgetChart and categoryBudgetChart IDs)
  const ctx = (
    document.getElementById("stateBudgetChart") || 
    document.getElementById("categoryBudgetChart")
  )?.getContext("2d");

  if (!ctx) return;

  // Distinct theme bar colors
  const barColors = [
    "#3b82f6", "#10b981", "#f59e0b", "#ef4444", 
    "#8b5cf6", "#ec4899", "#06b6d4", "#6366f1",
    "#14b8a6", "#f97316"
  ];

  new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Budget Allocation (₹ Lakhs)",
          data: dataValues,
          backgroundColor: barColors.slice(0, labels.length),
          borderRadius: 6
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: function (context) {
              return ` Budget: ₹ ${context.parsed.y.toLocaleString("en-IN")} Lakhs`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { font: { size: 11 } }
        },
        y: {
          beginAtZero: true,
          grid: { color: "#f1f5f9" },
          ticks: {
            callback: function (value) {
              return "₹ " + value.toLocaleString("en-IN") + " L";
            }
          }
        }
      }
    }
  });
}