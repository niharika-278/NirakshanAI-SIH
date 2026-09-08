// Data Loader Layer - Direct Fresh Fetching without Local Caching
const RESULTS_JSON_PATH = "data/results.json";

let memoryCache = null;

/**
 * Loads and normalizes project data directly from JSON file.
 * Appends a timestamp query to prevent browser HTTP caching during development.
 */
async function loadProjectsData() {
  if (memoryCache) {
    return memoryCache;
  }

  try {
    // Cache-busting query parameter attached to always fetch latest JSON
    const response = await fetch(`${RESULTS_JSON_PATH}?t=${Date.now()}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch data: ${response.statusText}`);
    }

    const rawData = await response.json();
    
    // Support both direct array format or wrapped { projects: [...] } structure
    const projectsArray = Array.isArray(rawData) 
      ? rawData 
      : (rawData.projects || rawData.data || []);

    memoryCache = projectsArray.map(normalizeProject);
    return memoryCache;
  } catch (error) {
    console.error("Error loading project dataset:", error);
    throw error;
  }
}

/**
 * Normalizes individual project records to ensure consistent property access across the app.
 */
function normalizeProject(project) {
  const sanitize = (val) =>
    val !== undefined && val !== null ? val : "";

  // Status
  const rawStatus = String(
    project.work_completion_status || project.status || ""
  ).toUpperCase();

  const normalizedStatus =
    rawStatus.includes("COMPLET") && !rawStatus.includes("NOT")
      ? "COMPLETED"
      : "NOT COMPLETED";

  // Risk
  const riskScore =
    project.risk_score !== undefined
      ? Number(project.risk_score)
      : null;

  let riskLevel = String(project.risk_level || "").toUpperCase();

  if (!["HIGH", "MEDIUM", "LOW", "CRITICAL"].includes(riskLevel)) {
    if (riskScore >= 70) riskLevel = "HIGH";
    else if (riskScore >= 40) riskLevel = "MEDIUM";
    else riskLevel = "LOW";
  }

  // Expenditure
  // Prefer top-level total_disbursed if available.
  // Otherwise use the value currently present in rule_evidence.
  const totalDisbursed =
    project.total_disbursed_amount ??
    project.rule_evidence?.R1?.evidence?.total_disbursed ??
    project.rule_evidence?.R2?.evidence?.payment_total ??
    0;

  // Rule score is inside explanation
  const ruleScore =
    project.rule_score ??
    project.explanation?.rule_score ??
    project.alert?.evidence?.rule_score ??
    null;

  // Anomaly signal comes from alert_title
  const alerts = [];

  if (project.alert?.alert_title) {
    alerts.push(project.alert.alert_title);
  }

  return {
    work_id: sanitize(project.work_id),

    project_name: sanitize(
      project.work_title ||
      project.project_name ||
      "Unnamed Work"
    ),

    state: sanitize(project.state),
    district: sanitize(project.district),

    category: sanitize(
      project.category ||
      project.work_category ||
      "General"
    ),

    sanction_amount: Number(project.sanction_amount || 0),

    total_disbursed: Number(totalDisbursed),

    status: normalizedStatus,

    risk_score: riskScore,
    risk_level: riskLevel,

    ml_anomaly_score:
      project.ml_anomaly_score !== undefined
        ? Number(project.ml_anomaly_score)
        : null,

    rule_score: ruleScore,

    alerts: alerts,

    recommendation: sanitize(
      project.recommended_action ||
      project.alert?.recommended_action ||
      ""
    )
  };
}