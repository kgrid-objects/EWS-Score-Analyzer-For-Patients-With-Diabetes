/**
 * ews_analysis.js
 * ===============
 * Early Warning System (EWS) Diagnostic Accuracy Analyzer
 *
 * Processes patient-level EWS score data to:
 *   1. Compute the highest EWS score per patient across all time-point columns
 *   2. Apply a user-defined threshold to classify patients as high-risk or low-risk
 *   3. Compute diagnostic accuracy metrics (Sensitivity, Specificity, PPV, NPV,
 *      Accuracy with 95% Wilson CI) against a binary mortality outcome
 *   4. Optionally stratify all metrics by a subgroup column
 *
 * Input data format:
 *   An array of plain objects (e.g. parsed from CSV), one object per patient.
 *   Each object should have:
 *     - One or more numeric EWS score fields (e.g. ews_001 … ews_096)
 *     - A binary outcome field: 1 = survived/discharged, 0 = died in hospital
 *     - (Optional) a group field for subgroup analysis (e.g. "Medical", "Surgical")
 *
 * Usage (Node.js):
 *   const { analyzeEWS } = require('./ews_analysis');
 *
 *   const results = analyzeEWS({
 *     data        : rows,          // array of patient objects
 *     scoreCols   : ['ews_001', 'ews_002', ...],  // EWS score column names
 *     outcomeCol  : 'outcome',     // binary outcome column (1=survived, 0=died)
 *     threshold   : 60,            // risk cutoff (default: 60)
 *     groupCol    : 'group',       // optional subgroup column
 *     verbose     : true           // print summary to console (default: true)
 *   });
 *
 * Returns:
 *   {
 *     patients     : [...],   // enriched patient array with highestScore added
 *     metrics      : {...},   // overall diagnostic metrics
 *     groupMetrics : {...},   // per-group metrics (if groupCol provided)
 *   }
 *
 * Compatible with Node.js (CommonJS & ESM) and modern browsers.
 * No external dependencies required.
 */

'use strict';

// ─────────────────────────────────────────────────────────────────────────────
// STATISTICS UTILITIES
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Approximation of the standard normal inverse CDF (probit function).
 * Uses the Abramowitz & Stegun rational approximation (formula 26.2.17).
 * Accurate to ~3×10⁻³ across the full range.
 *
 * @param {number} p - Probability (0 < p < 1)
 * @returns {number} z-score corresponding to p
 */
function normInvCDF(p) {
  if (p <= 0 || p >= 1) throw new RangeError('p must be in (0, 1)');
  const a = [2.515517, 0.802853, 0.010328];
  const b = [1.432788, 0.189269, 0.001308];
  const sign = p < 0.5 ? -1 : 1;
  const pp   = p < 0.5 ? p : 1 - p;
  const t    = Math.sqrt(-2 * Math.log(pp));
  const num  = a[0] + a[1] * t + a[2] * t * t;
  const den  = 1 + b[0] * t + b[1] * t * t + b[2] * t * t * t;
  return sign * (t - num / den);
}

/**
 * Wilson score confidence interval for a proportion.
 *
 * @param {number} successes - Number of successes (e.g. correct predictions)
 * @param {number} n         - Total observations
 * @param {number} [alpha]   - Significance level (default 0.05 → 95% CI)
 * @returns {{ lower: number, upper: number }}
 */
function wilsonCI(successes, n, alpha = 0.05) {
  if (n === 0) return { lower: NaN, upper: NaN };
  const z      = normInvCDF(1 - alpha / 2);
  const p      = successes / n;
  const denom  = 1 + (z * z) / n;
  const center = (p + (z * z) / (2 * n)) / denom;
  const margin = (z * Math.sqrt(p * (1 - p) / n + (z * z) / (4 * n * n))) / denom;
  return {
    lower: Math.max(0, center - margin),
    upper: Math.min(1, center + margin),
  };
}


// ─────────────────────────────────────────────────────────────────────────────
// CORE METRIC COMPUTATION
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Compute diagnostic accuracy metrics for a set of enriched patient objects.
 *
 * @param {object[]} patients  - Array of patient objects, each with:
 *                               { highestScore: number, outcome: number (0|1) }
 * @param {number}   threshold - Score cutoff; score >= threshold → predicted death
 * @returns {object} Metrics object (see below)
 */
function computeMetrics(patients, threshold) {
  let tp = 0, tn = 0, fp = 0, fn = 0;

  for (const p of patients) {
    const predictedDeath = p.highestScore >= threshold;
    const actualDeath    = p.outcome === 0;

    if ( actualDeath &&  predictedDeath) tp++;
    if (!actualDeath && !predictedDeath) tn++;
    if (!actualDeath &&  predictedDeath) fp++;
    if ( actualDeath && !predictedDeath) fn++;
  }

  const n           = patients.length;
  const sensitivity = (tp + fn) > 0 ? tp / (tp + fn) : NaN;
  const specificity = (tn + fp) > 0 ? tn / (tn + fp) : NaN;
  const ppv         = (tp + fp) > 0 ? tp / (tp + fp) : NaN;
  const npv         = (tn + fn) > 0 ? tn / (tn + fn) : NaN;
  const accuracy    = n > 0 ? (tp + tn) / n : NaN;
  const prevalence  = n > 0 ? (tp + fn) / n : NaN;
  const accuracyCI  = wilsonCI(tp + tn, n);

  return {
    n,
    threshold,
    prevalence,
    tp, tn, fp, fn,
    sensitivity,
    specificity,
    ppv,
    npv,
    accuracy,
    accuracyCI,             // { lower, upper }
    confusionMatrix: {      // 2×2 as nested object for clarity
      actualSurvived: { predictedSurvived: tn, predictedDied: fp },
      actualDied:     { predictedSurvived: fn, predictedDied: tp },
    },
  };
}


// ─────────────────────────────────────────────────────────────────────────────
// HIGHEST SCORE EXTRACTION
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Enrich each patient row with their highest EWS score across all score columns.
 *
 * @param {object[]} data       - Raw patient rows (array of plain objects)
 * @param {string[]} scoreCols  - Column names to scan for EWS scores
 * @param {string}   outcomeCol - Column name for binary outcome (1=survived, 0=died)
 * @returns {object[]} Enriched patient array; invalid rows are filtered out
 */
function extractHighestScores(data, scoreCols, outcomeCol) {
  const enriched = [];

  for (const row of data) {
    const scores = scoreCols
      .map(col => parseFloat(row[col]))
      .filter(v => !isNaN(v));

    if (scores.length === 0) continue;         // skip rows with no valid scores

    const outcome = parseInt(row[outcomeCol], 10);
    if (isNaN(outcome)) continue;              // skip rows with missing outcome

    enriched.push({
      ...row,                                  // preserve all original fields
      outcome,                                 // store as parsed integer (overrides string)
      highestScore: Math.max(...scores),
    });
  }

  return enriched;
}


// ─────────────────────────────────────────────────────────────────────────────
// SCORE DISTRIBUTION SUMMARY
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Compute a histogram of highest EWS scores (binned 0–100 in steps of 2.5).
 * Returned as an array of { binStart, binEnd, countSurvived, countDied, countTotal }.
 *
 * @param {object[]} patients - Enriched patient array
 * @param {number}   [bins]   - Number of bins (default 40 → width 2.5)
 * @returns {object[]} Histogram bins
 */
function scoreHistogram(patients, bins = 40) {
  const width  = 100 / bins;
  const counts = Array.from({ length: bins }, (_, i) => ({
    binStart:      parseFloat((i * width).toFixed(4)),
    binEnd:        parseFloat(((i + 1) * width).toFixed(4)),
    countSurvived: 0,
    countDied:     0,
    countTotal:    0,
  }));

  for (const p of patients) {
    const idx = Math.min(Math.floor(p.highestScore / width), bins - 1);
    counts[idx].countTotal++;
    if (p.outcome === 1) counts[idx].countSurvived++;
    else                 counts[idx].countDied++;
  }

  return counts;
}


// ─────────────────────────────────────────────────────────────────────────────
// CONSOLE SUMMARY PRINTER
// ─────────────────────────────────────────────────────────────────────────────

function pct(v, dec = 2) {
  return isNaN(v) ? 'N/A' : (v * 100).toFixed(dec) + '%';
}

function printMetricsBlock(label, m) {
  const ci = m.accuracyCI;
  const ciStr = (isNaN(ci.lower) || isNaN(ci.upper))
    ? 'N/A'
    : `${pct(ci.lower)} – ${pct(ci.upper)}`;

  console.log(`\n  [${label}]`);
  console.log(`  Patients      : ${m.n.toLocaleString()}`);
  console.log(`  Prevalence    : ${pct(m.prevalence)}`);
  console.log(`  Accuracy      : ${pct(m.accuracy)}  (95% CI: ${ciStr})`);
  console.log(`  Sensitivity   : ${pct(m.sensitivity)}`);
  console.log(`  Specificity   : ${pct(m.specificity)}`);
  console.log(`  PPV           : ${pct(m.ppv)}`);
  console.log(`  NPV           : ${pct(m.npv)}`);
  console.log(`  TP / TN / FP / FN : ${m.tp} / ${m.tn} / ${m.fp} / ${m.fn}`);
}

function printSummary(metrics, groupMetrics, threshold) {
  const sep = '='.repeat(62);
  console.log(`\n${sep}`);
  console.log(`  EWS DIAGNOSTIC ACCURACY  |  Threshold: ${threshold}`);
  console.log(sep);

  printMetricsBlock('OVERALL', metrics);

  for (const [grp, gm] of Object.entries(groupMetrics)) {
    printMetricsBlock(`GROUP: ${grp}`, gm);
  }

  const cm = metrics.confusionMatrix;
  console.log('\n  Confusion Matrix (Overall)');
  console.log('                       Pred Survived   Pred Died');
  console.log(`  Actual Survived  :   ${String(cm.actualSurvived.predictedSurvived).padStart(13)}   ${String(cm.actualSurvived.predictedDied).padStart(9)}`);
  console.log(`  Actual Died      :   ${String(cm.actualDied.predictedSurvived).padStart(13)}   ${String(cm.actualDied.predictedDied).padStart(9)}`);
  console.log(`${sep}\n`);
}


// ─────────────────────────────────────────────────────────────────────────────
// MAIN EXPORTED FUNCTION
// ─────────────────────────────────────────────────────────────────────────────

/**
 * analyzeEWS — Main entry point.
 *
 * @param {object}   options
 * @param {object[]} options.data        - Array of patient row objects
 * @param {string[]} options.scoreCols   - EWS score column names
 * @param {string}   options.outcomeCol  - Binary outcome column (1=survived, 0=died)
 * @param {number}   [options.threshold] - Risk cutoff score (default: 60)
 * @param {string}   [options.groupCol]  - Optional subgroup column name
 * @param {boolean}  [options.verbose]   - Print summary to console (default: true)
 *
 * @returns {{
 *   patients     : object[],   // enriched patient array with highestScore
 *   metrics      : object,     // overall diagnostic metrics
 *   groupMetrics : object,     // per-group metrics keyed by group value
 *   histogram    : object[],   // overall score histogram bins
 * }}
 */
function analyzeEWS({
  data,
  scoreCols,
  outcomeCol,
  threshold = 60,
  groupCol  = null,
  verbose   = true,
}) {
  // ── Validate inputs ──────────────────────────────────────────────────────
  if (!Array.isArray(data) || data.length === 0) {
    throw new Error('analyzeEWS: "data" must be a non-empty array of patient objects.');
  }
  if (!Array.isArray(scoreCols) || scoreCols.length === 0) {
    throw new Error('analyzeEWS: "scoreCols" must be a non-empty array of column name strings.');
  }
  if (typeof outcomeCol !== 'string' || !outcomeCol) {
    throw new Error('analyzeEWS: "outcomeCol" must be a non-empty string.');
  }
  if (typeof threshold !== 'number' || threshold < 0 || threshold > 100) {
    throw new Error('analyzeEWS: "threshold" must be a number between 0 and 100.');
  }

  // ── Step 1: Extract highest score per patient ────────────────────────────
  const patients = extractHighestScores(data, scoreCols, outcomeCol);

  if (patients.length === 0) {
    throw new Error('analyzeEWS: No valid patient rows after parsing. Check scoreCols and outcomeCol.');
  }

  // ── Step 2: Overall metrics ──────────────────────────────────────────────
  const metrics = computeMetrics(patients, threshold);

  // ── Step 3: Per-group metrics (optional) ─────────────────────────────────
  const groupMetrics = {};
  if (groupCol) {
    const groups = {};
    for (const p of patients) {
      const g = String(p[groupCol] ?? 'Unknown');
      if (!groups[g]) groups[g] = [];
      groups[g].push(p);
    }
    for (const [grp, gPatients] of Object.entries(groups)) {
      groupMetrics[grp] = computeMetrics(gPatients, threshold);
    }
  }

  // ── Step 4: Score distribution histogram ─────────────────────────────────
  const histogram = scoreHistogram(patients);

  // ── Step 5: Console output ───────────────────────────────────────────────
  if (verbose) {
    printSummary(metrics, groupMetrics, threshold);
  }

  return { patients, metrics, groupMetrics, histogram };
}


// ─────────────────────────────────────────────────────────────────────────────
// EXPORTS  (CommonJS + ESM compatible)
// ─────────────────────────────────────────────────────────────────────────────

if (typeof module !== 'undefined' && module.exports) {
  // CommonJS (Node.js require)
  module.exports = { analyzeEWS, computeMetrics, wilsonCI, normInvCDF, scoreHistogram };
} else if (typeof window !== 'undefined') {
  // Browser global
  window.EWSAnalysis = { analyzeEWS, computeMetrics, wilsonCI, normInvCDF, scoreHistogram };
}


// ─────────────────────────────────────────────────────────────────────────────
// DEMO  (runs only when executed directly: node ews_analysis.js)
// ─────────────────────────────────────────────────────────────────────────────

if (typeof require !== 'undefined' && require.main === module) {
  const fs  = require('fs');
  const path = require('path');

  // Try to load the sample CSV if available
  const csvPath = path.join(__dirname, 'ews_sample_data.csv');

  if (fs.existsSync(csvPath)) {
    console.log(`\nLoading sample data from: ${csvPath}`);

    const raw  = fs.readFileSync(csvPath, 'utf8');
    const lines = raw.trim().split('\n');
    const headers = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g, ''));

    function parseCSVLine(line) {
      const fields = [];
      let current = '', inQuotes = false;
      for (let i = 0; i < line.length; i++) {
        const ch = line[i];
        if (ch === '"') { inQuotes = !inQuotes; }
        else if (ch === ',' && !inQuotes) { fields.push(current); current = ''; }
        else { current += ch; }
      }
      fields.push(current);
      return fields;
    }

    const rows = lines.slice(1).map(line => {
      const values = parseCSVLine(line);
      const obj = {};
      headers.forEach((h, i) => { obj[h] = (values[i] ?? '').trim(); });
      return obj;
    });

    const scoreCols = headers.filter(h => h.startsWith('ews_'));
    console.log(`Patients: ${rows.length.toLocaleString()}  |  Score columns: ${scoreCols.length}`);

    const results = analyzeEWS({
      data       : rows,
      scoreCols,
      outcomeCol : 'outcome',
      threshold  : 60,
      groupCol   : 'group',
      verbose    : true,
    });

    // Show a sample of the histogram
    console.log('Score Distribution (bins with patients):');
    results.histogram
      .filter(b => b.countTotal > 0)
      .forEach(b => {
        const bar = '█'.repeat(Math.ceil(b.countTotal / 5));
        console.log(
          `  ${String(b.binStart.toFixed(1)).padStart(5)}–${String(b.binEnd.toFixed(1)).padEnd(5)} ` +
          `| ${String(b.countTotal).padStart(4)} patients  ${bar}`
        );
      });

  } else {
    // Minimal inline demo if no CSV present
    console.log('\nNo CSV found — running inline demo with synthetic data.\n');

    const demoData = [
      { ews_1: '72', ews_2: '80', ews_3: '65', outcome: '0', group: 'Medical'  },
      { ews_1: '20', ews_2: '35', ews_3: '28', outcome: '1', group: 'Medical'  },
      { ews_1: '45', ews_2: '55', ews_3: '62', outcome: '1', group: 'Medical'  },
      { ews_1: '88', ews_2: '91', ews_3: '70', outcome: '0', group: 'Surgical' },
      { ews_1: '15', ews_2: '22', ews_3: '18', outcome: '1', group: 'Surgical' },
      { ews_1: '60', ews_2: '58', ews_3: '63', outcome: '1', group: 'Surgical' },
    ];

    analyzeEWS({
      data      : demoData,
      scoreCols : ['ews_1', 'ews_2', 'ews_3'],
      outcomeCol: 'outcome',
      threshold : 60,
      groupCol  : 'group',
      verbose   : true,
    });
  }
}
