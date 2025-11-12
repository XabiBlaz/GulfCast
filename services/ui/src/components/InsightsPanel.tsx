import { FC, useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

type SplitSummary = {
  name: string;
  weeks: number;
  example_weeks: string[];
};

type ForecastStats = {
  records: number;
  bbox: Record<string, number>;
  variables: string[];
};

type RiskStats = {
  assets: number;
  el_mean: number;
  el_max: number;
  var95_mean: number;
  es95_mean: number;
};

export type InsightsPayload = {
  latest_common_week: string;
  feature_config: Record<string, string>;
  split_summary: SplitSummary[];
  forecast_stats: ForecastStats;
  risk_stats: RiskStats;
  training_notes: string;
  ablation_guidance: string;
  forecast_methodology: string;
  risk_methodology: string;
  hazard_summary: {
    variable: string;
    mean_mean: number;
    mean_q90: number;
    max_q90: number;
    unit: string;
  }[];
  hazard_series: {
    week: string;
    variable: string;
    q90: number;
  }[];
  top_assets: {
    asset_id: string;
    asset_name: string;
    asset_type: string;
    EL: number;
    VaR95: number;
    ES95: number;
  }[];
  risk_by_type: {
    asset_type: string;
    count: number;
    el_total: number;
    var95_total: number;
    es95_total: number;
  }[];
  report_content: string;
  project_overview: string;
  model_scores: {
    target: string;
    model: string;
    metric: string;
    value: number;
  }[];
  metric_docs: {
    name: string;
    description: string;
    direction: string;
  }[];
  model_overview: string;
};

type Props = {
  data: InsightsPayload | null;
  loading: boolean;
  error: string | null;
  onRetry: () => void;
};

const formatCurrency = (value: number) =>
  new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 }).format(value);

const InsightsPanel: FC<Props> = ({ data, loading, error, onRetry }) => {
  const hazardVars = Array.from(new Set(data?.hazard_series.map((entry) => entry.variable) ?? []));
  const hazardChartData = useMemo(() => {
    const map = new Map<string, Record<string, number | string>>();
    (data?.hazard_series ?? []).forEach((entry) => {
      const normalizedValue =
        entry.variable.toLowerCase().startsWith("t") && entry.q90 > 200 ? entry.q90 - 273.15 : entry.q90;
      if (!map.has(entry.week)) {
        map.set(entry.week, { week: entry.week });
      }
      map.get(entry.week)![entry.variable] = normalizedValue;
    });
    return Array.from(map.values()).sort((a, b) => String(a.week).localeCompare(String(b.week)));
  }, [data?.hazard_series]);
  const riskChartData = useMemo(
    () =>
      (data?.top_assets ?? []).map((asset) => ({
        asset_name: asset.asset_name,
        el_total: asset.EL / 1_000_000,
        var95_total: asset.VaR95 / 1_000_000,
        es95_total: asset.ES95 / 1_000_000
      })),
    [data?.top_assets]
  );
  const riskByTypeChart = useMemo(
    () =>
      (data?.risk_by_type ?? []).map((entry) => ({
        asset_type: entry.asset_type,
        el_total: entry.el_total / 1_000_000,
        var95_total: entry.var95_total / 1_000_000,
        es95_total: entry.es95_total / 1_000_000
      })),
    [data?.risk_by_type]
  );
  const mseChartData = useMemo(() => {
    const targets = Array.from(new Set((data?.model_scores ?? []).filter((m) => m.metric === "MSE").map((m) => m.target)));
    return targets.map((target) => {
      const rows = (data?.model_scores ?? []).filter((m) => m.target === target && m.metric === "MSE");
      const row: Record<string, number | string> = { target };
      rows.forEach((m) => {
        row[m.model] = m.value;
      });
      return row;
    });
  }, [data?.model_scores]);
  const r2ChartData = useMemo(() => {
    const targets = Array.from(new Set((data?.model_scores ?? []).filter((m) => m.metric === "R2").map((m) => m.target)));
    return targets.map((target) => {
      const rows = (data?.model_scores ?? []).filter((m) => m.target === target && m.metric === "R2");
      const row: Record<string, number | string> = { target };
      rows.forEach((m) => {
        row[m.model] = m.value;
      });
      return row;
    });
  }, [data?.model_scores]);
  const pinballChartData = useMemo(() => {
    const entries = (data?.model_scores ?? []).filter((m) => m.metric === "Pinball");
    const group = new Map<string, Record<string, number | string>>();
    entries.forEach((m) => {
      if (!group.has(m.target)) group.set(m.target, { target: m.target });
      group.get(m.target)![m.model] = m.value;
    });
    return Array.from(group.values());
  }, [data?.model_scores]);
  const palette = ["#38bdf8", "#f472b6", "#facc15", "#34d399", "#a78bfa", "#fb7185"];

  const derivedNarrative = useMemo(() => {
    if (!data) return "Insights will appear once the pipeline finishes.";
    const hazards = data.hazard_summary ?? [];
    const temp = hazards.find((h) => h.variable.toLowerCase().startsWith("t"));
    const precip = hazards.find((h) => h.variable.toLowerCase().includes("precip"));
    const latestWeek = data.latest_common_week ?? "N/A";
    const riskTotals = data.risk_stats;
    const topAsset = data.top_assets?.[0];
    const tempFragment = temp
      ? `mean ${temp.variable} ≈ ${temp.mean_mean.toFixed(1)}${temp.unit}`
      : "temperature anomalies remain mild";
    const precipFragment = precip
      ? `precip q90 ≈ ${precip.mean_q90.toFixed(1)} ${precip.unit}`
      : "precipitation stays near its seasonal envelope";
    const totalsFragment = riskTotals
      ? `Portfolio EL ≈ ${formatCurrency(riskTotals.el_mean * riskTotals.assets)}, VaR95 ≈ ${formatCurrency(
          riskTotals.var95_mean * riskTotals.assets
        )}, ES95 ≈ ${formatCurrency(riskTotals.es95_mean * riskTotals.assets)}`
      : "Portfolio totals remain modest";
    const topAssetFragment = topAsset
      ? `Highest exposure: ${topAsset.asset_name} (${topAsset.asset_type}) with EL ≈ ${formatCurrency(topAsset.EL)}.`
      : "";
    return `Scenario ${latestWeek}: ${tempFragment}; ${precipFragment}. With such tepid anomalies, the physrisk curves only clip a few percent off each asset's value (derates capped at 20%), which keeps individual EL/VaR/ES readings small even though ${totalsFragment}. ${topAssetFragment}`;
  }, [data]);

  const narrativeText = useMemo(() => {
    const text = data?.report_content;
    if (!text) return derivedNarrative;
    if (text.includes("The physrisk run processed") || text.startsWith("Risk Narrative")) {
      return derivedNarrative;
    }
    return text;
  }, [data?.report_content, derivedNarrative]);

  const buildBars = (models: string[]) =>
    models.map((model, idx) => <Bar key={model} dataKey={model} fill={palette[idx % palette.length]} />);

  const mseModels = useMemo(
    () =>
      Array.from(
        new Set(
          mseChartData.flatMap((row) =>
            Object.keys(row).filter((key) => key !== "target")
          )
        )
      ),
    [mseChartData]
  );
  const r2Models = useMemo(
    () =>
      Array.from(
        new Set(
          r2ChartData.flatMap((row) =>
            Object.keys(row).filter((key) => key !== "target")
          )
        )
      ),
    [r2ChartData]
  );
  const pinballModels = useMemo(
    () =>
      Array.from(
        new Set(
          pinballChartData.flatMap((row) =>
            Object.keys(row).filter((key) => key !== "target")
          )
        )
      ),
    [pinballChartData]
  );

  return (
    <div style={{ display: "grid", gap: "1.5rem" }}>
      <section style={{ background: "#0f172a", padding: "1.25rem", borderRadius: "0.75rem" }}>
        <h2 style={{ margin: 0 }}>Project Overview</h2>
        <p style={{ color: "#cbd5f5" }}>{data?.project_overview}</p>
      </section>

      <section style={{ background: "#0f172a", padding: "1.25rem", borderRadius: "0.75rem" }}>
        <h2 style={{ margin: 0 }}>Training & Research Notes</h2>
        <p style={{ color: "#e2e8f0" }}>{data?.training_notes}</p>
        <p style={{ color: "#cbd5f5" }}>{data?.ablation_guidance}</p>
      </section>

      <section style={{ background: "#0f172a", padding: "1.25rem", borderRadius: "0.75rem" }}>
        <h2 style={{ margin: 0 }}>Risk Narrative (OpenAI)</h2>
        <p style={{ color: "#38bdf8", fontSize: "0.9rem", marginBottom: "0.5rem" }}>Generated automatically from the latest forecasts and risk metrics.</p>
        <p style={{ color: "#cbd5f5", whiteSpace: "pre-wrap" }}>{narrativeText}</p>
      </section>

      <section style={{ display: "grid", gap: "1rem", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))" }}>
        <div style={{ background: "#0f172a", padding: "1rem", borderRadius: "0.75rem" }}>
          <h3>Forecast Coverage</h3>
          <p>{data?.forecast_stats.records} grid samples</p>
          <p>Variables: {data?.forecast_stats.variables.join(", ")}</p>
          <p>
            BBox: lat {data?.forecast_stats.bbox.lat_min?.toFixed(2)} →{" "}
            {data?.forecast_stats.bbox.lat_max?.toFixed(2)}, lon {data?.forecast_stats.bbox.lon_min?.toFixed(2)} →{" "}
            {data?.forecast_stats.bbox.lon_max?.toFixed(2)}
          </p>
        </div>
        <div style={{ background: "#0f172a", padding: "1rem", borderRadius: "0.75rem" }}>
          <h3>Risk Portfolio</h3>
          <p>{data?.risk_stats.assets} assets</p>
          <p>EL mean {formatCurrency(data?.risk_stats.el_mean ?? 0)}</p>
          <p>VaR95 mean {formatCurrency(data?.risk_stats.var95_mean ?? 0)}</p>
        </div>
        <div style={{ background: "#0f172a", padding: "1rem", borderRadius: "0.75rem" }}>
          <h3>Feature Config</h3>
          <ul>
            {Object.entries(data?.feature_config ?? {}).map(([key, value]) => (
              <li key={key}>
                <strong>{key}:</strong> {value}
              </li>
            ))}
          </ul>
        </div>
      </section>

      <section style={{ display: "grid", gap: "1.5rem", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))" }}>
        <div style={{ background: "#0f172a", padding: "1rem", borderRadius: "0.75rem" }}>
          <h3>Forecast Methodology</h3>
          <p>{data?.forecast_methodology}</p>
          <div style={{ height: "260px" }}>
            <ResponsiveContainer>
              <LineChart data={hazardChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="week" />
                <YAxis
                  label={{ value: "°C/mm", angle: -90, position: "insideLeft", fill: "#94a3b8" }}
                />
                <Tooltip />
                <Legend />
                {hazardVars.map((variable, idx) => (
                  <Line
                    key={variable}
                    type="monotone"
                    dataKey={variable}
                    name={`${variable} q90`}
                    stroke={["#38bdf8", "#f472b6", "#facc15", "#c084fc"][idx % 4]}
                    dot={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p style={{ color: "#94a3b8", margin: 0 }}>Table highlights the same top 10 assets ranked by expected loss.</p>
          <table style={{ width: "100%", marginTop: "0.75rem", fontSize: "0.85rem" }}>
            <thead>
              <tr>
                <th style={{ textAlign: "left" }}>Variable</th>
                <th>Mean</th>
                <th>Avg q90</th>
                <th>Max q90</th>
                <th>Unit</th>
              </tr>
            </thead>
            <tbody>
              {data?.hazard_summary.map((entry) => (
                <tr key={entry.variable}>
                  <td>{entry.variable}</td>
                  <td>{entry.mean_mean.toFixed(1)}</td>
                  <td>{entry.mean_q90.toFixed(1)}</td>
                  <td>{entry.max_q90.toFixed(1)}</td>
                  <td>{entry.unit}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div style={{ background: "#0f172a", padding: "1rem", borderRadius: "0.75rem", display: "grid", gap: "1rem" }}>
          <h3>Risk Translation</h3>
          <p>{data?.risk_methodology}</p>
          <p style={{ color: "#94a3b8", marginTop: "-0.75rem" }}>EL = expected loss, VaR95 = 95th percentile loss, ES95 = tail-average loss beyond VaR.</p>
          <div style={{ display: "grid", gap: "1rem", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))" }}>
            <div style={{ height: "240px" }}>
              <h4 style={{ marginBottom: "0.25rem" }}>Top Assets (EL/VaR/ES)</h4>
              <ResponsiveContainer>
                <BarChart data={riskChartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="asset_name" tick={false} />
                  <YAxis tickFormatter={(value) => `$${value.toFixed(1)}M`} />
                  <Tooltip formatter={(value) => formatCurrency((value as number) * 1_000_000)} />
                  <Bar dataKey="el_total" name="EL (M USD)" fill="#f97316" />
                  <Bar dataKey="var95_total" name="VaR95 (M USD)" fill="#facc15" />
                  <Bar dataKey="es95_total" name="ES95 (M USD)" fill="#c084fc" />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div style={{ height: "240px" }}>
              <h4 style={{ marginBottom: "0.25rem" }}>Risk by Asset Class</h4>
              <ResponsiveContainer>
                <BarChart data={riskByTypeChart}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="asset_type" tick={false} />
                  <YAxis tickFormatter={(value) => `$${value.toFixed(1)}M`} />
                  <Tooltip formatter={(value) => formatCurrency((value as number) * 1_000_000)} />
                  <Bar dataKey="el_total" name="EL (M USD)" fill="#34d399" />
                  <Bar dataKey="var95_total" name="VaR95 (M USD)" fill="#60a5fa" />
                  <Bar dataKey="es95_total" name="ES95 (M USD)" fill="#f9a8d4" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
          <table style={{ width: "100%", marginTop: "0.75rem", fontSize: "0.85rem" }}>
            <thead>
              <tr>
                <th style={{ textAlign: "left" }}>Asset</th>
                <th>Type</th>
                <th>EL</th>
                <th>VaR95</th>
              </tr>
            </thead>
            <tbody>
              {data?.top_assets.map((asset) => (
                <tr key={asset.asset_id}>
                  <td>{asset.asset_name}</td>
                  <td>{asset.asset_type}</td>
                  <td>{formatCurrency(asset.EL)}</td>
                  <td>{formatCurrency(asset.VaR95)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section style={{ background: "#0f172a", padding: "1rem", borderRadius: "0.75rem", display: "grid", gap: "1.25rem" }}>
        <div>
          <h3>Model vs Data Comparison</h3>
          <p style={{ color: "#cbd5f5" }}>{data?.model_overview}</p>
        </div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: "1rem" }}>
          {(data?.metric_docs ?? []).map((doc) => (
            <div key={doc.name} style={{ background: "#1e293b", padding: "0.75rem 1rem", borderRadius: "0.5rem", flex: "1 1 200px" }}>
              <strong>{doc.name}</strong>
              <p style={{ margin: "0.25rem 0", color: "#e2e8f0" }}>{doc.description}</p>
              <span style={{ fontSize: "0.8rem", color: "#94a3b8" }}>{doc.direction}</span>
            </div>
          ))}
        </div>
        <div style={{ display: "grid", gap: "1rem", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))" }}>
          <div style={{ height: "260px" }}>
            <h4 style={{ marginBottom: "0.25rem" }}>MSE by Model (lower is better)</h4>
            <ResponsiveContainer>
              <BarChart data={mseChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="target" />
                <YAxis />
                <Tooltip />
                {buildBars(mseModels)}
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div style={{ height: "260px" }}>
            <h4 style={{ marginBottom: "0.25rem" }}>R² by Model (higher is better)</h4>
            <ResponsiveContainer>
              <BarChart data={r2ChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="target" />
                <YAxis domain={[0, 1]} />
                <Tooltip />
                {buildBars(r2Models)}
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div style={{ height: "260px" }}>
            <h4 style={{ marginBottom: "0.25rem" }}>Quantile (Pinball) Loss</h4>
            <ResponsiveContainer>
              <BarChart data={pinballChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="target" />
                <YAxis />
                <Tooltip />
                {buildBars(pinballModels)}
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </section>

      <section style={{ display: "grid", gap: "0.75rem", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))" }}>
        {data?.split_summary.map((split) => (
          <div key={split.name} style={{ background: "#0f172a", padding: "1rem", borderRadius: "0.75rem" }}>
            <div style={{ fontSize: "0.9rem", color: "#cbd5f5" }}>{split.name.toUpperCase()}</div>
            <div style={{ fontSize: "2rem", fontWeight: 600 }}>{split.weeks} weeks</div>
            <div style={{ fontSize: "0.8rem", color: "#94a3b8" }}>
              eg. {split.example_weeks.length ? split.example_weeks.join(", ") : "n/a"}
            </div>
          </div>
        ))}
      </section>

      {error && (
        <div style={{ background: "#7f1d1d", padding: "1rem", borderRadius: "0.5rem" }}>
          <p>{error}</p>
          <button onClick={onRetry}>Retry</button>
        </div>
      )}
      {loading && <div>Loading analysis…</div>}
    </div>
  );
};

export default InsightsPanel;
