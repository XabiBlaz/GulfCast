import { useEffect, useMemo, useState, useCallback, useRef } from "react";
import axios from "axios";
import { MapContainer, TileLayer, CircleMarker, Tooltip } from "react-leaflet";

// Match the actual API response from /forecast/map
type GridResponse = {
  week: string;
  variable: string;
  mode: string;
  grid: Array<{
    lat: number;
    lon: number;
    value: number;
  }>;
};

type WeeksResponse = {
  splits: Record<string, string[]>;
  available_weeks: string[];
  next_forecast_week: string | null;
  weeks_by_variable?: Record<string, string[]>;
};

type Scenario = {
  delta_T_C: number;
  flood_return_period: number;
  drought_scale: number;
  slr_meters?: number;
};

type ScenarioJobStatus = {
  job_id: string;
  status: string;
  progress: number;
  current_step?: string;
  log_tail?: string[];
  error?: string | null;
};

type RiskSummary = {
  el: number;
  var95: number;
  es95: number;
};

const API_BASE = import.meta.env.VITE_API_BASE ?? "/api";

export default function ForecastMap() {
  const [weeks, setWeeks] = useState<string[]>([]);
  const [weeksByVar, setWeeksByVar] = useState<Record<string, string[]>>({});
  const [selectedWeek, setSelectedWeek] = useState<string>("");
  const [variable, setVariable] = useState<string>("t2m");
  const [mode, setMode] = useState<string>("weekly");
  const [grid, setGrid] = useState<GridResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [scenario, setScenario] = useState<Scenario>({ delta_T_C: 0, flood_return_period: 20, drought_scale: 1 });
  const [scenarioOptions, setScenarioOptions] = useState({ rebuildFeatures: false, retrainModels: false, rescoreForecasts: false });
  const [jobStatus, setJobStatus] = useState<ScenarioJobStatus | null>(null);
  const jobIntervalRef = useRef<number | null>(null);
  const [riskSummary, setRiskSummary] = useState<RiskSummary | null>(null);
  const [configDefaults, setConfigDefaults] = useState<{ members: string; posenc: string; lags: number; member_order: string }>({
    members: "full",
    posenc: "pe",
    lags: 1,
    member_order: "original"
  });

  useEffect(() => {
    const fetchWeeks = async () => {
      try {
        const resp = await axios.get<WeeksResponse>(`${API_BASE}/forecast/weeks`);
        setWeeks(resp.data.available_weeks || []);
        setWeeksByVar(resp.data.weeks_by_variable ?? {});
        const lastWeek = resp.data.available_weeks?.slice(-1)[0];
        if (lastWeek) {
          setSelectedWeek(lastWeek);
        }
      } catch (_) {
        setError("Unable to load available forecast weeks.");
      }
    };
    fetchWeeks();
  }, []);

  useEffect(() => {
    const fetchScenario = async () => {
      try {
        const resp = await axios.get(`${API_BASE}/risk/hazards`);
        if (resp.data?.scenario) {
          setScenario((prev) => ({ ...prev, ...resp.data.scenario }));
        }
      } catch (_) {
        /* ignore */
      }
    };
    fetchScenario();
  }, []);

  useEffect(() => {
    const fetchInsights = async () => {
      try {
        const resp = await axios.get(`${API_BASE}/insights`);
        const cfg = resp.data?.feature_config ?? {};
        setConfigDefaults({
          members: cfg.ensemble_mode ?? "full",
          posenc: cfg.positional_encoding ?? "pe",
          lags: Number(cfg.lags ?? 1),
          member_order: cfg.member_order ?? "original"
        });
      } catch (_) {
        /* ignore */
      }
    };
    fetchInsights();
  }, []);

  useEffect(() => {
    const fetchGrid = async () => {
      if (!selectedWeek) return;
      setLoading(true);
      setError(null);
      try {
        const resp = await axios.get<GridResponse>(`${API_BASE}/forecast/map`, {
          params: { week: selectedWeek, var: variable, mode },
        });
        setGrid(resp.data);
      } catch (err) {
        setError("Unable to fetch forecast grid for the selected week.");
        setGrid(null);
      } finally {
        setLoading(false);
      }
    };
    fetchGrid();
  }, [selectedWeek, variable, mode]);

  const fetchRiskSummary = useCallback(async () => {
    try {
      const resp = await axios.get(`${API_BASE}/risk/latest`, { params: { limit: 500 } });
      const totals = (resp.data.assets ?? []).reduce(
        (acc: RiskSummary, asset: any) => ({
          el: acc.el + (asset.EL ?? 0),
          var95: acc.var95 + (asset.VaR95 ?? 0),
          es95: acc.es95 + (asset.ES95 ?? 0)
        }),
        { el: 0, var95: 0, es95: 0 }
      );
      setRiskSummary(totals);
    } catch (_) {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    fetchRiskSummary();
  }, [fetchRiskSummary]);

useEffect(() => {
  return () => {
    if (jobIntervalRef.current) {
      window.clearInterval(jobIntervalRef.current);
    }
  };
}, []);

  useEffect(() => {
    const options = weeksByVar[variable] ?? weeks;
    if (options.length && !options.includes(selectedWeek)) {
      setSelectedWeek(options.slice(-1)[0]);
    }
  }, [variable, weeksByVar, weeks, selectedWeek]);

  const floodFactor = useMemo(() => 20 / Math.max(1, scenario.flood_return_period), [scenario.flood_return_period]);

  const circles = useMemo(() => {
    if (!grid?.grid || grid.grid.length === 0) return [];
    return grid.grid.map((point) => {
      let adjusted = point.value;
      if (grid.variable === "t2m") {
        adjusted += scenario.delta_T_C;
      } else if (grid.variable === "precip") {
        adjusted *= scenario.drought_scale;
        adjusted *= floodFactor;
      }
      return { ...point, value: adjusted };
    });
  }, [grid, scenario.delta_T_C, scenario.drought_scale, floodFactor]);

  const center = useMemo<[number, number]>(() => {
    if (!grid?.grid || grid.grid.length === 0) return [29.5, -90.0];
    const lats = grid.grid.map((p) => p.lat);
    const lons = grid.grid.map((p) => p.lon);
    const midLat = (Math.min(...lats) + Math.max(...lats)) / 2;
    const midLon = (Math.min(...lons) + Math.max(...lons)) / 2;
    return [midLat, midLon];
  }, [grid]);

  const variableWeeks = weeksByVar[variable] ?? weeks;

  // Color scale for markers
  const getColor = (value: number, variable: string): string => {
    if (variable === "t2m") {
      if (value < 10) return "#3b82f6";
      if (value < 20) return "#22c55e";
      if (value < 30) return "#eab308";
      return "#ef4444";
    } else {
      if (value < 10) return "#f3f4f6";
      if (value < 30) return "#93c5fd";
      if (value < 60) return "#3b82f6";
      return "#1e3a8a";
    }
  };

  const getUnit = (variable: string): string => {
    return variable === "t2m" ? "°C" : "mm";
  };

  const pollJobStatus = useCallback(
    async (jobId: string) => {
      try {
        const resp = await axios.get<ScenarioJobStatus>(`${API_BASE}/scenarios/status/${jobId}`, { params: { tail: 20 } });
        setJobStatus(resp.data);
        if (resp.data.status === "completed" || resp.data.status === "failed") {
          if (jobIntervalRef.current) window.clearInterval(jobIntervalRef.current);
          jobIntervalRef.current = null;
          if (resp.data.status === "completed") {
            fetchRiskSummary();
          }
        }
      } catch (_) {
        if (jobIntervalRef.current) window.clearInterval(jobIntervalRef.current);
        jobIntervalRef.current = null;
        setJobStatus({
          job_id: jobId,
          status: "failed",
          progress: 1,
          error: "Unable to fetch job status.",
          log_tail: []
        });
      }
    },
    [fetchRiskSummary]
  );

  const runScenarioJob = async () => {
    try {
      const payload = {
        delta_T_C: scenario.delta_T_C,
        flood_return_period: scenario.flood_return_period,
        drought_scale: scenario.drought_scale,
        rebuild_features: scenarioOptions.rebuildFeatures,
        retrain_models: scenarioOptions.retrainModels,
        rescore_forecasts: scenarioOptions.rescoreForecasts,
        risk_method: "physrisk",
        members: configDefaults.members,
        posenc: configDefaults.posenc,
        lags: configDefaults.lags,
        member_order: configDefaults.member_order
      };
      const resp = await axios.post<{ job_id: string; status: string }>(`${API_BASE}/scenarios/run`, payload);
      const jobId = resp.data.job_id;
      setJobStatus({ job_id: jobId, status: resp.data.status, progress: 0, log_tail: [] });
      if (jobIntervalRef.current) window.clearInterval(jobIntervalRef.current);
      const interval = window.setInterval(() => pollJobStatus(jobId), 4000);
      jobIntervalRef.current = interval;
    } catch (_) {
      setJobStatus({
        job_id: "n/a",
        status: "failed",
        progress: 1,
        error: "Failed to submit scenario job.",
        log_tail: []
      });
    }
  };

  const formatCurrency = (value: number) =>
    new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 }).format(value);

  return (
    <div style={{ display: "grid", gap: "1rem", gridTemplateColumns: "minmax(0, 1fr)" }}>
      <section style={{ background: "#0f172a", padding: "1rem", borderRadius: "0.75rem", display: "grid", gap: "1rem" }}>
        <div style={{ display: "grid", gap: "0.75rem", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))" }}>
          <label style={{ display: "flex", flexDirection: "column", gap: "0.25rem" }}>
            Forecast Week
            <select value={selectedWeek} onChange={(e) => setSelectedWeek(e.target.value)} disabled={!variableWeeks.length}>
              {variableWeeks.length === 0 && <option>Loading...</option>}
              {variableWeeks.map((week) => (
                <option key={week} value={week}>
                  {week}
                </option>
              ))}
            </select>
          </label>
          <label style={{ display: "flex", flexDirection: "column", gap: "0.25rem" }}>
            Variable
            <select value={variable} onChange={(e) => setVariable(e.target.value)}>
              <option value="t2m">Temperature (t2m)</option>
              <option value="precip">Precipitation</option>
            </select>
          </label>
          <label style={{ display: "flex", flexDirection: "column", gap: "0.25rem" }}>
            Mode
            <select value={mode} onChange={(e) => setMode(e.target.value)}>
              <option value="weekly">Mean</option>
              <option value="quantile">Q90</option>
            </select>
          </label>
        </div>
        <div style={{ display: "grid", gap: "1rem", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))" }}>
          <label>
            ΔT (°C): {scenario.delta_T_C.toFixed(1)}
            <input
              type="range"
              min={-2}
              max={4}
              step={0.1}
              value={scenario.delta_T_C}
              onChange={(e) => setScenario((prev) => ({ ...prev, delta_T_C: Number(e.target.value) }))}
            />
          </label>
          <label>
            Flood RP (yrs): {scenario.flood_return_period}
            <input
              type="range"
              min={5}
              max={100}
              step={5}
              value={scenario.flood_return_period}
              onChange={(e) => setScenario((prev) => ({ ...prev, flood_return_period: Number(e.target.value) }))}
            />
          </label>
          <label>
            Drought stress: {scenario.drought_scale.toFixed(1)}
            <input
              type="range"
              min={0.5}
              max={2.0}
              step={0.1}
              value={scenario.drought_scale}
              onChange={(e) => setScenario((prev) => ({ ...prev, drought_scale: Number(e.target.value) }))}
            />
          </label>
        </div>
        <div style={{ background: "#0b1220", padding: "0.75rem", borderRadius: "0.5rem", color: "#cbd5f5", fontSize: "0.9rem" }}>
          <strong>Scenario impact preview</strong>
          <ul style={{ margin: "0.4rem 0 0 1rem", padding: 0 }}>
            <li>
              ΔT adds {scenario.delta_T_C.toFixed(1)}°C to the temperature field. When you apply the scenario, that offset is written into the
              risk config so heat-related EL/VaR/ES are recomputed with the new baseline.
            </li>
            <li>
              Flood return period rescales precipitation by <code>20 ÷ RP = {floodFactor.toFixed(2)}</code>, mimicking more or less extreme flood stress.
              The automated job updates the scenario section of <code>config/risk.yaml</code> with that RP before running risk.
            </li>
            <li>
              Drought stress multiplies precipitation by {scenario.drought_scale.toFixed(1)}; values &gt;1 dry out the grid. The same factor is passed into
              the backend run so drought-sensitive assets see the adjusted totals.
            </li>
          </ul>
        </div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: "1rem", alignItems: "center" }}>
          <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
            <label style={{ display: "flex", alignItems: "center", gap: "0.35rem", color: "#cbd5f5" }}>
              <input
                type="checkbox"
                checked={scenarioOptions.rebuildFeatures}
                onChange={(e) => setScenarioOptions((prev) => ({ ...prev, rebuildFeatures: e.target.checked }))}
              />
              Rebuild features
            </label>
            <label style={{ display: "flex", alignItems: "center", gap: "0.35rem", color: "#cbd5f5" }}>
              <input
                type="checkbox"
                checked={scenarioOptions.retrainModels}
                onChange={(e) => setScenarioOptions((prev) => ({ ...prev, retrainModels: e.target.checked }))}
              />
              Retrain models
            </label>
            <label style={{ display: "flex", alignItems: "center", gap: "0.35rem", color: "#cbd5f5" }}>
              <input
                type="checkbox"
                checked={scenarioOptions.rescoreForecasts}
                onChange={(e) => setScenarioOptions((prev) => ({ ...prev, rescoreForecasts: e.target.checked }))}
              />
              Rescore forecasts
            </label>
          </div>
          <button
            onClick={runScenarioJob}
            disabled={jobStatus !== null && jobStatus.status === "running"}
            style={{
              background: "#2563eb",
              color: "#f8fafc",
              border: "none",
              borderRadius: "0.5rem",
              padding: "0.6rem 1.5rem",
              cursor: "pointer",
              fontWeight: 600
            }}
          >
            {jobStatus && jobStatus.status === "running" ? "Scenario running..." : "Apply scenario to portfolio"}
          </button>
        </div>
        <p style={{ color: "#94a3b8", margin: 0 }}>
          <strong>Options:</strong> Rebuild features re-syncs the datasets before training; Retrain models fits new LR/RF/stack heads; Rescore forecasts
          refreshes the parquet tiles so risk consumes the latest predictions. Leave unchecked if you only need the risk recalculation step.
        </p>
        {jobStatus && (
          <div style={{ background: "#0b1220", padding: "0.75rem", borderRadius: "0.5rem", color: "#cbd5f5", fontSize: "0.9rem" }}>
            <div>
              <strong>Job status:</strong> {jobStatus.status} {jobStatus.current_step ? `(${jobStatus.current_step})` : ""}
            </div>
            <div style={{ height: "8px", background: "#1e293b", borderRadius: "999px", margin: "0.5rem 0" }}>
              <div
                style={{
                  width: `${Math.round((jobStatus.progress ?? 0) * 100)}%`,
                  height: "100%",
                  borderRadius: "999px",
                  background: jobStatus.status === "failed" ? "#f87171" : "#38bdf8"
                }}
              />
            </div>
            {jobStatus.error && <div style={{ color: "#f87171" }}>{jobStatus.error}</div>}
            {jobStatus.log_tail && jobStatus.log_tail.length > 0 && (
              <pre style={{ background: "#030712", padding: "0.5rem", borderRadius: "0.5rem", maxHeight: "160px", overflow: "auto" }}>
                {jobStatus.log_tail.join("\n")}
              </pre>
            )}
          </div>
        )}
      </section>
      <section style={{ background: "#0f172a", borderRadius: "0.75rem", padding: "0.5rem" }}>
        {error && <div style={{ color: "#f87171" }}>{error}</div>}
        {loading && <div style={{ color: "#94a3b8" }}>Loading forecast map…</div>}
        {!loading && grid && circles.length > 0 && (
          <MapContainer 
            key={`${selectedWeek}-${variable}-${mode}-${scenario.delta_T_C}-${scenario.drought_scale}`}
            center={center} 
            zoom={6} 
            style={{ height: "600px", width: "100%" }}
          >
            <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
            {circles.map((cell, idx) => (
              <CircleMarker
                key={`${cell.lat}-${cell.lon}-${idx}`}
                center={[cell.lat, cell.lon]}
                radius={5}
                fillColor={getColor(cell.value, grid.variable)}
                color={getColor(cell.value, grid.variable)}
                fillOpacity={0.7}
                weight={1}
              >
                <Tooltip>
                  <strong>{grid.variable === "t2m" ? "Temperature" : "Precipitation"}</strong>
                  <br />
                  {cell.value.toFixed(1)} {getUnit(grid.variable)}
                  <br />
                  ({cell.lat.toFixed(2)}, {cell.lon.toFixed(2)})
                </Tooltip>
              </CircleMarker>
            ))}
          </MapContainer>
        )}
        {!loading && !grid && !error && <div style={{ color: "#94a3b8" }}>No grid data available.</div>}
        {!loading && grid && circles.length === 0 && <div style={{ color: "#94a3b8" }}>No valid data points in grid.</div>}
        {!loading && grid && (
          <div style={{ marginTop: "0.75rem", color: "#94a3b8", fontSize: "0.9rem" }}>
            When the automated job finishes, EL/VaR/ES and the datasets in the dashboard reflect the scenario values above.
          </div>
        )}
      </section>
      {riskSummary && (
        <section style={{ background: "#0f172a", padding: "1rem", borderRadius: "0.75rem" }}>
          <h3 style={{ marginTop: 0 }}>Latest risk totals</h3>
          <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap" }}>
            <div style={{ flex: "1 1 160px" }}>
              <div style={{ color: "#94a3b8" }}>Expected Loss</div>
              <div style={{ fontSize: "1.5rem", fontWeight: 600 }}>{formatCurrency(riskSummary.el)}</div>
            </div>
            <div style={{ flex: "1 1 160px" }}>
              <div style={{ color: "#94a3b8" }}>VaR95</div>
              <div style={{ fontSize: "1.5rem", fontWeight: 600 }}>{formatCurrency(riskSummary.var95)}</div>
            </div>
            <div style={{ flex: "1 1 160px" }}>
              <div style={{ color: "#94a3b8" }}>ES95</div>
              <div style={{ fontSize: "1.5rem", fontWeight: 600 }}>{formatCurrency(riskSummary.es95)}</div>
            </div>
          </div>
        </section>
      )}
    </div>
  );
}
