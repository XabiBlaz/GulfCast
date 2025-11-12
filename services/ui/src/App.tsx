import { useCallback, useEffect, useMemo, useState } from "react";
import axios from "axios";
import MapView from "./components/MapView";
import AssetTable from "./components/AssetTable";
import AblationControls, { AblationState } from "./components/AblationControls";
import InsightsPanel, { InsightsPayload } from "./components/InsightsPanel";
import ForecastMap from "./components/ForecastMap";

type AssetRisk = {
  asset_id: string;
  asset_name: string;
  asset_type: string;
  lat: number;
  lon: number;
  EL: number;
  VaR95: number;
  ES95: number;
};

const DEFAULT_ABLATION: AblationState = {
  ensemble: "full",
  positionalEncoding: "pe",
  memberOrder: "original"
};

const API_BASE = import.meta.env.VITE_API_BASE ?? "/api";

function App() {
  const [ablation, setAblation] = useState<AblationState>(DEFAULT_ABLATION);
  const [riskData, setRiskData] = useState<AssetRisk[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<"dashboard" | "insights" | "forecast">("dashboard");
  const [insightsData, setInsightsData] = useState<InsightsPayload | null>(null);
  const [insightsLoading, setInsightsLoading] = useState(false);
  const [insightsError, setInsightsError] = useState<string | null>(null);

  useEffect(() => {
    const fetchRisk = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await axios.get(`${API_BASE}/risk/latest`, { params: { limit: 1000 } });
        setRiskData(response.data.assets || []);
      } catch (err) {
        setError("Failed to load risk results. Ensure the pipeline has produced data/risk/latest.parquet.");
      } finally {
        setLoading(false);
      }
    };
    fetchRisk();
  }, []);

  const fetchInsights = useCallback(async () => {
    setInsightsLoading(true);
    setInsightsError(null);
    try {
      const response = await axios.get(`${API_BASE}/insights`);
      setInsightsData(response.data);
    } catch (err) {
      setInsightsError("Unable to load pipeline insights. Ensure the FastAPI server has generated metadata.");
    } finally {
      setInsightsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (view === "insights" && !insightsData && !insightsLoading) {
      fetchInsights();
    }
  }, [view, insightsData, insightsLoading, fetchInsights]);

  const mapMarkers = useMemo(
    () =>
      riskData.map((asset) => ({
        id: asset.asset_id,
        name: asset.asset_name,
        lat: asset.lat,
        lon: asset.lon,
        value: asset.EL,
        type: asset.asset_type
      })),
    [riskData]
  );

  return (
    <div style={{ padding: "1.5rem", display: "grid", gap: "1.25rem" }}>
      <header>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "1rem" }}>
          <div>
            <h1 style={{ margin: 0, fontSize: "2.25rem" }}>GulfCast Dashboard</h1>
            <p style={{ marginTop: "0.25rem", color: "#94a3b8" }}>
              Subseasonal week-3/4 hazard translation from ensemble forecasts to asset risk metrics across the globe.
            </p>
          </div>
          <nav style={{ display: "flex", gap: "0.5rem" }}>
            {[
              { key: "dashboard", label: "Risk Dashboard" },
              { key: "forecast", label: "Forecast Map" },
              { key: "insights", label: "Research & Data Insights" }
            ].map((item) => (
              <button
                key={item.key}
                onClick={() => setView(item.key as typeof view)}
                style={{
                  padding: "0.6rem 1rem",
                  borderRadius: "0.5rem",
                  border: "none",
                  cursor: "pointer",
                  background: view === item.key ? "#2563eb" : "#1e293b",
                  color: "#f8fafc"
                }}
              >
                {item.label}
              </button>
            ))}
          </nav>
        </div>
        <p style={{ marginTop: "0.25rem", color: "#94a3b8" }}>
          Toggle between the live asset risk view and deeper research context, training notes, and dataset analysis.
        </p>
      </header>
      {view === "dashboard" ? (
        <>
          <AblationControls state={ablation} onChange={setAblation} />
          <div style={{ background: "#0f172a", borderRadius: "0.75rem", padding: "1rem", color: "#cbd5f5" }}>
            <h3 style={{ marginTop: 0 }}>Ablation Switches</h3>
            <p style={{ marginBottom: "0.5rem" }}>
              These toggles describe the modelling configuration used when you rebuild features/models:
            </p>
            <ul style={{ margin: 0, paddingLeft: "1.2rem" }}>
              <li>
                <strong>Ensemble usage</strong>: sets <code>--members</code> (full member tensors vs mean/std compression).
              </li>
              <li>
                <strong>Positional encoding</strong>: sets <code>--posenc</code> (<code>pe</code>/<code>latlon</code>/<code>none</code>).
              </li>
              <li>
                <strong>Member order</strong>: sets <code>--member-order</code> (original, sorted, shuffled) before stacking.
              </li>
            </ul>
            <p style={{ marginTop: "0.5rem" }}>
              Adjust them, rerun <code>services.features.build</code> and <code>services.model.train</code> (copy the commands above), then refresh this dashboard to see the updated results.
            </p>
          </div>
          {error && <div style={{ background: "#7f1d1d", padding: "1rem", borderRadius: "0.5rem" }}>{error}</div>}
          <div style={{ display: "grid", gap: "1.5rem", gridTemplateColumns: "minmax(0, 1fr)" }}>
            <MapView markers={mapMarkers} loading={loading} />
            <AssetTable assets={riskData} loading={loading} />
          </div>
        </>
      ) : view === "forecast" ? (
        <ForecastMap />
      ) : (
        <InsightsPanel data={insightsData} loading={insightsLoading} error={insightsError} onRetry={fetchInsights} />
      )}
    </div>
  );
}

export default App;
