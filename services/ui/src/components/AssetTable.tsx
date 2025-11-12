type AssetRisk = {
  asset_id: string;
  asset_name: string;
  asset_type: string;
  EL: number;
  VaR95: number;
  ES95: number;
};

type AssetTableProps = {
  assets: AssetRisk[];
  loading: boolean;
};

function toCurrency(value: number) {
  return `$${value.toLocaleString("en-US", { maximumFractionDigits: 0 })}`;
}

export default function AssetTable({ assets, loading }: AssetTableProps) {
  const rows = [...assets].sort((a, b) => b.EL - a.EL).slice(0, 10);

  return (
    <div style={{ background: "#1e293b", borderRadius: "1rem", padding: "1.5rem" }}>
      <header style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1rem" }}>
        <div>
          <h2 style={{ margin: 0, fontSize: "1.5rem" }}>Asset Risk Breakdown</h2>
          <span style={{ color: "#94a3b8" }}>
            Top 10 assets ranked by Expected Loss (EL). VaR95 captures the 95th percentile tail and ES95 averages the losses beyond that.
          </span>
        </div>
        <button
          onClick={() => {
            const csvRows = [
              ["asset_id", "asset_name", "asset_type", "EL", "VaR95", "ES95"],
              ...assets.map((asset) => [
                asset.asset_id,
                asset.asset_name,
                asset.asset_type,
                asset.EL,
                asset.VaR95,
                asset.ES95
              ])
            ];
            const blob = new Blob([csvRows.map((row) => row.join(",")).join("\n")], { type: "text/csv" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = "risk-export.csv";
            a.click();
            URL.revokeObjectURL(url);
          }}
          style={{
            background: "#38bdf8",
            border: "none",
            color: "#0f172a",
            fontWeight: 600,
            padding: "0.5rem 1rem",
            borderRadius: "0.5rem",
            cursor: "pointer"
          }}
        >
          Export CSV
        </button>
      </header>
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead style={{ background: "#0f172a" }}>
            <tr>
              {["Asset", "Type", "EL", "VaR95", "ES95"].map((col) => (
                <th key={col} style={{ textAlign: "left", padding: "0.75rem", fontWeight: 600 }}>
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
          {rows.map((asset) => (
            <tr key={asset.asset_id} style={{ borderBottom: "1px solid rgba(148, 163, 184, 0.15)" }}>
                <td style={{ padding: "0.75rem" }}>{asset.asset_name}</td>
                <td style={{ padding: "0.75rem", textTransform: "capitalize" }}>{asset.asset_type}</td>
                <td style={{ padding: "0.75rem" }}>{toCurrency(asset.EL)}</td>
                <td style={{ padding: "0.75rem" }}>{toCurrency(asset.VaR95)}</td>
                <td style={{ padding: "0.75rem" }}>{toCurrency(asset.ES95)}</td>
              </tr>
            ))}
            {!loading && assets.length === 0 && (
              <tr>
                <td colSpan={5} style={{ padding: "1.5rem", textAlign: "center", color: "#94a3b8" }}>
                  Submit a forecast request to populate asset metrics.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      {loading && <div style={{ marginTop: "1rem", color: "#94a3b8" }}>Loading portfolio metrics…</div>}
    </div>
  );
}
