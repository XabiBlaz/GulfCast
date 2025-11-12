import { useState } from "react";

export type AblationState = {
  ensemble: "full" | "meanstd";
  positionalEncoding: "pe" | "latlon" | "none";
  memberOrder: "original" | "sorted" | "shuffled";
};

type Props = {
  state: AblationState;
  onChange: (state: AblationState) => void;
};

const chipStyle: React.CSSProperties = {
  padding: "0.5rem 0.9rem",
  borderRadius: "999px",
  border: "1px solid rgba(148, 163, 184, 0.25)",
  cursor: "pointer",
  fontWeight: 600,
  fontSize: "0.9rem"
};

function Chip({
  active,
  label,
  onClick
}: {
  active: boolean;
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      style={{
        ...chipStyle,
        background: active ? "#38bdf8" : "transparent",
        color: active ? "#0f172a" : "#e2e8f0",
        marginRight: "0.5rem"
      }}
    >
      {label}
    </button>
  );
}

export default function AblationControls({ state, onChange }: Props) {
  const [copied, setCopied] = useState(false);
  const update = (partial: Partial<AblationState>) => onChange({ ...state, ...partial });
  const membersFlag = state.ensemble === "full" ? "full" : "meanstd";
  const buildCommand = `poetry run python -m services.features.build --inputs data/raw --out data/proc --members ${membersFlag} --posenc ${state.positionalEncoding}`;
  const trainCommand = `poetry run python -m services.model.train --data data/proc --out models --features ensemble_${membersFlag} --posenc ${state.positionalEncoding} --member_order ${state.memberOrder}`;

  const copyCommands = () => {
    navigator.clipboard
      .writeText(`${buildCommand}\n${trainCommand}`)
      .then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      })
      .catch(() => setCopied(false));
  };

  return (
    <section style={{ background: "#1e293b", borderRadius: "1rem", padding: "1.25rem" }}>
      <h2 style={{ margin: "0 0 0.5rem 0" }}>Ablation Switches</h2>
      <p style={{ marginTop: 0, color: "#94a3b8" }}>
        Toggle feature flags to compare ensemble usage, positional encodings, and member ordering sensitivity—mirroring
        the Beyond Ensemble Averages ablation study.
      </p>
      <div style={{ display: "grid", gap: "1rem" }}>
        <div>
          <strong>Ensemble usage</strong>
          <div style={{ marginTop: "0.5rem" }}>
            <Chip active={state.ensemble === "full"} label="Full ensemble" onClick={() => update({ ensemble: "full" })} />
            <Chip
              active={state.ensemble === "meanstd"}
              label="Mean + Std"
              onClick={() => update({ ensemble: "meanstd" })}
            />
          </div>
        </div>
        <div>
          <strong>Positional encoding</strong>
          <div style={{ marginTop: "0.5rem" }}>
            <Chip active={state.positionalEncoding === "pe"} label="PE" onClick={() => update({ positionalEncoding: "pe" })} />
            <Chip
              active={state.positionalEncoding === "latlon"}
              label="Lat/Lon"
              onClick={() => update({ positionalEncoding: "latlon" })}
            />
            <Chip
              active={state.positionalEncoding === "none"}
              label="None"
              onClick={() => update({ positionalEncoding: "none" })}
            />
          </div>
        </div>
        <div>
          <strong>Member order</strong>
          <div style={{ marginTop: "0.5rem" }}>
            <Chip
              active={state.memberOrder === "original"}
              label="Original"
              onClick={() => update({ memberOrder: "original" })}
            />
            <Chip
              active={state.memberOrder === "sorted"}
              label="Sorted"
              onClick={() => update({ memberOrder: "sorted" })}
            />
            <Chip
              active={state.memberOrder === "shuffled"}
              label="Shuffled"
              onClick={() => update({ memberOrder: "shuffled" })}
            />
          </div>
        </div>
        <div style={{ background: "#0f172a", padding: "0.85rem", borderRadius: "0.75rem" }}>
          <strong style={{ display: "block", marginBottom: "0.35rem" }}>Ready-to-run commands</strong>
          <pre style={{ margin: 0, fontSize: "0.8rem", whiteSpace: "pre-wrap", color: "#cbd5f5" }}>
            {buildCommand}
            {"\n"}
            {trainCommand}
          </pre>
          <button
            onClick={copyCommands}
            style={{
              marginTop: "0.5rem",
              background: copied ? "#22c55e" : "#38bdf8",
              border: "none",
              color: "#0f172a",
              fontWeight: 600,
              padding: "0.4rem 0.8rem",
              borderRadius: "0.4rem",
              cursor: "pointer"
            }}
          >
            {copied ? "Copied!" : "Copy commands"}
          </button>
        </div>
      </div>
    </section>
  );
}
