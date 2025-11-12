import { useEffect, useMemo, useRef } from "react";
import { MapContainer, TileLayer, CircleMarker, Tooltip } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";

type Marker = {
  id: string;
  name: string;
  lat: number;
  lon: number;
  value: number;
  type: string;
};

type MapViewProps = {
  markers: Marker[];
  loading: boolean;
};

const DEFAULT_CENTER: [number, number] = [29.6, -92.5];

export default function MapView({ markers, loading }: MapViewProps) {
  const mapRef = useRef<L.Map | null>(null);
  useEffect(() => {
    const icon = L.icon({
      iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
      iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
      shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
      iconSize: [25, 41],
      iconAnchor: [12, 41]
    });
    L.Marker.prototype.options.icon = icon;
  }, []);

  useEffect(() => {
    if (!mapRef.current || !markers.length) return;
    const bounds = L.latLngBounds(markers.map((m) => [m.lat, m.lon]));
    if (bounds.isValid()) {
      mapRef.current.fitBounds(bounds, { padding: [40, 40] });
    }
  }, [markers]);

  const sortedMarkers = useMemo(() => [...markers].sort((a, b) => b.value - a.value), [markers]);
  const topIds = useMemo(
    () => new Set(sortedMarkers.slice(0, 10).map((marker) => marker.id)),
    [sortedMarkers]
  );
  const maxValue = useMemo(() => Math.max(...markers.map((m) => m.value), 1), [markers]);
  const assetTypes = useMemo(
    () => Array.from(new Set(markers.map((m) => m.type || "unknown"))).sort(),
    [markers]
  );
  const palette = ["#38bdf8", "#fb7185", "#fde047", "#34d399", "#c084fc", "#f97316", "#e879f9"];
  const colorMap = useMemo(() => {
    const map = new Map<string, string>();
    assetTypes.forEach((type, idx) => {
      map.set(type, palette[idx % palette.length]);
    });
    return map;
  }, [assetTypes]);

  return (
    <div style={{ position: "relative", height: "420px", borderRadius: "1rem", overflow: "hidden", background: "#1e293b" }}>
      <MapContainer
        center={DEFAULT_CENTER}
        zoom={5}
        style={{ height: "100%", width: "100%" }}
        whenCreated={(map) => {
          mapRef.current = map;
        }}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a> contributors'
          url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        {markers.map((marker) => {
          const color = colorMap.get(marker.type || "unknown") ?? "#94a3b8";
          const normalized = Math.max(marker.value / maxValue, 0);
          const baseRadius = 6 + normalized * 10;
          const isTop = topIds.has(marker.id);
          return (
          <CircleMarker
            key={marker.id}
            center={[marker.lat, marker.lon]}
            radius={isTop ? baseRadius + 4 : baseRadius}
            pathOptions={{
              color,
              weight: isTop ? 2.5 : 1,
              fillOpacity: isTop ? 0.9 : 0.6
            }}
          >
            <Tooltip direction="top">
              <div>
                <strong>{marker.name}</strong>
                <div>{marker.type}</div>
                <div>EL ${marker.value.toLocaleString(undefined, { maximumFractionDigits: 0 })}</div>
                {isTop && <div style={{ color: "#facc15", fontWeight: 600 }}>Top 10 risk</div>}
              </div>
            </Tooltip>
          </CircleMarker>
        );})}
      </MapContainer>
      {assetTypes.length > 0 && (
        <div
          style={{
            position: "absolute",
            top: "1rem",
            right: "1rem",
            background: "rgba(15,23,42,0.85)",
            padding: "0.75rem 1rem",
            borderRadius: "0.75rem",
            color: "#e2e8f0",
            fontSize: "0.85rem"
          }}
        >
          <div style={{ fontWeight: 600, marginBottom: "0.4rem" }}>Asset legend</div>
          {assetTypes.map((type) => (
            <div key={type} style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <span style={{ width: "12px", height: "12px", borderRadius: "50%", background: colorMap.get(type) ?? "#94a3b8" }} />
              <span style={{ textTransform: "capitalize" }}>{type}</span>
            </div>
          ))}
          <div style={{ marginTop: "0.5rem", color: "#94a3b8" }}>Top 10 assets show larger markers & halos.</div>
        </div>
      )}
      {loading && (
        <div
          style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: "rgba(15, 23, 42, 0.65)"
          }}
        >
          <span>Loading risk tiles...</span>
        </div>
      )}
    </div>
  );
}
