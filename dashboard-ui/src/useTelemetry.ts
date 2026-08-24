import { useEffect, useState } from "react";
import type { ConnectionState, DashboardSnapshot } from "./types";

const SUPPORTED_SCHEMA = 3;

function parseSnapshot(value: unknown): DashboardSnapshot {
  if (!value || typeof value !== "object") {
    throw new Error("Telemetry payload is not an object");
  }
  const snapshot = value as DashboardSnapshot;
  if (snapshot.schema_version !== SUPPORTED_SCHEMA) {
    throw new Error(`Unsupported telemetry schema: ${snapshot.schema_version}`);
  }
  return snapshot;
}

export function useTelemetry() {
  const [snapshot, setSnapshot] = useState<DashboardSnapshot | null>(null);
  const [connection, setConnection] = useState<ConnectionState>("connecting");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;

    fetch("/api/snapshot")
      .then((response) => {
        if (!response.ok) throw new Error(`Snapshot request failed: ${response.status}`);
        return response.json();
      })
      .then((value) => {
        if (!active) return;
        setSnapshot(parseSnapshot(value));
        setError(null);
      })
      .catch((reason: unknown) => {
        if (!active) return;
        setConnection("offline");
        setError(reason instanceof Error ? reason.message : "Snapshot request failed");
      });

    const events = new EventSource("/api/events");
    events.onmessage = (event) => {
      if (!active) return;
      try {
        setSnapshot(parseSnapshot(JSON.parse(event.data)));
        setConnection("live");
        setError(null);
      } catch (reason) {
        setError(reason instanceof Error ? reason.message : "Invalid telemetry event");
      }
    };
    events.onerror = () => {
      if (active) setConnection((current) => (current === "connecting" ? "offline" : "reconnecting"));
    };

    return () => {
      active = false;
      events.close();
    };
  }, []);

  return { snapshot, connection, error };
}
