// ============================================================================
// AICOS — AI City Operating System · Command Center (Prototype v0.1)
// ----------------------------------------------------------------------------
// Single-file working prototype of the multi-agent architecture:
//   · 5 detection agents (Security, Emergency, Traffic, Infrastructure, Env)
//     spawn incidents with confidence scores + sensor-level explainability
//   · Intelligence agent correlates events (patterns, links, forecasts)
//   · Command agent triages: auto-dispatch for low severity / high confidence,
//     human-in-the-loop approval gate for SEV 4-5 or low-confidence detections
//   · Live map, 17 response units, KPIs, injectable crisis scenarios
// All state is in-memory React state. No network calls. No storage APIs.
// ============================================================================

import { useState, useEffect, useMemo, useCallback } from "react";
import {
  Shield, Siren, Car, Wrench, Wind, Brain, Radio, Activity,
  MapPin, CheckCircle2, XCircle, Pause, Play, Zap, ChevronLeft, Eye, Users,
} from "lucide-react";

// ----------------------------------------------------------------------------
// Static configuration
// ----------------------------------------------------------------------------

const DISTRICTS = [
  { id: "D1", name: "Old Harbor", cx: 20, cy: 66, pop: 0.8 },
  { id: "D2", name: "Central", cx: 50, cy: 46, pop: 1.0 },
  { id: "D3", name: "North Gate", cx: 44, cy: 15, pop: 0.6 },
  { id: "D4", name: "East Fields", cx: 81, cy: 32, pop: 0.5 },
  { id: "D5", name: "Southside", cx: 66, cy: 80, pop: 0.7 },
  { id: "D6", name: "West Ridge", cx: 14, cy: 27, pop: 0.4 },
];

const AGENTS = {
  security: { label: "Security", icon: Shield, color: "#fb7185" },
  emergency: { label: "Emergency", icon: Siren, color: "#f97316" },
  traffic: { label: "Traffic", icon: Car, color: "#fbbf24" },
  infrastructure: { label: "Infrastructure", icon: Wrench, color: "#38bdf8" },
  environmental: { label: "Environmental", icon: Wind, color: "#34d399" },
};

const UNIT_META = {
  police: { label: "Police", color: "#7da7f5" },
  medical: { label: "Medical", color: "#fb7185" },
  fire: { label: "Fire", color: "#f97316" },
  works: { label: "Works", color: "#c8d2e4" },
  enviro: { label: "Enviro", color: "#34d399" },
};

const CATALOG = {
  security: [
    { t: "Weapon detected", sev: [4, 5], unit: "police" },
    { t: "Assault in progress", sev: [4, 5], unit: "police" },
    { t: "Robbery alarm", sev: [3, 5], unit: "police" },
    { t: "Break-in attempt", sev: [3, 4], unit: "police" },
    { t: "Vandalism detected", sev: [2, 3], unit: "police" },
    { t: "Suspicious loitering", sev: [1, 2], unit: "police" },
  ],
  emergency: [
    { t: "Vehicle collision with injuries", sev: [3, 5], unit: "medical" },
    { t: "Cardiac arrest call", sev: [4, 5], unit: "medical" },
    { t: "Structure fire", sev: [4, 5], unit: "fire" },
    { t: "Gas leak reported", sev: [3, 5], unit: "fire" },
    { t: "Person collapsed in public", sev: [2, 4], unit: "medical" },
    { t: "Flood rescue request", sev: [4, 5], unit: "fire" },
  ],
  traffic: [
    { t: "Heavy congestion building", sev: [1, 2], unit: "police" },
    { t: "Multi-car crash", sev: [3, 4], unit: "police" },
    { t: "Road obstruction", sev: [2, 3], unit: "works" },
    { t: "Signal fault at junction", sev: [2, 3], unit: "works" },
    { t: "Wrong-way driver", sev: [4, 4], unit: "police" },
  ],
  infrastructure: [
    { t: "Water main leak", sev: [2, 4], unit: "works" },
    { t: "Power outage", sev: [3, 5], unit: "works" },
    { t: "Road surface damage", sev: [1, 3], unit: "works" },
    { t: "Streetlight grid fault", sev: [1, 2], unit: "works" },
    { t: "Bridge sensor anomaly", sev: [3, 4], unit: "works" },
  ],
  environmental: [
    { t: "PM2.5 spike detected", sev: [2, 3], unit: "enviro" },
    { t: "Noise threshold exceeded", sev: [1, 2], unit: "enviro" },
    { t: "Flash flood risk rising", sev: [3, 4], unit: "enviro" },
    { t: "Chemical odor reports", sev: [3, 4], unit: "enviro" },
    { t: "Heat island alert", sev: [2, 3], unit: "enviro" },
  ],
};

const SIGNALS = {
  security: ["CCTV vision model", "Acoustic gunshot array", "Citizen report via app", "Access-control sensor", "Patrol bodycam AI"],
  emergency: ["112 call transcript NLP", "CCTV vision model", "Wearable health alert", "Smoke and heat IoT node", "Crash impact sensor"],
  traffic: ["Loop detector grid", "Traffic cam analytics", "Connected-vehicle pings", "Signal controller telemetry"],
  infrastructure: ["SCADA telemetry", "Pressure sensor mesh", "Grid load monitor", "Vibration IoT node", "Drone survey pass"],
  environmental: ["Air-quality sensor grid", "Hydrology gauges", "Weather radar feed", "Noise sensor mesh", "Satellite pass"],
};

const STATUS_META = {
  detected: { label: "DETECTED", color: "#8b9ab5" },
  pending: { label: "NEEDS APPROVAL", color: "#f43f5e" },
  queued: { label: "QUEUED", color: "#fbbf24" },
  dispatched: { label: "DISPATCHED", color: "#38bdf8" },
  onscene: { label: "ON SCENE", color: "#a78bfa" },
  resolved: { label: "RESOLVED", color: "#34d399" },
  dismissed: { label: "DISMISSED", color: "#5b6578" },
};

const SEV_COLORS = { 1: "#64748b", 2: "#38bdf8", 3: "#fbbf24", 4: "#fb923c", 5: "#f43f5e" };

const ACTIVE_STATUSES = ["detected", "pending", "queued", "dispatched", "onscene"];

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

let SEQ = 1;
const nid = (p) => p + "-" + SEQ++;
const ri = (a, b) => a + Math.floor(Math.random() * (b - a + 1));
const rf = (a, b) => a + Math.random() * (b - a);
const pick = (arr) => arr[Math.floor(Math.random() * arr.length)];
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const dist = (ax, ay, bx, by) => Math.hypot(ax - bx, ay - by);

const fmtClock = (tick) => {
  const s = 6 * 3600 + (tick || 0) * 30;
  const h = Math.floor(s / 3600) % 24;
  const m = Math.floor((s % 3600) / 60);
  return String(h).padStart(2, "0") + ":" + String(m).padStart(2, "0");
};

const districtName = (id) => {
  const d = DISTRICTS.find((x) => x.id === id);
  return d ? d.name : id;
};

const priorityOf = (inc, tick) =>
  clamp(Math.round(inc.sev * 14 + inc.conf * 22 + (tick - inc.createdTick) * 0.4), 1, 99);

// ----------------------------------------------------------------------------
// Simulation core (pure functions over a plain-data world object)
// ----------------------------------------------------------------------------

function makeIncident(agentKey, tick, forced) {
  const f = forced || {};
  const tpl = f.tpl || pick(CATALOG[agentKey]);
  const d = f.district || pick(DISTRICTS);
  const sev = f.sev || ri(tpl.sev[0], tpl.sev[1]);
  const conf = f.conf != null ? f.conf : rf(0.62, 0.97);
  const pool = SIGNALS[agentKey];
  const count = ri(2, Math.min(3, pool.length));
  const chosen = [...pool].sort(() => Math.random() - 0.5).slice(0, count);
  const weights = chosen.map(() => Math.random() + 0.35);
  const wSum = weights.reduce((a, b) => a + b, 0);
  const signals = chosen.map((src, i) => ({ src, w: weights[i] / wSum }));
  const confRounded = Math.round(conf * 100) / 100;
  return {
    id: nid("INC"),
    agent: agentKey,
    type: tpl.t,
    unit: tpl.unit,
    district: d.id,
    x: clamp(d.cx + rf(-9, 9), 4, 96),
    y: clamp(d.cy + rf(-8, 8), 4, 96),
    sev,
    conf: confRounded,
    signals,
    status: "detected",
    needsHuman: sev >= 4 || confRounded < 0.72,
    approved: false,
    unitId: null,
    linked: [],
    createdTick: tick,
    dispatchedTick: null,
    arrivedTick: null,
    resolvedTick: null,
  };
}

function initWorld() {
  const units = [];
  const add = (type, n) => {
    for (let i = 0; i < n; i++) {
      const x = rf(8, 92);
      const y = rf(8, 92);
      units.push({
        id: nid(type.slice(0, 2).toUpperCase()),
        type, x, y, hx: x, hy: y,
        status: "idle", targetId: null, timer: 0,
      });
    }
  };
  add("police", 5);
  add("medical", 4);
  add("fire", 3);
  add("works", 3);
  add("enviro", 2);
  return {
    tick: 0,
    incidents: [],
    units,
    intel: [],
    intelCooldown: {},
    linkedPairs: [],
    stats: {
      spawned: 0, resolved: 0, dismissed: 0,
      autoDispatched: 0, humanApproved: 0,
      respSum: 0, respCount: 0,
    },
    agentCounts: { security: 0, emergency: 0, traffic: 0, infrastructure: 0, environmental: 0 },
  };
}

function assignNearest(w, inc) {
  let best = null;
  let bestD = Infinity;
  for (const u of w.units) {
    if (u.type !== inc.unit || u.status !== "idle") continue;
    const dd = dist(u.x, u.y, inc.x, inc.y);
    if (dd < bestD) { bestD = dd; best = u; }
  }
  if (!best) { inc.status = "queued"; return false; }
  best.status = "enroute";
  best.targetId = inc.id;
  inc.status = "dispatched";
  inc.unitId = best.id;
  inc.dispatchedTick = w.tick;
  return true;
}

function runIntelligence(w) {
  const windowStart = w.tick - 40;
  const recent = w.incidents.filter((i) => i.createdTick >= windowStart && i.status !== "dismissed");

  // Pattern detection: clustered security incidents in one district
  for (const d of DISTRICTS) {
    const hits = recent.filter((i) => i.agent === "security" && i.district === d.id);
    const cool = w.intelCooldown[d.id] || 0;
    if (hits.length >= 3 && w.tick >= cool) {
      w.intelCooldown[d.id] = w.tick + 60;
      w.intel.unshift({
        id: nid("IQ"), tick: w.tick, kind: "pattern",
        conf: Math.round(rf(0.78, 0.94) * 100),
        text: "Crime pattern in " + d.name + ": " + hits.length + " security incidents inside 20 min. Recommend patrol surge and camera priority.",
        linked: hits.map((i) => i.id),
      });
    }
  }

  // Cross-agency link: traffic crash near a medical/fire emergency
  const crashes = recent.filter((i) => i.agent === "traffic" && (i.type.indexOf("crash") >= 0 || i.type.indexOf("Wrong-way") >= 0));
  const emergencies = recent.filter((i) => i.agent === "emergency");
  for (const c of crashes) {
    for (const e of emergencies) {
      const key = [c.id, e.id].sort().join("|");
      if (w.linkedPairs.indexOf(key) >= 0) continue;
      if (dist(c.x, c.y, e.x, e.y) <= 12 && Math.abs(c.createdTick - e.createdTick) <= 15) {
        w.linkedPairs.push(key);
        if (c.linked.indexOf(e.id) < 0) c.linked.push(e.id);
        if (e.linked.indexOf(c.id) < 0) e.linked.push(c.id);
        w.intel.unshift({
          id: nid("IQ"), tick: w.tick, kind: "link",
          conf: Math.round(rf(0.82, 0.96) * 100),
          text: "Multi-agency event: " + c.type + " linked to " + e.type + " in the same zone. Coordinated dispatch advised.",
          linked: [c.id, e.id],
        });
      }
    }
  }

  // Risk forecast: district load approaching cascade threshold
  for (const d of DISTRICTS) {
    const load = w.incidents
      .filter((i) => i.district === d.id && ACTIVE_STATUSES.indexOf(i.status) >= 0)
      .reduce((a, i) => a + i.sev, 0);
    const cool = w.intelCooldown["F" + d.id] || 0;
    if (load >= 12 && w.tick >= cool) {
      w.intelCooldown["F" + d.id] = w.tick + 80;
      w.intel.unshift({
        id: nid("IQ"), tick: w.tick, kind: "forecast",
        conf: Math.round(rf(0.7, 0.88) * 100),
        text: "Forecast: elevated cascade risk in " + d.name + " over the next 30 min (load " + load + "). Pre-stage units nearby.",
        linked: [],
      });
    }
  }
  w.intel = w.intel.slice(0, 30);
}

function stepWorld(prev) {
  const w = structuredClone(prev);
  w.tick += 1;

  // 1 · Detection agents spawn incidents
  if (Math.random() < 0.5) {
    const agentKey = pick(Object.keys(CATALOG));
    const inc = makeIncident(agentKey, w.tick);
    w.incidents.push(inc);
    w.stats.spawned += 1;
    w.agentCounts[agentKey] += 1;
  }

  // 2 · Command agent triage
  for (const inc of w.incidents) {
    if (inc.status === "detected") {
      if (inc.needsHuman && !inc.approved) {
        inc.status = "pending";
      } else {
        const ok = assignNearest(w, inc);
        if (ok && !inc.approved) w.stats.autoDispatched += 1;
      }
    } else if (inc.status === "queued") {
      if (!inc.needsHuman || inc.approved) assignNearest(w, inc);
    }
  }

  // 3 · Unit movement and resolution
  for (const u of w.units) {
    if (u.status === "enroute") {
      const inc = w.incidents.find((i) => i.id === u.targetId);
      if (!inc || inc.status === "dismissed" || inc.status === "resolved") {
        u.status = "returning";
        u.targetId = null;
        continue;
      }
      const d = dist(u.x, u.y, inc.x, inc.y);
      if (d < 2.5) {
        u.status = "onscene";
        u.timer = inc.sev * 3 + 2;
        inc.status = "onscene";
        inc.arrivedTick = w.tick;
        w.stats.respSum += w.tick - inc.createdTick;
        w.stats.respCount += 1;
      } else {
        u.x += ((inc.x - u.x) / d) * 3.4;
        u.y += ((inc.y - u.y) / d) * 3.4;
      }
    } else if (u.status === "onscene") {
      u.timer -= 1;
      if (u.timer <= 0) {
        const inc = w.incidents.find((i) => i.id === u.targetId);
        if (inc) {
          inc.status = "resolved";
          inc.resolvedTick = w.tick;
          w.stats.resolved += 1;
        }
        u.status = "returning";
        u.targetId = null;
      }
    } else if (u.status === "returning") {
      const d = dist(u.x, u.y, u.hx, u.hy);
      if (d < 2.5) {
        u.status = "idle";
      } else {
        u.x += ((u.hx - u.x) / d) * 3.4;
        u.y += ((u.hy - u.y) / d) * 3.4;
      }
    }
  }

  // 4 · Intelligence agent
  if (w.tick % 5 === 0) runIntelligence(w);

  // 5 · Prune closed incidents so memory stays bounded
  if (w.incidents.length > 90) {
    const closed = w.incidents.filter((i) => i.status === "resolved" || i.status === "dismissed");
    const removable = closed.slice(0, w.incidents.length - 90);
    const ids = new Set(removable.map((i) => i.id));
    w.incidents = w.incidents.filter((i) => !ids.has(i.id));
  }
  return w;
}

function injectScenario(prev, key) {
  const w = structuredClone(prev);
  const push = (agent, tplText, district, sev, conf) => {
    const tpl = CATALOG[agent].find((t) => t.t === tplText);
    if (!tpl) return;
    const d = DISTRICTS.find((x) => x.id === district) || pick(DISTRICTS);
    const inc = makeIncident(agent, w.tick, { tpl, district: d, sev, conf });
    w.incidents.push(inc);
    w.stats.spawned += 1;
    w.agentCounts[agent] += 1;
  };
  if (key === "fire") {
    push("emergency", "Structure fire", "D2", 5, 0.95);
    push("emergency", "Gas leak reported", "D2", 4, 0.83);
    push("environmental", "PM2.5 spike detected", "D2", 3, 0.9);
    push("traffic", "Heavy congestion building", "D2", 2, 0.85);
  } else if (key === "crime") {
    push("security", "Robbery alarm", "D1", 4, 0.88);
    push("security", "Break-in attempt", "D1", 3, 0.8);
    push("security", "Weapon detected", "D1", 5, 0.91);
    push("security", "Vandalism detected", "D1", 2, 0.77);
  } else if (key === "storm") {
    push("environmental", "Flash flood risk rising", "D5", 4, 0.86);
    push("emergency", "Flood rescue request", "D5", 5, 0.9);
    push("infrastructure", "Power outage", "D5", 4, 0.84);
    push("traffic", "Road obstruction", "D5", 3, 0.82);
  } else if (key === "blackout") {
    push("infrastructure", "Power outage", "D2", 5, 0.93);
    push("traffic", "Signal fault at junction", "D2", 3, 0.87);
    push("security", "Break-in attempt", "D2", 3, 0.68);
  }
  return w;
}

// ----------------------------------------------------------------------------
// Small presentational pieces
// ----------------------------------------------------------------------------

function SevBadge({ sev }) {
  const c = SEV_COLORS[sev] || "#64748b";
  return (
    <span
      className="px-1.5 py-0.5 rounded font-semibold shrink-0"
      style={{ fontSize: 10, color: c, border: "1px solid " + c + "55", background: c + "14", letterSpacing: 1 }}
    >
      SEV {sev}
    </span>
  );
}

function StatusChip({ status }) {
  const m = STATUS_META[status] || STATUS_META.detected;
  return (
    <span className="uppercase font-semibold shrink-0" style={{ fontSize: 9, color: m.color, letterSpacing: 1.2 }}>
      {m.label}
    </span>
  );
}

function ConfidenceBar({ value, color }) {
  const pct = Math.round((value || 0) * 100);
  return (
    <div className="flex items-center gap-2 w-full">
      <div className="flex-1 rounded-full overflow-hidden" style={{ height: 4, background: "rgba(125,150,190,0.15)" }}>
        <div style={{ width: pct + "%", height: "100%", background: color || "#5eead4", transition: "width .4s" }} />
      </div>
      <span className="font-semibold" style={{ fontSize: 10, color: "#aebad0" }}>{pct}%</span>
    </div>
  );
}

function Panel({ title, right, children, className, style }) {
  return (
    <div
      className={"rounded-lg flex flex-col min-h-0 " + (className || "")}
      style={{ background: "#0b111e", border: "1px solid rgba(125,150,190,0.14)", ...style }}
    >
      <div className="flex items-center justify-between px-3 py-2" style={{ borderBottom: "1px solid rgba(125,150,190,0.12)" }}>
        <span className="uppercase font-semibold" style={{ fontSize: 10, letterSpacing: 2, color: "#7c8aa5", fontFamily: "'Chakra Petch', sans-serif" }}>
          {title}
        </span>
        {right}
      </div>
      <div className="flex-1 min-h-0">{children}</div>
    </div>
  );
}

function KpiCard({ label, value, sub, accent }) {
  return (
    <div
      className="rounded-lg px-3 py-2 flex flex-col justify-between shrink-0"
      style={{ background: "#0b111e", border: "1px solid rgba(125,150,190,0.14)", minWidth: 128 }}
    >
      <span className="uppercase" style={{ fontSize: 9, letterSpacing: 1.8, color: "#7c8aa5", fontFamily: "'Chakra Petch', sans-serif" }}>
        {label}
      </span>
      <div className="flex items-baseline gap-2 mt-1">
        <span className="font-bold" style={{ fontSize: 22, color: accent || "#e7edf8", lineHeight: 1 }}>{value}</span>
        {sub ? <span style={{ fontSize: 10, color: "#7c8aa5" }}>{sub}</span> : null}
      </div>
    </div>
  );
}

// ----------------------------------------------------------------------------
// City map (SVG radar plate)
// ----------------------------------------------------------------------------

function CityMap({ world, selectedId, onSelect }) {
  return (
    <svg viewBox="0 0 100 100" className="w-full h-full block select-none" role="img" aria-label="Live city map">
      <defs>
        <pattern id="aicosGrid" width="5" height="5" patternUnits="userSpaceOnUse">
          <path d="M 5 0 L 0 0 0 5" fill="none" stroke="rgba(94,234,212,0.06)" strokeWidth="0.3" />
        </pattern>
        <radialGradient id="aicosGlow" cx="50%" cy="42%" r="75%">
          <stop offset="0%" stopColor="rgba(94,234,212,0.05)" />
          <stop offset="100%" stopColor="rgba(6,10,18,0)" />
        </radialGradient>
      </defs>

      <rect x="0" y="0" width="100" height="100" fill="#070c16" />
      <rect x="0" y="0" width="100" height="100" fill="url(#aicosGrid)" />
      <rect x="0" y="0" width="100" height="100" fill="url(#aicosGlow)" />

      {/* arterial roads */}
      <line x1="0" y1="46" x2="100" y2="46" stroke="rgba(148,163,184,0.14)" strokeWidth="1.1" />
      <line x1="44" y1="0" x2="44" y2="100" stroke="rgba(148,163,184,0.14)" strokeWidth="1.1" />
      <line x1="0" y1="88" x2="100" y2="62" stroke="rgba(148,163,184,0.1)" strokeWidth="0.7" />
      <line x1="66" y1="0" x2="96" y2="100" stroke="rgba(148,163,184,0.1)" strokeWidth="0.7" />
      <line x1="0" y1="18" x2="30" y2="46" stroke="rgba(148,163,184,0.08)" strokeWidth="0.6" />

      {/* districts */}
      {DISTRICTS.map((d) => (
        <g key={d.id}>
          <rect
            x={d.cx - 13} y={d.cy - 9} width="26" height="18" rx="1.5"
            fill="rgba(94,234,212,0.025)" stroke="rgba(94,234,212,0.16)" strokeWidth="0.35"
          />
          <text
            x={d.cx} y={d.cy - 10.8} textAnchor="middle"
            style={{ fontSize: 2.7, letterSpacing: 0.6, fill: "#5c6d8a", fontFamily: "'Chakra Petch', sans-serif", textTransform: "uppercase" }}
          >
            {d.name}
          </text>
        </g>
      ))}

      {/* units */}
      {world.units.map((u) => {
        const meta = UNIT_META[u.type] || UNIT_META.works;
        const busy = u.status === "enroute" || u.status === "onscene";
        return (
          <g key={u.id} transform={"rotate(45 " + u.x + " " + u.y + ")"}>
            <rect
              x={u.x - 1.1} y={u.y - 1.1} width="2.2" height="2.2"
              fill={busy ? meta.color : "#0b111e"}
              stroke={meta.color} strokeWidth="0.35"
              opacity={u.status === "idle" ? 0.7 : 1}
            >
              <title>{meta.label + " unit " + u.id + " — " + u.status}</title>
            </rect>
          </g>
        );
      })}

      {/* incidents */}
      {world.incidents.map((inc) => {
        if (inc.status === "dismissed") return null;
        const closed = inc.status === "resolved";
        const color = AGENTS[inc.agent] ? AGENTS[inc.agent].color : "#94a3b8";
        const r = 1.1 + inc.sev * 0.34;
        return (
          <g key={inc.id} onClick={() => onSelect(inc.id)} style={{ cursor: "pointer" }}>
            {!closed && inc.sev >= 4 ? (
              <circle cx={inc.x} cy={inc.y} r={r} fill="none" stroke={color} strokeWidth="0.4" opacity="0.8">
                <animate attributeName="r" values={r + ";" + (r + 4)} dur="1.6s" repeatCount="indefinite" />
                <animate attributeName="opacity" values="0.8;0" dur="1.6s" repeatCount="indefinite" />
              </circle>
            ) : null}
            <circle
              cx={inc.x} cy={inc.y} r={r}
              fill={color} opacity={closed ? 0.22 : 0.9}
              stroke={selectedId === inc.id ? "#ffffff" : "#070c16"}
              strokeWidth={selectedId === inc.id ? 0.6 : 0.3}
            >
              <title>{inc.type + " · SEV " + inc.sev + " · " + districtName(inc.district)}</title>
            </circle>
          </g>
        );
      })}
    </svg>
  );
}

// ----------------------------------------------------------------------------
// Incident detail (explainability + human-in-the-loop actions)
// ----------------------------------------------------------------------------

function IncidentDetail({ inc, tick, onApprove, onDismiss, onClose, onJump }) {
  if (!inc) return null;
  const agent = AGENTS[inc.agent];
  const color = agent ? agent.color : "#94a3b8";
  const active = ACTIVE_STATUSES.indexOf(inc.status) >= 0;
  const steps = [
    { label: "Detected", t: inc.createdTick },
    { label: "Dispatched", t: inc.dispatchedTick },
    { label: "On scene", t: inc.arrivedTick },
    { label: "Resolved", t: inc.resolvedTick },
  ];
  return (
    <Panel
      title={"Incident " + inc.id}
      right={
        <button onClick={onClose} className="flex items-center gap-1 hover:opacity-80" style={{ fontSize: 10, color: "#7c8aa5" }}>
          <XCircle size={13} /> CLOSE
        </button>
      }
    >
      <div className="p-3 flex flex-col gap-3">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="w-2 h-2 rounded-full shrink-0" style={{ background: color }} />
          <span className="font-semibold" style={{ fontSize: 13, color: "#e7edf8", fontFamily: "'Chakra Petch', sans-serif" }}>
            {inc.type}
          </span>
          <SevBadge sev={inc.sev} />
          <StatusChip status={inc.status} />
          <span className="flex items-center gap-1" style={{ fontSize: 10, color: "#7c8aa5" }}>
            <MapPin size={11} /> {districtName(inc.district)}
          </span>
          <span style={{ fontSize: 10, color: "#7c8aa5" }}>PRIORITY {priorityOf(inc, tick)}</span>
        </div>

        <div>
          <div className="uppercase mb-1" style={{ fontSize: 9, letterSpacing: 1.6, color: "#7c8aa5" }}>
            Model confidence
          </div>
          <ConfidenceBar value={inc.conf} color={color} />
        </div>

        <div>
          <div className="uppercase mb-1" style={{ fontSize: 9, letterSpacing: 1.6, color: "#7c8aa5" }}>
            Contributing signals · explainability
          </div>
          <div className="flex flex-col gap-1">
            {inc.signals.map((s, i) => (
              <div key={inc.id + "-sig-" + i} className="flex items-center gap-2">
                <span className="flex-1 truncate" style={{ fontSize: 11, color: "#c4cede" }}>{s.src}</span>
                <span style={{ fontSize: 10, color: "#7c8aa5" }}>{Math.round(s.w * 100)}%</span>
              </div>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-1 flex-wrap">
          {steps.map((s, i) => (
            <div key={inc.id + "-st-" + i} className="flex items-center gap-1">
              {i > 0 ? <span style={{ color: "#3b4560", fontSize: 10 }}>·</span> : null}
              <span
                style={{
                  fontSize: 9.5,
                  color: s.t != null ? "#5eead4" : "#57637d",
                  letterSpacing: 0.6,
                }}
              >
                {s.label} {s.t != null ? fmtClock(s.t) : "—"}
              </span>
            </div>
          ))}
        </div>

        {inc.linked.length > 0 ? (
          <div className="flex items-center gap-2 flex-wrap">
            <span className="uppercase" style={{ fontSize: 9, letterSpacing: 1.6, color: "#a78bfa" }}>Linked</span>
            {inc.linked.map((lid) => (
              <button
                key={inc.id + "-lnk-" + lid}
                onClick={() => onJump(lid)}
                className="px-1.5 py-0.5 rounded hover:opacity-80"
                style={{ fontSize: 10, color: "#a78bfa", border: "1px solid rgba(167,139,250,0.35)" }}
              >
                {lid}
              </button>
            ))}
          </div>
        ) : null}

        <div className="flex gap-2 flex-wrap">
          {inc.status === "pending" ? (
            <button
              onClick={() => onApprove(inc.id)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded font-semibold hover:opacity-90"
              style={{ fontSize: 11, background: "#5eead4", color: "#04211c", fontFamily: "'Chakra Petch', sans-serif", letterSpacing: 0.5 }}
            >
              <CheckCircle2 size={13} /> APPROVE DISPATCH
            </button>
          ) : null}
          {active ? (
            <button
              onClick={() => onDismiss(inc.id)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded font-semibold hover:opacity-90"
              style={{ fontSize: 11, color: "#fda4af", border: "1px solid rgba(244,63,94,0.4)", fontFamily: "'Chakra Petch', sans-serif", letterSpacing: 0.5 }}
            >
              <XCircle size={13} /> MARK FALSE POSITIVE
            </button>
          ) : null}
        </div>
      </div>
    </Panel>
  );
}

// ----------------------------------------------------------------------------
// Main application
// ----------------------------------------------------------------------------

export default function AICOSCommandCenter() {
  const [world, setWorld] = useState(initWorld);
  const [paused, setPaused] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [tab, setTab] = useState("feed");
  const [filter, setFilter] = useState(null);
  const [selectedId, setSelectedId] = useState(null);

  useEffect(() => {
    if (paused) return undefined;
    const id = setInterval(() => setWorld((w) => stepWorld(w)), 1100 / speed);
    return () => clearInterval(id);
  }, [paused, speed]);

  const approve = useCallback((id) => {
    setWorld((prev) => {
      const w = structuredClone(prev);
      const inc = w.incidents.find((i) => i.id === id);
      if (inc && (inc.status === "pending" || inc.status === "queued")) {
        inc.approved = true;
        w.stats.humanApproved += 1;
        assignNearest(w, inc);
      }
      return w;
    });
  }, []);

  const dismiss = useCallback((id) => {
    setWorld((prev) => {
      const w = structuredClone(prev);
      const inc = w.incidents.find((i) => i.id === id);
      if (!inc) return prev;
      if (inc.unitId) {
        const u = w.units.find((x) => x.id === inc.unitId);
        if (u) { u.status = "returning"; u.targetId = null; }
      }
      inc.status = "dismissed";
      w.stats.dismissed += 1;
      return w;
    });
  }, []);

  const runScenario = useCallback((key) => {
    setWorld((prev) => injectScenario(prev, key));
  }, []);

  const jumpTo = useCallback((id) => setSelectedId(id), []);

  // Derived data ------------------------------------------------------------
  const active = useMemo(
    () => world.incidents.filter((i) => ACTIVE_STATUSES.indexOf(i.status) >= 0),
    [world]
  );
  const pendings = useMemo(
    () =>
      world.incidents
        .filter((i) => i.status === "pending")
        .sort((a, b) => priorityOf(b, world.tick) - priorityOf(a, world.tick)),
    [world]
  );
  const feed = useMemo(() => {
    const list = filter ? world.incidents.filter((i) => i.agent === filter) : world.incidents;
    return [...list].sort((a, b) => b.createdTick - a.createdTick).slice(0, 40);
  }, [world, filter]);

  const idleUnits = world.units.filter((u) => u.status === "idle").length;
  const avgResp = world.stats.respCount > 0
    ? ((world.stats.respSum / world.stats.respCount) * 0.5).toFixed(1)
    : "—";
  const risk = clamp(Math.round(active.reduce((a, i) => a + i.sev * i.conf, 0) * 3), 0, 100);
  const riskColor = risk >= 66 ? "#f43f5e" : risk >= 33 ? "#fbbf24" : "#5eead4";
  const selected = selectedId ? world.incidents.find((i) => i.id === selectedId) || null : null;

  const scenarioBtns = [
    { key: "fire", label: "Fire outbreak" },
    { key: "crime", label: "Crime wave" },
    { key: "storm", label: "Storm cell" },
    { key: "blackout", label: "Blackout" },
  ];

  const tabBtn = (key, label, badge) => (
    <button
      key={key}
      onClick={() => setTab(key)}
      className="flex-1 py-1.5 uppercase font-semibold transition-colors"
      style={{
        fontSize: 10,
        letterSpacing: 1.5,
        fontFamily: "'Chakra Petch', sans-serif",
        color: tab === key ? "#5eead4" : "#7c8aa5",
        borderBottom: tab === key ? "2px solid #5eead4" : "2px solid transparent",
      }}
    >
      {label}
      {badge > 0 ? (
        <span
          className="ml-1.5 px-1.5 rounded-full font-bold"
          style={{ fontSize: 9, background: "#f43f5e", color: "#fff" }}
        >
          {badge}
        </span>
      ) : null}
    </button>
  );

  return (
    <div
      className="min-h-screen w-full p-3 md:p-4 flex flex-col gap-3"
      style={{ background: "#060a12", color: "#dbe4f3", fontFamily: "'IBM Plex Mono', monospace" }}
    >
      <style>{"@import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');" +
        "@keyframes aicosBlink { 0%,100% { opacity: 1 } 50% { opacity: .25 } }" +
        ".aicos-scroll::-webkit-scrollbar { width: 5px }" +
        ".aicos-scroll::-webkit-scrollbar-thumb { background: rgba(125,150,190,0.25); border-radius: 4px }" +
        ".aicos-scroll::-webkit-scrollbar-track { background: transparent }" +
        "@media (prefers-reduced-motion: reduce) { * { animation: none !important } }"}
      </style>

      {/* ── Header ─────────────────────────────────────────────────────── */}
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <div className="flex items-center gap-3">
          <div
            className="px-2.5 py-1.5 rounded font-bold"
            style={{ background: "#5eead4", color: "#04211c", fontFamily: "'Chakra Petch', sans-serif", fontSize: 15, letterSpacing: 2 }}
          >
            AICOS
          </div>
          <div className="leading-tight hidden sm:block">
            <div className="uppercase font-semibold" style={{ fontSize: 10, letterSpacing: 2.5, color: "#c4cede", fontFamily: "'Chakra Petch', sans-serif" }}>
              City Operating System
            </div>
            <div style={{ fontSize: 9, color: "#7c8aa5", letterSpacing: 1.5 }}>COMMAND LAYER · PROTOTYPE v0.1 · SIMULATED FEED</div>
          </div>
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          <div className="flex items-center gap-1.5 px-2 py-1 rounded" style={{ border: "1px solid rgba(125,150,190,0.18)" }}>
            <span
              className="w-1.5 h-1.5 rounded-full"
              style={{ background: paused ? "#fbbf24" : "#5eead4", animation: paused ? "none" : "aicosBlink 1.4s infinite" }}
            />
            <span style={{ fontSize: 10, color: paused ? "#fbbf24" : "#5eead4", letterSpacing: 1.5 }}>
              {paused ? "PAUSED" : "LIVE"}
            </span>
            <span className="font-semibold ml-1" style={{ fontSize: 13, color: "#e7edf8" }}>{fmtClock(world.tick)}</span>
          </div>
          <button
            onClick={() => setPaused((p) => !p)}
            className="p-1.5 rounded hover:bg-white/5"
            style={{ border: "1px solid rgba(125,150,190,0.18)", color: "#c4cede" }}
            aria-label={paused ? "Resume simulation" : "Pause simulation"}
          >
            {paused ? <Play size={13} /> : <Pause size={13} />}
          </button>
          <div className="flex rounded overflow-hidden" style={{ border: "1px solid rgba(125,150,190,0.18)" }}>
            {[1, 2, 4].map((s) => (
              <button
                key={"spd-" + s}
                onClick={() => setSpeed(s)}
                className="px-2 py-1 font-semibold"
                style={{
                  fontSize: 10,
                  color: speed === s ? "#04211c" : "#7c8aa5",
                  background: speed === s ? "#5eead4" : "transparent",
                }}
              >
                {s}x
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* ── KPI strip ──────────────────────────────────────────────────── */}
      <div className="flex gap-2 overflow-x-auto pb-1 aicos-scroll">
        <KpiCard label="Active incidents" value={active.length} sub={world.stats.spawned + " total"} />
        <KpiCard label="Awaiting operator" value={pendings.length} accent={pendings.length > 0 ? "#f43f5e" : "#e7edf8"} sub="human-in-the-loop" />
        <KpiCard label="Units available" value={idleUnits + " / " + world.units.length} />
        <KpiCard label="Avg response" value={avgResp} sub="min" />
        <KpiCard label="City risk index" value={risk} accent={riskColor} sub="/ 100" />
      </div>

      {/* ── Main grid ──────────────────────────────────────────────────── */}
      <div className="flex flex-col xl:flex-row gap-3 flex-1 min-h-0">

        {/* Agent roster */}
        <div className="xl:w-56 shrink-0 grid grid-cols-2 xl:grid-cols-1 gap-2 content-start">
          {Object.keys(AGENTS).map((k) => {
            const a = AGENTS[k];
            const Icon = a.icon;
            const recent = world.incidents.filter((i) => i.agent === k && i.createdTick > world.tick - 30).length;
            return (
              <button
                key={"agent-" + k}
                onClick={() => { setFilter(filter === k ? null : k); setTab("feed"); }}
                className="text-left rounded-lg px-3 py-2 transition-colors hover:bg-white/5"
                style={{
                  background: "#0b111e",
                  border: "1px solid " + (filter === k ? a.color + "88" : "rgba(125,150,190,0.14)"),
                }}
              >
                <div className="flex items-center gap-2">
                  <Icon size={14} style={{ color: a.color }} />
                  <span className="font-semibold flex-1 truncate" style={{ fontSize: 11, color: "#e7edf8", fontFamily: "'Chakra Petch', sans-serif", letterSpacing: 0.5 }}>
                    {a.label}
                  </span>
                  <span className="w-1.5 h-1.5 rounded-full" style={{ background: "#5eead4", animation: "aicosBlink 2s infinite" }} />
                </div>
                <div className="flex items-center justify-between mt-1.5">
                  <span style={{ fontSize: 9, color: "#7c8aa5", letterSpacing: 1 }}>DETECTIONS</span>
                  <span className="font-semibold" style={{ fontSize: 11, color: "#c4cede" }}>{world.agentCounts[k]}</span>
                </div>
                <div className="mt-1.5 rounded-full overflow-hidden" style={{ height: 3, background: "rgba(125,150,190,0.15)" }}>
                  <div style={{ width: Math.min(100, recent * 18) + "%", height: "100%", background: a.color, transition: "width .5s" }} />
                </div>
              </button>
            );
          })}

          <div className="rounded-lg px-3 py-2" style={{ background: "#0b111e", border: "1px solid rgba(167,139,250,0.3)" }}>
            <div className="flex items-center gap-2">
              <Brain size={14} style={{ color: "#a78bfa" }} />
              <span className="font-semibold flex-1" style={{ fontSize: 11, color: "#e7edf8", fontFamily: "'Chakra Petch', sans-serif", letterSpacing: 0.5 }}>
                Intelligence
              </span>
            </div>
            <div className="flex items-center justify-between mt-1.5">
              <span style={{ fontSize: 9, color: "#7c8aa5", letterSpacing: 1 }}>CORRELATIONS</span>
              <span className="font-semibold" style={{ fontSize: 11, color: "#a78bfa" }}>{world.intel.length}</span>
            </div>
          </div>

          <div className="rounded-lg px-3 py-2" style={{ background: "#0b111e", border: "1px solid rgba(94,234,212,0.3)" }}>
            <div className="flex items-center gap-2">
              <Radio size={14} style={{ color: "#5eead4" }} />
              <span className="font-semibold flex-1" style={{ fontSize: 11, color: "#e7edf8", fontFamily: "'Chakra Petch', sans-serif", letterSpacing: 0.5 }}>
                Command
              </span>
            </div>
            <div className="flex items-center justify-between mt-1.5">
              <span style={{ fontSize: 9, color: "#7c8aa5", letterSpacing: 1 }}>AUTO / HUMAN</span>
              <span className="font-semibold" style={{ fontSize: 11, color: "#5eead4" }}>
                {world.stats.autoDispatched} / {world.stats.humanApproved}
              </span>
            </div>
          </div>
        </div>

        {/* Map + detail */}
        <div className="flex-1 min-w-0 flex flex-col gap-3">
          <div
            className="relative rounded-lg overflow-hidden"
            style={{ background: "#070c16", border: "1px solid rgba(125,150,190,0.14)", height: "min(62vw, 520px)" }}
          >
            <CityMap world={world} selectedId={selectedId} onSelect={setSelectedId} />

            {/* scenario injector */}
            <div className="absolute top-2 left-2 flex items-center gap-1.5 flex-wrap pr-2">
              <span className="flex items-center gap-1 uppercase" style={{ fontSize: 9, letterSpacing: 1.5, color: "#7c8aa5" }}>
                <Zap size={11} /> Simulate
              </span>
              {scenarioBtns.map((s) => (
                <button
                  key={"sc-" + s.key}
                  onClick={() => runScenario(s.key)}
                  className="px-2 py-0.5 rounded hover:bg-white/10 transition-colors"
                  style={{ fontSize: 10, color: "#c4cede", border: "1px solid rgba(125,150,190,0.25)", background: "rgba(11,17,30,0.8)" }}
                >
                  {s.label}
                </button>
              ))}
            </div>

            {/* unit legend */}
            <div className="absolute bottom-2 left-2 flex items-center gap-2 flex-wrap px-2 py-1 rounded" style={{ background: "rgba(11,17,30,0.8)", border: "1px solid rgba(125,150,190,0.15)" }}>
              <Users size={11} style={{ color: "#7c8aa5" }} />
              {Object.keys(UNIT_META).map((k) => (
                <span key={"lg-" + k} className="flex items-center gap-1" style={{ fontSize: 9, color: "#9aa8c0" }}>
                  <span className="inline-block" style={{ width: 6, height: 6, background: UNIT_META[k].color, transform: "rotate(45deg)" }} />
                  {UNIT_META[k].label}
                </span>
              ))}
            </div>
          </div>

          {selected ? (
            <IncidentDetail
              inc={selected}
              tick={world.tick}
              onApprove={approve}
              onDismiss={dismiss}
              onClose={() => setSelectedId(null)}
              onJump={jumpTo}
            />
          ) : null}
        </div>

        {/* Right panel: feed / approvals / intel */}
        <Panel
          title="Operations"
          className="xl:w-80 shrink-0"
          style={{ height: 480 }}
          right={
            filter ? (
              <button onClick={() => setFilter(null)} className="flex items-center gap-1 hover:opacity-80" style={{ fontSize: 9, color: "#5eead4", letterSpacing: 1 }}>
                <ChevronLeft size={11} /> CLEAR FILTER
              </button>
            ) : null
          }
        >
          <div className="flex flex-col h-full min-h-0">
            <div className="flex" style={{ borderBottom: "1px solid rgba(125,150,190,0.12)" }}>
              {tabBtn("feed", "Feed", 0)}
              {tabBtn("approvals", "Approve", pendings.length)}
              {tabBtn("intel", "Intel", 0)}
            </div>

            <div className="flex-1 overflow-y-auto aicos-scroll">
              {tab === "feed" ? (
                feed.length === 0 ? (
                  <div className="p-4" style={{ fontSize: 11, color: "#7c8aa5" }}>
                    No detections yet. The agents are watching — events will appear here, or inject a scenario from the map.
                  </div>
                ) : (
                  feed.map((inc) => {
                    const a = AGENTS[inc.agent];
                    return (
                      <button
                        key={"feed-" + inc.id}
                        onClick={() => setSelectedId(inc.id)}
                        className="w-full text-left px-3 py-2 hover:bg-white/5 transition-colors"
                        style={{
                          borderBottom: "1px solid rgba(125,150,190,0.08)",
                          borderLeft: "2px solid " + (a ? a.color : "#64748b"),
                          background: selectedId === inc.id ? "rgba(94,234,212,0.05)" : "transparent",
                        }}
                      >
                        <div className="flex items-center gap-2">
                          <span className="flex-1 truncate font-semibold" style={{ fontSize: 11, color: "#e7edf8" }}>
                            {inc.type}
                          </span>
                          <SevBadge sev={inc.sev} />
                        </div>
                        <div className="flex items-center gap-2 mt-1">
                          <span style={{ fontSize: 9.5, color: "#7c8aa5" }}>
                            {fmtClock(inc.createdTick)} · {districtName(inc.district)}
                          </span>
                          <span className="flex-1" />
                          <StatusChip status={inc.status} />
                        </div>
                      </button>
                    );
                  })
                )
              ) : null}

              {tab === "approvals" ? (
                pendings.length === 0 ? (
                  <div className="p-4 flex items-start gap-2" style={{ fontSize: 11, color: "#7c8aa5" }}>
                    <CheckCircle2 size={14} style={{ color: "#5eead4", marginTop: 1 }} className="shrink-0" />
                    <span>Queue clear. High-severity or low-confidence detections stop here for your decision before any unit moves.</span>
                  </div>
                ) : (
                  pendings.map((inc) => {
                    const a = AGENTS[inc.agent];
                    return (
                      <div
                        key={"pnd-" + inc.id}
                        className="px-3 py-2.5"
                        style={{ borderBottom: "1px solid rgba(125,150,190,0.08)", borderLeft: "2px solid #f43f5e" }}
                      >
                        <div className="flex items-center gap-2">
                          <span className="flex-1 truncate font-semibold" style={{ fontSize: 11, color: "#e7edf8" }}>
                            {inc.type}
                          </span>
                          <SevBadge sev={inc.sev} />
                        </div>
                        <div className="mt-1" style={{ fontSize: 9.5, color: "#7c8aa5" }}>
                          {districtName(inc.district)} · {inc.sev >= 4 ? "High severity" : "Low confidence"} · conf {Math.round(inc.conf * 100)}% · needs {UNIT_META[inc.unit] ? UNIT_META[inc.unit].label : inc.unit}
                        </div>
                        <div className="flex gap-2 mt-2">
                          <button
                            onClick={() => approve(inc.id)}
                            className="flex-1 py-1 rounded font-semibold hover:opacity-90"
                            style={{ fontSize: 10, background: "#5eead4", color: "#04211c", letterSpacing: 0.5 }}
                          >
                            APPROVE
                          </button>
                          <button
                            onClick={() => dismiss(inc.id)}
                            className="flex-1 py-1 rounded font-semibold hover:opacity-90"
                            style={{ fontSize: 10, color: "#fda4af", border: "1px solid rgba(244,63,94,0.4)", letterSpacing: 0.5 }}
                          >
                            DISMISS
                          </button>
                          <button
                            onClick={() => setSelectedId(inc.id)}
                            className="px-2 rounded hover:bg-white/5"
                            style={{ border: "1px solid rgba(125,150,190,0.2)", color: "#9aa8c0" }}
                            aria-label="View details"
                          >
                            <Eye size={12} />
                          </button>
                        </div>
                      </div>
                    );
                  })
                )
              ) : null}

              {tab === "intel" ? (
                world.intel.length === 0 ? (
                  <div className="p-4" style={{ fontSize: 11, color: "#7c8aa5" }}>
                    No correlations yet. The Intelligence agent scans every cycle for crime patterns, cross-agency links, and cascade risk.
                  </div>
                ) : (
                  world.intel.map((it) => {
                    const kindColor = it.kind === "pattern" ? "#fb7185" : it.kind === "link" ? "#a78bfa" : "#fbbf24";
                    return (
                      <button
                        key={"iq-" + it.id}
                        onClick={() => { if (it.linked.length > 0) setSelectedId(it.linked[0]); }}
                        className="w-full text-left px-3 py-2.5 hover:bg-white/5 transition-colors"
                        style={{ borderBottom: "1px solid rgba(125,150,190,0.08)", borderLeft: "2px solid " + kindColor }}
                      >
                        <div className="flex items-center gap-2">
                          <Activity size={11} style={{ color: kindColor }} className="shrink-0" />
                          <span className="uppercase font-semibold" style={{ fontSize: 9, letterSpacing: 1.5, color: kindColor }}>
                            {it.kind}
                          </span>
                          <span className="flex-1" />
                          <span style={{ fontSize: 9.5, color: "#7c8aa5" }}>{fmtClock(it.tick)} · {it.conf}%</span>
                        </div>
                        <div className="mt-1" style={{ fontSize: 10.5, color: "#c4cede", lineHeight: 1.5 }}>
                          {it.text}
                        </div>
                      </button>
                    );
                  })
                )
              ) : null}
            </div>
          </div>
        </Panel>
      </div>

      <div style={{ fontSize: 9, color: "#57637d", letterSpacing: 1 }} className="text-center">
        SIMULATED DATA · PRIVACY-FIRST BY DESIGN · ALL SEV 4-5 ACTIONS REQUIRE HUMAN APPROVAL
      </div>
    </div>
  );
}
