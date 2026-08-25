import { Card, Chip, ProgressBar, Skeleton } from "@heroui/react";
import {
  Activity,
  Bot,
  CircleDot,
  Clock3,
  Cpu,
  Database,
  GitCommitHorizontal,
  Gauge,
  Network,
  ShieldCheck,
  Swords,
  Trophy,
  Wifi,
  WifiOff,
  type LucideIcon,
} from "lucide-react";
import { useEffect, useMemo, useState, type ReactNode } from "react";
import type { ConnectionState, DashboardSnapshot, HistoryPoint } from "./types";
import { useTelemetry } from "./useTelemetry";

const integerFormatter = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
const decimalFormatter = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 });

function formatNumber(value: number | null | undefined, digits = 0) {
  if (value == null) return "—";
  return digits ? decimalFormatter.format(value) : integerFormatter.format(value);
}

function formatLoss(value: number | null | undefined) {
  return value == null ? "—" : value.toFixed(3);
}

function formatSigned(value: number | null | undefined) {
  if (value == null) return "—";
  return `${value >= 0 ? "+" : ""}${Math.round(value)}`;
}

function formatPercent(value: number | null | undefined, digits = 1) {
  return value == null ? "—" : `${(value * 100).toFixed(digits)}%`;
}

function formatSpeedup(
  baseline: number | null,
  optimized: number | null,
  lowerIsBetter = false,
) {
  if (baseline == null || optimized == null || baseline <= 0 || optimized <= 0) return "—";
  return `${(lowerIsBetter ? baseline / optimized : optimized / baseline).toFixed(2)}×`;
}

function formatDuration(seconds: number) {
  const total = Math.max(0, Math.floor(seconds));
  const days = Math.floor(total / 86_400);
  const hours = Math.floor((total % 86_400) / 3_600);
  const minutes = Math.floor((total % 3_600) / 60);
  const secs = total % 60;
  const clock = [hours, minutes, secs].map((part) => String(part).padStart(2, "0")).join(":");
  return days > 0 ? `${days}d ${clock}` : clock;
}

function modeLabel(mode: string) {
  return mode.replaceAll("_", " ").toLowerCase().replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function useCurrentTime() {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1_000);
    return () => window.clearInterval(timer);
  }, []);
  return now;
}

interface PanelProps {
  title: string;
  description?: string;
  icon?: LucideIcon;
  action?: ReactNode;
  className?: string;
  children: ReactNode;
}

function Panel({ title, description, icon: Icon, action, className = "", children }: PanelProps) {
  return (
    <Card className={`panel ${className}`}>
      <Card.Header className="panel-header">
        <div className="panel-heading">
          {Icon ? (
            <span className="panel-icon" aria-hidden="true">
              <Icon size={16} strokeWidth={1.75} />
            </span>
          ) : null}
          <div>
            <Card.Title className="panel-title">{title}</Card.Title>
            {description ? <Card.Description className="panel-description">{description}</Card.Description> : null}
          </div>
        </div>
        {action}
      </Card.Header>
      <Card.Content className="panel-content">{children}</Card.Content>
    </Card>
  );
}

interface StatCardProps {
  label: string;
  value: string;
  detail: string;
  icon: LucideIcon;
}

function StatCard({ label, value, detail, icon: Icon }: StatCardProps) {
  return (
    <Card className="stat-card">
      <Card.Content className="stat-content">
        <div className="stat-label-row">
          <span>{label}</span>
          <Icon aria-hidden="true" size={15} strokeWidth={1.75} />
        </div>
        <strong>{value}</strong>
        <small>{detail}</small>
      </Card.Content>
    </Card>
  );
}

type StatusColor = "default" | "accent" | "success" | "warning" | "danger";

function ConnectionChip({ state, stale }: { state: ConnectionState; stale: boolean }) {
  const effectiveState = stale && state === "live" ? "reconnecting" : state;
  const config: Record<ConnectionState, { label: string; color: StatusColor; icon: LucideIcon }> = {
    connecting: { label: "Connecting", color: "default", icon: CircleDot },
    live: { label: "Live", color: "success", icon: Wifi },
    reconnecting: { label: stale ? "Stale" : "Reconnecting", color: "warning", icon: WifiOff },
    offline: { label: "Offline", color: "danger", icon: WifiOff },
  };
  const status = config[effectiveState];
  const Icon = status.icon;
  return (
    <Chip color={status.color} size="sm" variant="soft">
      <Icon aria-hidden="true" size={12} />
      <Chip.Label>{status.label}</Chip.Label>
    </Chip>
  );
}

interface ChartPoint {
  value: number;
  low?: number;
  high?: number;
}

function TelemetryChart({ points, label }: { points: ChartPoint[]; label: string }) {
  const geometry = useMemo(() => {
    if (points.length < 2) return null;
    const allValues = points.flatMap((point) => [point.value, point.low, point.high]).filter((value): value is number => value != null);
    let minimum = Math.min(...allValues);
    let maximum = Math.max(...allValues);
    const padding = Math.max((maximum - minimum) * 0.12, 1);
    minimum -= padding;
    maximum += padding;
    const x = (index: number) => 8 + (index / (points.length - 1)) * 584;
    const y = (value: number) => 12 + ((maximum - value) / (maximum - minimum)) * 136;
    const line = points.map((point, index) => `${index === 0 ? "M" : "L"}${x(index)} ${y(point.value)}`).join(" ");
    const hasBand = points.every((point) => point.low != null && point.high != null);
    const band = hasBand
      ? [
          ...points.map((point, index) => `${x(index)},${y(point.high!)}`),
          ...[...points].reverse().map((point, reverseIndex) => `${x(points.length - reverseIndex - 1)},${y(point.low!)}`),
        ].join(" ")
      : null;
    return { line, band, lastX: x(points.length - 1), lastY: y(points.at(-1)!.value) };
  }, [points]);

  if (!geometry) return <div className="chart-empty">Waiting for enough history</div>;

  return (
    <svg className="telemetry-chart" viewBox="0 0 600 160" preserveAspectRatio="none" role="img" aria-label={label}>
      <title>{label}</title>
      <desc>{`${points.length} telemetry samples. Latest value ${points.at(-1)?.value}.`}</desc>
      {[46, 92, 138].map((y) => <line key={y} className="chart-grid-line" x1="0" x2="600" y1={y} y2={y} />)}
      {geometry.band ? <polygon className="chart-band" points={geometry.band} /> : null}
      <path className="chart-line" d={geometry.line} />
      <circle className="chart-point" cx={geometry.lastX} cy={geometry.lastY} r="4" />
    </svg>
  );
}

function TrendBlock({ label, value, detail, points }: { label: string; value: string; detail: string; points: ChartPoint[] }) {
  return (
    <div className="trend-block">
      <div className="trend-heading">
        <div>
          <span>{label}</span>
          <small>{detail}</small>
        </div>
        <strong>{value}</strong>
      </div>
      <TelemetryChart label={`${label} history`} points={points} />
    </div>
  );
}

const pieces: Record<string, string> = {
  p: "♟", r: "♜", n: "♞", b: "♝", q: "♛", k: "♚",
  P: "♙", R: "♖", N: "♘", B: "♗", Q: "♕", K: "♔",
};

function ChessBoard({ fen, lastMove }: { fen: string; lastMove: string }) {
  const squares = useMemo(() => {
    const highlights = new Set(lastMove.length >= 4 ? [lastMove.slice(0, 2), lastMove.slice(2, 4)] : []);
    const result: { name: string; piece: string; dark: boolean; highlighted: boolean }[] = [];
    const rows = (fen || "8/8/8/8/8/8/8/8").split(" ")[0].split("/");
    rows.forEach((row, visualRank) => {
      let file = 0;
      for (const token of row) {
        const count = Number(token);
        if (Number.isInteger(count) && count > 0) {
          for (let index = 0; index < count; index += 1) addSquare("");
        } else {
          addSquare(token);
        }
      }
      function addSquare(piece: string) {
        const name = `${"abcdefgh"[file]}${8 - visualRank}`;
        result.push({ name, piece, dark: (visualRank + file) % 2 === 1, highlighted: highlights.has(name) });
        file += 1;
      }
    });
    return result;
  }, [fen, lastMove]);

  return (
    <div className="board" role="img" aria-label={`Live chess position. Last move ${lastMove || "not available"}.`}>
      {squares.map((square) => (
        <span
          key={square.name}
          aria-hidden="true"
          className={`board-square ${square.dark ? "is-dark" : "is-light"} ${square.highlighted ? "is-highlighted" : ""}`}
          title={square.name}
        >
          {pieces[square.piece] || ""}
        </span>
      ))}
      <span className="sr-only">FEN: {fen}</span>
    </div>
  );
}

function ProgressMetric({ label, value, maxValue, display }: { label: string; value: number; maxValue: number; display: string }) {
  return (
    <div className="progress-metric">
      <div className="progress-heading">
        <span>{label}</span>
        <strong>{display}</strong>
      </div>
      <ProgressBar aria-label={label} maxValue={maxValue} value={Math.min(value, maxValue)}>
        <ProgressBar.Track className="metric-track">
          <ProgressBar.Fill className="metric-fill" />
        </ProgressBar.Track>
      </ProgressBar>
    </div>
  );
}

function LoadingDashboard() {
  return (
    <main className="dashboard-shell" aria-label="Loading dashboard">
      <div className="loading-header">
        <Skeleton className="h-9 w-48 rounded-md" />
        <Skeleton className="h-7 w-24 rounded-full" />
      </div>
      <div className="loading-grid">
        <Skeleton className="h-48 rounded-lg" />
        <Skeleton className="h-48 rounded-lg" />
        {Array.from({ length: 4 }).map((_, index) => <Skeleton key={index} className="h-28 rounded-lg" />)}
      </div>
    </main>
  );
}

function snapshotClocks(snapshot: DashboardSnapshot, now: number) {
  const updatedAt = Date.parse(snapshot.updated_at);
  const delta = Number.isFinite(updatedAt) ? Math.max(0, (now - updatedAt) / 1_000) : 0;
  return {
    session: snapshot.session_elapsed_seconds + delta,
    training: snapshot.training_elapsed_seconds + (snapshot.mode === "TRAINING" ? delta : 0),
    stale: !snapshot.demo && delta > 10,
    age: delta,
  };
}

function App() {
  const { snapshot, connection, error } = useTelemetry();
  const now = useCurrentTime();

  if (!snapshot) return <LoadingDashboard />;

  const clocks = snapshotClocks(snapshot, now);
  const history = snapshot.history ?? [];
  const eloPoints: ChartPoint[] = history
    .filter((point) => point.elo_delta != null)
    .map((point) => ({ value: point.elo_delta!, low: point.elo_low ?? undefined, high: point.elo_high ?? undefined }));
  const lossPoints = history.filter((point) => point.total_loss != null).map((point) => ({ value: point.total_loss! }));
  const productionPoints = history.map((point) => ({ value: point.games_per_hour }));
  const latestHistory = history.at(-1);
  const game = snapshot.live_game;
  const [win = 0, draw = 1, loss = 0] = game.wdl ?? [];
  const diversity = snapshot.diversity;
  const pilotColor: StatusColor = snapshot.pilot_status === "PASSED"
    ? "success"
    : snapshot.pilot_status === "FAILED"
      ? "danger"
      : snapshot.pilot_status === "TRAINING"
        ? "accent"
        : "default";
  const checkpointColor: StatusColor = snapshot.checkpoint_status === "VERIFIED"
    ? "success"
    : snapshot.checkpoint_status === "FAILED"
      ? "danger"
      : snapshot.checkpoint_status === "WRITING"
        ? "warning"
        : "default";

  return (
    <div className="app-frame">
      <header className="app-header">
        <div className="header-inner">
          <div className="brand-lockup">
            <div className="brand-mark" aria-hidden="true">H</div>
            <div>
              <h1>HarbiChess</h1>
              <p>Training dashboard</p>
            </div>
          </div>
          <div className="header-status" aria-live="polite">
            {snapshot.demo ? <Chip className="demo-chip" size="sm" variant="soft">Demo data</Chip> : null}
            <ConnectionChip stale={clocks.stale} state={connection} />
            <Chip color="accent" size="sm" variant="soft">{modeLabel(snapshot.mode)}</Chip>
          </div>
        </div>
      </header>

      <main className="dashboard-shell">
        {error ? <div className="error-banner" role="alert">Telemetry warning: {error}</div> : null}

        <section className="overview-grid" aria-labelledby="operation-title">
          <Card className="operation-card">
            <Card.Content className="operation-content">
              <div className="operation-copy">
                <span className="section-kicker">Current operation</span>
                <h2 id="operation-title">{snapshot.mode_detail}</h2>
                <p>
                  Run <span className="mono">{snapshot.run_id}</span> · {snapshot.demo ? "demo snapshot" : clocks.age < 2 ? "updated just now" : `updated ${Math.round(clocks.age)}s ago`}
                </p>
              </div>
              <div className="clock-pair">
                <div><span>Run time</span><strong>{formatDuration(clocks.session)}</strong></div>
                <div><span>Training time</span><strong>{formatDuration(clocks.training)}</strong></div>
              </div>
              <div className="checkpoint-row">
                <div><span>Active</span><strong>{snapshot.active_checkpoint}</strong></div>
                <div><span>Candidate</span><strong>{snapshot.candidate_checkpoint}</strong></div>
                <div><span>Promoted</span><strong>{snapshot.promoted_checkpoint}</strong></div>
                <div><span>Source</span><strong className="mono"><GitCommitHorizontal size={13} />{snapshot.source_commit}</strong></div>
              </div>
            </Card.Content>
          </Card>

          <Card className="lifetime-card">
            <Card.Content className="lifetime-content">
              <div className="lifetime-heading"><span>Lifetime self-play</span><Bot size={18} aria-hidden="true" /></div>
              <strong className="lifetime-number">{formatNumber(snapshot.lifetime_games)}</strong>
              <p>Total completed games</p>
              <div className="lifetime-detail">
                <div><span>This run</span><strong>{formatNumber(snapshot.run_games)}</strong></div>
                <div><span>Generation</span><strong>{formatNumber(snapshot.generation_games)}</strong></div>
                <div><span>Active</span><strong>{formatNumber(snapshot.active_games)}</strong></div>
              </div>
            </Card.Content>
          </Card>
        </section>

        <section className="stat-grid" aria-label="Key performance metrics">
          <StatCard icon={Activity} label="Training step" value={formatNumber(snapshot.training_step)} detail="optimizer updates" />
          <StatCard icon={Gauge} label="Positions / sec" value={formatNumber(snapshot.positions_per_second)} detail="encoded states" />
          <StatCard icon={Cpu} label="Neural evals / sec" value={formatNumber(snapshot.neural_evals_per_second)} detail="MLX inference" />
          <StatCard icon={Swords} label="MCTS nodes / sec" value={formatNumber(snapshot.mcts_nodes_per_second)} detail="search throughput" />
        </section>

        <section className="guardrail-grid" aria-label="OCAK sanity guardrails">
          <Panel
            className="pilot-panel"
            description="Low-budget learner gate before recursive training"
            icon={ShieldCheck}
            title="OCAK sanity pilot"
            action={<Chip color={pilotColor} size="sm" variant="soft">{modeLabel(snapshot.pilot_status)}</Chip>}
          >
            <ProgressMetric
              display={`${formatNumber(snapshot.pilot_steps_completed)} / ${formatNumber(snapshot.pilot_steps_planned)}`}
              label="Pilot steps"
              maxValue={Math.max(1, snapshot.pilot_steps_planned)}
              value={snapshot.pilot_steps_completed}
            />
            <div className="gate-metrics">
              <div><span>Train loss</span><strong>{formatLoss(snapshot.pilot_initial_train_loss)} <small>→</small> {formatLoss(snapshot.pilot_final_train_loss ?? snapshot.total_loss)}</strong></div>
              <div><span>Validation loss</span><strong>{formatLoss(snapshot.pilot_initial_validation_loss)} <small>→</small> {formatLoss(snapshot.pilot_final_validation_loss ?? latestHistory?.validation_loss)}</strong></div>
              <div><span>Max gradient norm</span><strong>{formatLoss(snapshot.pilot_max_gradient_norm)}</strong></div>
              <div><span>Validation samples</span><strong>{formatNumber(snapshot.validation_samples)}</strong></div>
              <div><span>Best validation</span><strong>{formatLoss(snapshot.pilot_best_validation_loss)} <small>@ step {formatNumber(snapshot.pilot_best_validation_step)}</small></strong></div>
              <div><span>Early stopping</span><strong>{snapshot.pilot_stopped_early ? "Triggered" : "Not triggered"} <small>· {formatNumber(snapshot.pilot_steps_attempted)} attempted</small></strong></div>
              <div><span>Training stopped by</span><strong>{modeLabel(snapshot.pilot_stop_reason)}</strong></div>
              <div><span>Last improvement</span><strong>step {formatNumber(snapshot.pilot_last_improvement_step)} <small>· last validation {formatLoss(snapshot.pilot_last_validation_loss)} @ {formatNumber(snapshot.pilot_last_validation_step)}</small></strong></div>
              <div><span>Validation patience</span><strong>{formatNumber(snapshot.pilot_stale_validation_evaluations)} / {formatNumber(snapshot.pilot_early_stopping_patience)} <small>· every {formatNumber(snapshot.pilot_validation_interval_steps)} steps</small></strong></div>
              <div><span>Arena selected</span><strong>{snapshot.pilot_arena_selected_step == null ? "Not evaluated" : `step ${formatNumber(snapshot.pilot_arena_selected_step)}`} <small>· {modeLabel(snapshot.pilot_arena_selection_reason)}</small></strong></div>
              <div><span>Arena candidates</span><strong>{formatNumber(snapshot.validation_checkpoint_count)}</strong></div>
              <div><span>Continuation samples</span><strong>{formatNumber(snapshot.continuation_replay_samples)}</strong></div>
            </div>
            {snapshot.pilot_reasons.length ? (
              <div className="guardrail-message is-danger">{snapshot.pilot_reasons.join(" · ")}</div>
            ) : (
              <div className="guardrail-message">No guardrail violation recorded</div>
            )}
            <div className="guardrail-message">{snapshot.pilot_stop_detail}</div>
            <div className="checkpoint-state">
              <div>
                <span>Checkpoint</span>
                <strong title={snapshot.checkpoint_path}>{snapshot.candidate_checkpoint || "Not written yet"}</strong>
              </div>
              <Chip color={checkpointColor} size="sm" variant="soft">{modeLabel(snapshot.checkpoint_status)}</Chip>
            </div>
          </Panel>

          <Panel
            className="diversity-panel"
            description={`${formatNumber(diversity.games)} games · ${formatNumber(diversity.positions)} positions`}
            icon={Network}
            title="Self-play diversity"
            action={<Chip size="sm" variant="soft">{formatNumber(snapshot.replay_shards)} shards</Chip>}
          >
            <div className="diversity-metrics">
              <div><span>Unique games</span><strong>{formatPercent(diversity.unique_game_ratio)}</strong></div>
              <div><span>Duplicate games</span><strong>{formatPercent(diversity.duplicate_game_ratio)}</strong></div>
              <div><span>Unique positions</span><strong>{formatPercent(diversity.unique_position_ratio)}</strong></div>
              <div><span>Action coverage</span><strong>{formatPercent(diversity.action_space_coverage, 2)}</strong></div>
              <div><span>Effective branches</span><strong>{formatNumber(diversity.effective_policy_branches, 1)}</strong></div>
              <div><span>Mean game plies</span><strong>{formatNumber(diversity.mean_game_plies, 1)}</strong></div>
              <div><span>Decisive games</span><strong>{formatNumber(diversity.decisive_games)} · {formatPercent(diversity.decisive_game_ratio)}</strong></div>
              <div><span>Max-ply draws</span><strong>{formatNumber(diversity.max_ply_draws)} · {formatPercent(diversity.max_ply_draw_ratio)}</strong></div>
              <div><span>Repetition redirects</span><strong>{formatNumber(diversity.repetition_redirects)} · {formatPercent(diversity.repetition_redirect_ratio)}</strong></div>
            </div>
            <div className="termination-list">
              <span className="subsection-label">Termination distribution</span>
              {(diversity.terminations ?? []).map((termination) => (
                <div className="termination-row" key={termination.termination}>
                  <span>{modeLabel(termination.termination)}</span>
                  <strong>{formatNumber(termination.count)}</strong>
                  <small>{formatPercent(termination.ratio)}</small>
                </div>
              ))}
              {!diversity.terminations?.length ? <p className="empty-copy">Waiting for terminal outcomes</p> : null}
            </div>
            <div className="opening-list">
              <span className="subsection-label">Opening prefix coverage</span>
              {(diversity.openings ?? []).map((opening) => (
                <div className="opening-row" key={opening.ply}>
                  <span>Ply {opening.ply}</span>
                  <strong>{formatNumber(opening.unique_prefixes)} unique</strong>
                  <small>entropy {opening.entropy.toFixed(2)} · effective {formatNumber(opening.effective_prefixes, 1)}</small>
                </div>
              ))}
              {!diversity.openings?.length ? <p className="empty-copy">Waiting for completed games</p> : null}
            </div>
            <div className="diversity-wdl">
              <span>W / D / L</span>
              <strong>{formatNumber(diversity.white_wins)} / {formatNumber(diversity.draws)} / {formatNumber(diversity.black_wins)}</strong>
            </div>
          </Panel>
        </section>

        <section className="primary-grid">
          <Panel
            className="quality-panel"
            description={`${snapshot.arena_games} color-balanced arena games`}
            icon={Trophy}
            title="Candidate vs champion"
            action={
              <Chip color={snapshot.promotion_ready ? "success" : "default"} size="sm" variant="soft">
                {snapshot.promotion_ready ? "Promotion ready" : "Collecting games"}
              </Chip>
            }
          >
            <div className="quality-metrics">
              <div><span>Estimated gain</span><strong>{formatSigned(snapshot.arena_elo_delta)} <small>Elo</small></strong></div>
              <div><span>95% interval</span><strong>{snapshot.arena_elo_low == null ? "—" : `${formatSigned(snapshot.arena_elo_low)} to ${formatSigned(snapshot.arena_elo_high)}`}</strong></div>
              <div><span>Arena score</span><strong>{snapshot.arena_games ? `${formatNumber(snapshot.arena_score_rate * 100, 1)}%` : "—"}</strong></div>
            </div>
            <TelemetryChart label="Candidate Elo gain and 95 percent confidence interval" points={eloPoints} />
            <div className="wdl-summary" aria-label={`${snapshot.arena_wins} wins, ${snapshot.arena_draws} draws, ${snapshot.arena_losses} losses`}>
              <div><span>Wins</span><strong>{formatNumber(snapshot.arena_wins)}</strong></div>
              <div><span>Draws</span><strong>{formatNumber(snapshot.arena_draws)}</strong></div>
              <div><span>Losses</span><strong>{formatNumber(snapshot.arena_losses)}</strong></div>
            </div>
            <div className="arena-termination-summary">
              <div><span>Decisive</span><strong>{formatNumber(snapshot.arena_decisive_games)}</strong></div>
              <div><span>Threefold</span><strong>{formatNumber(snapshot.arena_threefold_repetitions)} <small>· {formatNumber(snapshot.arena_avoidable_threefold_repetitions)} avoidable</small></strong></div>
              <div><span>Continuation replay</span><strong>{formatNumber(snapshot.arena_continuation_replay_samples)} roots</strong></div>
              <div><span>Max-ply</span><strong>{formatNumber(snapshot.arena_max_ply_draws)}</strong></div>
              <div><span>Other draws</span><strong>{formatNumber(snapshot.arena_other_draws)}</strong></div>
            </div>
          </Panel>

          <Panel className="trajectory-panel" description="Latest recorded training history" icon={Activity} title="Training trajectory">
            <TrendBlock detail="optimization objective" label="Total loss" points={lossPoints} value={formatLoss(latestHistory?.total_loss)} />
            <TrendBlock detail="self-play production" label="Games / hour" points={productionPoints} value={formatNumber(latestHistory?.games_per_hour)} />
          </Panel>
        </section>

        <section className="work-grid">
          <Panel
            className="game-panel"
            description={game.game_id || "No active game"}
            icon={Swords}
            title="Live self-play sample"
            action={<Chip size="sm" variant="soft">Ply {game.ply || 0}</Chip>}
          >
            <div className="game-layout">
              <ChessBoard fen={game.fen} lastMove={game.last_move} />
              <div className="game-analysis">
                <div className="last-move-row"><span>Last move</span><strong className="mono">{game.last_move || "—"}</strong></div>
                <div className="position-wdl">
                  <div><span>Win</span><strong>{Math.round(win * 100)}%</strong></div>
                  <div><span>Draw</span><strong>{Math.round(draw * 100)}%</strong></div>
                  <div><span>Loss</span><strong>{Math.round(loss * 100)}%</strong></div>
                </div>
                <div className="top-moves-block">
                  <span className="subsection-label">Policy / MCTS top moves</span>
                  <div className="move-list">
                    {(game.top_moves ?? []).map(([move, probability]) => (
                      <div className="move-row" key={move}>
                        <strong className="mono">{move}</strong>
                        <div className="move-track"><span style={{ width: `${Math.max(0, Math.min(100, probability * 100))}%` }} /></div>
                        <span>{Math.round(probability * 100)}%</span>
                      </div>
                    ))}
                    {!game.top_moves?.length ? <p className="empty-copy">Waiting for search data</p> : null}
                  </div>
                </div>
              </div>
            </div>
          </Panel>

          <Panel className="pipeline-panel" description="Actor–learner pipeline" icon={Database} title="System pulse">
            <div className="progress-stack">
              <ProgressMetric display={formatNumber(snapshot.inference_batch_size)} label="Inference batch" maxValue={128} value={snapshot.inference_batch_size} />
              <ProgressMetric display={formatNumber(snapshot.inference_queue_depth)} label="Queue depth" maxValue={128} value={snapshot.inference_queue_depth} />
              <ProgressMetric
                display={`${formatNumber(snapshot.replay_samples)} / ${formatNumber(snapshot.replay_capacity)}`}
                label="Replay buffer"
                maxValue={Math.max(1, snapshot.replay_capacity)}
                value={snapshot.replay_samples}
              />
            </div>
            <div className="loss-grid">
              <div><span>Policy loss</span><strong>{formatLoss(snapshot.policy_loss)}</strong></div>
              <div><span>WDL loss</span><strong>{formatLoss(snapshot.value_loss)}</strong></div>
              <div><span>Total loss</span><strong>{formatLoss(snapshot.total_loss)}</strong></div>
              <div><span>Learning rate</span><strong>{snapshot.learning_rate == null ? "—" : snapshot.learning_rate.toExponential(2)}</strong></div>
            </div>
            <span className="subsection-label">Profiler comparison</span>
            <div className="gate-metrics">
              <div><span>Self-play time</span><strong>{formatNumber(snapshot.profile_self_play_baseline_seconds, 1)}s <small>→</small> {formatNumber(snapshot.profile_self_play_optimized_seconds, 1)}s <small>· {formatSpeedup(snapshot.profile_self_play_baseline_seconds, snapshot.profile_self_play_optimized_seconds, true)}</small></strong></div>
              <div><span>Self-play throughput</span><strong>{formatNumber(snapshot.profile_self_play_baseline_positions_per_second, 1)} <small>→</small> {formatNumber(snapshot.profile_self_play_optimized_positions_per_second, 1)} <small>pos/s</small></strong></div>
              <div><span>Training throughput</span><strong>{formatNumber(snapshot.profile_training_baseline_positions_per_second, 1)} <small>→</small> {formatNumber(snapshot.profile_training_optimized_positions_per_second, 1)} <small>· {formatSpeedup(snapshot.profile_training_baseline_positions_per_second, snapshot.profile_training_optimized_positions_per_second)}</small></strong></div>
              <div><span>Batched MCTS search</span><strong>{formatNumber(snapshot.profile_inference_baseline_positions_per_second, 1)} <small>→</small> {formatNumber(snapshot.profile_inference_optimized_positions_per_second, 1)} <small>sim/s · {formatSpeedup(snapshot.profile_inference_baseline_positions_per_second, snapshot.profile_inference_optimized_positions_per_second)}</small></strong></div>
              <div><span>Measured worker optimum</span><strong>{formatNumber(snapshot.profile_optimal_workers)} <small>parallel games</small></strong></div>
            </div>
            <div className="pipeline-footer">
              <span><Clock3 size={13} />Last update</span>
              <strong>{new Date(snapshot.updated_at).toLocaleTimeString("en-US")}</strong>
            </div>
          </Panel>
        </section>
      </main>
    </div>
  );
}

export default App;
