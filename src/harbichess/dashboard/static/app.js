const $ = (id) => document.getElementById(id);
const number = (value, digits = 0) => Number(value || 0).toLocaleString(undefined, { maximumFractionDigits: digits });
const loss = (value) => value == null ? "—" : Number(value).toFixed(3);
let latest = null;
let receivedAt = performance.now();

function duration(seconds) {
  const total = Math.max(0, Math.floor(seconds || 0));
  const days = Math.floor(total / 86400);
  const hours = Math.floor((total % 86400) / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  const clock = [hours, minutes, secs].map(v => String(v).padStart(2, "0")).join(":");
  return days ? `${days}d ${clock}` : clock;
}

function setText(id, value) { $(id).textContent = value; }
function setBar(id, value) { $(id).style.width = `${Math.max(0, Math.min(100, value))}%`; }
function signed(value, digits = 0) {
  if (value == null) return "—";
  return `${value >= 0 ? "+" : ""}${Number(value).toFixed(digits)}`;
}

function drawChart(canvasId, history, valueOf, color, bandOf = null) {
  const canvas = $(canvasId);
  const width = Math.max(240, canvas.clientWidth);
  const height = canvas.classList.contains("compact") ? 105 : 190;
  const scale = window.devicePixelRatio || 1;
  canvas.width = width * scale; canvas.height = height * scale;
  const context = canvas.getContext("2d"); context.scale(scale, scale);
  const points = (history || []).map(point => ({ value: valueOf(point), band: bandOf?.(point) })).filter(point => point.value != null);
  context.clearRect(0, 0, width, height);
  if (points.length < 2) {
    context.fillStyle = "#858d94"; context.font = "11px ui-monospace";
    context.fillText("Waiting for history", 12, height / 2); return;
  }
  const values = points.flatMap(point => point.band ? [point.value, ...point.band.filter(value => value != null)] : [point.value]);
  let min = Math.min(...values), max = Math.max(...values);
  const padding = Math.max((max - min) * .14, 1); min -= padding; max += padding;
  const x = index => 10 + index / (points.length - 1) * (width - 20);
  const y = value => 8 + (max - value) / (max - min) * (height - 18);
  context.strokeStyle = "rgba(255,255,255,.07)"; context.lineWidth = 1;
  for (let line = 1; line < 4; line++) { const lineY = line * height / 4; context.beginPath(); context.moveTo(0,lineY); context.lineTo(width,lineY); context.stroke(); }
  if (bandOf && points.every(point => point.band?.every(value => value != null))) {
    context.beginPath(); points.forEach((point, index) => index ? context.lineTo(x(index),y(point.band[1])) : context.moveTo(x(index),y(point.band[1])));
    [...points].reverse().forEach((point, reverseIndex) => context.lineTo(x(points.length - 1 - reverseIndex),y(point.band[0])));
    context.closePath(); context.fillStyle = "rgba(231,183,95,.12)"; context.fill();
  }
  context.beginPath(); points.forEach((point, index) => index ? context.lineTo(x(index),y(point.value)) : context.moveTo(x(index),y(point.value)));
  context.strokeStyle = color; context.lineWidth = 2; context.stroke();
  const last = points.at(-1); context.beginPath(); context.arc(x(points.length - 1),y(last.value),3.5,0,Math.PI*2); context.fillStyle = color; context.fill();
}

const pieces = { p:"♟", r:"♜", n:"♞", b:"♝", q:"♛", k:"♚", P:"♙", R:"♖", N:"♘", B:"♗", Q:"♕", K:"♔" };
function fenBoard(fen, lastMove) {
  const board = $("board");
  board.replaceChildren();
  const rows = (fen || "8/8/8/8/8/8/8/8").split(" ")[0].split("/");
  const highlights = new Set(lastMove && lastMove.length >= 4 ? [lastMove.slice(0,2), lastMove.slice(2,4)] : []);
  rows.forEach((row, visualRank) => {
    let file = 0;
    for (const token of row) {
      if (/\d/.test(token)) { for (let i=0; i<Number(token); i++) addSquare(visualRank, file++, ""); }
      else addSquare(visualRank, file++, pieces[token] || "");
    }
  });
  function addSquare(visualRank, file, piece) {
    const rank = 8 - visualRank;
    const name = `${"abcdefgh"[file]}${rank}`;
    const square = document.createElement("div");
    square.className = `square ${(visualRank + file) % 2 ? "dark" : "light"}${highlights.has(name) ? " last" : ""}`;
    square.textContent = piece;
    square.title = name;
    board.appendChild(square);
  }
}

function renderTopMoves(moves) {
  const root = $("top-moves"); root.replaceChildren();
  (moves || []).forEach(([move, probability]) => {
    const row = document.createElement("div"); row.className = "move-row";
    row.innerHTML = `<span>${move}</span><div class="move-track"><i style="width:${probability*100}%"></i></div><span>${Math.round(probability*100)}%</span>`;
    root.appendChild(row);
  });
  if (!moves?.length) root.textContent = "Waiting for search data";
}

function render(data) {
  latest = data; receivedAt = performance.now();
  $("connection").className = "connection live"; setText("connection", "LIVE");
  const mode = data.mode || "IDLE"; setText("mode", mode.replace("_", " "));
  $("mode").className = `mode ${mode.toLowerCase()}`;
  $("demo-badge").className = data.demo ? "demo" : "demo hidden";
  setText("mode-detail", data.mode_detail); setText("active-checkpoint", data.active_checkpoint);
  setText("promoted-checkpoint", data.promoted_checkpoint); setText("candidate-checkpoint", data.candidate_checkpoint); setText("source-commit", data.source_commit);
  setText("lifetime-games", number(data.lifetime_games)); setText("run-games", number(data.run_games)); setText("generation-games", number(data.generation_games)); setText("active-games", number(data.active_games)); setText("games-hour", number(data.games_per_hour));
  setText("training-step", number(data.training_step)); setText("positions-sec", number(data.positions_per_second)); setText("evals-sec", number(data.neural_evals_per_second)); setText("nodes-sec", number(data.mcts_nodes_per_second));
  setText("batch-size", number(data.inference_batch_size)); setText("queue-depth", number(data.inference_queue_depth)); setBar("batch-bar", data.inference_batch_size / 128 * 100); setBar("queue-bar", data.inference_queue_depth / 128 * 100);
  setText("replay-label", `${number(data.replay_samples)} / ${number(data.replay_capacity)}`); setBar("replay-bar", data.replay_capacity ? data.replay_samples / data.replay_capacity * 100 : 0);
  setText("policy-loss", loss(data.policy_loss)); setText("value-loss", loss(data.value_loss)); setText("total-loss", loss(data.total_loss)); setText("learning-rate", data.learning_rate == null ? "—" : Number(data.learning_rate).toExponential(2));
  setText("elo-delta", signed(data.arena_elo_delta));
  setText("elo-interval", data.arena_elo_low == null ? "—" : `${signed(data.arena_elo_low)} / ${signed(data.arena_elo_high)}`);
  setText("arena-score", data.arena_games ? `${number(data.arena_score_rate * 100, 1)}%` : "—");
  setText("arena-games", `${number(data.arena_games)} games`); setText("arena-wins", number(data.arena_wins)); setText("arena-draws", number(data.arena_draws)); setText("arena-losses", number(data.arena_losses));
  const promotion = $("promotion-status"); setText("promotion-status", data.promotion_ready ? "PROMOTION READY" : "COLLECTING"); promotion.className = data.promotion_ready ? "pill promotion-ready" : "pill";
  const history = data.history || [];
  setText("loss-trend-value", loss(history.at(-1)?.total_loss)); setText("throughput-trend-value", number(history.at(-1)?.games_per_hour));
  drawChart("elo-chart", history, point => point.elo_delta, "#e7b75f", point => [point.elo_low, point.elo_high]);
  drawChart("loss-chart", history, point => point.total_loss, "#64e9dc");
  drawChart("throughput-chart", history, point => point.games_per_hour, "#87e38d");
  setText("run-id", data.run_id); setText("updated-at", new Date(data.updated_at).toLocaleTimeString());
  const game = data.live_game || {}; setText("game-id", game.game_id || "No active game"); setText("ply", `PLY ${game.ply || 0}`); setText("last-move", game.last_move || "—");
  const [win=0, draw=1, lose=0] = game.wdl || []; setText("wdl-win", `${Math.round(win*100)}%`); setText("wdl-draw", `${Math.round(draw*100)}%`); setText("wdl-loss", `${Math.round(lose*100)}%`);
  fenBoard(game.fen, game.last_move); renderTopMoves(game.top_moves);
  updateClocks();
}

function updateClocks() {
  if (!latest) return;
  const delta = (performance.now() - receivedAt) / 1000;
  setText("session-time", duration(latest.session_elapsed_seconds + delta));
  const trainingDelta = latest.mode === "TRAINING" ? delta : 0;
  setText("training-time", duration(latest.training_elapsed_seconds + trainingDelta));
}
setInterval(updateClocks, 1000);

const events = new EventSource("/api/events");
events.onmessage = (event) => render(JSON.parse(event.data));
events.onerror = () => { $("connection").className = "connection"; setText("connection", "RECONNECTING"); };
fetch("/api/snapshot").then(r => r.json()).then(render).catch(() => setText("connection", "OFFLINE"));
window.addEventListener("resize", () => latest && render(latest));
