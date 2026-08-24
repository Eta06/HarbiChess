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

