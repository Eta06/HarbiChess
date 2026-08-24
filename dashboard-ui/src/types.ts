export type RunMode =
  | "SELF_PLAY"
  | "TRAINING"
  | "EVALUATION"
  | "CHECKPOINTING"
  | "PAUSED"
  | "IDLE";

export interface LiveGame {
  game_id: string;
  white: string;
  black: string;
  fen: string;
  last_move: string;
  ply: number;
  top_moves: [string, number][];
  wdl: [number, number, number];
}

export interface HistoryPoint {
  training_step: number;
  training_elapsed_seconds: number;
  lifetime_games: number;
  total_loss: number | null;
  elo_delta: number | null;
  elo_low: number | null;
  elo_high: number | null;
  games_per_hour: number;
  positions_per_second: number;
}

export interface DashboardSnapshot {
  schema_version: number;
  updated_at: string;
  mode: RunMode;
  mode_detail: string;
  run_id: string;
  session_elapsed_seconds: number;
  training_elapsed_seconds: number;
  active_checkpoint: string;
  promoted_checkpoint: string;
  candidate_checkpoint: string;
  source_commit: string;
  training_step: number;
  lifetime_games: number;
  run_games: number;
  generation_games: number;
  active_games: number;
  completed_games: number;
  failed_games: number;
  games_per_hour: number;
  positions_per_second: number;
  neural_evals_per_second: number;
  mcts_nodes_per_second: number;
  inference_batch_size: number;
  inference_queue_depth: number;
  replay_samples: number;
  replay_capacity: number;
  policy_loss: number | null;
  value_loss: number | null;
  total_loss: number | null;
  learning_rate: number | null;
  demo: boolean;
  live_game: LiveGame;
  arena_games: number;
  arena_wins: number;
  arena_draws: number;
  arena_losses: number;
  arena_score_rate: number;
  arena_elo_delta: number | null;
  arena_elo_low: number | null;
  arena_elo_high: number | null;
  promotion_ready: boolean;
  history: HistoryPoint[];
}

export type ConnectionState = "connecting" | "live" | "reconnecting" | "offline";
