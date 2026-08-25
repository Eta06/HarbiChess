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
  policy_loss: number | null;
  value_loss: number | null;
  validation_loss: number | null;
}

export type PilotStatus =
  | "NOT_STARTED"
  | "SELF_PLAY"
  | "REPLAY"
  | "TRAINING"
  | "PASSED"
  | "FAILED";

export type CheckpointStatus = "NONE" | "WRITING" | "VERIFIED" | "FAILED";

export interface OpeningDiversity {
  ply: number;
  eligible_games: number;
  unique_prefixes: number;
  entropy: number;
  effective_prefixes: number;
}

export interface TerminationSnapshot {
  termination: string;
  count: number;
  ratio: number;
}

export interface DiversitySnapshot {
  games: number;
  positions: number;
  unique_game_ratio: number;
  duplicate_game_ratio: number;
  unique_position_ratio: number;
  selected_actions: number;
  action_space_coverage: number;
  mean_policy_entropy: number;
  effective_policy_branches: number;
  mean_game_plies: number;
  white_wins: number;
  draws: number;
  black_wins: number;
  decisive_games: number;
  decisive_game_ratio: number;
  max_ply_draws: number;
  max_ply_draw_ratio: number;
  repetition_redirects: number;
  repetition_redirect_ratio: number;
  terminations: TerminationSnapshot[];
  openings: OpeningDiversity[];
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
  profile_self_play_baseline_seconds: number | null;
  profile_self_play_optimized_seconds: number | null;
  profile_self_play_baseline_positions_per_second: number | null;
  profile_self_play_optimized_positions_per_second: number | null;
  profile_training_baseline_positions_per_second: number | null;
  profile_training_optimized_positions_per_second: number | null;
  profile_inference_baseline_positions_per_second: number | null;
  profile_inference_optimized_positions_per_second: number | null;
  profile_optimal_workers: number | null;
  replay_samples: number;
  replay_capacity: number;
  policy_loss: number | null;
  value_loss: number | null;
  total_loss: number | null;
  learning_rate: number | null;
  pilot_status: PilotStatus;
  pilot_steps_planned: number;
  pilot_steps_completed: number;
  pilot_steps_attempted: number;
  pilot_best_validation_step: number;
  pilot_best_validation_loss: number | null;
  pilot_stopped_early: boolean;
  pilot_initial_train_loss: number | null;
  pilot_final_train_loss: number | null;
  pilot_initial_validation_loss: number | null;
  pilot_final_validation_loss: number | null;
  pilot_max_gradient_norm: number | null;
  pilot_reasons: string[];
  checkpoint_status: CheckpointStatus;
  checkpoint_path: string;
  checkpoint_verified: boolean;
  replay_shards: number;
  validation_samples: number;
  validation_checkpoint_count: number;
  continuation_replay_samples: number;
  diversity: DiversitySnapshot;
  demo: boolean;
  live_game: LiveGame;
  arena_games: number;
  arena_wins: number;
  arena_draws: number;
  arena_losses: number;
  arena_decisive_games: number;
  arena_threefold_repetitions: number;
  arena_avoidable_threefold_repetitions: number;
  arena_continuation_replay_samples: number;
  arena_max_ply_draws: number;
  arena_other_draws: number;
  arena_score_rate: number;
  arena_elo_delta: number | null;
  arena_elo_low: number | null;
  arena_elo_high: number | null;
  promotion_ready: boolean;
  history: HistoryPoint[];
}

export type ConnectionState = "connecting" | "live" | "reconnecting" | "offline";
