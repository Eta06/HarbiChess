from pathlib import Path
p=Path(__file__).resolve().parent
old=p.parent/'pv-history-search-v1/search_v2.py'
s=old.read_text().replace('Budgeted PVS/history prototype. Conventional engine methods, no strength claim.','Progressive-width PVS/history prototype: declared root subsets, no strength claim.')
s=s.replace('    completed_root_passes: int = 0','''    completed_root_passes: int = 0
    root_width_mode: str = "progressive"
    full_legal_completed_depth: int = 0
    completed_root_coverage: tuple[tuple[int, tuple[str, ...]], ...] = ()
    attempted_depth: int = 0
    attempted_root_actions: tuple[str, ...] = ()
    completed_attempted_root_actions: int = 0
    depth_semantics: str = "completed declared root subset; not full-legal minimax"
    root_estimate_semantics: str = "previous completed-pass fail-soft estimates; may be bounds"''')
s=s.replace('        use_pvs=True,','        use_pvs=True,\n        root_width="progressive",')
s=s.replace('            or incheck_extensions not in (0, 1)','            or incheck_extensions not in (0, 1)\n            or root_width not in ("progressive", "full")')
s=s.replace('        self.evaluator = evaluator','        self.root_width = root_width\n        self.evaluator = evaluator')
s=s.replace('            return Result(None, terminal, self.nodes, 0, 0, 0)','            return Result(None, terminal, self.nodes, 0, 0, 0, root_width_mode=self.root_width)')
s=s.replace('        passes = 0\n        for depth', '''        passes = 0
        active_moves = moves
        coverage = []
        full_depth = 0
        attempted_depth = 0
        attempted_actions = ()
        completed_attempted_actions = 0
        for depth''')
s=s.replace('                    moves, key=lambda m: (0 if m == winner else 1, -scores[m], m.uci())','                    active_moves, key=lambda m: (0 if m == winner else 1, -scores[m], m.uci())')
s=s.replace('                for index, move in enumerate(ordered):','''                if self.root_width == "progressive" and depth >= 2:
                    width = 8 if depth == 2 else 4 if depth == 3 else 2
                    ordered = ordered[:width]
                attempted_depth = depth
                attempted_actions = tuple(m.uci() for m in ordered)
                completed_attempted_actions = 0
                for index, move in enumerate(ordered):''')
s=s.replace('                    iteration[move] = value','                    iteration[move] = value\n                    completed_attempted_actions += 1')
s=s.replace('            scores = iteration','''            scores = iteration
            active_moves = list(iteration)
            coverage.append((depth, attempted_actions))
            if len(iteration) == len(moves):
                full_depth = depth''')
s=s.replace('            passes,\n        )','''            passes,
            self.root_width,
            full_depth,
            tuple(coverage),
            attempted_depth,
            attempted_actions,
            completed_attempted_actions,
        )''')
(p/'search.py').write_text(s)
