# Post ADX Ablation Design

## Goal
Add a repeatable ablation experiment group for the current winning trendability representation: post-COVID `adx_parts`.

## Scope
- Add a new experiment group `post_adx_ablation`
- Restrict it to post-COVID only
- Support all session names through the existing CLI
- Initial variants:
  - base `adx_parts`
  - `adx_parts_minus_persistence`
  - `adx_parts_minus_crossover`
  - `adx_parts_minus_log_vr`
  - optional `adx_parts_minus_irr`
- Keep the existing representation sweep unchanged

## Design
- Extend `features/trend/modeling/registry.py` with a focused ablation spec builder.
- Leave the main registry intact; the new group is additive, not a replacement.
- Extend the CLI `--experiment-group` choices to include `post_adx_ablation`.
- Reuse the existing walk-forward runner and summary flow unchanged.

## Success Criteria
- `python3 -m features.trend.modeling.cli run-experiments --experiment-group post_adx_ablation ...` writes experiment outputs for the selected session.
- Summary tables show the reduced variants beside the base `adx_parts` model.
- Existing modeling tests continue to pass.
