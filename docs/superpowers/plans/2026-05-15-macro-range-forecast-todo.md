# Macro Range Forecast TODO

## Immediate

- [ ] Build a daily target table for macro range pct: `(macro_high - macro_low) / close_at_15_49`
- [ ] Define full-history and post-COVID experiment slices
- [ ] Implement fixed 2-year walk-forward splits

## Baselines

- [ ] Rolling empirical quantile baseline
- [ ] HAR-RV baseline for range/volatility forecasting

## Main Model

- [ ] Quantile regression / boosting model for 10/25/50/75/90 bands
- [ ] Same-day pre-15:50 feature set
- [ ] Rolling history feature set
- [ ] Economic calendar feature set

## Evaluation

- [ ] Pinball loss by quantile
- [ ] Coverage / calibration checks
- [ ] Full-history vs post-COVID comparison
- [ ] Event-day vs non-event-day breakdown

## Future Ideas

- [ ] Predict 15:50–15:54 range from first macro minute + pre-15:50 context
- [ ] Predict 15:55–15:59 from 15:50–15:54 behavior
