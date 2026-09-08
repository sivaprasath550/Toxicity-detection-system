# Results log

Running log of every evaluated run, in order. Each entry links a config
(commit/YAML) to the metrics it produced, so the README's summary tables
are always traceable back to a specific run rather than hand-typed.

Fill in one block per run:

```
## <date> — <short description>
- config: config/config.yaml (commit <hash> or diff from default)
- data: <full 1.8M | dev subsample N=...>
- weighting: none | inverse_frequency | metric_derived
- multitask heads: on | off

overall_auc: 
subgroup_auc_power_mean: 
bpsn_auc_power_mean: 
bnsp_auc_power_mean: 
final_score: 

per-subgroup table: reports/<run_name>_per_subgroup.csv
notes:
```
