set -e
VPY=/home/ian/.venvs/venv/bin/python
CFG=configs/sde_lstm_default.yaml
echo "### C4 TRAIN ###";            $VPY scripts/run_train_sde_lstm.py --config $CFG
echo "### C4 SCORE ###";            $VPY scripts/run_score_sde_lstm.py --config $CFG
echo "### C4 STRESS p99 ###";       $VPY scripts/run_stress_test_sde_lstm.py --config $CFG --score-name total_nll --quantile 0.99
echo "### C4 STRESS p95 ###";       $VPY scripts/run_stress_test_sde_lstm.py --config $CFG --score-name total_nll --quantile 0.95
echo "### C4 COMPARE ###";          $VPY scripts/run_compare_thresholds.py
echo "### C4 ROLLOUT ###";          $VPY scripts/run_rollout_sde_lstm.py --config $CFG
echo "### C4 STATIONARY RULE ###";  $VPY scripts/run_stationary_rule.py --config $CFG
echo "### C4 STATIONARY STRESS ###";$VPY scripts/run_stress_test_stationary_rule.py --config $CFG --max-samples 50000
echo "### C4 FUSED ###";            $VPY scripts/run_fused_sde_stationary.py --config $CFG --score-name total_nll --quantile 0.99 --max-samples 50000
echo "### C4 CHAIN COMPLETE ###"
