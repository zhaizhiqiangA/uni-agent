DEPLOYMENT=openyuanrong \
AGENT_CONFIG_PATH=examples/swe_agent_blackbox/config/agent_config_yr.yaml \
OPENYUANRONG_SERVER_ADDRESS=124.70.166.142:443 \
OPENYUANRONG_ENV_PREPARE_CMD="unset http_proxy && unset https_proxy && unset no_proxy && pip config set global.index-url https://repo.huaweicloud.com/repository/pypi/simple && pip config set install.trusted-host repo.huaweicloud.com && python3 -m pip install -q swe-rex" \
DEBUG_MODE=1 \
HYDRA_FULL_ERROR=1 \
MODEL_PATH=/data1/models/Qwen/Qwen3.5-4B \
TRAIN_DATA=/home/wtc/dataset/yr/swe_rebench_filtered_openyuanrong.parquet \
VAL_DATA=/home/wtc/dataset/yr/swe_bench_verified_openyuanrong.parquet \
TRAIN_BATCH_SIZE=2 \
PROMPT_LENGTH=4096 \
RESPONSE_LENGTH=4096 \
TP=4 \
N=1 \
TOTAL_EPOCHS=1 \
MAX_TURNS=1 \
ROLLOUT_GPU_MEM_UTIL=0.35 \
SAVE_FREQ=999 \
TEST_FREQ=999 \
bash examples/swe_agent_blackbox/scripts/run_train.sh \
> ./train_yr.log 2>&1 &
