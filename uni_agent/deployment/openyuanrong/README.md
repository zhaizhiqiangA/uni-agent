使用元戎沙箱训练说明

## 1. docker login到华为云

```bash
docker login -u cn-east-3@HPUAQEX0WVY3TQIGMKPZ -p 11ae52cdf16b7b9e5f6da8746646582ca043e18ab0bea2561ebd9d8df6673f5e swr.cn-east-3.myhuaweicloud.com
```

## 2. 安装依赖包

在8.92.9.155机器的`/home/wtc/download/yuanrong`目录下，安装akernel-sdk的whl包。该包会同时安装akernel（元戎沙箱）和openyuanrong-sdk. 目前尚未开源，预计615会开。
参照https://uni-agent.readthedocs.io/en/latest/start/installation.html 安装其他依赖

## 3. 设置环境变量

```bash
export DEPLOYMENT=openyuanrong
export OPENYUANRONG_SERVER_ADDRESS=124.70.166.142:443
# 该token有一年多的有效期，足以在开发阶段使用
export OPENYUANRONG_TOKEN=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE4MTU2NzgwNjUsInJvbGUiOiJkZXZlbG9wZXIiLCJzdWIiOiJkZWZhdWx0In0.YWJjODAxZGQzZTJiN2U1NWNjNWY0MjAwMTg1ZTI1NmM2M2E3OTIxN2QxNDQxODVkMzU5YmM0N2YyNGJlYWY0Yg
# 部分docker image设置了http_proxy等环境变量，需要取消才能正常访问网络
export OPENYUANRONG_ENV_PREPARE_CMD="unset http_proxy && unset https_proxy && unset no_proxy && pip config set global.index-url https://repo.huaweicloud.com/repository/pypi/simple && pip config set install.trusted-host repo.huaweicloud.com && python3 -m pip install -q swe-rex"
# 看到更多日志
export DEBUG_MODE=1
# 仅demo.py需要
export OPENYUANRONG_DEPLOYMENT_IMAGE=swr.cn-east-3.myhuaweicloud.com/openyuanrong/swe-rebench/0b01001001_1776_spectree-64:latest
```

## 4. 跑通demo.py

指定一个镜像：

```bash
export OPENYUANRONG_DEPLOYMENT_IMAGE=swr.cn-east-3.myhuaweicloud.com/openyuanrong/swe-rebench/0b01001001_1776_spectree-64:latest
```
然后运行`python examples/agent_env/demo.py`。如要看更多debug信息可以设置`export DEBUG_MODE=1`

`swr.cn-east-3.myhuaweicloud.com/openyuanrong/`下已经上传了swe-bench-verified，swe-rebench-filtered和r2e-gym-subset-filtered数据集的全部配套镜像。镜像命名和vefaas一致。如果要检查某个镜像是否存在，可以在docker login之后运行如下命令。如打印出信息，说明镜像已上传。

```bash
docker manifest inspect swr.cn-east-3.myhuaweicloud.com/openyuanrong/swe-rebench/0b01001001_1776_spectree-64:latest
```

可以在.parquet文件中看到所有题目对应的image路径。


## 5. 跑通端到端训练

先用examples/data_preprocess下的脚本生成适配元戎环境的.parquet文件，例如：

```bash
DEPLOYMENT=openyuanrong python3 examples/data_preprocess/swe_bench_verified.py --local-save-dir /your/data/path
```

### uni-agent

参考以下脚本。和元戎相关的环境变量不要改，模型和数据路径需指定，训练参数根据硬件配置调整。
```bash
   DEPLOYMENT=openyuanrong \
   OPENYUANRONG_SERVER_ADDRESS=124.70.166.142:443 \
   OPENYUANRONG_TOKEN=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE4MTU2NzgwNjUsInJvbGUiOiJkZXZlbG9wZXIiLCJzdWIiOiJkZWZhdWx0In0.YWJjODAxZGQzZTJiN2U1NWNjNWY0MjAwMTg1ZTI1NmM2M2E3OTIxN2QxNDQxODVkMzU5YmM0N2YyNGJlYWY0Yg \
   OPENYUANRONG_ENV_PREPARE_CMD="unset http_proxy && unset https_proxy && unset no_proxy && pip config set global.index-url https://repo.huaweicloud.com/repository/pypi/simple && pip config set install.trusted-host repo.huaweicloud.com && python3 -m pip install -q swe-rex" \
   DEBUG_MODE=1 \
   HYDRA_FULL_ERROR=1 \
   MODEL_PATH=/data1/models/Qwen/Qwen3.5-4B \
   AGENT_CONFIG_PATH=examples/swe_agent_blackbox/config/agent_config_yr.yaml \
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
   > train_yr.log 2>&1 &
```

### swe-agent

TODO

## 6. 黄绿区的额外配置

设置好http_proxy, https_proxy后，先`curl 124.70.166.142:443`，如果返回404 page not found，代表可以连通元戎服务器。
设置环境变量
```bash
export YR_ENABLE_HTTP_PROXY=true
```
在demo.py的 impl == "openyuanrong"分支，设置proxy为os.getenv('http_proxy')。然后跑demo.py, 看结果是否和蓝区一致。
