# MiniOneRec-5090D GitHub 发布指南

本指南用于把 `D:\Document\OneminiRec\MiniOneRec` 发布为一个独立 GitHub
仓库。当前 `MiniOneRec` 目录位于外层 Git 仓库中，但它自己还不是独立仓库，
因此需要在该目录内重新初始化 Git。

## 发布范围

GitHub 仓库应包含：

- MiniOneRec 源码及 Apache-2.0 `LICENSE`
- `repro/` 下的复现、诊断、训练和评估脚本
- `repro/archive/` 下的小型实验指标与归档说明
- 研究主页、benchmark 文档和恢复训练说明

GitHub 仓库不应直接包含：

- `output_dir/` 中约 69 GB 的模型 checkpoint
- `data/Amazon/` 中的下载或派生数据
- `results/` 中的逐样本预测结果
- `repro/logs/` 中的本地训练日志
- Hugging Face 缓存、conda 环境、API token 或其他凭据

模型权重以后可单独发布到 Hugging Face Hub 或 GitHub Release，并在 README
中添加下载地址。

## 第一步：在 GitHub 创建空仓库

1. 登录 GitHub，点击右上角 `+`，选择 `New repository`。
2. 推荐仓库名：`MiniOneRec-5090D` 或 `minionerec-qwen25-3b-5090d`。
3. 选择 `Public`，如果暂时不希望公开实验细节则选择 `Private`。
4. 不要勾选 `Add a README file`、`.gitignore` 或 `Choose a license`。
5. 点击 `Create repository`，保留页面上显示的仓库 URL。

## 第二步：在 WSL 中进入项目

```bash
cd /mnt/d/Document/OneminiRec/MiniOneRec
pwd
```

`pwd` 应输出：

```text
/mnt/d/Document/OneminiRec/MiniOneRec
```

## 第三步：初始化独立仓库

```bash
git init -b main
git status --short
```

如果 Git 提示需要用户信息，只需设置一次：

```bash
git config --global user.name "你的英文名或 GitHub 用户名"
git config --global user.email "你的 GitHub 邮箱"
```

## 第四步：检查大文件是否被排除

```bash
git check-ignore -v output_dir/qwen25_3b_Industrial_and_Scientific_single5090d_finalonly_sft/final_checkpoint/model-00001-of-00002.safetensors
git check-ignore -v data/Amazon/index/Industrial_and_Scientific.index.json
git check-ignore -v results/a0_finalonly_single5090d/final_result_Industrial_and_Scientific.json
```

三条命令都应显示由 `.gitignore` 命中的规则。再检查工作区中超过 90 MB 的
文件：

```bash
find . -type f -size +90M -not -path './.git/*' -print
```

该命令可以列出本地大文件，但这些文件不应进入暂存区。

## 第五步：暂存并复核

```bash
git add .
git status --short
git diff --cached --stat
```

重点确认暂存列表中没有以下路径：

```text
output_dir/
data/Amazon/
results/
repro/logs/
```

确认暂存区没有超过 GitHub 单文件限制的文件：

```bash
git diff --cached --name-only -z | xargs -0 -r du -h | sort -h | tail -20
```

## 第六步：创建首个提交

```bash
git commit -m "Add RTX 5090D MiniOneRec reproduction and research benchmark"
```

## 第七步：连接 GitHub

把下面的 `<YOUR_USERNAME>` 和仓库名替换成你的实际信息：

```bash
git remote add origin https://github.com/<YOUR_USERNAME>/MiniOneRec-5090D.git
git remote -v
```

如果之前误加了远程地址，使用：

```bash
git remote set-url origin https://github.com/<YOUR_USERNAME>/MiniOneRec-5090D.git
```

## 第八步：登录并推送

推荐使用 GitHub CLI 完成浏览器授权：

```bash
sudo apt update
sudo apt install gh -y
gh auth login
git push -u origin main
```

在 `gh auth login` 中依次选择：

1. `GitHub.com`
2. `HTTPS`
3. `Login with a web browser`
4. 浏览器中输入终端显示的一次性验证码并授权

推送完成后刷新 GitHub 仓库页面即可。

## 第九步：确认 GitHub 项目首页

根目录 `README.md` 已经是本地 RTX 5090D 研究首页，包含本地代码入口、实验
结果、消融结论和研究边界。上游项目说明保存在 `README_UPSTREAM.md`，用于
学术署名、原始使用说明和许可证追溯，无需再执行 README 改名操作。

## 后续更新

每完成一个实验，先更新 `repro/BENCHMARK_5090D.md` 和
`repro/archive/`，然后提交：

```bash
git add README.md RESEARCH_CONTRIBUTIONS.md repro
git commit -m "Record D1 seed stability experiment"
git push
```

不要提交 checkpoint。需要公开模型时，将 checkpoint 上传到 Hugging Face
Hub，并在研究 README 中记录模型版本、基线 checkpoint、训练配置和 SHA256。
