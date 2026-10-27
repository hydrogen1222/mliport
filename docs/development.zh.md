# 开发

这一页面向要修改 mliport 代码的人。只是使用的话，看[安装](installation.md)就够了。

## 两种环境

开发时有两类环境：一个**核心开发环境**，和若干个独立的**后端运行环境**，两者用途不同。

核心环境用来跑单元测试、集成测试、lint 和文档构建。仓库根目录只放了工具配置，在那里执行 `uv sync`
装不上任何有用的东西（没有 numpy、pytest，也没有任何 MLIP 后端）。请按 CI 的做法，从包的元数据建
环境：

```bash
uv venv .venv-dev
uv pip install --python .venv-dev/bin/python -e './mliport[dev]'
uv pip install --python .venv-dev/bin/python torch --index-url https://download.pytorch.org/whl/cpu
.venv-dev/bin/python -m pytest -q
.venv-dev/bin/ruff check mliport/mliport tests scripts
.venv-dev/bin/ruff format --check mliport/mliport tests scripts
python scripts/check_repository_hygiene.py
```

核心环境里没有真实的 UMA、MACE、DPA、GRACE，相关测试用的是合成的计算器。要测真实后端，需要用
安装脚本建的环境：

```bash
./scripts/install_mliport.sh --engines mace dpa grace uma --device auto
.venv-mace/bin/mliport doctor --engine mace --device auto
```

完整的 GPU 验证在 `validation/` 下，用的就是这些环境；见[验证](validation.md)。

## 目录结构

```text
mliport/mliport/           包的源码
  calculators/             后端适配器与工厂
  config/                  解析器、schema、默认值、设置
  runners/                 SP、优化、MD、NEB 与批量执行
  confsearch/             基于 MD 的构型搜索（adsorbate 模式）
  analysis/                轨迹分析
  tui/                     Textual 界面
  install/                 安装计划、兼容性注册表、启动器
tests/mliport/             单元与集成测试
validation/science/        验证活动、证据、报告、模型清单
docs/                      本文档（英文与中文）
scripts/                   仓库工具：安装器、文档生成、卫生检查
```

可安装的包在 `mliport/` 子目录，有自己的 `mliport/pyproject.toml`。根目录的 `pyproject.toml` 只是开发
工具的配置（ruff、pytest、uv），直接安装根目录会报包查找错误，这是有意的。请用
`./scripts/install_mliport.sh`，手动安装时指向 `mliport/` 子目录。

## 写文档

文档站用 MkDocs Material 构建，从 `main` 分支发布到 GitHub Pages。每个页面有两份：`page.md` 是英文，
`page.zh.md` 是中文。先写英文，再按相同结构翻译成中文。中文请写成自然的中文，不要逐句直译。

```bash
pip install -r requirements-docs.txt
mkdocs serve
```

下面几条规则保证两种语言一致，测试会自动检查：

- 中文页和英文页的 `##` 小节数、代码块数、链接目标相同，自动生成的区块也相同。
- 页面之间的链接写英文文件名（`installation.md`，不要写 `installation.zh.md`），站点会自动切换语言。
  指向 `docs/` 以外的链接必须写完整的 GitHub 地址。
- 新页面要加到 `mkdocs.yml` 的 `nav` 里，也要加到文档测试的页面列表里。

有些内容来自代码，页面里用自动生成的区块展示：

```markdown
<!-- BEGIN GENERATED: backend-matrix -->
...不要手动编辑...
<!-- END GENERATED -->
```

环境名、Python 版本范围、框架版本和 GPU 安装方案由兼容性注册表（`mliport/install/compatibility.py`）
决定，固定的模型 profile 由模型清单决定。要改这些内容，请改源文件再重新生成：

```bash
python scripts/generate_docs.py          # 刷新生成块和 mliport/README.md
python scripts/generate_docs.py --check  # CI 运行的就是这条
```

## 测试

- 推送前运行 `.venv-dev/bin/python -m pytest -q`，应当全部通过。
- 核心代码支持 Python 3.10 到 3.13，各后端支持的范围更窄，见[安装](installation.md)。
- 文档测试会检查链接和锚点，用 `--help` 运行文档里出现的命令，编译 Python 代码片段，并核对自动生成
  的区块。
- 修改时先写一个会失败的测试，再写修复。

## 构建包

```bash
uv build mliport              # sdist 与 wheel
```

检查元数据（README、引用信息、包名），并确认 wheel 里没有 `mlipx` 命名空间或临时的计划文档。

## 提交与拉取请求

- 一个 PR 只做一件事。
- 提交信息要写清楚：原来哪里出错、怎么修的，以及**数据影响**（写"没有科学记录失效"，或者说明重新
  生成了哪些验证记录、为什么）。
- CI 必须全部通过：Python 3.10 到 3.13 的测试、lint、打包和文档构建。
- 不要为了让报告好看而放宽阈值，也不要删掉如实标注的 `not_run` 记录。
