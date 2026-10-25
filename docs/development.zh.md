# 开发

这一页写给要修改 mliport 本身的人。如果你只是想运行它，需要的是[安装指南](installation.md)。

## 两种环境

mliport 有一个**核心开发环境**和若干独立的**后端运行时环境**，它们不能互相替代。

核心环境用来跑单元测试和集成测试、lint 以及文档构建。仓库根目录只是工具配置，所以在那里执行
`uv sync` 什么有用的都装不上（没有 numpy，没有 pytest），更不会装上任何 MLIP 运行时。请像 CI 那样，
从包的元数据来建环境：

```bash
uv venv .venv-dev
uv pip install --python .venv-dev/bin/python -e './mliport[dev]'
uv pip install --python .venv-dev/bin/python torch --index-url https://download.pytorch.org/whl/cpu
.venv-dev/bin/python -m pytest -q
.venv-dev/bin/ruff check mliport/mliport tests scripts
.venv-dev/bin/ruff format --check mliport/mliport tests scripts
python scripts/check_repository_hygiene.py
```

这个环境对 UMA、MACE、DPA、GRACE 什么也证明不了，核心测试在这些路径上用的是合成计算器。要对真实
后端做检查，需要安装器建出来的环境：

```bash
./scripts/install_mliport.sh --engines mace dpa grace uma --device auto
.venv-mace/bin/mliport doctor --engine mace --device auto
```

完整的 GPU 验证活动放在 `validation/` 下，使用的就是这些环境；见[验证](validation.md)。

## 东西都在哪

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

可安装的包是 `mliport/`，带有它自己的 `mliport/pyproject.toml`。仓库根目录的 `pyproject.toml` 只是开发
者工具（ruff、pytest、uv），所以直接安装仓库根目录是有意不支持的：setuptools 看到的是仓库布局，会以
包发现错误停下。请使用 `./scripts/install_mliport.sh`，或者让手动管理的环境指向 `mliport/` 子目录。

## 写文档

站点用 MkDocs Material 构建，从 `main` 发布到 GitHub Pages。每个页面都有两份：`page.md` 是英文，
`page.zh.md` 是对应的中文。没有工具会替你翻译，所以请用相同的结构亲手写两份。

```bash
pip install -r requirements-docs.txt
mkdocs serve
```

下面几条规则让两种语言保持一致，测试会强制检查：

- 翻译页与英文页的 `##` 小节数、代码块数、链接数相同，生成块也相同。
- 页面之间的链接写英文文件名（`installation.md`，不要写 `installation.zh.md`）；语言由站点解析。离开
  `docs/` 的链接必须是完整的 GitHub URL。
- 新页面要加进 `mkdocs.yml` 的 `nav`，也要加进文档契约测试里的页面列表。

有些事实存在于代码里，页面通过生成块来展示它们：

```markdown
<!-- BEGIN GENERATED: backend-matrix -->
...不要手动编辑...
<!-- END GENERATED -->
```

兼容性注册表（`mliport/install/compatibility.py`）掌管环境名、Python 范围、框架固定版本和 GPU 路线；
模型清单掌管固定的 profile。要改其中任何一项，请改源头再重新生成：

```bash
python scripts/generate_docs.py          # 刷新生成块和 mliport/README.md
python scripts/generate_docs.py --check  # CI 运行的就是这条
```

## 测试

- 推送之前运行 `.venv-dev/bin/python -m pytest -q`，应当全部通过。
- 核心代码支持 Python 3.10 到 3.13。后端的范围更窄；见[安装](installation.md)。
- 文档契约测试会检查链接和锚点、用 `--help` 运行文档里的 CLI 命令、编译 Python 片段，并把生成块与它
  们的来源对比。
- 每个改动都从一个失败的测试开始，然后才是修复。

## 构建包

```bash
uv build mliport              # sdist 与 wheel
```

请检查元数据（README、引用、包名），并确认 wheel 里没有混进 `mlipx` 命名空间或临时的规划文档。

## 提交与拉取请求

- 一个拉取请求只做一件逻辑上完整的事。
- 提交信息写清楚什么失败了、什么修好了它，以及**数据影响**：要么是"没有科学记录失效"，要么说明重新
  生成了哪些证据、为什么。
- CI 必须通过：Python 3.10 到 3.13 的测试、lint、包构建和文档构建。
- 不要为了让报告好看而放宽阈值，或者删掉诚实的 `not_run` 记录。
