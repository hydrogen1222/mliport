# TUI reference

mliport ships a terminal user interface (TUI) built on [Textual](https://textual.textualize.io/).
It is a guided alternative to the CLI — the same calculations, but with
interactive menus instead of flags.

Launch it with:

```bash
mliport tui
```

## Screens

The TUI is organised as a set of screens. The main screen shows the top-level
menu; pressing a key or selecting an item navigates to the relevant screen.

| Screen | Purpose |
|---|---|
| **Main** | Top-level menu — start a new task or open advanced screens |
| **New Task (Wizard)** | Step-by-step form for sp, opt, md, or neb in a folder |
| **Run** | Build and launch an arbitrary `mliport run` command |
| **Analysis** | Browse analysis tasks and configure options |
| **Config** | Inspect and edit the active settings.ini |
| **Jobs** | Monitor running and queued calculations |

## Keyboard shortcuts

| Key | Action |
|---|---|
| `q` | Quit |
| `Esc` | Go back to the previous screen |
| `F1` | Help |
| `Enter` | Confirm / activate |
| `Tab` / `Shift+Tab` | Move between fields |
| Arrow keys | Navigate menus and lists |

## Workflow: start a new task

1. Run `mliport tui` and you land on the **Main** screen.
2. Select **New Task** — the wizard opens.
3. Choose a calculation type (`sp`, `opt`, `md`, `neb`).
4. Fill in the structure file, model file/type, output directory, and options.
5. Review the generated command, then press **Run**.

The wizard writes an INCAR to your output directory and launches the job.
Progress is shown live. The **Jobs** screen lists all running and recently
finished jobs.

## Workflow: analysis

1. From the **Main** screen, select **Analysis**.
2. Choose the run directory (the folder with `trajectory/` or the XDATCAR).
3. Select an analysis task.
4. Set options (mobile species, temperature, frame range, etc.).
5. Press **Run** to execute.

Results land in `<run>/analysis/<task>/<analysis-id>/` just as with the CLI.

## Relationship to the CLI

The TUI generates and runs the same `mliport` commands you would type yourself.
Every option available in the TUI corresponds to a CLI flag; the CLI reference
documents all flags, defaults, and edge cases.

The TUI does not expose every flag — advanced options (custom drift indices,
MSD method override, etc.) require the CLI directly.

## Tips

- Use `mliport doctor` from the CLI first to verify your installation before
  launching the TUI on a remote machine.
- The TUI inherits the same `settings.ini` search path as the CLI. Use
  `mliport config show` to verify which settings file is active.
- For batch processing and queue management, the CLI (`mliport batch`,
  `mliport queue`) is more flexible than the TUI.

## See also

- [CLI reference](cli.md) — all commands and flags
- [Quickstart](quickstart.md) — first calculation in five minutes
- [Configuration](configuration.md) — settings.ini and default overrides
