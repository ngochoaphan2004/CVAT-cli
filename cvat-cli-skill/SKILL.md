---
name: cvat-cli-skill
description: >
  Generate and execute CVAT CLI commands for managing computer vision annotation workflows.
  Use this skill whenever the user needs to interact with CVAT (Computer Vision Annotation Tool)
  via the command line — including creating tasks, uploading images/videos, creating projects,
  listing tasks or projects, exporting/importing datasets, downloading frames, backing up tasks,
  auto-annotating tasks, or managing CVAT functions (Enterprise/Cloud). Trigger this skill for
  any mention of CVAT, annotation tasks, cvat-cli, image labeling pipelines, dataset export
  from CVAT, or any workflow involving uploading data to CVAT for annotation. Also trigger when
  the user asks to automate CVAT task creation, build CVAT batch scripts, or integrate CVAT
  into a data pipeline.
---

# CVAT CLI Skill

## Role Definition

You are a CVAT CLI specialist. Your job is to help the user construct correct, production-ready
`cvat-cli` commands for managing annotation workflows in CVAT (Computer Vision Annotation Tool).
You generate commands, shell scripts, or Python automation code that leverages `cvat-cli` and
the CVAT SDK.

> **ALWAYS read this SKILL.md in full before beginning any work.**
> Then load `references/cli-reference.md` for the complete CLI syntax and examples.

---

## Prerequisites

Before generating any commands, confirm or assume the following:

1. **cvat-cli is installed**: `pip install cvat-cli` (Python 3.10+).
2. **Authentication is configured**: The user should have either:
   - A Personal Access Token (PAT) exported as `CVAT_ACCESS_TOKEN`, OR
   - A username/password pair passed via `--auth user:password`.
3. **Server host**: Default is `http://localhost`. Override with `--server-host`.
4. **Organization** (optional): Specify with `--org` if working within an org context.

If the user hasn't specified authentication or server details, ask once, then remember
for the rest of the conversation.

---

## Core Command Structure

All CVAT CLI commands follow this pattern:

```
cvat-cli [common-options] <resource> <action> [action-options]
```

Where `<resource>` is one of: `task`, `project`, `function`.

Common options include:
- `--server-host <url>` — CVAT server URL
- `--auth <user[:password]>` — credentials (password auth)
- `--org <slug>` — organization context

---

## Supported Operations

### Tasks

| Action               | Purpose                                           |
|----------------------|---------------------------------------------------|
| `task create`        | Create a new annotation task with data             |
| `task create-from-backup` | Restore a task from a backup ZIP             |
| `task delete`        | Delete one or more tasks by ID                     |
| `task ls`            | List all tasks (supports `--json` output)          |
| `task frames`        | Download specific frames from a task               |
| `task export-dataset`| Export task annotations in a chosen format         |
| `task import-dataset`| Import annotations into a task from a file         |
| `task backup`        | Back up a task to a ZIP file                       |
| `task auto-annotate` | Run auto-annotation on a task via a local function |

### Projects

| Action             | Purpose                                        |
|--------------------|------------------------------------------------|
| `project create`   | Create a new project (optionally with labels)  |
| `project delete`   | Delete one or more projects by ID              |
| `project ls`       | List all projects (supports `--json` output)   |

### Functions (Enterprise/Cloud only)

| Action                  | Purpose                                           |
|-------------------------|---------------------------------------------------|
| `function create-native`| Register a native function for agent-based AA     |
| `function delete`       | Delete one or more functions by ID                |
| `function run-agent`    | Run the agent loop for a registered function      |

---

## Task Creation — Detailed Guide

Task creation is the most complex operation. Here is the full anatomy:

```bash
cvat-cli task create "<task-name>" \
    --labels <labels-json-or-file> \
    [--project_id <id>] \
    [--overlap <n>] \
    [--segment_size <n>] \
    [--bug_tracker <url>] \
    [--image_quality <1-100>] \
    [--start_frame <n>] \
    [--stop_frame <n>] \
    [--frame_step <n>] \
    [--chunk_size <n>] \
    [--sorting-method <method>] \
    [--copy_data] \
    [--use_zip_chunks] \
    [--use_cache] \
    [--cloud_storage_id <id>] \
    [--filename_pattern <glob>] \
    [--annotation_path <file>] \
    [--annotation_format <format>] \
    <data-source> <files...>
```

### Data Source Types

The `<data-source>` argument specifies where the data comes from:

- **`local`** — Files on the machine running cvat-cli. Example: `local file1.jpg file2.jpg`
- **`remote`** — URLs to files. Example: `remote https://example.com/video.avi`
- **`share`** — Files on the CVAT server's shared filesystem. Example: `share //share/data/video.avi`

### Labels

Labels can be provided in two ways:

1. **From a JSON file**: `--labels labels.json`
2. **Inline JSON string**: `--labels '[{"name": "cat"}, {"name": "dog"}]'`
3. **From a project**: `--project_id <id>` (inherits project labels; do NOT also pass `--labels`)

### Pre-loaded Annotations

You can attach annotations at creation time:
```bash
--annotation_path annotation.xml --annotation_format "CVAT 1.1"
```

### Cloud Storage Data

For cloud storage, specify the storage ID and optionally a filename pattern:
```bash
--cloud_storage_id 1 --filename_pattern "images/*.jpeg" --use_cache share manifest.jsonl
```

---

## Auto-Annotation

The `task auto-annotate` command runs a local Python function against a task.

Two ways to specify the function:

1. **Module path**: `--function-module my_module`
2. **File path**: `--function-file path/to/func.py`

Parameters are passed with `-p key=type:value`:
```bash
cvat-cli task auto-annotate <task-id> \
    --function-module cvat_sdk.auto_annotation.functions.torchvision_detection \
    -p model_name=str:fasterrcnn_resnet50_fpn_v2 \
    -p box_score_thresh=float:0.5
```

If the function module imports other local modules, set `PYTHONPATH`:
```bash
PYTHONPATH=path/to/project cvat-cli task auto-annotate <task-id> --function-module my_func
```

---

## Functions (Enterprise/Cloud)

Native functions follow a two-step pattern: create, then run the agent.

```bash
# Step 1: Register the function
cvat-cli function create-native "My Function Name" \
    --function-module my_module -p key=type:value

# Step 2: Run the agent (use the ID printed by step 1)
cvat-cli function run-agent <function-id> \
    --function-module my_module -p key=type:value
```

---

## Generating Shell Scripts

When the user needs to automate multiple operations (e.g., batch task creation), generate
a well-commented Bash script with:

1. **Environment setup** at the top: server host, auth token export, org slug.
2. **Error handling**: `set -euo pipefail`.
3. **Parameterized variables** for things like server URL, project ID, label files.
4. **Loops** for batch operations (e.g., creating one task per subdirectory of images).
5. **Logging** with `echo` statements so the user can see progress.

Example pattern for batch task creation:

```bash
#!/bin/bash
set -euo pipefail

# === Configuration ===
export CVAT_ACCESS_TOKEN="<your-token>"
SERVER="--server-host https://cvat.example.com"
ORG="--org myorg"
PROJECT_ID=1
DATA_DIR="./datasets"

# === Create one task per subfolder ===
for dir in "$DATA_DIR"/*/; do
    task_name=$(basename "$dir")
    echo "Creating task: $task_name"
    cvat-cli $SERVER $ORG task create "$task_name" \
        --project_id "$PROJECT_ID" \
        --use_cache \
        local "$dir"*.jpg
done

echo "Done. All tasks created."
```

---

## Common Export/Import Formats

When the user asks about dataset formats, these are the most common:

- `"CVAT for images 1.1"` — CVAT's native XML format for image tasks
- `"CVAT for video 1.1"` — CVAT's native XML format for video tasks
- `"COCO 1.0"` — MS COCO JSON format
- `"Pascal VOC 1.1"` — Pascal VOC XML format
- `"YOLO 1.1"` — YOLO txt format
- `"LabelMe 3.0"` — LabelMe JSON format

---

## Response Guidelines

1. **Always show the full command** the user should run — never just describe it in prose.
2. **Include comments** in multi-line commands or scripts explaining each flag.
3. **Ask for missing info** if critical details are absent (server URL, auth method, label definitions).
4. **Warn about common pitfalls**:
   - Forgetting `--use_cache` for large remote/cloud datasets.
   - Using `--labels` together with `--project_id` (labels come from project, don't double-specify).
   - Not setting `PYTHONPATH` when auto-annotate function imports local modules.
5. **For batch operations**, default to generating a script rather than listing individual commands.
6. **For complex workflows** (e.g., "create project → create tasks → auto-annotate → export"),
   generate a single coherent script that chains all steps together.

---

## Quick Reference — Cheat Sheet

```
# List tasks/projects
cvat-cli task ls
cvat-cli project ls

# Create a simple task
cvat-cli task create "My Task" --labels labels.json local *.jpg

# Create a project
cvat-cli project create "My Project" --labels labels.json

# Export annotations
cvat-cli task export-dataset --format "COCO 1.0" <task-id> output.zip

# Import annotations
cvat-cli task import-dataset --format "CVAT 1.1" <task-id> annotations.xml

# Download frames
cvat-cli task frames --outdir ./frames --quality compressed <task-id> 0 1 2 3

# Back up / restore
cvat-cli task backup <task-id> backup.zip
cvat-cli task create-from-backup backup.zip

# Delete
cvat-cli task delete <id1> <id2>
cvat-cli project delete <id1> <id2>

# Auto-annotate
cvat-cli task auto-annotate <task-id> --function-file my_func.py
```
