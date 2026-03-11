# CVAT CLI Reference — Full Syntax and Examples

This file contains the complete CVAT CLI command reference extracted from official documentation.
The SKILL.md provides the operational guide; this file provides exhaustive syntax details and
examples for when you need precise flag combinations.

---

## Table of Contents

1. [Installation](#installation)
2. [Authentication](#authentication)
3. [Task Examples](#task-examples)
4. [Project Examples](#project-examples)
5. [Function Examples](#function-examples)
6. [Auto-Annotation Function Protocol](#auto-annotation-function-protocol)

---

## Installation

```bash
pip install cvat-cli
```

Requires Python 3.10+.

---

## Authentication

Two authentication methods are supported:

### Personal Access Token (PAT) — Recommended

Set via environment variable (takes priority over password auth):

```bash
export CVAT_ACCESS_TOKEN="your-token-value"
cvat-cli task ls
```

PATs are configured in the CVAT user settings UI.
Reference: https://docs.cvat.ai/docs/api_sdk/access_tokens/

### Password Authentication

Passed via the `--auth` common option:

```bash
cvat-cli --auth username:password task ls
```

If only a username is given (no colon + password), the CLI will prompt for the password.

---

## Task Examples

### Create — Basic

```bash
# Local images, labels from file
cvat-cli task create "new task" --labels labels.json local file1.jpg file2.jpg

# Custom server and user
cvat-cli --server-host https://example.com --auth user-1 task create "task 1" \
    --labels labels.json local image1.jpg

# Within an organization
cvat-cli --org myorg task create "task 1" --labels labels.json local file1.jpg

# Labels from a project (no --labels needed)
cvat-cli --auth user-1:password task create "task 1" --project_id 1 \
    remote https://github.com/opencv/opencv/blob/master/samples/data/vtest.avi?raw=true
```

### Create — Advanced Options

```bash
# Inline labels, chunk size, random sorting, frame step, copy data, zip chunks, shared data
cvat-cli task create "task 1 sort random" \
    --labels '[{"name": "cat"},{"name": "dog"}]' \
    --chunk_size 8 \
    --sorting-method random \
    --frame_step 10 \
    --copy_data \
    --use_zip_chunks \
    share //share/dataset_1/video.avi

# With bug tracker, reduced image quality, and pre-loaded annotations
cvat-cli --auth user-2 task create "task from dataset_1" \
    --labels labels.json \
    --bug_tracker https://bug-tracker.com/0001 \
    --image_quality 75 \
    --annotation_path annotation.xml \
    --annotation_format "CVAT 1.1" \
    local dataset_1/images/

# Segmented task with frame range and caching
cvat-cli task create "segmented task 1" \
    --labels labels.json \
    --overlap 5 \
    --segment_size 100 \
    --start_frame 5 \
    --stop_frame 705 \
    --use_cache \
    remote https://github.com/opencv/opencv/blob/master/samples/data/vtest.avi?raw=true
```

### Create — Cloud Storage

```bash
# Filtered cloud storage data
cvat-cli task create "task with filtered cloud storage data" \
    --labels '[{"name": "car"}]' \
    --use_cache \
    --cloud_storage_id 1 \
    --filename_pattern "test_images/*.jpeg" \
    share manifest.jsonl

# All data from cloud storage
cvat-cli task create "task with filtered cloud storage data" \
    --labels '[{"name": "car"}]' \
    --use_cache \
    --cloud_storage_id 1 \
    --filename_pattern "*" \
    share manifest.jsonl
```

### Delete Tasks

```bash
cvat-cli --auth user-1:password task delete 100 101 102
```

### List Tasks

```bash
cvat-cli task ls
cvat-cli --org myorg task ls
cvat-cli task ls --json > list_of_tasks.json
```

### Download Frames

```bash
cvat-cli task frames --outdir images --quality compressed 119 12 15 22
```

### Export Dataset

```bash
cvat-cli task export-dataset --format "CVAT for images 1.1" 103 output.zip
cvat-cli task export-dataset --format "COCO 1.0" 104 output.zip
```

### Import Dataset

```bash
cvat-cli task import-dataset --format "CVAT 1.1" 105 annotation.xml
```

### Back Up a Task

```bash
cvat-cli task backup 136 task_136.zip
```

### Create from Backup

```bash
cvat-cli task create-from-backup task_backup.zip
```

### Auto-Annotate

```bash
# Using a built-in torchvision function with parameters
cvat-cli task auto-annotate 137 \
    --function-module cvat_sdk.auto_annotation.functions.torchvision_detection \
    -p model_name=str:fasterrcnn_resnet50_fpn_v2 \
    -p box_score_thresh=float:0.5

# Using a custom function file
cvat-cli task auto-annotate 138 --function-file path/to/my_func.py

# With PYTHONPATH for local module imports
PYTHONPATH=path/to/my-project cvat-cli task auto-annotate 139 --function-module my_func
```

---

## Project Examples

### Create

```bash
# Basic project with labels
cvat-cli project create "new project" --labels labels.json

# Project from a dataset
cvat-cli project create "new project" --dataset_file coco.zip --dataset_format "COCO 1.0"
```

### Delete

```bash
cvat-cli project delete 100 101 102
```

### List

```bash
cvat-cli project ls
cvat-cli project ls --json > list_of_projects.json
```

---

## Function Examples (Enterprise/Cloud Only)

### Create and Run

```bash
# Torchvision detection function
cvat-cli function create-native "Faster R-CNN" \
    --function-module cvat_sdk.auto_annotation.functions.torchvision_detection \
    -p model_name=str:fasterrcnn_resnet50_fpn_v2
cvat-cli function run-agent <ID> \
    --function-module cvat_sdk.auto_annotation.functions.torchvision_detection \
    -p model_name=str:fasterrcnn_resnet50_fpn_v2

# SAM2 tracking function
cvat-cli function create-native "SAM2" \
    --function-file=<CVAT_DIR>/ai-models/tracker/sam2/func.py \
    -p model_id=str:facebook/sam2.1-hiera-tiny
cvat-cli function run-agent <ID> \
    --function-file=<CVAT_DIR>/ai-models/tracker/sam2/func.py \
    -p model_id=str:facebook/sam2.1-hiera-tiny
```

### Delete

```bash
cvat-cli function delete 100 101
```

---

## Auto-Annotation Function Protocol

Functions passed to `task auto-annotate` or `function create-native` must implement the
CVAT auto-annotation protocol.

### Direct Module Implementation

The module defines `spec` and `detect` at the top level:

```python
import cvat_sdk.auto_annotation as cvataa

spec = cvataa.DetectionFunctionSpec(...)

def detect(context, image):
    ...
```

### Factory Pattern

The module defines a `create()` function that returns an object implementing the protocol.
Parameters from `-p` flags are forwarded to `create()`:

```python
import cvat_sdk.auto_annotation as cvataa

class _MyFunction:
    def __init__(self, ...):
        ...

    spec = cvataa.DetectionFunctionSpec(...)

    def detect(self, context, image):
        ...

def create(...) -> cvataa.DetectionFunction:
    return _MyFunction(...)
```

### Reference

Full auto-annotation API docs: https://docs.cvat.ai/docs/api_sdk/sdk/auto-annotation/
