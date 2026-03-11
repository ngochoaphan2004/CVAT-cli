# cvat-cli-skill

A [Claude Skill](https://docs.anthropic.com/) that helps you manage [CVAT](https://www.cvat.ai/) (Computer Vision Annotation Tool) annotation workflows through natural language. Just tell Claude what you need — it'll figure out the right commands for you.

## What This Skill Does

CVAT is a popular open-source tool for labeling images and videos for machine learning. It has a powerful command-line interface, but remembering all the flags and syntax can be a pain. This skill teaches Claude (or Claude Code) how to generate the correct commands so you don't have to.

You describe what you want in plain language, and Claude handles the rest — whether that's creating a single task, batch-uploading hundreds of image folders, exporting your annotations, or setting up an entire project from scratch.

## Installation

Download `cvat-cli-skill.skill` from the [Releases](../../releases) page and add it to your Claude project or workspace. The skill activates automatically whenever your conversation involves CVAT.

## Prerequisites

Before running the commands Claude generates, make sure you have:

1. **cvat-cli** installed — `pip install cvat-cli` (Python 3.10 or newer)
2. **A way to log in** — either a Personal Access Token (recommended) or a username and password
3. **A running CVAT server** — could be your local setup or a team server

## Usage Examples

Here are some things you can ask Claude (in Claude.ai, Claude Code, or any Claude-powered tool) once the skill is installed.

### Creating tasks

> *"Create a CVAT task called 'chest-xray-batch-12' with labels 'normal' and 'abnormal' from the JPG files in my images folder."*

> *"I have a folder called datasets/ with one subfolder per patient study, each containing PNG files. Create one task per subfolder under project 5 on our server at https://cvat.example.com."*

> *"Make a new task from a remote video URL and assign it to project 3."*

### Listing and checking what's there

> *"Show me all the tasks on our CVAT server."*

> *"List all projects in the radiology-ai organization."*

> *"Save the full task list as a JSON file."*

### Exporting and importing annotations

> *"Export task 42 annotations in COCO format."*

> *"I have a CVAT XML annotation file. Import it into task 105."*

> *"Export task 200 in YOLO format so I can use it for training."*

### Downloading frames

> *"Download frames 10, 20, and 30 from task 119 into a folder called samples."*

### Backup and restore

> *"Back up task 136 so I can restore it later."*

> *"Restore a task from the backup file task_backup.zip."*

### Auto-annotation

> *"Run auto-annotation on task 137 using the torchvision Faster R-CNN model."*

> *"Auto-annotate task 200 using my custom detection function in detect.py."*

### Projects

> *"Create a new project called 'ICH Detection' with labels for the four hemorrhage types."*

> *"Delete projects 100 and 101."*

### Full workflows

> *"Set up everything from scratch: create a project called 'Lung Screening', create tasks for each subfolder in my data directory, and give me the export commands I'll need when annotation is done."*

Claude will generate a complete, commented script that chains all the steps together.

## Skill Structure

```
cvat-cli-skill/
├── SKILL.md                        # Main instructions Claude reads
└── references/
    └── cli-reference.md            # Full CLI syntax and examples
```

## Contributing

If you spot a missing feature, have a useful workflow pattern, or run into an issue, feel free to open an issue or submit a PR.

## License

MIT

## Acknowledgments

Built on the official [CVAT CLI documentation](https://docs.cvat.ai/docs/api_sdk/cli/). CVAT is developed by [CVAT.ai](https://www.cvat.ai/).

---

> **Heads up** — this skill is still under active development. Things may change, break, or behave unexpectedly. Use with caution and feel free to report anything weird you run into. 🚧
