# Prompt Templates

Built-in prompt templates live in this folder.

- `index.yaml` lists available templates and default selection.
- `templates/*.yaml` contains one template per file.

Template fields:

- `id` (stable identifier)
- `name` (display label)
- `version` (integer)
- `description`
- `tags` (list of strings)
- `output_format` (`markdown` or `json`)
- `enabled` (bool)
- `system_prompt` (multiline string)
- `user_prompt_scaffold` (multiline string)
