# _legacy/

These are the v1 prototype scripts preserved for git history reference.
They have been superseded by the structured `src/` package.

| Legacy File | Replaced By |
|---|---|
| `onboard_sku.py` | `src/pipeline/orchestrator.py` + `src/pipeline/stages/` |
| `bproc_generator.py` | `src/rendering/bproc_generator.py` |
| `generate_synthetic.py` | `src/rendering/bproc_generator.py` (raw bpy prototype) |

Entry point is now:
```bash
python -m src.pipeline.orchestrator --stage all --image product.jpg
```
