# PPE mask and footwear model comparison

A public-data classification baseline for an ongoing task-aware PPE capstone. The notebook compares ResNet-50, EfficientNet-B0, and YOLOv8n-cls on two separate tasks: mask/no-mask and shoes/no-shoes.

## Open and run

Open `PPE_Mask_Shoes_Model_Comparison.ipynb` in Google Colab or Jupyter. Run the installation cell, restart the kernel if needed, and continue in order. A GPU is recommended for the full six-run comparison. The supplied saved run used a Tesla T4.

For local use, install a matching PyTorch/torchvision pair for your hardware before running. The notebook installs the other dependencies in its first cell. Default local outputs go to `ppe_classification_work/`; Colab optionally saves results to Google Drive. Dataset downloads total about 2.27 GB, with additional space required for crops and checkpoints.

Use `RUN_MODE = 'quick'` for a smoke run. Full mode allows up to 20 epochs per model with early stopping. Preserve a run's configuration when resuming, or choose a new experiment name.

## Recorded results

These are existing notebook outputs, not a new execution during publication. Model selection uses validation macro F1.

| Task | Validation-selected model | Test accuracy | Test macro F1 | Test crops / inferred groups |
| --- | --- | --- | --- | --- |
| Mask | ResNet-50 | 96.55% | 0.9537 | 87 / 3 |
| Footwear | EfficientNet-B0 | 100% | 1.0000 | 120 / 4 |

The notebook retains all six model results, learning curves, confusion matrices, error examples, and recorded GPU forward-pass timings. Comment and Markdown edits preserve the executable statements, execution counts, and saved outputs.

## Dataset and evaluation protocol

Source: [PPEs v7 by Personal Protective Equipment](https://universe.roboflow.com/personal-protective-equipment/ppes-kaxsi/dataset/7), via the [keremberke Hugging Face mirror](https://huggingface.co/datasets/keremberke/protective-equipment-detection). The dataset card states CC BY 4.0. The notebook pins revision `90d2c6da950a6168fcee20ec69e194a034f44eef` and archive checksums. Dataset files are downloaded separately and are not included here.

The original training archive has no mask examples. The experiment pools the archives and rebuilds a shared grouped split, targeting 70/15/15 percent. Source images, duplicate pixels, and temporal blocks are grouped; frame-index guards separate retained blocks. Scores are not comparable to benchmarks using the publisher's original split.

## Scope and limitations

This is a within-recording temporal-block baseline. People, cameras, or recordings can recur across splits; the small number of inferred test groups does not establish independent-session performance. A shoe crop can depict a detached shoe, and the label does not certify protective footwear or correct wearing.

The models classify image regions. A complete workshop pipeline still needs detection, person association, task rules, uncertain-view handling, and independent workshop evaluation. GPU forward timings exclude capture, cropping, preprocessing, and rule logic. They do not establish Raspberry Pi or end-to-end performance.

The notebook can export best checkpoints, predictions, manifests, configuration, and plots. That results ZIP and the trained weights were not supplied for this repository; embedded notebook outputs are retained instead. Load PyTorch checkpoints only from trusted sources.

## Credits

Published by Alazar Wake Kore as capstone experiment material. Dataset credit belongs to the original provider and mirror maintainers. Model implementations and pretrained weights are subject to their upstream licenses; consult PyTorch/torchvision and Ultralytics terms before reuse. No new open-source license is asserted for this source edition.
