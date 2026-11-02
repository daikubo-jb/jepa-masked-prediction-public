# Third-party software and model notices

This file inventories third-party software referenced by the source in this repository. It is not a complete lockfile: model-runtime dependency versions and the full transitive dependency set are not pinned here. No Python dependency package or model checkpoint is redistributed in this repository; the only copied third-party source is the limited V-JEPA 2 subset listed below.

## Copied source

| Component | Version / source | Use here | License and notice |
|---|---|---|---|
| V-JEPA 2 | commit `204698b45b3712590f06245fbfba32d3be539812`; see [`SOURCE-MANIFEST.json`](vendor/vjepa2-204698b45b3712590f06245fbfba32d3be539812/SOURCE-MANIFEST.json) | Native predictor, masks, utility modules, and two ViT-L/16 configs used for implementation checks | Upstream MIT. The upstream `LICENSE` is retained at [`vendor/vjepa2-204698b45b3712590f06245fbfba32d3be539812/LICENSE`](vendor/vjepa2-204698b45b3712590f06245fbfba32d3be539812/LICENSE). The vendored file list does not include the three upstream dataset utilities the repository README identifies as Apache-2.0 (`randaugment.py`, `randerase.py`, `worker_init_fn.py`). |

The upstream V-JEPA 2 repository says most files use MIT and names those three exceptions as Apache-2.0. See its [license statement](https://github.com/facebookresearch/vjepa2/tree/204698b45b3712590f06245fbfba32d3be539812#license) and [pinned LICENSE](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/LICENSE). The source manifest records pinned file hashes; it does not claim a cryptographic signature attestation.

## Python dependencies referenced by code

| Package | Version evidence | Use | Published license | Distributed here? |
|---|---|---|---|---|
| PyYAML | `6.0.3` pinned in `requirements-test.txt` | Parse pinned YAML configs in the parity contract/test path | MIT ([PyPI release](https://pypi.org/project/PyYAML/6.0.3/)) | No; installed by the user for tests |
| PyTorch (`torch`) | No version pin in this snapshot. A prior authorized runtime record used `2.7.1`; this is not a supported environment lock. | Tensor operations, predictor/model construction, safe checkpoint loading, and tests | BSD-3-Clause ([PyTorch license](https://github.com/pytorch/pytorch/blob/main/LICENSE); [PyPI 2.7.1 metadata](https://pypi.org/project/torch/2.7.1/)) | No |
| NumPy | No version pin or exact runtime version recorded in this snapshot | Positional-embedding utilities and processor array conversion | BSD-3-Clause ([upstream license](https://github.com/numpy/numpy/blob/main/LICENSE.txt)) | No |
| `timm` | No version pin in this snapshot. A prior authorized runtime record used `1.0.24`; this is not a supported environment lock. | Upstream V-JEPA 2 module imports `timm.models.layers.drop_path` | Apache-2.0 ([PyPI 1.0.24 metadata](https://pypi.org/project/timm/1.0.24/)) | No |
| Transformers | No version pin or exact runtime version recorded in this snapshot | `VJEPA2VideoProcessor` preprocessing path | Apache-2.0 ([upstream LICENSE](https://github.com/huggingface/transformers/blob/main/LICENSE)) | No |

The table covers direct imports found in the included Python source. It does not enumerate transitive packages pulled in by Transformers or PyTorch; their versions and licenses must be checked against the chosen environment before distributing those packages or a bundled environment. Package use is distinct from redistributing their code or binaries.

## Model artifact (not included)

The experiment report records a local checkpoint identity whose SHA-256 and size match the public `original/model.pth` LFS pointer for [`facebook/vjepa2-vitl-fpc64-256`](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256/commit/995ed017ecd861661bf87bf134adf0e76927198c). That model card labels the artifact MIT ([model card](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256)). The checkpoint itself is not included. Hash/size agreement supports byte identity with that published artifact but does not establish the local file's acquisition path or a signature verification.

## Project license boundary

The root [`LICENSE`](LICENSE) applies only to the original project code, project-authored documentation/reports, and generated synthetic fixture metadata described in the README. It does not relicense the copied V-JEPA 2 source, dependency packages, model checkpoints, or unrelated third-party material.
