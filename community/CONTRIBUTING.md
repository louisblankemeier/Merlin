# Adding your project to the Merlin community index

The community index points people to work derived from Merlin: fine-tuned models, derived datasets, embeddings, benchmarks, and tutorials. You keep hosting and maintaining your artifact wherever you like. This index just makes it discoverable.

## How to add an entry

1. Fork this repository.
1. Copy [`template.yaml`](template.yaml) to `community/entries/<your-project-name>.yaml`. The filename must be lowercase, e.g. `merlin-nnunet.yaml`.
1. Fill in the fields (see the reference below) and delete the optional ones you do not need.
1. Regenerate the index table and check your links:
   ```bash
   pip install pyyaml
   python community/validate.py --write --check-links
   ```
1. Commit both your YAML file and the updated `community/README.md`, then open a pull request.

CI re-runs the same validation on every pull request that touches `community/`, so if you skip step 4 the check will tell you what to fix.

## Field reference

| Field | Required | Notes |
| --- | --- | --- |
| `name` | yes | Display name, 80 characters max. Must be unique in the index. |
| `description` | yes | One or two sentences, 300 characters max, single line. |
| `category` | yes | One of `model`, `dataset`, `tool`, `benchmark`, `tutorial`, `other`. |
| `authors` | yes | List of `name`, with optional `affiliation` and `email`. |
| `license` | yes | License of your artifact, e.g. `MIT`, `Apache-2.0`, `CC BY-NC 4.0`, or the name of your data use agreement. |
| `links` | yes | Mapping with a required `homepage`, plus optional `code`, `data`, `paper`, `demo`, `docs`. All must be public http(s) URLs. |
| `added` | yes | Date (`YYYY-MM-DD`) you added the entry. |
| `merlin_components` | no | Which parts of Merlin you build on, e.g. `image encoder`, `Merlin Abdominal CT Dataset`. |
| `tags` | no | Free-form keywords. |
| `citation` | no | BibTeX for your work, as a YAML `\|` block. |
| `contact` | no | Email or URL for questions and link-rot follow-up. Recommended. |

## What we ask

- **Host your own artifacts.** This repository does not store models or data. If you are sharing data, point to a host that can serve it, for example Hugging Face, Zenodo, or an institutional repository.
- **Links must be public.** URLs that require a login cannot be checked and will fail CI. A landing page describing how to request access is fine; a private URL is not.
- **Be clear about licensing and provenance.** State the license of what you are sharing, and make sure you have the right to share it. Derived data carries the obligations of the source data — if your work uses the Merlin Abdominal CT Dataset, its [data use agreement](../documentation/download.md) still applies to anything you redistribute.
- **No patient-identifiable data.** Anything you link must be appropriately de-identified.
- **Keep it working.** A weekly CI job re-checks every link. If yours goes dead we will open an issue and, if we cannot reach you, remove the entry. To update or remove your own entry, just open a pull request.

Questions about an entry are best raised as a [GitHub issue](https://github.com/StanfordMIMI/Merlin/issues).
