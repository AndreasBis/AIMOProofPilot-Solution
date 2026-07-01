# Utilities

This directory contains notebooks and supporting notes for moving large model artifacts and packaging project folders without manual 65 GB uploads.

| File | Purpose |
| --- | --- |
| `colab_hf_to_kaggle_olmo_upload.ipynb` | Uses Colab secrets to download `allenai/Olmo-3.1-32B-Think` from Hugging Face and publish it as a Kaggle Dataset for inference. |
| `zip_training_drive_folder.ipynb` | Streams a ZIP archive from a mounted Google Drive folder to Google Drive through rclone without first creating the full archive on Colab local disk. |
| `upload_dummy_to_drive.ipynb` | Uploads the dummy model folder into the Google Drive training package area. |
| `upload_judge_to_drive.ipynb` | Uploads the judge model folder into the Google Drive training package area. |
| `upload_contestant_to_drive.ipynb` | Uploads the contestant model folder into the Google Drive training package area. |
| `rclone_google_drive_oauth.md` | Documents the Google Cloud OAuth client setup needed by rclone for Google Drive access. |

## OLMo Kaggle Dataset Upload

Run `colab_hf_to_kaggle_olmo_upload.ipynb` in Colab with these secrets configured:

| Secret | Used for |
| --- | --- |
| `HF_TOKEN` | Authenticates the Hugging Face model download. |
| `KAGGLE_TOKEN` | Authenticates Kaggle Dataset creation. |

The notebook stages the model at `/content/olmo-3-1-32b-think`, removes local Hugging Face cache metadata, verifies the expected safetensors count, writes `dataset-metadata.json`, and creates the Kaggle Dataset `andreasbis/olmo-3-1-32b-think`.

## Google Drive ZIP Workflow

Run `zip_training_drive_folder.ipynb` cells in order.

When the rclone config cell opens, create a Google Drive remote named exactly:

```text
gdrive
```

Use these choices:

```text
n/s/q> n
name> gdrive
Storage> drive
client_id>
client_secret>
scope> 1
service_account_file>
Edit advanced config? y/n> n
Use web browser to automatically authenticate rclone with remote? y/n> n
```

At the `config_token` prompt, keep the Colab cell waiting. On a local machine with a browser and rclone installed, run the exact `rclone authorize "drive" ...` command printed by Colab. Authorize the same Google account that owns the Drive folder, copy the full token output, and paste it back into the Colab prompt.

Finish the rclone prompts with the defaults unless the folder is in a Shared Drive. When rclone shows the remote summary, choose `y`. At the main config menu, choose `q`.

Then run the check cell. It should find `gdrive:` before the streaming ZIP cell starts.

The streaming cell uploads:

```text
gdrive:AIMOProofPilot_Container.zip
gdrive:AIMOProofPilot_Container.zip.sha256
```

Those files are written to the root of the `gdrive:` remote, typically My Drive, alongside the source folder rather than inside it.

Observed runtime for the large training package: 1 hour 20 minutes.
