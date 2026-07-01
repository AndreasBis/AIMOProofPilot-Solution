# Notebook Outputs

This directory was previously named `sandbox`. It stores captured proof outputs and status JSON files from the notebook iteration process.

The status JSON files are summarized below so readers do not need to inspect raw JSON for the main run statistics.

## Artifact Map

| Run | Status JSON | Proof output |
| --- | --- | --- |
| v1 | `aimo_notebook_status_1.json` | `aimo_proof_outputs_1.txt` |
| v2 | `aimo_notebook_status_2.json` | `aimo_proof_outputs_2.txt` |
| v3 | `aimo_notebook_status_3.json` | `aimo_proof_outputs_3.txt` |
| v4 | `aimo_notebook_status_4.json` | `aimo_proof_outputs_4.txt` |
| v5 | `aimo_notebook_status_5.json` | `aimo_proof_outputs_5.txt` |
| v6 | `aimo_notebook_status_6.json` | `aimo_proof_outputs_6.txt` |

The `output_path` field inside the JSON files keeps the original RunPod path from capture time.

## Run Totals

| Run | Problems | Passes | Input tokens | Generated tokens | Total tokens | Python calls | Python errors | Elapsed seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 | 2 | 4 | 3,581 | 66,218 | 69,799 | 0 | 0 | 3,567 |
| v2 | 2 | 4 | 3,064 | 60,752 | 63,816 | Not recorded | Not recorded | 3,109 |
| v3 | 2 | 2 | 825 | 38,693 | 39,518 | Not recorded | Not recorded | 1,987 |
| v4 | 2 | 2 | 1,067 | 33,125 | 34,192 | Not recorded | Not recorded | 1,747 |
| v5 | 16 | 16 | 6,962 | 237,354 | 244,316 | Not recorded | Not recorded | 12,465 |
| v6 | 16 | 16 | 7,570 | 233,468 | 241,038 | Not recorded | Not recorded | 11,921 |

## Problem-Level Stats

| Run | Problem | Passes | Input tokens | Generated tokens | Total tokens | Elapsed seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| v1 | `0dwr` | 2 | 1,584 | 14,316 | 15,900 | 809 |
| v1 | `0a0g` | 2 | 1,997 | 51,902 | 53,899 | 2,755 |
| v2 | `0dwr` | 2 | 1,584 | 15,751 | 17,335 | 797 |
| v2 | `0a0g` | 2 | 1,480 | 45,001 | 46,481 | 2,311 |
| v3 | `0dwr` | 1 | 334 | 7,283 | 7,617 | 366 |
| v3 | `0a0g` | 1 | 491 | 31,410 | 31,901 | 1,621 |
| v4 | `0dwr` | 1 | 455 | 6,801 | 7,256 | 352 |
| v4 | `0a0g` | 1 | 612 | 26,324 | 26,936 | 1,395 |
| v5 | `0dwr` | 1 | 388 | 7,489 | 7,877 | 388 |
| v5 | `0a0g` | 1 | 545 | 27,671 | 28,216 | 1,467 |
| v5 | `0bro` | 1 | 412 | 15,102 | 15,514 | 792 |
| v5 | `0dmr` | 1 | 476 | 14,654 | 15,130 | 768 |
| v5 | `0emh` | 1 | 418 | 14,908 | 15,326 | 781 |
| v5 | `0b0u` | 1 | 383 | 11,803 | 12,186 | 616 |
| v5 | `09jl` | 1 | 472 | 20,264 | 20,736 | 1,068 |
| v5 | `0jjj` | 1 | 419 | 4,748 | 5,167 | 242 |
| v5 | `0hbj` | 1 | 456 | 9,810 | 10,266 | 510 |
| v5 | `07r9` | 1 | 375 | 10,936 | 11,311 | 570 |
| v5 | `0ahj` | 1 | 385 | 10,558 | 10,943 | 550 |
| v5 | `0d42` | 1 | 477 | 14,128 | 14,605 | 740 |
| v5 | `01r6` | 1 | 463 | 25,622 | 26,085 | 1,356 |
| v5 | `00ml` | 1 | 460 | 16,274 | 16,734 | 855 |
| v5 | `08kr` | 1 | 430 | 19,286 | 19,716 | 1,016 |
| v5 | `04a8` | 1 | 403 | 14,101 | 14,504 | 739 |
| v6 | `0dwr` | 1 | 426 | 4,803 | 5,229 | 239 |
| v6 | `0a0g` | 1 | 583 | 36,465 | 37,048 | 1,888 |
| v6 | `0bro` | 1 | 450 | 15,091 | 15,541 | 769 |
| v6 | `0dmr` | 1 | 514 | 13,236 | 13,750 | 673 |
| v6 | `0emh` | 1 | 456 | 10,371 | 10,827 | 525 |
| v6 | `0b0u` | 1 | 421 | 14,081 | 14,502 | 716 |
| v6 | `09jl` | 1 | 510 | 17,375 | 17,885 | 888 |
| v6 | `0jjj` | 1 | 457 | 5,429 | 5,886 | 270 |
| v6 | `0hbj` | 1 | 494 | 11,468 | 11,962 | 582 |
| v6 | `07r9` | 1 | 413 | 6,894 | 7,307 | 345 |
| v6 | `0ahj` | 1 | 423 | 11,412 | 11,835 | 578 |
| v6 | `0d42` | 1 | 515 | 14,617 | 15,132 | 745 |
| v6 | `01r6` | 1 | 501 | 24,998 | 25,499 | 1,285 |
| v6 | `00ml` | 1 | 498 | 22,192 | 22,690 | 1,138 |
| v6 | `08kr` | 1 | 468 | 17,226 | 17,694 | 880 |
| v6 | `04a8` | 1 | 441 | 7,810 | 8,251 | 393 |
