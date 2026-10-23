# Public real-child pose dataset availability

## Decision

COCO-UniHuman v2 provides a usable real-child evaluation cohort with human body keypoints and downloadable COCO images. The prepared child cohort contains 134 unique source images and is disjoint by source image ID from the project's existing 921-image COCO subset. It uses COCO imagery, so it is a non-overlapping child subset rather than an independent image source. Ages 3–17 are manual apparent-age annotations and are not verified chronological ages. SVIRO is synthetic vehicle-interior imagery and is excluded from active runs and the paper package at the user's request.

## Candidate review

| Candidate | Real child imagery | Keypoints | Image access | COCO-17 mapping | Decision |
| --- | --- | --- | --- | --- | --- |
| COCO-UniHuman v2 val | Real COCO val2017 photographs with manually annotated apparent ages | 17 human-labeled COCO keypoints; valid labels retained from source annotations | Public COCO image URLs and public v2 annotation download | Identity mapping to COCO-17; person crop only translates visible x/y coordinates | **Selected.** Age 3–17 filter yields 262 eligible people across 134 source images after removing all 921 project COCO image IDs. The preparation step also checks SHA-256 against every existing COCO image and across the child cohort; it found 134 unique source files with no exact byte duplicates. Keep the largest eligible person per unique source image, producing 134 crops. |
| H3D v1.01 | The extracted release has 471 unique Flickr image names and 1,240 person annotations. Its XML labels contain 664 `male`, 501 `female`, 70 `baby`, and 5 `unknown` instances; there are no `child` labels. | Human-provided 2D points plus inferred 3D | Images are included in the public archive | All 17 COCO points map directly by name; two toe points are extra | Excluded from the child domain because its released records identify babies, not a distinct child cohort. Of 70 baby instances, 63 lack at least one of the 17 mapped points, leaving only 7 complete poses. Do not merge it into SyRIP or relabel babies as children. |
| Children Gait Pose Sequence (CGPS / ChildGait) | Real clinical gait video of 110 children | 133 COCO-WholeBody points; Sapiens-2B estimates manually adjusted by annotators | Raw videos are explicitly withheld for child privacy; hosted records contain pose metadata only | COCO-17 body points correspond to the first 17 WholeBody points | Useful for pose-sequence or skeleton analysis, but cannot run image-input models or verify image predictions without source frames. |
| ARAN | Real images of 512 children aged 16–98 months (2,048 views) | Body measurements and age, not body keypoints | Images and metadata are published for research | No pose mapping is possible until keypoints are annotated | Suitable source images for a future manually annotated child set, but not a pose benchmark as released. |
| “On the Estimation of Children's Poses” | Real images from 104 children; the paper reports 1,176 images | 22 manually labeled body points | The dataset is described as available from the authors on request; no direct public image/annotation archive was found | Likely mappable by semantics, but a direct verified mapping requires the actual annotation schema | Strongest research match, but it is not currently a directly downloadable dataset. |

## COCO-17 mapping contract

The following semantic mapping is verified from the H3D `config_person.xml` keypoint declarations and per-instance XML annotations. It demonstrates that non-COCO labels can be converted without inventing locations:

| COCO-17 index | COCO name | H3D XML name |
| ---: | --- | --- |
| 0 | nose | `Nose` |
| 1 | left_eye | `L_Eye` |
| 2 | right_eye | `R_Eye` |
| 3 | left_ear | `L_Ear` |
| 4 | right_ear | `R_Ear` |
| 5 | left_shoulder | `L_Shoulder` |
| 6 | right_shoulder | `R_Shoulder` |
| 7 | left_elbow | `L_Elbow` |
| 8 | right_elbow | `R_Elbow` |
| 9 | left_wrist | `L_Wrist` |
| 10 | right_wrist | `R_Wrist` |
| 11 | left_hip | `L_Hip` |
| 12 | right_hip | `R_Hip` |
| 13 | left_knee | `L_Knee` |
| 14 | right_knee | `R_Knee` |
| 15 | left_ankle | `L_Ankle` |
| 16 | right_ankle | `R_Ankle` |

H3D stores a visibility boolean. For a COCO JSON export, a present point with `visible=1` maps to `(x, y, 2)`; a present point with `visible=0` maps to `(x, y, 1)`; and an absent point maps to `(0, 0, 0)`. Do not use the 3D `z` value in 2D evaluation. `L_Toes` and `R_Toes` are not COCO-17 points and are discarded. A local pass over all 1,240 XML annotations confirmed that coordinates and visibility are valid for each present mapped point; 1,069 annotations are sparse and lack at least one COCO-17 point. This mapping is appropriate for H3D's baby records, but does not change their age-group label.

For CGPS, its 133-point COCO-WholeBody array starts with the COCO body skeleton, so the mapping is identity for indices 0–16:

`0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear, 5 left_shoulder, 6 right_shoulder, 7 left_elbow, 8 right_elbow, 9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip, 13 left_knee, 14 right_knee, 15 left_ankle, 16 right_ankle`.

Its downloadable pose records contain model-generated coordinates and confidence scores rather than independent manually annotated ground-truth visibility for each point. The mapping is syntactically possible, but the missing RGB frames and estimator-derived labels make it unsuitable for this project's image-model COCOeval comparison.

## Reproduction and paper wording

The local H3D archive is cached under the ignored `02_datasets/` directory. Its XML records were counted locally; the archive itself and child images are not added to Git or the paper bundle. Official H3D project information and its Flickr CC-BY terms are at <https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/shape/h3d/>.

Describe the three domains as:

- COCO: adult-dominant general human pose reference set.
- SyRIP: infant domain.
- COCO-UniHuman child cohort: real COCO val2017 child crops with apparent-age labels 3–17 and COCO-17 ground truth. It is image-ID disjoint from the project's 921-image COCO subset, but shares the COCO source collection. Do not describe apparent ages as verified birth ages.
- SVIRO: excluded from active runs and paper package at the user's request; synthetic vehicle-interior imagery.

The annotation file is `02_datasets/source_archives/coco_unihuman_val_v2.json` and is intentionally ignored by Git. The preparation script verifies its SHA-256 before use, excludes source image IDs and byte-identical image content from the COCO project subset, selects one person per image, crops around that person with 12% context, and translates the original keypoints into crop coordinates. The source COCO images, crop images, generated annotations, and model outputs remain local and ignored by Git. See `scripts/prepare_coco_unihuman_child.py` and `config/datasets.yaml` for the frozen selection contract.

Other reviewed sources: [CGPS/ChildGait](https://www.kaggle.com/competitions/cvpr-2026-the-first-ai-children-challenge/data), [ARAN](https://aranchilddata.uvt.nl/), and [On the Estimation of Children's Poses](https://iris.cnr.it/handle/20.500.14243/371177). CGPS withholds raw videos; ARAN has child images without pose-keypoint labels; and the 2017 child-pose paper's annotated imagery was not available as a direct public download.
