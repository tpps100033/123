# Public real-child pose dataset availability

## Decision

The project still needs a real-image child evaluation domain that is separate from both COCO and the infant SyRIP domain. The public sources checked for this migration do not currently provide one usable package that contains (1) real child images, (2) human body keypoints suitable as evaluation ground truth, and (3) downloadable image files. SVIRO is synthetic vehicle-interior imagery and was removed from active runs and the paper package at the user's request. It is not the real-child benchmark.

## Candidate review

| Candidate | Real child imagery | Keypoints | Image access | COCO-17 mapping | Decision |
| --- | --- | --- | --- | --- | --- |
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

Until a usable real-child keypoint set is obtained, describe results as:

- COCO: adult-dominant general human pose reference set.
- SyRIP: infant domain.
- SVIRO: excluded from active runs and paper package at the user's request; synthetic vehicle-interior imagery.
- Real-child image benchmark: not yet available from the public candidates reviewed here; do not report a score for it.

Potential sources: [CGPS/ChildGait](https://www.kaggle.com/competitions/cvpr-2026-the-first-ai-children-challenge/data), [ARAN](https://aranchilddata.uvt.nl/), and [On the Estimation of Children's Poses](https://iris.cnr.it/handle/20.500.14243/371177). The CGPS challenge page states that raw video is withheld. The ARAN project describes 2,048 images from 512 children aged 16–98 months but does not provide pose-keypoint labels. The 2017 child-pose paper describes an annotated dataset but its repository page does not expose a downloadable image/annotation archive.
