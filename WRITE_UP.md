# Anat Shkolyar - Person-Vehicle Interaction Task

This is a detailed write-up of the work process.

NOTE: I worked on my personal laptop which is CPU-only. 
This affected some decisions and (of course) runtime.
My focus was on an offline pipeline, not real-time, due to these limitations.

## Overall Development with Claude Code

I defined each step and reviewed every suggestion — **there was no "paste the task into Claude and let it run free".** I drove the design in [PLAN.md](docs/PLAN.md), logged decisions in [WORKLOG.md](docs/WORKLOG.md) (the source for this write-up), and had Claude split the code into small modules and work TDD (hence the many small files + tests), landing each feature as a small PR I reviewed as I would a colleague's.

## Ground Truth

I reviewed all clips and frames myself across several GT sessions (Claude recorded my observations):

1. Watch and describe each clip — scene, camera, and a first estimate of interaction start/end from the video timestamp (see [ground_truth/ground_truth.md](ground_truth/ground_truth.md)).
2. Review the tracked objects per clip, noting where tracking broke down (see [ground_truth/tracking_notes.md](ground_truth/tracking_notes.md)).
3. Fine-tune the GT — Claude showed me frames around each estimated boundary; I picked the exact start/end frames.

I also recorded the **scope decisions** in the GT file: interaction *types* are enter / exit / other (with `other` reportable behind a flag; default broad); a "vehicle" is the COCO `car` class only (every interaction here is with a car); and a *merge rule* treats consecutive engagements by the same person with the same vehicle whose boxes never separate as a single interaction.

## Pipeline Description

After going over the videos and describing them, I defined the work plan.

### Detection and Tracking
I use a pre-trained Ultralytics **YOLO11** detector — a framework I've worked with before and find flexible — with the built-in **BoT-SORT** tracker (Claude's suggestion, accepted for its camera-motion compensation). Tracks are cached simply: **CSV** per-frame, a JSON metadata sidecar, and a JSONL raw-detection cache (over Claude's parquet suggestion). Detection dominates runtime on CPU and scales with frame count, not resolution (see [docs/timing_findings.txt](docs/timing_findings.txt)).

(NOTE: Ultralytics is AGPL-3.0, so a license is required for commercial use; this also covers the YOLO-World model used for the `--fast` descriptions. The `--detailed` captioner, `moondream2`, is Apache-2.0. Everything runs locally — no external services or network APIs.)

### Note on Detection quality
Some objects were missed at detection. In the low-res grayscale night CCTV clip (`HIu4lM4B8hA_1`) the car wasn't detected under the "car" class alone — unfiltered detection revealed YOLO had labelled it **"boat"**. As a workaround I default the vehicle class to car + truck + bus, and added a per-clip override counting "boat" as a vehicle for this clip only (with a TODO to drive this from video properties — "low-quality → expand vehicle definition" — rather than hard-coding by name). In the same clip the man entering the car is barely detected (a couple of frames, not during the entry); in `iMGR_0AG3a8_2_3` (indoor parking garage) a woman exiting a car isn't detected at all.

YOLO was likely not trained primarily on CCTV, so fine-tuning on customer-representative footage — and swapping in a stronger tracker — are natural improvements (see Future Improvements).

## Extracting Interactions
The first question here was "What defines a person-vehicle interaction?".

Programatically - it must have the person and vehicle bounding boxes in close proximity, and for a meaningful amount of time.
It also required the model to be confident in the object detection
I defined several parameters that may indicate an interaction and asked Claude to extract those per clip.

The parameters, per (person, vehicle) pair over time: **normalized overlap** (box intersection ÷ person-box area — the fraction of the person inside the vehicle box), **normalized center-distance** (÷ vehicle-box diagonal, so it is scale-invariant), and the per-frame **detection confidence**; plus temporal gates — a minimum **duration** in contact and a maximum **gap** bridged. These signals per pair vs the GT windows can be plotted with `scripts.plot_signals`. For example, the woman entering the gray car in `mKzCQKTHizw_0` (person 45 × vehicle 49): the normalized overlap climbs toward ~1.0 right over the GT interaction window — **the dark shaded band in the plot** — while the normalized distance drops. That separation is what the thresholds key on.

![Signals vs frame for an entering person; the dark shaded band is the ground-truth interaction window](docs/images/example_signal_plot.png)

Next, I ran a LOSO (leave-one-scene-out) to find the thresholds per-fold. This showed the approach had merit.
I fit a set of global thresholds on all clips together - these are the thresholds set in config ([`config.SHIPPED_THRESHOLDS`](person_vehicle_interactions/config.py)): `min_overlap=0.2`, `max_distance=0.0`, `min_duration_frames=10`, `min_confidence=0.3`, `max_gap_frames=15`.

I also took a video of people exiting a car with my phone, downsampled it and used it as an external sanity test for the thresholds. There were no surprises there, but I won't show the images here as the people in the video did not consent to being filmed.

The pipeline reports interaction **candidates** (the person↔vehicle contact window); classifying each as enter / exit / other was scoped out - the brief only asks to *list* interactions - though the GT annotates the type, so it is a natural next step.

Each output record is **one person paired with one vehicle**: concurrent people around the same car become separate records (each with its own person description), rather than a single record listing several people. The brief allows "person(s)", so grouping simultaneous participants into one interaction is a reasonable alternative - deferred (see Future Improvements).

## Analyzing the Interactions + Descriptions

On the ground truth, the pipeline finds most real interactions: leave-one-scene-out (leakage-free) gives overall **precision 0.62 / recall 0.77 / F1 0.69** (10 TP, 6 FP, 3 FN across the 6 scenes). The recall miss is mostly the two detection failures noted above; the false positives are mostly **split interactions** (one real interaction fragmented across track-ID switches) plus a couple of hard pass-bys.

Going over the interactions highlighted the broken tracking - same interaction was split into several sections.
My idea was to use the Description section to help filtering FPs and unifying segmented interactions.
My assumption was that same person + same vehicle, in a given time range = same interaction, despite tracking issues, and that if the descriptions are detailed enough they may indicate "person walking past a car" or "person getting into a car", or even just answer a yes/no question of "is there a person touching a car in this image".

I tasked Claude to find suitable models for image descriptions. We started from YOLO-World, whose attributes proved unreliable - it does emit colors and gender guesses, but they are frequently wrong (it labelled a man in a green shirt as "a woman" and a red sedan as "a red SUV"); it is more dependable on object *type* (sedan/SUV/truck). I asked to switch to a captioner model, Claude suggested `moondream2` from HuggingFace, which showed promise (it described that same clip as "Male, wearing green" and "Red four-door sedan" - matching the GT), but was very slow on CPU - more than a minute per image (per crop; descriptions are deduplicated per track, so each unique person/vehicle is described once, not once per interaction).

For this task, I created 2 options for descriptions - one `--fast` using YOLO-World, just for the feeling of sane runtime and `--detailed` where `moondream2` was used, for usable descriptions.

I decided not to filter out FPs (for example, a woman walking in front of a car and not interacting with it in video `NmlzoaDcOuI_1`, which `moondream2` correctly described as "passing by" when asked, on the union crop of the person + vehicle boxes, whether she was interacting or just passing by).
I also decided not to unify split interactions at this time, as YOLO-world cannot be depended on and `moondream2` is too slow.

## Limitations

- **Single-camera ambiguity / identity-blind scoring:** a person passing *in front of* a parked car overlaps its box and can be scored as an interaction (the `NmlzoaDcOuI_1` passer-by, below). Evaluation also matches predictions to GT by temporal overlap, not identity, so it can mislabel in either direction.

  ![passer-by false positive](docs/images/limitation_passerby.jpg)

- **Broken tracking → fragmentation:** ID switches split one real interaction into several windows (the aerial `gt1125_06` driver, below), which is the main source of false positives. The same driver on the same car `v5` is tracked as `p18` and then re-identified as `p111`:

  ![driver as p18](docs/images/limitation_fragmentation_a.jpg)
  ![same driver re-ID'd as p111](docs/images/limitation_fragmentation_b.jpg)

- **Static occupant:** a person already seated inside a car has sustained box overlap with no mount/dismount, so overlap alone can mistake them for someone entering (`mKzCQKTHizw_0` `p51`, seated in `v49`):

  ![static occupant](docs/images/limitation_static_occupant.jpg)

- **CCTV detection recall:** low-res / grayscale clips (`HIu4lM4B8hA_1`) miss people and misclassify cars (the `boat` case), capping recall regardless of the interaction logic.

## Reproducibility

Models and seeds are pinned and the detector/tracker run CPU-deterministically; the `moondream2`
descriptions rely on the model's default greedy decode (no sampling) rather than an explicitly
fixed generation seed. The detection **tracks** and the **results** are committed under `outputs/`,
so a reviewer can reproduce the classify + description stage (and the contact sheets) without
re-running the ~73-minute detection.

The committed `outputs/` hold **15 interactions across the 8 clips** under the single shipped
threshold set (`config.SHIPPED_THRESHOLDS`); `HIu4lM4B8hA_1` contributes **0** (the documented
CCTV detection-recall miss, not a broken run). The precision/recall/F1 above is a separate,
leakage-free LOSO **evaluation** device and is not what produces these shipped outputs — so its
window counts (16 predictions vs 13 GT) are not expected to match the 15 committed interactions.

# Future Improvements

1. Detection / tracking
    a. Detection model trained on CCTV and other suitable footage
    b. Better tracking module
    c. A video-native model that detects + tracks jointly (e.g. transformer trackers such as MOTR / TrackFormer, or open-vocabulary video models) instead of per-frame detection + a separate tracker.
2. Definition of Interaction 
    a. Is a person removing a car cover considered an interaction? 
    b. Tuning / replacing selected interaction parameters based on more data
    c. Address more edge cases - multiple people interacting simultaneously with a vehicle - is it one or multiple interactions?
3. Descriptions
    a. Using GPU for faster image descriptions
    b. Using descriptions to filter out FPs
4. A completely other direction - train a model that identifies interactions within videos - requires large amounts of labeled data.
