# Anat Shkolyar - Person-Vehicle Interaction Task

(<TODO> make sure it is under 2 pages in total)

This is a detailed write-up of the work process.

NOTE: I worked on my personal laptop which is CPU-only. 
This affected some decisions and (of course) runtime.
My focus was on an offline pipeline, not real-time, due to these limitations.

## Overall Development with Claude Code

I made sure to define what I want to do, then I reviewed any suggestions made by Claude.
**There was no "paste the task into Claude and just let it run free".**

Lots of code files - I asked Claude to separate as much as possible to make it easy for me to review.
And I also asked Claude to work TDD, so this explains the amount of test files.

I used [PLAN.md](PLAN.md) to define what I want to do, then had a local (untracked) handover file to track the current status of all tasks between sessions.
Any decisions were logged in WORKLOG.md, so they will be available for summarization in this write-up.

I asked Claude to create small PRs for each feature and reviewed those as I would for any colleague.

## Ground Truth

I had several GT overview sessions, where I asked Claude to track my observations.
I reviewed all the clips and other images myself.

1. Watching and describing each of the video clips provided (see [ground_truth/ground_truth.md](ground_truth/ground_truth.md)), including descriptions of the scene, the camera used, and estimating start/end of interaction based on video timestamp - this was done first.
2. Going over all of the relevant objects detected in each clip, noting where tracking broke down. (see [ground_truth/tracking_notes.md](ground_truth/tracking_notes.md))
3. Fine-tuning the GT - asked Claude to show me frames around the start/end of the GT I indicated before, selected exact frames for interaction start/end.

## Pipeline Description

After going over the videos and describing them, I defined the work plan.

### Detection and Tracking
I have chosen to use a pre-trained model, Ultralytics' YOLO11, for object detections.
I have worked with these models in the past and feel that their framework is easy and flexible enough for many tasks.
(NOTE: license required for commercial use).

Claude suggested using the built-in tracking option with **BoT-SORT** tracker, as it has compensation for camera motion --> I accepted the suggestion.

There were several discussions on how to save the tracking outputs, I tried to keep it simple (JSON as opposed to Claude suggesting parquet) and defined the output format.

Claude wrote the code for the detection and tracking, and ran it over all the videos.

The detection takes most of the runtime (on CPU) and is heavily dependent on frame count (not resolution — see [docs/timing_findings.txt](docs/timing_findings.txt)).

### Note on Detection quality
Some objects were missed at the detection stage.

Example - in the low quality CCTV video (`HIu4lM4B8hA_1`, low-res grayscale night), the car was not detected at first (when used the "car" class alone). When I examined the full, unfiltered detection, it turns out it was identified as "boat".
As a workaround, I made sure that "truck" and "bus" are added to "car" as the default vehicle class.
For the specific video, I added an override that allows "boat" be counted as a vehicle for this video alone, along with a TODO that in the future this workaround should use the video properties to select the "low-quality-so-expand-definition-of-vehicle" route instead of hard-coding by video name.

In the same video, the man stealing the vehicle is not detected at all.
In another video (`iMGR_0AG3a8_2_3`, the indoor parking garage) - a woman exiting a car is not detcted at all.

The YOLO model was probably **NOT** trained on CCTV videos, at least not exclusively.
In the future, it may be beneficial to use models specifically trained on CCTV footage (or any other footage that the customers provide) to improve detection.
(<TODO> in the future improvements section below, make sure to include "improve detection by fine-tuning models" - look at our plan and todos in the code to collect info on that).

Additionally, I'm no expert on tracking, but there are many tracking modules (or even video-ingesting models) to choose from, so this is another possible future optimization.

## Extracting Interactions
The first question here was "What defines a person-vehicle interaction?".

Programatically - it must have the person and vehicle bounding boxes in close proximity, and for a meaningful amount of time.
It also required the model to be confident in the object detection
I defined several parameters that may indicate an interaction and asked Claude to extract those per clip.

The parameters, per (person, vehicle) pair over time: **normalized overlap** (box intersection ÷ person-box area — the fraction of the person inside the vehicle box), **normalized center-distance** (÷ vehicle-box diagonal, so it is scale-invariant), and the per-frame **detection confidence**; plus temporal gates — a minimum **duration** in contact and a maximum **gap** bridged. These signals per pair vs the GT windows can be plotted with `scripts.plot_signals`. <TODO> embed an example graph.

Next, I ran a LOSO (leave-one-scene-out) to find the thresholds per-fold. This showed the approach had merit.
I fit a set of global thresholds on all clips together - these are the thresholds set in config ([`config.SHIPPED_THRESHOLDS`](person_vehicle_interactions/config.py)): `min_overlap=0.2`, `max_distance=0.0`, `min_duration_frames=10`, `min_confidence=0.3`, `max_gap_frames=15`.

I also took a video of people exiting a car with my phone, downsampled it and used it as an external sanity test for the thresholds.

## Analyzing the Interactions + Descriptions

Most interactions were detected correctly (<TODO> is that true? compare detected interactions with GT, note split interactions).

Going over the interactions highlighted the broken tracking - same interaction was split into several sections.
My idea was to use the Description section to help filtering FPs and unifying segmented interactions.
My assumption was that same person + same vehicle, in a given time range = same interaction, despite tracking issues, and that if the descriptions are detailed enough they may indicate "person walking past a car" or "person getting into a car", or even just answer a yes/no question of "is there a person touching a car in this image".

I tasked Claude to find suitable models for image descriptios. We started from YOLO-World, which did not deliver, as it could not identify car or clothes colors, not to mention genders. I asked to switch to a captioner model, Claude suggested `moondream2` from HuggingFace, which showed promise, but was very slow on CPU - more than a minute per image (per crop; descriptions are deduplicated per track, so each unique person/vehicle is described once, not once per interaction).

For this task, I created 2 options for descriptions - one `--fast` using YOLO-World, just for the feeling of sane runtime and `--detailed` where `moondream2` was used, for usable descriptions.

I decided not to filter out FPs (for example, a woman walking in front of a car and not interacting with it in video `NmlzoaDcOuI_1`, which `moondream2` correctly described as "passing by" when asked, on the union crop of the person + vehicle boxes, whether she was interacting or just passing by).
I also decided not to unify split interactions at this time, as YOLO-world cannot be depended on and `moondream2` is too slow.

# Future Improvements
<TODO> make sure all future improvements listed above are summarized in the section below.

1. Detection / tracking
    a. Detection model trained on CCTV and other suitable footage
    b. Better tracking module
    c. A video-ingester model that does tracking at the same time as detection <TODO> any suggestions for such models?
2. Definition of Interaction 
    a. Is a person removing a car cover considered an interaction? 
    b. Tuning / replacing selected interaction parameters based on more data
    c. Address more edge cases - multiple people interacting simultaneously with a vehicle - is it one or multiple interactions?
3. Descriptions
    a. Using GPU for faster image descriptions
    b. Using descriptions to filter out FPs
4. A completely other direction - train a model that identifies interactions within videos - requires large amounts of labeled data.







