# Anat Shkolyar - Person-Vehicle Interaction Task

This is a detailed write-up of the work process.

NOTE: I worked on my personal laptop which is CPU-only. 
This affected some decisions and (of course) runtime.

## Overall Development with Claude Code

I made sure to define what I want to do, then I reviewed any suggestions made by Claude.
**There was no "paste the task into Claude and just let it run free".**

Lots of code files - I asked Claude to separate as much as possible to make it easy for me to review.
And I also asked Claude to work TDD, so this explains the amount of test files.

I used PLAN.md (<TODO>link to it) to define what I want to do, then had a local (untracked) handover file to track the current status of all tasks between sessions.
Any decisions were logged in WORKLOG.md, so they will be available for summarization in this write-up.

I asked Claude to create small PRs for each feature and reviewed those as I would for any colleague.

## Ground Truth

I had several GT overview sessions, where I asked Claude to track my observations.

1. Watching and describing each of the video clips provided (<TODO> link to ground_truth.md) - this was done first.
2. Going over all of the relevant objects detected in each clip, noting where tracking broke down. (<TODO> add link to tracking_notes.md)

## Pipeline Description

After going over the videos and describing them, I defined the work plan.

### Detection and Tracking
I have chosen to use a pre-trained model, Ultralytics' YOLO11, for object detections.
I have worked with these models in the past and feel that their framework is easy and flexible enough for many tasks.
(NOTE: license required for commercial use).

Claude suggested using the built-in tracking option with **BoT-SORT** tracker, as it has compensation for camera motion --> I accepted the suggestion.

There were several discussions on how to save the tracking outputs, I tried to keep it simple (JSON as opposed to Claude suggesting parquet) and defined the output format.

Claude wrote the code for the detection and tracking, and ran it over all the videos.

## Extracting Interactions



