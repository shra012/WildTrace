"""Narrate WildTrace_Demo1.pptx with the slide notes and export an mp4.

The submitted deck is left unchanged. Audio is added to a temporary copy.
"""
from __future__ import annotations

import shutil
import time
import wave
from pathlib import Path

import win32com.client
from pptx import Presentation

HERE = Path(__file__).resolve().parent
SRC = HERE / "WildTrace_Demo1.pptx"
WORK = HERE / "figures" / "narration"
NARRATED = HERE / "figures" / "WildTrace_Demo1_narrated.pptx"
MP4 = HERE / "WildTrace_Demo1.mp4"


# Spoken cut of the speaker notes. The deck keeps the longer notes for a live presentation.
SPOKEN = [
    "WildTrace turns a wildlife photograph into a pen path a robot can draw, then measures how closely the arm followed it. This demo covers data preparation, the train and test split, the analytics, and the models we have already run.",
    "There are two layers. The dataset layer fetches an Open Images photo and mask, keeps a drawable subject, redraws it with FLUX, and exports at most six strokes. Isaac Sim then drives an xArm7 and records tracking error. Ahead you will see 540 samples reduced to 186 subjects, 132 accepted drawings, and 39 gold trajectories.",
    "The source is Open Images version 7. One sample is one mask instance, not one file. Seven categories were requested. Butterfly has no training masks, so it contributes nothing. The fetch ledger shows 641 successful downloads and 5 corrupt archives, and the latest view holds 540 samples. We drop duplicates, small or blurry images, weak masks, and rear or top-down views. BioCLIP then names the species.",
    "Of those 540 samples, 186 were accepted. That is 34.4 percent. Of those subjects, 132 drawings passed validation, 71 percent, leaving 39 gold trajectories across 6 categories. The main rejection flags were low mask coverage, a top-heavy pose, and blur. A sample can carry more than one. Every rejected drawing used all five attempts and failed the OpenCV line check.",
    "These are accepted drawings the arm has already been given: a bird, a cat, a dog, a fish, a frog, and a horse. Export keeps the longest contours, at most six strokes, at most 128 points each, normalized between zero and one. Across the 39 gold files the average is 4.46 strokes. The horse still has eight strokes from an earlier export. A new export of that same drawing would now be rejected.",
    "The development set is the curated corpus. Generator scores use a frozen test, evalset v2: 20 subjects, seed 14, and both models see the same images. A 30-subject harness and a 60-diagram label set are also prepared. Those human labels have not been collected, so we do not report an accuracy. The robot baseline is eight frozen trajectories, plus one later frog as a reproduction check. This test slice comes from the curated data. It is not the official Open Images test split.",
    "On the accepted subjects, BioCLIP confidence averages 0.885, and viewpoint confidence averages 0.674. That is confidence, not checked accuracy. Just over half the drawings, 51 percent, passed on the first attempt. A draw-time equation fit on seven finished robot runs has an R squared of 0.989. Stroke count and path length already predict how long the arm will take.",
    "On the same 20 subjects, a shared referee, Qwen3-VL 235 billion, passed FLUX on 20 of 20 drawings and OmniGen2 on 6 of 20. The paired sign test p-value is 0.0001. BioCLIP named the right animal for 75 percent of FLUX drawings and 30 percent of OmniGen2. Every FLUX drawing exported. Sixty percent of OmniGen2 drawings did. The exception is pose. When OmniGen2 actually draws, it follows the photograph more closely. FLUX often substitutes a generic stance.",
    "Three failures are measured. OmniGen2 left 9 of 20 final drawings blank, and the local 7 billion judge called a blank a simple cartoon and passed it. FLUX passed the referee every time and still changed the pose. Facing direction matched 65 percent of the time. And OmniGen2's closest sketches, a betta and a frog, were rejected because the lines broke apart. A CPU check on 10 stored drawings passed every original and none of the blurred or speckled copies. The gate sees broken lines. It does not score pose.",
    "We keep FLUX as the production generator and BioCLIP for species labels. We do not keep OmniGen2 as the default, because 45 percent of its finals were blank. We replace the 7 billion judge. It passed all 40 drawings, so agreement with the referee is a kappa of zero. The 8 billion judge agrees at a kappa of 0.83. SDXL with a scribble ControlNet is in the code, on the same validation path, and its drawing quality has not been measured.",
    "On six paired drawings, tracking error fell from 1.45 millimeters to 0.34, about 4.3 times lower. Across eight frozen runs the mean is 0.330 millimeters, and all eight finished. The longest, the horse, took 563 seconds. These runs are zero gravity and contact free. They show the controller is in a plausible range. They are not a real-robot result. The scene we train in now uses Earth gravity, 9.81 meters per second squared.",
    "PPO has finished two sessions of 300 updates, about nineteen and a half minutes each, and we could not show a gain. The path had 957 targets and the episode allowed only 800 steps, so finishing was impossible. Episodes are now 3,000 steps, with a curriculum from a short path at 3 millimeters down to the full path at 2. Behavior cloning is implemented and not trained. We have one demonstration, and training stays off until there are two. A contour bug that blanked the conditioning mask is fixed. Benchmark r2 is the record from before that fix.",
    "This is one subject, dog 96ab6db6, accepted on the first attempt. The figure is the exported pen path: four strokes, 210 points, on a 689 by 616 canvas. Under the frozen baseline the arm drew it at 0.366 millimeters root mean square error, in 380 seconds. A different file, a two-stroke puppy, has been run in the current gravity scene at 0.738 millimeters. That is a status check. It is not a replay of this dog.",
    "Four results. Preparation takes 540 samples to 186 subjects, 132 drawings, and 39 trajectories. The test is a frozen set of 20 shared subjects, and human labels are still ahead. FLUX leads the referee 20 to 6. The controller baseline is 0.33 millimeters, and the learning policies are not ready to claim a gain. Next we rerun the generators after the contour fix, with one shared judge, and we add SDXL on the same 20 subjects.",
]


def notes() -> list[str]:
    prs = Presentation(SRC)
    if len(prs.slides) != len(SPOKEN):
        raise SystemExit(f"expected {len(SPOKEN)} slides, found {len(prs.slides)}")
    return SPOKEN


def synthesize(rows: list[str]) -> list[Path]:
    WORK.mkdir(parents=True, exist_ok=True)
    ps = WORK / "speak.ps1"
    ps.write_text(
        """
param([string]$InFile, [string]$OutFile)
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SelectVoice("Microsoft Zira Desktop")
$synth.Rate = -1
$synth.SetOutputToWaveFile($OutFile)
$synth.Speak((Get-Content -Raw -Encoding UTF8 $InFile))
$synth.Dispose()
""".strip(),
        encoding="utf-8",
    )
    paths = []
    for i, text in enumerate(rows, 1):
        src = WORK / f"slide{i:02d}.txt"
        wav = WORK / f"slide{i:02d}.wav"
        src.write_text(text, encoding="utf-8")
        import subprocess
        subprocess.run(
            ["powershell", "-NoProfile", "-File", str(ps), "-InFile", str(src), "-OutFile", str(wav)],
            check=True,
        )
        paths.append(wav)
        print(f"audio {i:02d} {duration(wav):.1f}s", flush=True)
    return paths


def duration(path: Path) -> float:
    with wave.open(str(path)) as handle:
        return handle.getnframes() / float(handle.getframerate())


def export(wavs: list[Path]) -> None:
    if MP4.exists():
        MP4.unlink()
    shutil.copyfile(SRC, NARRATED)
    ppt = win32com.client.Dispatch("PowerPoint.Application")
    ppt.Visible = True
    pres = ppt.Presentations.Open(str(NARRATED), WithWindow=True)
    try:
        for i, wav in enumerate(wavs, 1):
            slide = pres.Slides(i)
            seconds = duration(wav) + 0.8
            shape = slide.Shapes.AddMediaObject2(str(wav), 0, -1, -40, -40, 12, 12)
            seq = slide.TimeLine.MainSequence
            seq.AddEffect(shape, 83, 0, 2)  # media play, with previous
            slide.SlideShowTransition.AdvanceOnClick = 0
            slide.SlideShowTransition.AdvanceOnTime = -1
            slide.SlideShowTransition.AdvanceTime = seconds
            print(f"timed {i:02d} {seconds:.1f}s", flush=True)
        pres.Save()
        pres.CreateVideo(str(MP4), True, 5, 720, 24, 70)
        deadline = time.time() + 45 * 60
        while time.time() < deadline:
            status = pres.CreateVideoStatus
            print(f"encode status {status}", flush=True)
            if status == 3:
                print("VIDEO_DONE", flush=True)
                return
            if status == 4:
                print("VIDEO_FAILED", flush=True)
                return
            time.sleep(5)
        print("VIDEO_FAILED timeout", flush=True)
    finally:
        pres.Close()
        ppt.Quit()


if __name__ == "__main__":
    rows = notes()
    print(f"slides {len(rows)} words {sum(len(r.split()) for r in rows)}", flush=True)
    wavs = synthesize(rows)
    total = sum(duration(p) for p in wavs)
    print(f"narration {total/60:.1f} min", flush=True)
    export(wavs)
    print(f"mp4 {MP4} bytes {MP4.stat().st_size if MP4.exists() else 0}", flush=True)
