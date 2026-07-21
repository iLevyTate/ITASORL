# Outreach video design: "Detectable is not learned"

Date: 2026-07-14
Status: approved (brainstormed in session; user selected format, audience, scope, and renderer)

## Purpose

A polished 60-90 second MP4 for a general/LinkedIn audience that shows the
research progression: an outside observer detects the surrogate world almost
perfectly, the from-scratch creature does not bother to learn it, and survival
pressure changes that. Visuals are driven by real recorded runs (hybrid: real
data, designed motion-graphics framing). Rendered by a dependency-free web
player in the Soft Substrate design language (promo/DESIGN-SYSTEM.md) and
captured headlessly to MP4.

Considered and rejected: game engines (new toolchain, off-brand realism) and
commercial video generation (Veo/Sora/Runway - cannot render the real data,
garbles numbers, and undercuts the credibility of a project about detecting
fake worlds).

## Narrative beats (~80-85s, three numbers only: 0.99 -> 0.50 -> 0.73)

1. The creature (0-10s). The patch fades in: height field as soft lavender
   relief, water as pressed pools, food pellets as raised dots. The organism is
   an abstract concentric-ellipse glyph (landing-page eye motif), breathing
   subtly, moving on a real recorded trajectory. Caption: "Raise a creature
   from nothing. No labels. No instructions. It just lives here."
2. The trick (10-25s). Split screen, two visually identical patches labeled
   A / B in mono type. Caption: "Sometimes its world is the real thing.
   Sometimes we swap in a flawed copy." Sub-line: "Can you tell? Neither can
   we, by eye."
3. The outside observer (25-40s). A pressed-well instrument gauge sweeps to
   0.99 while two point clouds separate. Caption: "A privileged observer,
   reading the raw dynamics, catches the copy almost every time."
4. The creature does not care (40-55s). The same gauge wired to the creature's
   recurrent state: needle sits at 0.50. Caption: "We read its mind - never
   reward it. Nothing. Detectable is not learned."
5. Survival changes that (55-75s). Energy bar appears; the world's physics
   becomes a learned fingerprint the creature must cope with to eat. The gauge
   climbs to 0.73. Caption: "When its life depends on coping with the copy,
   the difference shows up in its head - unasked, unrewarded, a side effect of
   surviving."
6. Close (75-85s). Card: "Detectable is not learned. Survival changes that."
   plus project URL, with one mono footnote line (AUROC, n=10, pre-specified
   bar).

## Anti-cheese rules (binding)

- No faces, no mascot; the organism is a glyph with life (breathing scale,
  heading tick, fading trail).
- Editorial motion only: opacity/position/scale eases; no spins, particles, or
  lens flares.
- Numbers are instrument readouts (IBM Plex Mono in pressed wells), shown as
  published: 0.993 -> "0.99"; 0.488-0.517 -> "0.50"; 0.726/0.722 -> "0.73".
- Color per DESIGN-SYSTEM.md: substrate #E8E5F1; the signal gradient
  (#9079C8 -> #5F82C6) reserved for the gauge needle and one gradient phrase
  per beat.
- Text-on-screen only (muted autoplay); no voiceover in v1; music optional at
  mux time.
- Master 1080x1350 (4:5); the player is resolution-agnostic, so 16:9 is a
  later render flag, not a redesign.

## Architecture (new viz/ directory, three decoupled units)

    viz/
      collect.py           exporter: sim -> viz/data/*.json (render-state)
      data/                committed JSON: terrain grids, trajectories, numbers
      player/
        index.html         stage: DOM/CSS chrome + canvas world
        player.js          Canvas 2D renderer + virtual-clock timeline
        beats.json         director script: beat times, captions, gauges
        style.css          Soft Substrate tokens
      render_video.py      Playwright frame stepper -> ffmpeg -> MP4

- Collector: builds PatchOfEarthV0 on the frozen world P params, samples
  height/wetness/ambient fields on a 160x160 grid, records pellets and agent
  pos/vel/heading/energy per step. Trajectories come from a saved survival
  agent (fullruns/l3_h8_heldout/agents, the first run with --save-agents),
  rolled in authentic and L3-surrogate worlds via setup_l3_surrogate /
  install_l3_surrogate. Fallback (flagged in the JSON): untrained-policy
  rollout for beats 1-2. Beat numbers are read from
  artifacts/expB2/behavior_audit_l3_h8_traces.json and FINDINGS; the collector
  fails loudly if caption numbers drift from artifact values.
- Player: static, no build step. Canvas world layer (pre-shaded height field,
  top-left light), DOM chrome layer (captions, gauges, labels). Deterministic
  virtual clock: ?t=<ms> renders that exact frame; ?play=1 free-runs.
- Capture: Playwright steps ~2550 frames at 30 fps, ffmpeg assembles
  viz/out/itasorl_v1.mp4 (H.264, CRF 18). Same inputs, same MP4.

## Constraints

- fullruns/ is strictly read-only; do not load agents/ until the in-flight
  holdout run exits. Build player first, collector last.
- Branch feat/outreach-video in .worktrees/outreach-video; nothing pushed
  without an explicit ask.
- ASCII punctuation everywhere, including captions.
- viz/out/ and captured frames gitignored; viz/data/*.json committed.

## Verification

- Browser pass at each beat boundary and a full ?play=1 run at 1080x1350.
- Determinism: byte-compare a repeated ?t= frame capture.
- Number honesty: collector asserts captions equal artifact values.
- Output MP4: 80-90s, 1080x1350, plays cleanly, under ~60 MB.
- ruff check . and pytest -q stay green.
