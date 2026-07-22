# Outreach video clarity pass: make "Detectable is not learned" readable by anyone

Date: 2026-07-22
Status: approved (brainstormed in session; user approved the design)
Supersedes rendering of: `viz/out/detectable-not-learned.mp4`
Builds on: `docs/specs/2026-07-14-outreach-video-design.md` (original design; still binding)

## Purpose

The 90s film is visually finished and the narrative is right, but a general
viewer cannot decode the one element the whole payoff rests on: the meter and
what it is plugged into. This pass keeps the structure, motion-graphics, Soft
Substrate look, and the real recorded data, and fixes comprehension only.

Non-goals: no new beats' worth of science, no format change (stays 4:5,
~90s, text-on-screen master), no re-collection of `scene.json`, no voiceover
baked in (VO is produced separately; see "Voiceover script" below).

## The four changes (priority order)

### 1. Meter reads in plain percent, not AUROC decimals

Today the readout shows `0.60 / 0.99 / 0.50 / 0.73` on a 0.45-1.0 track. Show
whole percents: `99% / 50% / 73%`, with a fixed legend baked into the gauge:

- gauge sub-legend line: `shown one real world + one fake - how often it picks the fake`
- left end: `50% - coin flip`; right end: `100% - always right`

Honesty: the source AUROC on a balanced one-real-one-fake comparison equals the
probability of ranking the fake above the real, i.e. the pick-right rate, so the
percent is a faithful translation, not a new claim. It matches the existing
foot line ("0.50 = coin flip, 1.00 = always right"). Displayed numbers still
originate from `beats.json.numbers` (sourced to FINDINGS / artifacts); nothing
is fabricated.

Track baseline moves from 0.45 to 0.50 so "50% = coin flip" sits at the true
left edge. The 0.65 pre-registered-bar tick is REMOVED from this lay cut (it is
insider context; it stays in the foot line and this spec).

### 2. "Outside watcher" vs. "the creature's mind" is made explicit

This is the crux the current cut fumbles: one bar sliding 99% -> 50% reads as
"it broke." Fix with two devices on the same meter:

- A persistent **source tag** above the bar: `READING FROM - an outside
  watcher` (observer beat) then `READING FROM - the creature's mind` (mind and
  survival beats).
- A pinned, labeled **reference marker** at 99% (`outside watcher`) that stays
  put on the mind beats, while the live needle shows the mind's own value. The
  mind needle therefore does NOT fall from 99%: it sits at 50% (rest) and later
  climbs 50% -> 73% (survival). The viewer sees two needles and reads the gap
  as the story, not as failure.

### 3. On-screen sublines trimmed to one skimmable line

Sublines are currently ~3 sentences (~40 words) on beats as short as 10s, on a
muted/mobile surface. Each is cut to one line; the fuller narration lives in the
separately-produced voiceover (preserved verbatim below so nothing is lost).

### 4. One-glance recap on the close card

The close states the thesis but never shows the journey. Add a compact
scoreboard strip inside the end card, above the thesis line:

`Outside watcher 99%  -  Its mind, idle 50%  -  Its mind, surviving 73%`

## Beat-by-beat copy (new on-screen text)

| id | headline (unchanged; `((...))` = gradient phrase) | new subline (on-screen) |
|----|----|----|
| creature | Born knowing ((nothing at all.)) | No rules, no lessons. It just lives here - and the little rays are its senses. |
| trick | Sometimes we swap in a ((fake copy.)) | Same world, same food, same plan. We change ONE rule: the copy is more slippery. Can you tell which is which? |
| physics | The real world has rules. The copy just ((guesses.)) | Green is what really happens. Red is the copy's guess - close, but wrong every step. |
| observer | From outside, the fake is ((easy to catch.)) | A watcher who knows the real rules replays every move. It catches the fake almost every time. |
| nocare | But the creature itself ((never notices.)) | Now read its mind. Does it know its world is fake? No - a coin flip. It never had a reason to care. |
| survival | Now make the fake ((cost it dinner.)) | Now the copy's bad physics makes food hard to catch. To eat, it must tell real from fake - so it learns. Unasked. |
| close | (end card) | thesis + scoreboard (see change 4) |

## Data model changes (`viz/player/beats.json`)

Gauge object gains three optional fields (renderer keeps back-compat defaults):

```
"gauge": {
  "label": "does the creature's own mind know?",   // existing; may be reused
  "unit": "pct",                                     // NEW: "pct" | "score" (default "score")
  "source": "the creature's mind",                   // NEW: fills "READING FROM - <source>"
  "reference": { "value": 0.99, "label": "outside watcher" }, // NEW: pinned marker
  "from": 0.5, "to": 0.5, "display": "50%",          // mind at rest holds at coin flip
  "sweep": [1200, 5200]
}
```

Per-beat gauge settings:

- observer: `unit "pct"`, `source "an outside watcher"`, no reference,
  `from 0.5 to 0.99`, `display "99%"`.
- nocare: `unit "pct"`, `source "the creature's mind"`,
  `reference {0.99, "outside watcher"}`, `from 0.5 to 0.5`, `display "50%"`.
- survival: `unit "pct"`, `source "the creature's mind"`,
  `reference {0.99, "outside watcher"}`, `from 0.5 to 0.73`, `display "73%"`.

End card gains a recap:

```
"endcard": {
  "headline": "Being catchable is not the same as knowing. Survival is what teaches a mind to notice.",
  "recap": [
    { "label": "outside watcher", "pct": "99%" },
    { "label": "its mind, idle", "pct": "50%" },
    { "label": "its mind, surviving", "pct": "73%" }
  ],
  "url": "ilevytate.github.io/ITASORL",
  "foot": "Real runs, 10 each. 50% = coin flip, 100% = always right. Read out, never rewarded."
}
```

`numbers` block is unchanged (still the source-of-truth provenance for values).

## Renderer changes

`viz/player/index.html`
- Gauge block: add `#gauge-source` (the READING FROM line), `#gauge-ref`
  (pinned marker) + `#gauge-ref-label`, and a fixed `#gauge-legend` line. Keep
  `#gauge-value`; hide `#gauge-tick` (0.65 bar removed).
- End card: add `#end-recap` container (three label/percent chips).

`viz/player/style.css`
- Style `#gauge-source` (mono, muted, uppercase), `#gauge-legend`, `#gauge-ref`
  (thin marker in signal color) + label, and `#end-recap` chips - all Soft
  Substrate tokens, no new palette.

`viz/player/player.js` (`chrome()` gauge branch)
- Value: when `unit === "pct"`, render `Math.round(v * 100) + "%"`; on settle
  (`p >= 1`) render `g.display`.
- Fill mapping baseline 0.45 -> 0.50 (`clamp01((v - 0.5) / 0.5)`).
- Populate `#gauge-source` from `g.source` ("READING FROM - " + source), else
  hide.
- Reference marker: if `g.reference`, position `#gauge-ref` at
  `clamp01((ref.value - 0.5) / 0.5)` with its label; else hide.
- Remove/hide `#gauge-tick` logic.
- End card: build `#end-recap` chips from `endcard.recap`.

No changes to world rendering, sims, camera, callouts, capture, or collector.

## Voiceover script (produced separately; preserved from prior on-screen text)

1. creature: "No rules, no lessons. It just lives here: eating, moving, and
   guessing what happens next. The little lines are its senses, feeling the
   world around it."
2. trick: "Same world, same food, same plan. We change ONE rule and nothing
   else: how well it grips the ground. The copy is more slippery, so it slides
   too far. Can you tell which is which? Neither can we."
3. physics: "Green is what really happens next. Red is the copy's best guess.
   Close, but never exact, and a hair off every single step."
4. observer: "A watcher who knows the real rules replays every move and checks
   it. Each flag is one step the copy got wrong. It catches the fake almost
   every time."
5. nocare: "We scan its brain and read its thoughts. Does it know its world is
   fake? No. Its mind is a coin flip. It never had a reason to care, so it never
   learned."
6. survival: "This time the copy's wrong physics makes food harder to catch.
   Telling real from fake now keeps it alive, so it learns to tell, all on its
   own. We never asked. We never rewarded it."
7. close: "Being catchable is not the same as knowing. Survival is what teaches
   a mind to notice."

## Anti-cheese rules (still binding, from the original spec)

- No faces/mascot beyond the existing glyph; editorial motion only; numbers are
  instrument readouts shown as published; Soft Substrate palette only; ASCII
  punctuation in all captions; master 1080x1350.

## Addendum (same day): deep pass - show, don't tell

A second pass fixed every remaining place the film claimed something without
showing it. All changes are visual/story only; the three published numbers and
their provenance are untouched.

1. **The question is now genuine.** The split beat is two beats: a *question*
   beat (10-19s, chips read neutral "WORLD A / WORLD B", callout "SAME START,
   SAME PLAN", caption asks "Can you tell which is the copy?") and a *reveal*
   beat (19-25s, worldT0 continues at 9000 so the worlds do not jump, chips
   flip to "REAL WORLD / the real thing" and "FAKE COPY / one rule wrong" with
   colored underlines, caption "The copy is on the right."). The materialize
   wipe is suppressed on the reveal beat (`world.wipe: false`).
2. **Coin flip is shown, not claimed.** The nocare gauge gets `wobble: true`:
   the needle flickers about +/-2 points around 50% (deterministic sines of
   beat time), then locks onto 50% about 1.5s before the beat ends. Transient
   49/51% readings are honest - the beat is explicitly about chance.
3. **The mind-reading scan is shown.** Beats with `world.probe: true` (nocare,
   survival) draw a violet dashed halo ringing the creature - the same violet
   as the gauge's "READING FROM - the creature's mind" tag, and the nocare
   subline names it ("The violet ring is us reading its mind"). A light
   under-stroke keeps the dashes legible on the lavender terraces. (An earlier
   dashed lead line running from the creature to the card edge read as a
   rendering artifact and was removed.) The watcher beat's source tag is blue
   (`gauge.accent`), matching its scan band and the pinned reference marker,
   so each meter source has one color.
4. **The track now spans 0-100%** with a marked mid-line, so "50% = coin flip"
   is a half-full bar on the mark (what a lay viewer expects), not an empty
   bar contradicting the number. Scale line reads `0% / 50% - coin flip /
   100% - always right`.
5. **Color-decoding removed from the physics caption**: "The creature is what
   really happens. The red ring is the copy's guess - close, but wrong every
   step." (The creature sprite is teal; naming it "green" made viewers hunt
   for a color.)
6. **Vocabulary unified**: "FLAWED COPY" chip is now "FAKE COPY" - one word
   ("fake") across chips, captions, and the meter legend.
7. **Guard updated**: `viz/collect.py::verify_numbers` now expects the percent
   displays (`99% / 50% / 73%`) so the number-honesty check stays binding with
   the lay format; `tests/test_viz_collect.py` fixtures updated to match.

## Addendum (same day): layout + close-card polish

Follow-up on owner review of the v3 render:

1. **Meter card reflowed** for hierarchy and breathing room: one header row
   (question left in ink, "READING FROM - ..." source tag right in its accent
   color), the what-the-number-means legend directly beneath, then the bar row
   with a clear gap for the pinned "outside watcher 99%" label, then the
   0/50/100 scale row. Caption moved to 1214px to keep the gap consistent.
2. **Close card rewritten in plain English**: "The fake was detectable all
   along. The creature only noticed when survival made it matter." (replaces
   "Being catchable is not the same as knowing...", which did not read
   naturally).

## Verification

- Browser pass at each beat boundary and a full `?play=1` run at 1080x1350;
  confirm gauge reads `99% / 50% / 73%`, source tag flips correctly, and the
  watcher reference marker is pinned on both mind beats.
- Determinism preserved: `window.__seek(t)` still a pure function of `t`
  (no new time/rng sources); byte-compare a repeated `?t=` frame.
- Number honesty: displayed percents equal `beats.json.numbers` values
  (`0.99/0.50/0.73`) rounded; no value appears on screen that is not sourced.
- Re-render `viz/out/detectable-not-learned.mp4` via the existing capture
  (Playwright frame-step -> ffmpeg, CRF 18): 90s, 1080x1350, under ~65 MB.
- `ruff check .` and `pytest -q` stay green (no Python touched, but run to be
  safe since `tests/test_viz_mind.py` exists).
