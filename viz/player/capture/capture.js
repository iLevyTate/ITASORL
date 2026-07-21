// Headless frame capture for the deterministic outreach film (player.js).
// Drives window.__seek(t) frame-by-frame and pipes PNG frames straight into
// ffmpeg's stdin (image2pipe) so nothing but the final video touches disk.
//
// Env:
//   CHROME_EXE  path to a cached chromium chrome.exe
//   CAP_URL     player URL (default http://127.0.0.1:8765/index.html)
//   CAP_FPS     frames per second (default 30)
//   CAP_OUT     output video path (default out.mp4)
//   CAP_MODE    "proof" (a few keyframe PNGs) or "full" (encode video)

const { chromium } = require("playwright");
const { spawn } = require("child_process");
const fs = require("fs");

const CHROME = process.env.CHROME_EXE;
const URL = process.env.CAP_URL || "http://127.0.0.1:8765/index.html";
const FPS = parseInt(process.env.CAP_FPS || "30", 10);
const OUT = process.env.CAP_OUT || "out.mp4";
const MODE = process.env.CAP_MODE || "full";
const W = 1080;
const H = 1350;

function writeChunk(stream, buf) {
  return new Promise((resolve) => {
    if (stream.write(buf)) resolve();
    else stream.once("drain", resolve);
  });
}

(async () => {
  const browser = await chromium.launch({
    executablePath: CHROME,
    headless: true,
    args: ["--force-color-profile=srgb", "--hide-scrollbars"],
  });
  const page = await browser.newPage({
    viewport: { width: W, height: H },
    deviceScaleFactor: 1,
  });
  page.on("console", (m) => {
    const t = m.type();
    if (t === "error" || t === "warning") console.log(`[page:${t}]`, m.text());
  });

  await page.goto(URL, { waitUntil: "load" });
  await page.waitForFunction("window.__ready === true", null, { timeout: 30000 });
  await page.evaluate("document.fonts && document.fonts.ready");

  const duration = await page.evaluate("window.__duration");
  const src = await page.evaluate("window.__sceneSource");
  const frameCount = Math.round((duration / 1000) * FPS);
  console.log(`duration_ms=${duration} fps=${FPS} frames=${frameCount} scene=${src}`);

  // Number honesty: the published film must come from the real recorded world,
  // never the deterministic placeholder fallback. Proof mode may run on either.
  if (MODE !== "proof" && src !== "collect.py") {
    console.error(
      `CAPTURE_ERROR refusing to encode: scene source is "${src}", expected "collect.py". ` +
      `Run viz/collect.py so viz/data/scene.json exists before rendering.`
    );
    await browser.close();
    process.exit(2);
  }

  if (MODE === "proof") {
    fs.mkdirSync("proof", { recursive: true });
    // One frame late in each beat (90% through), after gauge sweeps have
    // settled on their final displayed values; a mid-sweep frame can show a
    // transient number that contradicts the science (e.g. 0.97 during the
    // 0.99 -> 0.50 sweep).
    const beats = JSON.parse(fs.readFileSync("../beats.json", "utf8")).beats;
    const times = beats
      .map((b) => Math.round(b.t0 + 0.9 * (b.t1 - b.t0)))
      .filter((t) => t < duration);
    for (const t of times) {
      await page.evaluate((tt) => window.__seek(tt), t);
      await page.screenshot({ path: `proof/frame_${String(t).padStart(6, "0")}.png` });
      console.log("proof frame t=", t);
    }
    await browser.close();
    console.log("PROOF_DONE");
    return;
  }

  const ff = spawn(
    "ffmpeg",
    [
      "-y",
      "-f", "image2pipe",
      "-framerate", String(FPS),
      "-i", "pipe:0",
      "-c:v", "libx264",
      "-preset", "medium",
      "-crf", "18",
      "-pix_fmt", "yuv420p",
      "-movflags", "+faststart",
      OUT,
    ],
    { stdio: ["pipe", "inherit", "inherit"] }
  );
  const ffDone = new Promise((res, rej) => {
    ff.on("close", (code) => (code === 0 ? res() : rej(new Error("ffmpeg exit " + code))));
  });

  const t0 = Date.now();
  for (let i = 0; i < frameCount; i++) {
    const t = Math.round((i * 1000) / FPS);
    await page.evaluate((tt) => window.__seek(tt), t);
    const buf = await page.screenshot({ type: "png" });
    await writeChunk(ff.stdin, buf);
    if (i % 60 === 0 || i === frameCount - 1) {
      const el = (Date.now() - t0) / 1000;
      const rate = i > 0 ? i / el : 0;
      const eta = rate > 0 ? (frameCount - i) / rate : 0;
      console.log(
        `frame ${i + 1}/${frameCount} t=${t}ms  ${rate.toFixed(1)}fps elapsed=${el.toFixed(0)}s eta=${eta.toFixed(0)}s`
      );
    }
  }
  ff.stdin.end();
  await ffDone;
  await browser.close();
  console.log("FULL_DONE ->", OUT);
})().catch((e) => {
  console.error("CAPTURE_ERROR", e);
  process.exit(1);
});
