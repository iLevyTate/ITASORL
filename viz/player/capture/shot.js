// One-off still capture: navigate to any player URL, seek to one time, save PNG.
// Env: CHROME_EXE, SHOT_URL, SHOT_T (ms, default 6000), SHOT_OUT (png path).
const { chromium } = require("playwright");

const CHROME = process.env.CHROME_EXE;
const URL = process.env.SHOT_URL;
const T = parseInt(process.env.SHOT_T || "6000", 10);
const OUT = process.env.SHOT_OUT || "shot.png";

(async () => {
  const browser = await chromium.launch({
    executablePath: CHROME,
    headless: true,
    args: ["--force-color-profile=srgb", "--hide-scrollbars"],
  });
  const page = await browser.newPage({ viewport: { width: 1080, height: 1350 }, deviceScaleFactor: 1 });
  page.on("console", (m) => {
    const t = m.type();
    if (t === "error" || t === "warning") console.log(`[page:${t}]`, m.text());
  });
  await page.goto(URL, { waitUntil: "load" });
  await page.waitForFunction("window.__ready === true", null, { timeout: 30000 });
  const src = await page.evaluate("window.__sceneSource");
  await page.evaluate((tt) => window.__seek(tt), T);
  await page.screenshot({ path: OUT });
  console.log(`SHOT_DONE url=${URL} t=${T} src=${src} out=${OUT}`);
  await browser.close();
})().catch((e) => {
  console.error("SHOT_ERROR", e);
  process.exit(1);
});
