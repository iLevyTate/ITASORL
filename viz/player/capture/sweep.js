// Multi-still sweep with ONE browser: seek through a list of times and save
// numbered PNGs. Env: CHROME_EXE, SWEEP_URL, SWEEP_TIMES ("a,b,c" ms),
// SWEEP_DIR (output directory).
const { chromium } = require("playwright");
const fs = require("fs");

const CHROME = process.env.CHROME_EXE;
const URL = process.env.SWEEP_URL;
const TIMES = (process.env.SWEEP_TIMES || "").split(",").map((s) => parseInt(s, 10));
const DIR = process.env.SWEEP_DIR || "sweep";

(async () => {
  fs.mkdirSync(DIR, { recursive: true });
  const browser = await chromium.launch({
    executablePath: CHROME,
    headless: true,
    args: ["--force-color-profile=srgb", "--hide-scrollbars"],
  });
  const page = await browser.newPage({ viewport: { width: 1080, height: 1350 }, deviceScaleFactor: 1 });
  page.on("pageerror", (e) => console.log("[pageerror]", e.message));
  await page.goto(URL, { waitUntil: "load" });
  await page.waitForFunction("window.__ready === true", null, { timeout: 30000 });
  for (const t of TIMES) {
    await page.evaluate((tt) => window.__seek(tt), t);
    await page.screenshot({ path: `${DIR}/t${String(t).padStart(6, "0")}.png` });
  }
  await browser.close();
  console.log(`SWEEP_DONE n=${TIMES.length} dir=${DIR}`);
})().catch((e) => {
  console.error("SWEEP_ERROR", e);
  process.exit(1);
});
