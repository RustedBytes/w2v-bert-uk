const { spawnSync } = require("child_process");
const path = require("path");

const napiBin = process.platform === "win32" ? "napi.cmd" : "napi";
const napiPath = path.join(__dirname, "..", "node_modules", ".bin", napiBin);

const args = [
  "build",
  "--release",
  "--features",
  "nodejs,ort-dynamic",
  "--cargo-flags=--no-default-features --lib",
  ...process.argv.slice(2),
];

const result = spawnSync(napiPath, args, { stdio: "inherit" });

if (result.error) {
  console.error(result.error.message);
  process.exit(1);
}

process.exit(result.status ?? 1);
