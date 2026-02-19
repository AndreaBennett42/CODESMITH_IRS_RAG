/* eslint-disable no-console */
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// A function to select the transpilation from the latest scala version (in case someone forgets to run clean)
function selectLatestVersionFolder(folderPath: string): string | null {
  if (!fs.existsSync(folderPath)) {
    console.error(`SCRIPT ERROR: Directory '${folderPath}' does not exist.`);
    return null;
  }

  const versions = fs
    .readdirSync(folderPath)
    .filter((file) => fs.lstatSync(path.join(folderPath, file)).isDirectory())
    .map((folderName) => {
      // Parse the version number from the folder name
      const match = folderName.match(/scala-(\d+\.\d+\.\d+)/);
      return match ? match[1] : null;
    })
    .filter((version): version is string => version !== null)
    .sort((a, b) => {
      // Compare version numbers to find the latest one
      return a.localeCompare(b);
    });

  if (versions.length === 0) {
    console.error(`No version folders found in '${folderPath}'.`);
    return null;
  }

  const latestVersion = versions[versions.length - 1];
  const latestVersionFolder = path.join(folderPath, `scala-${latestVersion}`);
  return latestVersionFolder;
}

const fgsJsTargetPath = path.join(__dirname, `..`, `..`, `..`, `..`, `fact-graph-scala`, `js`, `target`);
console.info(`🚀 This script is using the following target folder path: \n`, fgsJsTargetPath);
const latestScalaVersionPath = selectLatestVersionFolder(fgsJsTargetPath);
console.info(`\n🚀 This script is using the following scala version path: \n`, latestScalaVersionPath);
console.log(`\n`);

if (latestScalaVersionPath) {
  const mainJsSrc = path.join(latestScalaVersionPath, `fact-graph-fastopt`, `main.js`);
  const mainJsMapSrc = path.join(latestScalaVersionPath, `fact-graph-fastopt`, `main.js.map`);

  const mainJsDest = path.join(__dirname, `..`, `..`, `..`, `js-factgraph-scala`, `src`, `main.js`);
  const mainJsMapDest = path.join(__dirname, `..`, `..`, `..`, `js-factgraph-scala`, `src`, `main.js.map`);

  fs.copyFile(mainJsSrc, mainJsDest, (err) => {
    if (err) {
      console.error(`SCRIPT ERROR: error copying main.js file:`, err);
    } else {
      console.info(`Main.js copied successfully!`);
    }
  });

  fs.copyFile(mainJsMapSrc, mainJsMapDest, (err) => {
    if (err) {
      console.error(`SCRIPT ERROR: Error copying main.js.map file:`, err);
    } else {
      console.info(`Main.js.map copied successfully!`);
    }
  });
} else {
  console.error(
    `SCRIPT ERROR: there are no scala folders in the "fact-graph-scala/js/target folder", did you run fastOptJS?`
  );
}
