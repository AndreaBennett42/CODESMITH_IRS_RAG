import yaml from 'js-yaml';
import fs from 'fs';
import { unmapFactAlias } from './dependencyGraph.js';
import { Path } from '../Path.js';

interface ConfigurationLocationYaml {
  'configuration-location': string;
}

interface PdfsYaml {
  'configured-pdfs': ConfigurationLocationYaml[];
}

interface DirectFileYaml {
  pdfs: PdfsYaml;
}

interface ApplicationYaml {
  'direct-file': DirectFileYaml;
}

interface HasOtherRequiredFacts {
  otherRequiredFacts: string[];
}

interface TableFactExpressionMapping {
  factExpression: string;
}

interface HasTable {
  table: Table;
}

interface Table {
  rowsCollectionPath: string;
  columns: TableFactExpressionMapping[];
  oncePerPage: TableFactExpressionMapping[];
}

function getFactExpressions(obj: object): string[] {
  const ret: string[] = [];

  for (const [property, value] of Object.entries(obj)) {
    if ([`includeWhen`, `includeForEach`].includes(property)) {
      // These top level keys have fact path values.
      ret.push(value);
    } else if (property === `otherRequiredFacts`) {
      // This key is just a list (array) of fact path values.
      const hasOtherRequiredFacts = obj as HasOtherRequiredFacts;
      for (const path of hasOtherRequiredFacts[property]) {
        ret.push(path);
      }
    } else if (property === `table`) {
      // This key has certain subkeys with fact expression values, but not all.
      const hasTable = obj as HasTable;
      const table = hasTable[property] as Table;
      ret.push(table.rowsCollectionPath);
      for (const column of table.columns) {
        ret.push(column.factExpression.replaceAll(/\.\.\//g, table.rowsCollectionPath + `/*/`));
      }
      if (table.oncePerPage) {
        for (const opp of table.oncePerPage) {
          ret.push(opp.factExpression.replaceAll(/\.\.\//g, table.rowsCollectionPath + `/*/`));
        }
      }
    } else if (property === `form`) {
      // This key is top level of a recursive structure.
      ret.push(...getLeafStrings(obj));
    } else if (property === `customMaps`) {
      // Custom data mappings are just passed thru to PDF service; no validation.
    } else {
      throw new Error(`Unknown top level key ${property}`);
    }
  }
  return ret;
}

function getLeafStrings(obj: object): string[] {
  const ret: string[] = [];

  for (const value of Object.values(obj)) {
    // eslint-disable-next-line eqeqeq
    if (typeof value == `string`) {
      ret.push(value);
      // eslint-disable-next-line eqeqeq
    } else if (value != null) {
      ret.push(...getLeafStrings(value));
    }
  }
  return ret;
}

export function readPdfFactPaths(): Path[] {
  const APPLICATION_YAML = `../../backend/src/main/resources/application.yaml`;
  const PDF_RESOURCES = `../../backend/src/main/resources/`;
  const factPathRegEx = /\/\S+/g;
  const factPaths = new Set();

  // Read `application.yml` ...
  // nosemgrep: nodejs_scan.javascript-eval-rule-yaml_deserialize
  const applicationYaml = yaml.load(fs.readFileSync(APPLICATION_YAML, `utf-8`)) as ApplicationYaml;
  const configs = applicationYaml[`direct-file`].pdfs[`configured-pdfs`] as ConfigurationLocationYaml[];

  // ... to get the location of each PDF config file ...
  for (let i = 0; i < configs.length; i++) {
    // nosemgrep: nodejs_scan.javascript-eval-rule-yaml_deserialize
    const yamlObject = yaml.load(
      // nosemgrep: eslint.detect-non-literal-fs-filename
      fs.readFileSync(`${PDF_RESOURCES}${configs[i][`configuration-location`]}`, `utf-8`)
    ) as object;

    // eslint-disable-next-line eqeqeq
    if (yamlObject == null) continue;

    // ... and extract the fact expressions that are values for terminal fields.
    const expressions = getFactExpressions(yamlObject);
    for (let expression of expressions) {
      // Remove any comment in fact expression.
      const index = expression.indexOf(`#`);
      if (index >= 0) {
        expression = expression.substring(0, index);
      }

      // Extract fact paths from the expression.
      for (const factPath of expression.matchAll(factPathRegEx)) {
        // Exclude "pseudo paths" that contain tildes.
        if (factPath[0].includes(`~`)) continue;
        // Convert "index paths" to wildcard form.
        factPaths.add(unmapFactAlias(factPath[0].replaceAll(/\[\d+\]/g, `*`)));
      }
    }
  }

  return [...factPaths.values()].sort() as Path[];
}
