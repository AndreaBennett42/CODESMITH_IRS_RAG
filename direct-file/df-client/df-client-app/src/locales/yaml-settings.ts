import { JSON_SCHEMA, type DumpOptions } from 'js-yaml';

export const YamlSettings: DumpOptions = { schema: JSON_SCHEMA, noRefs: true, quotingType: `"`, lineWidth: 120 };
