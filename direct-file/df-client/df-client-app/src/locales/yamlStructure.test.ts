import mockEnYaml from './en.yaml';
import mockEsYaml from './es.yaml';
import mockEnScreenerYaml from '../../../df-static-site/src/locales/en.yaml';
import mockEsScreenerYaml from '../../../df-static-site/src/locales/es.yaml';
import { z, ZodError } from 'zod';
import fs from 'fs';
import pathModule from 'path';

/* here we read the entire yaml file recursively, checking that the structure of the content is
what our components expect.  Then check that any <Link...> tags in the text have an
opening/closing pair, as well as a corresponding key in the "urls" object. */

/* the regex match returns an array that looks like this:
[
  '<Link1>Find other ways to file your taxes</Link1>',
  '1',
  'Find other ways to file your taxes',
  index: 0,
  input: '<Link1>Find other ways to file your taxes</Link1>',
  groups: undefined
]
so when we do match[1] throughout this file, we're getting the suffix for any given <Link> tag.
<Link1> gives us 1, <LinkPublicFiling> gives us PublicFiling, etc.  We then use the suffixes to match
tag pairs and match link tags to what's in the urls. */

/* you'll see the repeated use of the regex /<Link(\w+)>/g throughout the file.  You'll think, "We can
create this once at the top and reuse it."  If you do that, and don't reset the lastIndex property
whenever you use the regex, you'll run into mysterious bugs, due to the fact that global regexes
maintain state.  Ask me how I know.  I chose to create the regex anew each time, instead of
resetting the lastIndex property.
More info here: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/RegExp/lastIndex */

function structuredHtmlSchemaWith<AllowedTags extends [string, ...string[]]>(allowedTags: Readonly<AllowedTags>) {
  const TagEnum = z.enum(allowedTags);

  type RecursiveHtml = string | Partial<{ [Tag in AllowedTags[number]]: RecursiveHtml }> | RecursiveHtml[];
  const RecursiveHtmlSchema: z.ZodType<RecursiveHtml> = z.lazy(() =>
    z.union([z.string(), z.record(TagEnum, RecursiveHtmlSchema), z.array(RecursiveHtmlSchema)])
  );

  return RecursiveHtmlSchema;
}

const infoDisplayAllowedTags = [`p`, `ul`, `ol`, `li`] as const;
const modalAllowedTags = [...infoDisplayAllowedTags, `h2`] as const;

const urlsSchema = z.record(z.string());

const modalContentSchema = z.object({
  header: z.string(),
  body: structuredHtmlSchemaWith(modalAllowedTags).optional(),
  // TODO: we should also have a schema for conditional sections
  // https://git.irslabs.org/irslabs-prototypes/direct-file/-/issues/4608
  urls: urlsSchema.optional(),
});

const staticModalTextSchema = z.object({
  text: structuredHtmlSchemaWith(infoDisplayAllowedTags),
});
const staticModalLinkSchema = z.record(z.union([modalContentSchema, z.string()]));

const staticModalTextKeys = [`text`];

/** Manually pick which schema validator to use, see https://github.com/colinhacks/zod/issues/2573 */

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const staticModalSchemaValidator = (staticModal: any) => {
  if (typeof staticModal !== `object` || !staticModal) {
    throw new ZodError([
      {
        code: `invalid_type`,
        expected: `object`,
        received: staticModal,
        path: [`text`],
        fatal: true,
        message: `Not a valid object`,
      },
    ]);
  }

  for (const [key, value] of Object.entries(staticModal as Record<string, unknown>)) {
    if (staticModalTextKeys.includes(key)) {
      staticModalTextSchema.parse({ [key]: value });
      // eslint-disable-next-line eqeqeq
    } else if (key == `urls`) {
      // a modal may have external urls in its body as well
      urlsSchema.parse(value);
    } else {
      staticModalLinkSchema.parse({ [key]: value });
    }
  }
  return true;
};

const staticModalSchema =
  z.custom<z.infer<typeof staticModalTextSchema | typeof staticModalLinkSchema>>(staticModalSchemaValidator);

type Modal = z.infer<typeof staticModalSchema>;
type LinkModal = z.infer<typeof staticModalLinkSchema>;

const expandableHelperTextSchema = z.object({
  heading: z.string(),
  body: structuredHtmlSchemaWith(modalAllowedTags),
  urls: urlsSchema.optional(),
});

type ExpandableHelperText = z.infer<typeof expandableHelperTextSchema>;

const alertTextSchema = z
  .object({
    heading: z.string().optional(),
    body: z
      .union([
        z.object({
          helpText: z.object({
            modals: staticModalSchema.optional(),
          }),
        }),
        z.object({
          expandableHelperText: expandableHelperTextSchema,
        }),
        structuredHtmlSchemaWith(modalAllowedTags),
        urlsSchema.optional(),
      ])
      .optional(),
    urls: urlsSchema.optional(),
  })
  .refine((data) => data.heading || data.body, {
    message: `alertText must have either a heading or a body.`,
  });

type Alert = z.infer<typeof alertTextSchema>;

const helpLinkSchema = z.object({
  text: z.string(),
  urls: urlsSchema.optional(),
});

type HelpLink = z.infer<typeof helpLinkSchema>;

type HelpTextType = 'modals' | 'helpLink' | 'hint' | 'urls';

const helpTextValues: HelpTextType[] = [`modals`, `helpLink`, `hint`, `urls`];

const isHelpTextType = (value: string): value is HelpTextType => {
  return helpTextValues.includes(value as HelpTextType);
};

const NESTED_BODY_OBJECTS = [`expandableHelperText`, `helpText`];

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const deeplyValidateTags = (node: any, path = ``, parentKey: string | null = null, currentUrls: any = null): void => {
  if (Array.isArray(node)) {
    node.forEach((item, index) => deeplyValidateTags(item, `${path}[${index}]`, parentKey, currentUrls));
  } else if (typeof node === `object` && node !== null) {
    const hasBody = node.body !== undefined;
    const hasUrls = node.urls !== undefined;
    const hasComplexBody =
      hasBody && typeof node.body !== `string` && NESTED_BODY_OBJECTS.some((key) => key in node.body);
    const newCurrentUrls = hasUrls ? node.urls : currentUrls;

    if (hasBody && !hasComplexBody) {
      const bodyContent = Array.isArray(node.body) ? node.body.map(extractTextContent).join(` `) : node.body;
      if (hasUrls || hasLinkTags(bodyContent)) {
        if (newCurrentUrls) {
          validateLinkTagsWithUrls(bodyContent, newCurrentUrls, path);
        } else {
          throw new Error(`Error: 'body' at path '${path}' contains link tags without matching 'urls'.`);
        }
      }
    }

    Object.keys(node).forEach((key) => {
      if (!(hasBody && key === `body`)) {
        deeplyValidateTags(node[key], path ? `${path}.${key}` : key, key, newCurrentUrls);
      }
    });
  } else if (typeof node === `string`) {
    validateTagsPairing(node, path);
  }
};

const validateTagsPairing = (text: string, path: string): void => {
  const openTagPattern = /<(\w+)\s*>/g; // Matches opening tags
  const closeTagPattern = /<\/(\w+)>/g; // Matches closing tags
  const selfClosingTagPattern = /<(\w+)\s*\/>/g; // Matches self-closing tags

  const foundTags = new Map(); // Track the count of each tag
  let match;

  while ((match = openTagPattern.exec(text)) !== null) {
    const tag = match[1].toLowerCase();
    if (!selfClosingTagPattern.test(`<${match[0]}`)) {
      foundTags.set(tag, (foundTags.get(tag) || 0) + 1);
    }
  }

  while ((match = closeTagPattern.exec(text)) !== null) {
    const tag = match[1].toLowerCase();
    if (foundTags.has(tag)) {
      foundTags.set(tag, foundTags.get(tag) - 1);
    } else {
      throw new Error(`[Path: ${path}] Missing opening tag for </${tag}>`);
    }
  }

  // we counted opening tags and closing tags, so if their count differs, we have a mismatch
  foundTags.forEach((count, tag) => {
    if (count !== 0) {
      throw new Error(`[Path: ${path}] Mismatched tag count for <${tag}>. Count: ${count}`);
    }
  });
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const extractTextContent = (element: any): string => {
  if (typeof element === `string`) return element;

  if (Array.isArray(element) || typeof element === `object`) {
    return Object.values(element).map(extractTextContent).join(` `);
  }
  return ``;
};

const hasLinkTags = (text: string | object) => {
  // creating the regex anew to avoid state mutation in global js regexes
  const linkTagPattern = /<Link(\w+)>/gi;
  if (typeof text === `string`) {
    return linkTagPattern.test(text);
  } else {
    return Object.values(text).some(hasLinkTags);
  }
};

const extractLinkSuffix = (text: string, path: string): string[] => {
  validateTagsPairing(text, path);
  // creating the regex anew to avoid state mutation in global js regexes
  const linkTagPattern = /<Link(\w+)>/g;
  const allMatchesIterator = text.matchAll(linkTagPattern);
  const suffixes: string[] = [];
  for (const match of allMatchesIterator) {
    const suffix = match[1];
    suffixes.push(suffix);
  }
  return suffixes;
};

// do the link tags match the keys in "urls"?
const validateLinkTagsWithUrls = (text: string | object, urls: { [key: string]: string }, path: string) => {
  if (typeof text === `object`) {
    // Text is represented as a structured html, so we need to flatten into a single, searchable string
    validateLinkTagsWithUrls(JSON.stringify(text), urls, path);
    return;
  }

  for (const key in urls) {
    if (!key.startsWith(`Link`)) {
      throw new Error(`[Path: ${path}] Key error: '${key}' in 'urls' does not begin with 'Link'.`);
    }
  }

  const linkSuffixes = extractLinkSuffix(text, path);
  for (const suffix of linkSuffixes) {
    if (!suffix.includes(`Modal`)) {
      if (!Object.hasOwn(urls, `Link${suffix}`)) {
        throw new Error(`[Path: ${path}] Mismatch: Link${suffix} found in text but not in 'urls' keys.`);
      }
    }
  }

  // do the url keys have matching link tags?
  for (const key in urls) {
    const url = urls[key];
    const matchPattern = new RegExp(`<${key}>`);
    // If the URL is a local image, check if the file exists
    if (/\.(jpeg|jpg|gif|png)$/.test(url)) {
      const imagePath = pathModule.resolve(__dirname, `..`, `..`, `public`, `imgs`, url);
      if (!fs.existsSync(imagePath)) {
        throw new Error(`[Path: ${path}] Image file ${url} does not exist.`);
      }
    }

    if (!matchPattern.test(text)) {
      throw new Error(`[Path: ${path}] Mismatch: ${key} found in 'urls' keys but not in text.`);
    }
  }
};

/**
 * The model's `text` node needs to contain the trigger to launch the modal(s).
 *
 * Returns the number of each modal as a string (e.g. `["1", "2"]`)
 */
const recursivelyValidateModalText = (text: string | object, path: string, numbers: string[]): string[] => {
  if (typeof text === `string`) {
    validateTagsPairing(text, path);

    // Extract all LinkModal# tags from the top level text
    const modalTagsPattern = /<LinkModal(\w+)>/g;
    let match;
    while ((match = modalTagsPattern.exec(text)) !== null) {
      numbers.push(match[1]);
    }
  } else {
    Object.values(text).forEach((v) => recursivelyValidateModalText(v, path, numbers));
  }
  return numbers;
};

const validateModal = (modal: Modal, path: string) => {
  const parsedModal = staticModalSchema.safeParse(modal);

  if (!parsedModal.success) {
    throw new Error(`[Path: ${path}] Modal does not match expected structure. ${parsedModal.error.message}`);
  }

  const validatedModal = parsedModal.data as LinkModal;

  const modalNumbersFromText = recursivelyValidateModalText(validatedModal.text, path, []);

  const modalNumbersFromProperties: string[] = [];
  for (const prop in validatedModal) {
    const propMatch = prop.match(/^LinkModal(\w+)$/);
    if (propMatch) {
      modalNumbersFromProperties.push(propMatch[1]);
    }
  }

  // Validate that for every <LinkModal> tag in the text, there is a corresponding LinkModal# object
  for (const modalNumber of modalNumbersFromText) {
    const modalContent = validatedModal[`LinkModal${modalNumber}`];

    if (typeof modalContent === `string`) continue;

    // If the modal content's body text contains <Link> tags, there should be a urls object
    if (modalContent.body && hasLinkTags(modalContent.body) && !modalContent.urls) {
      throw new Error(
        `[Path: ${path}] Body text for <LinkModal${modalNumber}> contains <Link> tags but 'urls' property is missing.`
      );
    }

    deeplyValidateTags(modalContent, path);
  }

  // check that every LinkModal# object has a corresponding <LinkModal#> tag in the text
  for (const modalNumber of modalNumbersFromProperties) {
    if (!modalNumbersFromText.includes(modalNumber)) {
      throw new Error(
        `[Path: ${path}] <LinkModal${modalNumber}> tag missing for the existing LinkModal${modalNumber} property.`
      );
    }
  }
};

const validateHintNamespace = (path: string) => {
  // hints must be in the info namespace
  if (!path.includes(`info.`)) {
    throw new Error(`[Path: ${path}] Hints must be in the info namespace.`);
  }
};

const validateHelpLink = (helpLink: HelpLink, path: string) => {
  const parsedHelpLink = helpLinkSchema.safeParse(helpLink);

  if (!parsedHelpLink.success) {
    throw new Error(`[Path: ${path}] HelpLink does not match expected structure. ${parsedHelpLink.error.message}`);
  }

  const validatedHelpLink = parsedHelpLink.data;

  // Are there <Link> tags in the text, but no "urls" property?
  if (hasLinkTags(validatedHelpLink.text)) {
    if (!validatedHelpLink.urls) {
      throw new Error(`[Path: ${path}] HelpLink text contains <Link> tags but 'urls' property is missing.`);
    }
    validateLinkTagsWithUrls(validatedHelpLink.text, validatedHelpLink.urls, path);
  }
};

const validateExpandableHelperText = (expandableHelperText: ExpandableHelperText, path: string) => {
  const parsedResult = expandableHelperTextSchema.safeParse(expandableHelperText);

  if (!parsedResult.success) {
    throw new Error(
      `[Path: ${path}] ExpandableHelperText does not match expected structure. ${parsedResult.error.message}`
    );
  }

  const validatedExpandableHelperText = parsedResult.data;
  if (hasLinkTags(validatedExpandableHelperText.body)) {
    if (!validatedExpandableHelperText.urls) {
      throw new Error(`[Path: ${path}] ExpandableHelperText body contains <Link> tags but 'urls' property is missing.`);
    }
    validateLinkTagsWithUrls(validatedExpandableHelperText.body, validatedExpandableHelperText.urls, path);
  }
};

/**
 * An alert is a colored box on the screen:
 *   it can contain text, arbitrary HTML, an accordian,
 *   or an entire modal (trigger + hidden modal content).
 */
const validateAlertText = (alertText: Alert, path: string) => {
  const parsedResult = alertTextSchema.safeParse(alertText);
  if (!parsedResult.success) {
    throw new Error(`[Path: ${path}] alertText does not match expected structure. ${parsedResult.error.message}`);
  }

  const validatedAlertText = parsedResult.data;
  if (validatedAlertText.body) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { helpText, expandableHelperText } = validatedAlertText.body as any;

    if (!helpText?.modals && !expandableHelperText && hasLinkTags(validatedAlertText.body)) {
      // If the alert's body text contains <Link> tags, there should be a urls object
      if (!validatedAlertText.urls) {
        throw new Error(`[Path: ${path}] Body text in alertText contains <Link> tags but 'urls' property is missing.`);
      }
      // check that any links in the urls object appear in the text
      validateLinkTagsWithUrls(validatedAlertText.body, validatedAlertText.urls, path);
    }

    if (helpText?.modals) {
      validateModal(helpText.modals, path);
    }

    if (expandableHelperText) {
      validateExpandableHelperText(expandableHelperText, path);
    }
  }
};

// not going to worry about typing the yaml structure for a test that's validating the yaml anyway
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const recursivelyTraverse = (node: any, path = ``) => {
  if (!node || typeof node !== `object`) {
    return;
  }
  const hasHelpText = node.helpText !== undefined;
  const hasAlertText = node.alertText !== undefined;
  const hasExpandableHelperText = node.body?.expandableHelperText !== undefined;
  const hasHelpTextModals = node.helpText?.modals !== undefined;
  const hasHelpTextHelpLink = node.helpText?.helpLink !== undefined;
  const hasHelpTextHint = node.helpText?.hint !== undefined;
  const isKnownStructure =
    hasHelpText ||
    hasAlertText ||
    hasExpandableHelperText ||
    hasHelpTextModals ||
    hasHelpTextHelpLink ||
    hasHelpTextHint;

  if (hasHelpText) {
    for (const key in node.helpText) {
      if (!isHelpTextType(key)) {
        throw new Error(
          `Invalid helpText type '${key}' found at path ${path}. Expected 'modals', 'helpLink', or 'hint'.`
        );
      }
    }
    if (hasHelpTextModals) {
      validateModal(node.helpText.modals, path);
    }
    if (hasHelpTextHelpLink) {
      validateHelpLink(node.helpText.helpLink, path);
    }
    if (hasHelpTextHint) {
      validateHintNamespace(path);
      validateHelpLink(node.helpText.hint, path);
    }
  }
  if (hasAlertText) {
    validateAlertText(node.alertText, path);
  }

  if (hasExpandableHelperText) {
    validateExpandableHelperText(node.body.expandableHelperText, path);
  }

  if (!isKnownStructure) {
    deeplyValidateTags(node, path);
  }

  for (const key in node) {
    if (![`helpText`, `alertText`, `body`].includes(key)) {
      const newPath = path ? `${path}.${key}` : key;
      recursivelyTraverse(node[key], newPath);
    }
  }
};

describe(`English yaml structure`, () => {
  it(`DF English known yaml object types should have the shape and content we expect`, () => {
    recursivelyTraverse(mockEnYaml);
  });
  it(`Screener English known yaml object types should have the shape and content we expect`, () => {
    recursivelyTraverse(mockEnScreenerYaml);
  });
});

describe(`Spanish yaml structure`, () => {
  it(`DF Spanish known yaml object types should have the shape and content we expect`, () => {
    recursivelyTraverse(mockEsYaml);
  });
  it(`Screener Spanish known yaml object types should have the shape and content we expect`, () => {
    recursivelyTraverse(mockEsScreenerYaml);
  });
});
