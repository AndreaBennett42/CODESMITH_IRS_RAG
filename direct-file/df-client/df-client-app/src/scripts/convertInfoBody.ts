/**
 * This script is added for historical/review purposes
 */
import fs from 'fs';
import { load, dump } from 'js-yaml';
import { YamlSettings } from '../locales/yaml-settings.js';
import { CommonTranslation } from 'df-i18n';

const ALLOWED_TAGS = [`p`, `ul`, `ol`, `li`, `h2`];
type ContentDisplayItem<AllowedTag extends string> =
  | Record<AllowedTag, string | ContentDisplayItem<AllowedTag>[]>
  | string;

type LegacyModalEntry = Omit<CommonTranslation.ModalEntry, `body` | 'urls'> & {
  body: {
    text: string;
    urls: Record<string, string>;
  };
};
type LegacyModalData = { text: string } & {
  [key: string]: LegacyModalEntry;
};

interface InfoTranslations<AllowedTag extends string = (typeof ALLOWED_TAGS)[number]> {
  body: string | ContentDisplayItem<AllowedTag>[];
  helpText?: {
    modals?: CommonTranslation.ModalData | LegacyModalData;
  };
}

function isNotEmpty(val: unknown) {
  return !!val;
}

// NOTE: I don't like using a regex this awkward for this, but I didn't have much luck with alternatives
/**
 * Partial breakdown of regex:
 *    Opening tag: <(p|ul|ol|li)>
 *    Closing tag for the captured opening tag: <\/1>
 *    Content of captured tag: ((?!<\1>)[^]*?)
 *        Negative lookahead to avoid capturing the closing tag: (?!<\1>)
 *        Capture all content, including new lines: [^]*?
 *    Everything that is not a tag we are trying to capture: ((?:(?!<\/?(p|ul|ol|li|h2)>)[^])+)
 */
const HtmlRegex = /<(p|ul|ol|li|h2)>((?!<\1>)[^]*?)<\/\1>|((?:(?!<\/?(p|ul|ol|li|h2)>)[^])+)/gm;

function embeddedHtmlToContentArray<AllowedTags extends string[]>(
  body: string,
  allowedTags: AllowedTags
): ContentDisplayItem<AllowedTags[number]>[] | ContentDisplayItem<AllowedTags[number]> | string {
  if (body.includes(`<`)) {
    // There is (probably) html here
    const tokens = Array.from(body.matchAll(HtmlRegex));

    const newBody = tokens
      .map(([body, tag, content]) => {
        if (tag && allowedTags.includes(tag)) {
          const displayItems = embeddedHtmlToContentArray(content, allowedTags);
          // eslint-disable-next-line eqeqeq
          if (Array.isArray(displayItems) && displayItems.length == 0) {
            return ``;
          }

          return {
            [tag]:
              typeof displayItems === `string` ? embeddedHtmlToContentArray(displayItems, allowedTags) : displayItems,
          } as ContentDisplayItem<AllowedTags[number]>;
        } else return body.trim();
      })
      .filter(isNotEmpty);

    // Avoid unnecessary arrays
    if (newBody.length === 1) return newBody[0];
    else return newBody;
  } else return body.trim();
}

function convertBody(localeTranslations: { headings: Record<string, InfoTranslations> }) {
  return {
    ...localeTranslations,
    headings: Object.fromEntries(
      Object.entries(localeTranslations.headings as Record<string, InfoTranslations>).map(
        ([infoKey, infoTranslations]) => {
          const helpText = infoTranslations.helpText;

          if (helpText?.modals) {
            // NOTE: Modifying in place to simplify return statement
            const { text, ...modalEntries } = helpText.modals;
            helpText.modals = {
              text,
              ...Object.fromEntries(
                Object.entries(modalEntries).map(
                  ([modalKey, modalCopy]: [string, CommonTranslation.ModalEntry | LegacyModalEntry]) => {
                    const { header, body } = modalCopy;
                    if (typeof body === `object` && `text` in body && typeof body.text === `string`) {
                      const newBody = embeddedHtmlToContentArray(body.text, ALLOWED_TAGS);
                      return [
                        modalKey,
                        {
                          header,
                          urls: body.urls,
                          body: newBody,
                        },
                      ];
                    } else {
                      return [modalKey, modalCopy];
                    }
                  }
                )
              ),
            };
          }

          return [infoKey, infoTranslations];
        }
      )
    ),
  };
}

function main() {
  const localesToConvert = [`en`] as const;

  for (const locale of localesToConvert) {
    // nosemgrep: eslint.detect-non-literal-fs-filename
    const localeTranslations = load(fs.readFileSync(`./src/locales/${locale}.yaml`, `utf-8`), YamlSettings) as {
      headings: Record<string, InfoTranslations>;
    };

    const convertedLocale = convertBody(localeTranslations);

    // nosemgrep: eslint.detect-non-literal-fs-filename
    fs.writeFileSync(`./src/locales/${locale}.yaml`, dump(convertedLocale, YamlSettings));
  }
}

main();
