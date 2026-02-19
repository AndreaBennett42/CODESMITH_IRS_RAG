import flowNodes from './flow.js';
import { describe, it, expect } from 'vitest';
import {
  BooleanFactDeclaration,
  ConditionalListDeclaration,
  DFAlertDeclaration,
  EnumFactDeclaration,
  InfoDisplayDeclaration,
  MefAlertDeclaration,
} from './ContentDeclarations.js';
import en from '../locales/en.yaml';
import { findFlowNodesOfType } from './findFlowNodesOfType.js';
import { CollectionLoopDeclaration } from './flowDeclarations.js';
import { EnumFactProps } from '../types/core.js';
import { findFlowNodesWithProp } from './findFlowNodesWithProp.js';
import { extractTags, HamlTree } from '../components/HelperText/DFModal.js';
const REGEX_FOR_KEY_WITH_DOTSLASH = /^([a-zA-Z]*)\.(\/.*)/;
const REGEX_FOR_KEY_WITH_SLASH = /^([a-zA-Z]*)\/(.*)/;
const REGEX_FOR_KEY_WITH_DOT = /^([a-zA-Z]*)\.(.*)/;
const REGEX_FOR_KEY_WITH_LEADING_SLASH = /^([a-zA-Z]*)\/(.*)/;

describe(`Flow Nodes correspond to english translations`, () => {
  describe(`All Heading nodes have corresponding translations`, () => {
    const HeadingNodes = findFlowNodesOfType(flowNodes, `Heading`);
    for (const node of HeadingNodes) {
      const props = node.props as InfoDisplayDeclaration;
      it(`Has key for ${props.i18nKey}`, () => {
        expect(en.headings[props.i18nKey as keyof typeof en.headings]).toBeDefined();
      });
    }
  });
  describe(`All InfoDisplay nodes have corresponding translations`, () => {
    const InfoDisplayNodes = findFlowNodesOfType(flowNodes, `InfoDisplay`);
    for (const node of InfoDisplayNodes) {
      const props = node.props as InfoDisplayDeclaration;
      it(`Has key for ${props.i18nKey}`, () => {
        expect(en.info[props.i18nKey as keyof typeof en.info], `${props.i18nKey} was undefined`).toBeDefined();
        expect(
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          (en.info[props.i18nKey as keyof typeof en.info] as any).body,
          `${props.i18nKey}.body was undefined`
        ).toBeDefined();
      });
    }
  });
  describe(`All HelpLink nodes have corresponding translations`, () => {
    const HelpLinkNodes = findFlowNodesOfType(flowNodes, `HelpLink`);
    for (const node of HelpLinkNodes) {
      const props = node.props as InfoDisplayDeclaration;
      it(`Has key for ${props.i18nKey}`, () => {
        expect(en.info[props.i18nKey as keyof typeof en.info]).toBeDefined();
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        expect((en.info[props.i18nKey as keyof typeof en.info] as any).helpText).toBeDefined();
      });
    }
  });

  describe(`All DFModal nodes have corresponding translations`, () => {
    const DFModalNodes = findFlowNodesOfType(flowNodes, `DFModal`);
    for (const node of DFModalNodes) {
      const props = node.props as InfoDisplayDeclaration;
      it(`Has key for ${props.i18nKey}`, () => {
        expect(en.info[props.i18nKey as keyof typeof en.info]).toBeDefined();
      });
      const modalTrigger = extractTags(en.info[props.i18nKey as keyof typeof en.info].helpText.modals.text as HamlTree);
      const sharedModalKeys = modalTrigger
        .filter((tag) => tag.startsWith(`<sharedModal`))
        .map((tag) => {
          return tag.replace(/[<>]/g, ``);
        });
      sharedModalKeys.forEach((key) => {
        it(`Has key for <${key}> from ${props.i18nKey}`, () => {
          expect(en.modals[key as keyof typeof en.info]).toBeDefined();
        });
      });
    }
  });

  describe(`All DFAlert nodes have corresponding translations`, () => {
    const DFAlertNodes = findFlowNodesOfType(flowNodes, `DFAlert`);
    for (const node of DFAlertNodes) {
      const props = node.props as DFAlertDeclaration;
      if (props.i18nKey === null) {
        it(`has child configs when i18nKey is null`, () => {
          expect(props.children).toBeDefined();
        });
      } else {
        it(`Has key for ${props.i18nKey}`, () => {
          expect(en.info[props.i18nKey as keyof typeof en.info].alertText).toBeDefined();
        });
        it(`has no implicit modal content in ${props.i18nKey}`, () => {
          expect(en.info[props.i18nKey as keyof typeof en.info].helpText?.body?.helpText?.modals).not.toBeDefined();
          expect(JSON.stringify(en.info[props.i18nKey as keyof typeof en.info]).includes(`modals`)).toBeFalsy();
        });
      }
    }
  });

  describe(`All MefAlert nodes have corresponding translations`, () => {
    const MefAlertNodes = findFlowNodesOfType(flowNodes, `MefAlert`);
    for (const node of MefAlertNodes) {
      const props = node.props as MefAlertDeclaration;
      it(`Has key for ${props.i18nKey}`, () => {
        expect(en.mefAlerts[props.mefErrorCode][props.i18nKey].alertText.body).toBeDefined();
      });
    }
  });

  describe(`All TaxReturnAlert nodes have corresponding translations`, () => {
    const TaxReturnAlertNodes = findFlowNodesOfType(flowNodes, `TaxReturnAlert`);
    for (const node of TaxReturnAlertNodes) {
      const props = node.props;
      if (props.i18nKey === null) {
        it(`has child configs when i18nKey is null`, () => {
          expect(props.children).toBeDefined();
        });
      } else {
        it(`Has key for ${props.i18nKey}`, () => {
          expect(en.info[props.i18nKey as keyof typeof en.info].alertText).toBeDefined();
        });
        it(`has no implicit modal content in ${props.i18nKey}`, () => {
          expect(en.info[props.i18nKey as keyof typeof en.info].helpText?.body?.helpText?.modals).not.toBeDefined();
          expect(JSON.stringify(en.info[props.i18nKey as keyof typeof en.info]).includes(`modals`)).toBeFalsy();
        });
      }

      it(`Has the internalLink prop if the translation has internalLink defined for ${props.i18nKey}`, () => {
        const translationHasInternalLink = Object.hasOwn(
          en.info[props.i18nKey as keyof typeof en.info],
          `internalLink`
        );
        if (translationHasInternalLink) expect(props.internalLink).not.toBe(undefined);
      });

      it(`Has the data view alert key defined for ${props.i18nKey} with type ${props.type}`, () => {
        const dataViewKey = `${props.i18nKey}/data-view`;
        if (dataViewKey in en.info) {
          expect(en.info[`${props.i18nKey}/data-view`]).toBeDefined();
        } else {
          expect(en.dataviews.alerts[props.type]).toBeDefined();
        }
      });
    }
  });

  // TODO expand this to all fact types
  describe(`All Boolean fact nodes have corresponding translations`, () => {
    const booleanNodes = findFlowNodesOfType(flowNodes, `Boolean`);
    const nodesWithKey = booleanNodes.filter((node) => {
      const props = node.props as BooleanFactDeclaration;
      return props.i18nKeySuffixContext;
    });

    for (const node of nodesWithKey) {
      const props = node.props as BooleanFactDeclaration;
      it(`Has key for ${props.path}.${props.i18nKeySuffixContext}`, () => {
        expect(en.fields[props.path][props.i18nKeySuffixContext as string].boolean.yes).toBeDefined();
        expect(en.fields[props.path][props.i18nKeySuffixContext as string].boolean.no).toBeDefined();
      });
    }
  });

  describe(`All Enum fact nodes have corresponding translations`, () => {
    const enumNodes = findFlowNodesOfType(flowNodes, `Enum`);
    const nodesWithKey = enumNodes.filter((node) => {
      const props = node.props as EnumFactProps;
      return props.i18nKeySuffixContext;
    });

    for (const node of nodesWithKey) {
      const props = node.props as EnumFactDeclaration;
      // this is very hacky and self-referential
      const foundKey = en.fields[props.path];
      const optionsPath = Object.entries(foundKey)[0][0];
      it(`Has key for ${props.path}`, () => {
        expect(en.fields[props.path][optionsPath][props.i18nKeySuffixContext as string]).toBeDefined();
      });
    }
  });

  describe(`All CollectionItem fact nodes have corresponding translations`, () => {
    const collectionLoops = findFlowNodesOfType(flowNodes, `CollectionLoop`);
    const loops = collectionLoops.filter((loop) => loop.props.autoIterate !== true);

    for (const loop of loops) {
      const props = loop.props as CollectionLoopDeclaration;
      it(`Has primary key for ${props.loopName}`, () => {
        const keysExist = [
          en.fields[props.loopName]?.collectionListing.itemHeading1,
          en.fields[props.collection]?.collectionListing.itemHeading1,
        ];
        expect(keysExist.some(Boolean)).toBe(true);
      });
      it(`Has back up key for ${props.loopName}`, () => {
        const keysExist = [
          en.fields[props.loopName]?.collectionListing.itemHeading2,
          en.fields[props.collection]?.collectionListing.itemHeading2,
        ];
        expect(keysExist.some(Boolean)).toBe(true);
      });
    }
  });

  describe(`All conditional ConditionalList items have corresponding translations`, () => {
    const conditionalLists = findFlowNodesOfType(flowNodes, `ConditionalList`);
    for (const node of conditionalLists) {
      const props = node.props as ConditionalListDeclaration;
      for (const item of props.items) {
        it(`Has key for ${props.i18nKey}.${item.itemKey}`, () => {
          if (props.i18nKey.split(`/`)[1] === `heading`) {
            expect(en.headings[props.i18nKey as keyof typeof en.headings][item.itemKey]).toBeDefined();
          } else {
            expect(en.info[props.i18nKey as keyof typeof en.info][item.itemKey]).toBeDefined();
          }
        });
      }
    }
  });

  describe(`All flow nodes of any type with i18n key have valid corresponding translations`, () => {
    const contentNodesWithKey = findFlowNodesWithProp(flowNodes, `i18nKey`);
    for (const node of contentNodesWithKey) {
      if (node.props.i18nKey) {
        let translationKey = node.props.i18nKey as keyof typeof en;

        // TODO: standardize how we pass in i18n keys for Hints so we don't need this
        if (node.type.name === `Hint`) {
          translationKey = `/info${translationKey}` as keyof typeof en;
        }
        // possible key types and mappings:
        // /info/a --> en.info[/info/a]
        // /heading/a/b/c --> en.heading[/headings/a/b/c]
        // subheadings./subheadings/a --> en.subheadings[/subheadings/a]
        // button.abc --> en.button[abc]
        // about-you-ip-pin-input (mefErrorCode: IND-181-01) --> mefAlerts.IND-181-01.about-you-ip-pin-choice:xyz
        // /checkbox/complete/sign-and-submit/outro --> should have fields defined: label, errorMessages.RequiredField
        // /modal/a/b --> en.modals[/modal/a/b]
        // some additional hidden keys from error states:
        // Enum: enums.errorMessages.IncompleteData
        // TODO: Check each field component for error message states and other hidden keys

        let nameSpace: string;
        let match: RegExpMatchArray | null;

        if (translationKey.startsWith(`/info/`)) {
          nameSpace = `info`;
        } else if (translationKey.startsWith(`/heading/`)) {
          nameSpace = `headings`;
        } else if (translationKey.startsWith(`/iconList/`)) {
          nameSpace = `iconLists`;
        } else if (translationKey.startsWith(`/checkbox/`)) {
          nameSpace = `checkbox`;
        } else if (translationKey.startsWith(`/modal/`)) {
          nameSpace = `modals`;
        } else if (translationKey.match(REGEX_FOR_KEY_WITH_DOTSLASH)) {
          match = translationKey.match(REGEX_FOR_KEY_WITH_DOTSLASH);
        } else if (translationKey.match(REGEX_FOR_KEY_WITH_SLASH)) {
          match = translationKey.match(REGEX_FOR_KEY_WITH_SLASH);
        } else if (translationKey.match(REGEX_FOR_KEY_WITH_LEADING_SLASH)) {
          match = translationKey.match(REGEX_FOR_KEY_WITH_LEADING_SLASH);
        } else if (translationKey.match(REGEX_FOR_KEY_WITH_DOT)) {
          match = translationKey.match(REGEX_FOR_KEY_WITH_DOT);
        }

        it(`Has key for ${translationKey}`, () => {
          if (nameSpace) {
            if (nameSpace === `checkbox`) {
              expect(en.fields[translationKey].label).toBeDefined();
              expect(en.fields[translationKey].errorMessages.RequiredField).toBeDefined();
            } else {
              expect(en[nameSpace][translationKey]).toBeDefined();
            }
          } else if (match) {
            const prefix = match[1];
            const suffix = match[2];
            if (prefix === `dataviews`) {
              // to split into another test
              traversePath(translationKey, en);
            } else if (prefix === `fields`) {
              traversePath(translationKey, en);
            } else if (prefix === `datapreviews`) {
              traversePath(translationKey, en);
            } else {
              expect(en[prefix][suffix]).toBeDefined();
            }
          } else {
            if (node.props.mefErrorCode) {
              expect(en.mefAlerts[node.props.mefErrorCode][translationKey].alertText.body).toBeDefined();
            } else {
              expect(en[translationKey]).toBeDefined();
            }
          }
        });
      }
    }
  });
});

const traversePath = (fullPath: string, yaml: typeof en) => {
  const path = fullPath.split(`.`);
  let last = yaml;
  if (path.length) {
    for (const [i, key] of path.entries()) {
      if (i === path.length - 1) {
        expect(last[key], `${fullPath} is undefined`).toBeDefined();
      }
      last = last[key];
    }
  }
};
