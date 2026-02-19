import { assertNever } from 'assert-never';
import flowNodes from '../flow/flow.js';
import {
  FlowChild,
  isScreen,
  isAssertion,
  isCollectionLoop,
  isGate,
  isCategory,
  isSubcategory,
  isSubSubcategory,
  CategoryNode,
  SubcategoryNode,
  SubSubcategoryNode,
} from '../flow/flowDeclarations.js';
import {
  getExpectedAssertionKeys,
  getExpectedCategoryContentKeys,
  getExpectedCollectionLoopContentKeys,
  getExpectedScreenContentKeys,
  getExpectedSubCategoryContentKeys,
  getExpectedSubSubCategoryContentKeys,
} from './flowLocaleHelpers.js';

// Because of the parity tests, we only need to check the English yaml file
import rawEnLocale from '../locales/en.yaml';
import Locale from '../scripts/locale.js';

const enLocale = new Locale(rawEnLocale);

function wrapIfNotArray<T>(value: T | T[]): T[] {
  if (Array.isArray(value)) return value;
  else return [value];
}

function validateFlowKeys(
  node: FlowChild | FlowChild[],
  category?: CategoryNode,
  subCategory?: SubcategoryNode,
  subSubCategory?: SubSubcategoryNode
): void {
  // TODO should we generalize a "walk flow" variant of this?
  if (Array.isArray(node)) node.forEach((childNode) => validateFlowKeys(childNode, category, subCategory));
  else {
    if (isScreen(node)) {
      describe(`in Screen: ${node.props.route}`, () => {
        const screenContentNodes = wrapIfNotArray(node.props.children);
        it(`required keys exist for rendered nodes`, () => {
          if (!category) throw new Error(`Missing category for Screen: ${node.props.route}`);
          if (!subCategory) throw new Error(`Missing subCategory for Screen: ${node.props.route}`);

          const parentRoute = `/flow/${category.props.route}/${subCategory.props.route}`;

          for (const contentNode of screenContentNodes) {
            const { requiredKeys } = getExpectedScreenContentKeys(contentNode, parentRoute, subSubCategory);
            const missingKeys = requiredKeys.filter((i18nKey) => !enLocale.has(i18nKey));

            if (missingKeys.length > 0) {
              const nodeType = typeof contentNode.type === `string` ? contentNode.type : contentNode.type.name;
              // Throw error so we can provide extra context
              throw new Error(
                `Content node of type ${nodeType} is missing required keys:\n\t${missingKeys.join(`\n\t`)}`
              );
            }
          }
        });
      });
    } else if (isAssertion(node)) {
      it(`required keys exist for Assertion: ${node.props.i18nKey}`, () => {
        const { requiredKeys } = getExpectedAssertionKeys(node);
        const missingKeys = requiredKeys.filter((i18nKey) => !enLocale.has(i18nKey));

        expect(missingKeys).toEqual([]);
      });
    } else if (isCollectionLoop(node)) {
      describe(`in CollectionLoop: ${node.props.loopName}`, () => {
        it(`dataview keys exist: ${node.props.loopName}`, () => {
          const { requiredKeys } = getExpectedCollectionLoopContentKeys(node);
          const missingKeys = requiredKeys.filter((i18nKey) => !enLocale.has(i18nKey));

          expect(missingKeys).toEqual([]);
        });

        validateFlowKeys(node.props.children, category, subCategory);
      });
    } else if (isGate(node)) {
      // No gate specific copy, so just move onto the children
      validateFlowKeys(node.props.children, category, subCategory, subSubCategory);
    } else if (isCategory(node)) {
      describe(`in Category: ${node.props.route}`, () => {
        it(`Non-screen content exists`, () => {
          const { requiredKeys } = getExpectedCategoryContentKeys(node);
          const missingKeys = requiredKeys.filter((i18nKey) => !enLocale.has(i18nKey));

          expect(missingKeys).toEqual([]);
        });
        validateFlowKeys(node.props.children, node);
      });
    } else if (isSubcategory(node)) {
      describe(`in SubCategory: ${node.props.route}`, () => {
        it(`Non-screen content exists`, () => {
          const parentRoute = `/flow/${category?.props.route}`;
          const { requiredKeys } = getExpectedSubCategoryContentKeys(node, parentRoute);
          const missingKeys = requiredKeys.filter((i18nKey) => !enLocale.has(i18nKey));

          expect(missingKeys).toEqual([]);
        });
        validateFlowKeys(node.props.children, category, node);
      });
    } else if (isSubSubcategory(node)) {
      describe(`in SubSubCategory: ${node.props.route}`, () => {
        it(`Non-screen content exists`, () => {
          const parentRoute = `/flow/${category?.props.route}/${subCategory?.props.route}`;
          const { requiredKeys } = getExpectedSubSubCategoryContentKeys(node, parentRoute);
          const missingKeys = requiredKeys.filter((i18nKey) => !enLocale.has(i18nKey));

          expect(missingKeys).toEqual([]);
        });

        validateFlowKeys(node.props.children, category, subCategory, node);
      });
    } else {
      assertNever(node);
    }
  }
}

describe(`Flow content`, () => {
  validateFlowKeys(flowNodes.props.children);
});
