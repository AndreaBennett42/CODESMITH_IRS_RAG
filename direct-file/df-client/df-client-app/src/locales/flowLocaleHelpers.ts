import type {
  AssertionNode,
  CategoryNode,
  CollectionLoopNode,
  SubSubcategoryNode,
  SubcategoryNode,
} from '../flow/flowDeclarations.js';
import {
  ConditionalListConfig,
  FlowComponentConfig,
  ScreenButtonConfig,
  ScreenContentNode,
  FactConfig,
  contentConfigIsAggregatedAlertConfig,
  contentConfigIsCollectionDataViewInternalLinkConfig,
  contentConfigIsCollectionItemManagerConfig,
  contentConfigIsConditionalListConfig,
  contentConfigIsDFAlertConfig,
  contentConfigIsFactConfig,
  contentConfigIsIconDisplayConfig,
  contentConfigIsInfoDisplayConfig,
  contentConfigIsInternalLinkConfig,
  contentConfigIsScreenButtonConfig,
  InfoDisplayConfig,
  IconDisplayConfig,
  CollectionDataViewInternalLinkConfig,
  InternalLinkConfig,
  DFAlertConfig,
  CollectionItemManagerConfig,
  contentConfigIsDataPreviewConfig,
  contentConfigIsCollectionDataPreviewConfig,
} from '../flow/ContentDeclarations.js';
import { CommonTranslation } from 'df-i18n';
import { assertNever } from 'assert-never';
import { wrappedFacts } from '../fact-dictionary/generated/wrappedFacts.js';
import { Path } from '../fact-dictionary/Path.js';

// We use this reference object to ensure we check all the relevant keys
// Importing the blank object from `BankAccount.tsx` causes import errors because of `scss` modules in the file
import type { BankAccount } from '@irs/js-factgraph-scala';
import { buildScreenContentFromConfigNode } from '../flow/ScreenConfig.js';
const BLANK_BANK_ACCOUNT: BankAccount = {
  accountNumber: ``,
  accountType: `Checking`,
  routingNumber: ``,
};

const EXCLUDED_OPTION_PATHS: Path[] = [
  `/filingStatusInitialOptions`,
  `/k12EducatorOptions`,
  `/paidEstimatedTaxesWithFormerSpouseOptions`,
  `/signReturnIdentityOptions`,
  `/signReturnIdentitySpouseOptions`,
];
export interface AssociatedKeys {
  requiredKeys: string[];
  optionalKeys: string[];
}

export function getExpectedModalKeys(config: FlowComponentConfig<`DFModal`>): AssociatedKeys {
  const requiredKeys: string[] = [];
  const optionalKeys: string[] = [];

  if (Array.isArray(config.props.i18nKey)) throw new Error(`Unexpected array in DFModal props`);

  const baseKey = CommonTranslation.getNamespacedKey(config.props.i18nKey);

  requiredKeys.push(`${baseKey}.helpText.modals`);

  requiredKeys.push();

  if (config.props.inlineButtonI18nKey)
    requiredKeys.push(CommonTranslation.getNamespacedKey(config.props.inlineButtonI18nKey));
  if (config.props.inlinePDFButtonI18nKey)
    requiredKeys.push(CommonTranslation.getNamespacedKey(config.props.inlinePDFButtonI18nKey));

  return {
    requiredKeys,
    optionalKeys,
  };
}

export function getExpectedConditionalAccordionKeys(
  conditionalConfig: FlowComponentConfig<`ConditionalAccordion`>
): AssociatedKeys {
  const baseKey = CommonTranslation.getNamespacedKey(conditionalConfig.props.i18nKey);
  const requiredKeys = conditionalConfig.props.items.map(({ itemKey }) => `${baseKey}.${itemKey}`);

  requiredKeys.push(`${baseKey}.heading`);

  return {
    requiredKeys,
    optionalKeys: [`${baseKey}.subList`],
  };
}

export function getExpectedConditionalListKeys(conditionalConfig: ConditionalListConfig): AssociatedKeys {
  const baseKey = CommonTranslation.getNamespacedKey(conditionalConfig.props.i18nKey);
  const additionalNamespace = conditionalConfig.componentName === `SummaryTable` ? `.sections` : ``;

  const requiredKeys = conditionalConfig.props.items.map(
    ({ itemKey }) => `${baseKey}${additionalNamespace}.${itemKey}`
  );
  const optionalKeys: string[] = [];

  if (conditionalConfig.props.i18nPrefixKey) {
    requiredKeys.push(CommonTranslation.getNamespacedKey(conditionalConfig.props.i18nPrefixKey));
  }

  if (conditionalConfig.componentName === `SummaryTable`) {
    requiredKeys.push(`${baseKey}${additionalNamespace}.caption`);
    optionalKeys.push(`${baseKey}${additionalNamespace}.conclusion`, `${baseKey}${additionalNamespace}.explainer`);
  }

  return {
    requiredKeys,
    optionalKeys,
  };
}

export function getExpectedScreenButtonKeys(config: ScreenButtonConfig): AssociatedKeys {
  if (config.componentName === `ExitButton`) {
    // These buttons don't use dynamic keys
    if (config.props.i18nKey) {
      throw new Error(`${config.componentName} does not reference i18nKey but one was provided anyways`);
    }

    if (config.componentName === `ExitButton`) {
      return {
        requiredKeys: [`button.dashboard`],
        optionalKeys: [],
      };
    } else {
      assertNever(config);
    }
  } else if (config.props.i18nKey) {
    return {
      requiredKeys: [config.props.i18nKey],
      optionalKeys: [],
    };
  } else if (config.componentName === `SubmitButton`) {
    return {
      requiredKeys: [`button.submit`, `button.submitSigned`],
      optionalKeys: [],
    };
  } else if (config.componentName === `SaveAndOrContinueButton`) {
    // SaveAndContinue sometimes renders an alert instead of the button
    const alertKeys = getExpectedAlertKeys({
      componentName: `DFAlert`,
      props: {
        type: `error`,
        i18nKey: `/info/complete/review/review-and-confirm/info-alert`,
        headingLevel: `h1`,
        childConfigs: [],
      },
    });
    return {
      requiredKeys: [`button.go-to-checklist`, ...alertKeys.requiredKeys],
      optionalKeys: [...alertKeys.optionalKeys],
    };
  } else if (config.componentName === `StateTaxesButton`) {
    return {
      requiredKeys: [`stateInfoAlert.otherBenefit.linkText`],
      optionalKeys: [],
    };
  } else {
    throw new Error(`${config.componentName} had no i18nKey provided`);
  }
}

export function getEnumOptions(enumOptionsPath: Path): string[] {
  const optionsFact = wrappedFacts.find(
    (fact) => fact.path === enumOptionsPath && fact.derived?.typeName === `EnumOptions`
  );

  const values = optionsFact?.derived?.children.map((child, index) => {
    const unconditionalValue = child.options.value;

    if (unconditionalValue) return unconditionalValue;
    else {
      const valueChild = child.children.find((subchild) => subchild.typeName === `Value`);
      const valueChildActualValue = valueChild?.children.find((subChild) => subChild.typeName === `String`);

      const conditionalValue = valueChildActualValue?.options.value;

      if (conditionalValue) return conditionalValue;
      else throw new Error(`Failed to get value for ${enumOptionsPath}[${index}]`);
    }
  });

  if (values && values.length > 0) {
    return values;
  } else {
    throw new Error(`Could not find options for ${enumOptionsPath}`);
  }
}

export function getEnumOptionsPath(enumFactPath: Path): Path {
  const enumFact = wrappedFacts.find((fact) => fact.path === enumFactPath && fact.writable?.typeName === `Enum`);

  if (!enumFact) throw new Error(`Could not find enum with path: ${enumFactPath}`);

  const enumOptionsPath = enumFact?.writable?.options.optionsPath;

  if (enumOptionsPath) return enumOptionsPath as Path;
  else throw new Error(`Could not find options for enum: ${enumFactPath}`);
}

export function getExpectedFactControlKeys(
  config: FactConfig,
  parentRoute: string,
  subSubCategory?: SubSubcategoryNode
) {
  const requiredKeys: string[] = [];
  const optionalKeys: string[] = [];
  const { path, displayOnlyOn, hintKey } = config.props;
  if (displayOnlyOn !== `data-view`) {
    if (config.componentName === `Boolean`) {
      if (config.props.i18nKeySuffixContext) {
        requiredKeys.push(`fields.${path}.${config.props.i18nKeySuffixContext}.boolean`);
      } else {
        optionalKeys.push(`fields.${path}.boolean`);
      }
    } else if (config.componentName === `Enum`) {
      const enumOptionsPath = getEnumOptionsPath(config.props.path);
      if (config.props.i18nKeySuffixContext) {
        // If a suffix context is provided, then the path-specific content *must* exists
        requiredKeys.push(
          ...getEnumOptions(enumOptionsPath).map(
            (optionValue) => `fields.${path}.${enumOptionsPath}.${config.props.i18nKeySuffixContext}.${optionValue}`
          )
        );
      } else {
        // Enums usually don't have labels and fallback to the relevant `field./*Options` copy
        // They *may* have unique copy
        const enumOptions = getEnumOptions(enumOptionsPath);

        optionalKeys.push(`fields.${path}.name`);

        if (EXCLUDED_OPTION_PATHS.includes(enumOptionsPath)) {
          // These keys intentionally do not have general copy and must provide their own unique copy
          requiredKeys.push(...enumOptions.map((optionValue) => `fields.${path}.${enumOptionsPath}.${optionValue}`));
        } else {
          requiredKeys.push(...enumOptions.map((optionValue) => `fields.${enumOptionsPath}.${optionValue}`));
          optionalKeys.push(...enumOptions.map((optionValue) => `fields.${path}.${enumOptionsPath}.${optionValue}`));
        }
      }
    } else {
      requiredKeys.push(`fields.${path}`);
      optionalKeys.push(`info./info${path}`, `info.${path}`);
    }

    if (config.componentName === `BankAccount`) {
      // BankAccount has additional copy for the account type options
      requiredKeys.push(`fields.${path}/accountType`);
    }
  }

  if (displayOnlyOn !== `edit` && subSubCategory?.props.hidden !== true) {
    const resolvedPaths =
      config.componentName === `BankAccount`
        ? Object.keys(BLANK_BANK_ACCOUNT).map((field) => `${path}/${field}`)
        : [path];

    for (const resolvedPath of resolvedPaths) {
      const baseKey = `dataviews.${parentRoute}.${resolvedPath}`;
      if (subSubCategory) {
        requiredKeys.push(baseKey);
      } else {
        // This is a temporary workaround until exportLocales is able to reliably pass in the subSubCategory
        optionalKeys.push(baseKey);
      }
      optionalKeys.push(`${baseKey}_spouse`, `${baseKey}_has1099g`);
    }
  }

  if (config.componentName === `Address`) {
    // Address includes a fixed modal
    const modalKeys = getExpectedModalKeys({
      componentName: `DFModal`,
      props: {
        i18nKey: `/info/why-cant-i-change-country`,
      },
    });

    requiredKeys.push(...modalKeys.requiredKeys);
    optionalKeys.push(...modalKeys.optionalKeys);
  }
  if (hintKey) {
    requiredKeys.push(CommonTranslation.getNamespacedKey(hintKey));
  }

  return { requiredKeys, optionalKeys };
}

export function getExpectedInfoDisplayKeys(config: InfoDisplayConfig): AssociatedKeys {
  if (Array.isArray(config.props.i18nKey)) {
    throw new Error(`${config.componentName} config has multiple key: ${config.props.i18nKey.join(`, `)}`);
  } else if (config.props.i18nKey !== undefined) {
    if (config.componentName === `ConditionalAccordion`) {
      return getExpectedConditionalAccordionKeys(config);
    } else if (config.componentName === `Hint`) {
      return {
        requiredKeys: [`info./info${config.props.i18nKey}`],
        optionalKeys: [],
      };
    } else {
      const requiredKeys: string[] = [];

      requiredKeys.push(CommonTranslation.getNamespacedKey(config.props.i18nKey));

      if (config.componentName === `StateInfoCard` && config.props.stateLinki18nKey) {
        requiredKeys.push(CommonTranslation.getNamespacedKey(config.props.stateLinki18nKey));
      }

      return { requiredKeys, optionalKeys: [] };
    }
  } else if (config.componentName === `DFModal`) {
    return getExpectedModalKeys(config);
  } else if (config.componentName === `SectionReview`) {
    return {
      requiredKeys: [`dataviews.review`, `checklist`],
      optionalKeys: [],
    };
  } else if (config.componentName === `StateTaxReminderAlertWrapper`) {
    return {
      requiredKeys: [`enums.statesAndProvinces`, `stateInfoAlert`],
      optionalKeys: [],
    };
  } else {
    throw new Error(`${config.componentName} has no i18nKey`);
  }
}

export function getExpectedInternalLinkKeys(
  config: CollectionDataViewInternalLinkConfig | InternalLinkConfig
): AssociatedKeys {
  return {
    requiredKeys: [`${CommonTranslation.getNamespacedKey(config.props.i18nKey)}.internalLink`],
    optionalKeys: [],
  };
}

export function getExpectedIconDisplayKeys(config: IconDisplayConfig): AssociatedKeys {
  const requiredKeys: string[] = [];
  if (config.props.i18nKey) {
    requiredKeys.push(config.props.i18nKey);
  }

  return { requiredKeys, optionalKeys: [] };
}

export function getExpectedAlertKeys(config: DFAlertConfig | FlowComponentConfig<`TaxReturnAlert`>): AssociatedKeys {
  let requiredKeys: string[] = [];
  let optionalKeys: string[] = [];

  if (config.props.i18nKey) requiredKeys.push(`${CommonTranslation.getNamespacedKey(config.props.i18nKey)}`);

  if (`childConfigs` in config.props) {
    for (const childConfig of config.props.childConfigs) {
      let childKeys;
      if (contentConfigIsInfoDisplayConfig(childConfig)) {
        childKeys = getExpectedInfoDisplayKeys(childConfig);
      } else if (contentConfigIsConditionalListConfig(childConfig)) {
        childKeys = getExpectedConditionalListKeys(childConfig);
      } else {
        assertNever(childConfig);
      }
      requiredKeys = [...requiredKeys, ...childKeys.requiredKeys];
      optionalKeys = [...optionalKeys, ...childKeys.optionalKeys];
    }
  }

  if (requiredKeys.length === 0) {
    throw new Error(`${config.componentName} has no i18nKey or childConfigs, ${JSON.stringify(config)}`);
  }

  return {
    requiredKeys,
    optionalKeys,
  };
}

export function getExpectedMefAlertKeys(config: FlowComponentConfig<`MefAlert`>): AssociatedKeys {
  const baseKey = `mefAlerts.${config.props.mefErrorCode}.${config.props.i18nKey}`;
  return {
    requiredKeys: [`${baseKey}.alertText.body`],
    optionalKeys: [`${baseKey}.internalLink`],
  };
}

export function getExpectedCollectionItemManagerKeys(config: CollectionItemManagerConfig): AssociatedKeys {
  const baseKey = `fields.${config.props.path}`;

  const controlKeys = [`add`, `complete`].map((subKey) => `${baseKey}.controls.${subKey}`);
  const requiredCollectionListingKeys = [`itemHeading1`, `itemHeading2`, `label2`].map(
    (subKey) => `${baseKey}.collectionListing.${subKey}`
  );
  const optionalCollectionListingKeys = [`item1`, `value1`, `value2`].map(
    (subKey) => `${baseKey}.collectionListing.${subKey}`
  );

  const deleteConfirmationModalKeys = getExpectedConfirmationModalKeys(`${baseKey}.deleteControl`);

  return {
    requiredKeys: [...controlKeys, ...requiredCollectionListingKeys, ...deleteConfirmationModalKeys.requiredKeys],
    optionalKeys: [
      `${baseKey}.controls.back`,
      ...deleteConfirmationModalKeys.optionalKeys,
      ...optionalCollectionListingKeys,
    ],
  };
}

export function getExpectedScreenContentKeys(
  node: ScreenContentNode,
  parentRoute: string,
  subSubCategory?: SubSubcategoryNode
): AssociatedKeys {
  if (node.type.name === `SetFactAction`) {
    // Fact actions are not rendered on screen
    return { requiredKeys: [], optionalKeys: [] };
  }

  const config = buildScreenContentFromConfigNode(node);

  if (config === undefined) throw new Error(`Found invalid config of type ${node.type.name}`);

  if (contentConfigIsFactConfig(config)) return getExpectedFactControlKeys(config, parentRoute, subSubCategory);
  else if (contentConfigIsInfoDisplayConfig(config)) return getExpectedInfoDisplayKeys(config);
  else if (contentConfigIsConditionalListConfig(config)) return getExpectedConditionalListKeys(config);
  else if (contentConfigIsIconDisplayConfig(config)) return getExpectedIconDisplayKeys(config);
  else if (contentConfigIsCollectionDataViewInternalLinkConfig(config) || contentConfigIsInternalLinkConfig(config)) {
    return getExpectedInternalLinkKeys(config);
  } else if (contentConfigIsDFAlertConfig(config)) return getExpectedAlertKeys(config);
  else if (contentConfigIsAggregatedAlertConfig(config)) {
    if (config.componentName === `MefAlert`) {
      return getExpectedMefAlertKeys(config);
    } else if (config.componentName === `TaxReturnAlert`) {
      return getExpectedAlertKeys(config);
    } else {
      assertNever(config);
    }
  } else if (contentConfigIsScreenButtonConfig(config)) return getExpectedScreenButtonKeys(config);
  else if (contentConfigIsCollectionItemManagerConfig(config)) return getExpectedCollectionItemManagerKeys(config);
  else if (contentConfigIsDataPreviewConfig(config) || contentConfigIsCollectionDataPreviewConfig(config))
    return { requiredKeys: [], optionalKeys: [] };
  else {
    assertNever(config);
  }
}

export function getExpectedCategoryContentKeys(node: CategoryNode): AssociatedKeys {
  return {
    requiredKeys: [`checklist./flow/${node.props.route}.heading`],
    optionalKeys: [],
  };
}

export function getExpectedSubCategoryContentKeys(node: SubcategoryNode, parentRoute: string): AssociatedKeys {
  const route = `${parentRoute}/${node.props.route}`;
  return {
    requiredKeys: [`checklist.${route}.heading`],
    optionalKeys: [`dataviews.${route}`],
  };
}

export function getExpectedSubSubCategoryContentKeys(node: SubSubcategoryNode, parentRoute: string): AssociatedKeys {
  return {
    requiredKeys: node.props.hidden ? [] : [`subsubcategories.${parentRoute}.${node.props.route}`],
    optionalKeys: [],
  };
}

export function getExpectedCollectionLoopContentKeys(node: CollectionLoopNode): AssociatedKeys {
  const { dataViewSections } = node.props;

  const requiredKeys: string[] = [];
  const optionalKeys: string[] = [];

  if (dataViewSections) {
    for (const section of dataViewSections) {
      requiredKeys.push(CommonTranslation.getNamespacedKey(section.i18nKey));
      if (section.i18nModalKey) {
        requiredKeys.push(CommonTranslation.getNamespacedKey(section.i18nModalKey));
      }
    }
  }

  return {
    requiredKeys,
    optionalKeys,
  };
}

export function getExpectedAssertionKeys(node: AssertionNode): AssociatedKeys {
  return {
    requiredKeys: [node.props.i18nKey],
    optionalKeys: [],
  };
}

export function getExpectedConfirmationModalKeys(baseKey: string): AssociatedKeys {
  const modalKeys = [`header`, `controls.confirm`, `controls.cancel`].map(
    (subKey) => `${baseKey}.LinkModal1.${subKey}`
  );
  return {
    requiredKeys: [`${baseKey}.text`, ...modalKeys],
    optionalKeys: [],
  };
}
