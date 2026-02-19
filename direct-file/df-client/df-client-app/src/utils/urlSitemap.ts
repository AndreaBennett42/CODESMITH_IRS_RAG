import { TFunction } from 'i18next';
import { CollectionItemManagerConfig } from '../flow/ContentDeclarations.js';
import { ScreenConfig } from '../flow/ScreenConfig.js';
import { FlowConfig } from '../flow/flowConfig.js';
import { redactPII } from '../hooks/useTranslatePIIRedacted.js';
import { getCollectionDataviewTitleKey } from '../screens/data-view/CollectionItemDataView.js';
import { findComponentsOfType, getStringKey, getTitleKey } from './i18nUtils.js';

export interface UrlEntry {
  site: string;
  type: string;
  url: string;
  categoryRoute: string;
  categoryTitle: string;
  subcategoryRoute: string;
  subcategoryTitle: string;
  subcategorySequence?: number;
  uniqueTitle?: string;
  pageTitleKey?: string;
  pageTitle?: string;
}

const BASE_ROUTER_PATH = import.meta.env.VITE_PUBLIC_PATH || ``;

export const getDataViews = (flow: FlowConfig, t: TFunction): UrlEntry[] => {
  const dataviews: UrlEntry[] = [];
  flow.categories.forEach((cat) => {
    cat.subcategories.forEach((subcat) => {
      if (subcat.hasDataView) {
        const category = t(`checklist.${cat.route}.heading`);
        const subcategory = t(`checklist.${subcat.route}.heading`);
        const defaults: UrlEntry = {
          site: `df-client`,
          url: ``,
          categoryRoute: cat.route,
          categoryTitle: category,
          subcategoryRoute: subcat.route,
          subcategoryTitle: subcategory,
          type: `dataview`,
        };
        // We only want to seelct the dataview overrides, all others should show up in the flow screens
        const dataViewOverrideScreens = subcat.screens.filter((s) => s.actAsDataView);
        dataViewOverrideScreens.forEach((override: ScreenConfig, index: number) => {
          const pageTitleKey = getStringKey(getTitleKey(override.content), t);
          const pageTitle = pageTitleKey ? redactPII(t, pageTitleKey, true) : undefined;
          const url = BASE_ROUTER_PATH + override.fullRoute(``);
          const uniqueTitle = `${subcategory} Dataview ${index}`;
          const entry = { ...defaults, url, uniqueTitle, pageTitleKey, pageTitle };
          dataviews.push(entry);
        });
      }
    });
  });
  return dataviews;
};

// Get flow screens and collection loops
const getFlowScreens = (flow: FlowConfig, t: TFunction): UrlEntry[] => {
  const screens: UrlEntry[] = [];

  flow.categories.forEach((cat) => {
    cat.subcategories.forEach((subcat) => {
      let subcatSeq = 0;
      subcat.screens.forEach((screen) => {
        subcatSeq += 1;
        const url = `${BASE_ROUTER_PATH}${subcat.route}/${screen.route}`;
        const category = t(`checklist.${cat.route}.heading`);
        const subcategory = t(`checklist.${subcat.route}.heading`);
        const pageTitleKey = getStringKey(getTitleKey(screen.content), t);
        const pageTitle = pageTitleKey ? redactPII(t, pageTitleKey, true) : undefined;
        const entry = {
          site: `df-client`,
          url,
          pageTitleKey,
          pageTitle,
          categoryRoute: cat.route,
          categoryTitle: category,
          subcategoryRoute: subcat.route,
          subcategoryTitle: subcategory,
          subcategorySequence: subcatSeq,
          uniqueTitle: `${category} - ${subcatSeq}`,
          type: `flow`,
        };
        screens.push(entry);

        // Search for a CollectionLoop to include the loop views
        const results = findComponentsOfType(`CollectionItemManager`, screen.content);
        results.forEach((collection) => {
          const loopName = (collection as CollectionItemManagerConfig).props.loopName;
          const url = `${BASE_ROUTER_PATH}/data-view/loop/${encodeURIComponent(loopName)}`;
          const pageTitleKey = getStringKey(getCollectionDataviewTitleKey(loopName), t);
          const pageTitle = pageTitleKey ? redactPII(t, pageTitleKey, true) : undefined;
          const loopEntry = {
            ...entry,
            url,
            pageTitleKey,
            pageTitle,
            uniqueTitle: `${subcategory} loop ${loopName}`,
            type: `collection`,
          };
          screens.push(loopEntry);
        });
      });
    });
  });
  return screens;
};

function getUrlSitemap(flow: FlowConfig, t: TFunction) {
  let urls = getFlowScreens(flow, t);
  urls = urls.concat(getDataViews(flow, t));
  return urls;
}

export default getUrlSitemap;
