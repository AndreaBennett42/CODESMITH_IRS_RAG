import { ConcretePath } from '@irs/js-factgraph-scala';
import { useFactGraph } from '../../factgraph/FactGraphContext.js';
import { useEffectOnce } from '../../hooks/useEffectOnce.js';
import { useNavigate } from 'react-router-dom';
import { contentConfigIsFactConfig, DataPreviewDeclaration } from '../../flow/ContentDeclarations.js';
import { FlowSubSubcategory, useFlow } from '../../flow/flowConfig.js';
import { SubSubCategory } from '../../screens/data-view/SubSubCategory.js';
import { MutableRefObject, useMemo, useRef } from 'react';
import { findLast } from '../../utils/polyfills.js';
import { useAppSelector } from '../../redux/hooks.js';
import { saveImportedFacts } from '../../redux/slices/data-import/saveImportedFacts.js';

type DataPreviewProps = {
  collectionId: string | null;
  currentRoute: string;
};

const DataPreview = ({
  wasSavedFactPath,
  collectionId = null,
  currentRoute,
  subsubcategories,
}: DataPreviewDeclaration & DataPreviewProps) => {
  const navigate = useNavigate();
  const { factGraph } = useFactGraph();
  const flow = useFlow();
  const dataWasSavedResult = factGraph.get(wasSavedFactPath as ConcretePath);
  const dataWasSaved = dataWasSavedResult.complete ? dataWasSavedResult.get : false;
  useEffectOnce(() => {
    if (dataWasSaved) {
      // If the data was saved, redirect user to the last screen of the data preview.
      const path = lastScreen?.fullRoute(collectionId) || -1;
      navigate(path as string, {
        replace: true,
      });
    }
  }, []);
  const dataImportSlice = useAppSelector((state) => state.dataImportProfile);

  const baseRoute = currentRoute.substr(0, currentRoute.lastIndexOf(`/`) + 1);
  const sscs = subsubcategories
    .map((route) => flow.subsubcategoriesByRoute.get(`${baseRoute}${route}`))
    .filter((ssc) => ssc) as FlowSubSubcategory[];

  // Find the last non-null screen in the last non-null subsubcategory
  // I'm not sure if these should be nullable, but refactors shouldn't change the logic
  let lastScreenOfDataPreview;
  if (sscs) {
    // The !! is the idiomatic way to convert a value to its falsy or truthy equivalent
    // This is done to satisfy the type checker and not change the existing logic
    const screens = findLast(sscs, (ssc) => !!ssc)?.screens;
    if (screens) {
      lastScreenOfDataPreview = findLast(screens, (s) => !!s);
    }
  }

  const lastScreen = flow.screensByRoute.get(lastScreenOfDataPreview?.screenRoute || ``);

  // TODO: We can remove all this once this occurs upon initialization of the fact graph.
  // https://git.irslabs.org/irslabs-prototypes/direct-file/-/issues/12293
  const screens = sscs?.flatMap((ssc) => ssc?.screens);
  useMemo(() => {
    if (dataWasSaved) {
      return;
    }
    if (dataImportSlice.data.status !== `complete`) {
      // TODO - actual error handling when designs come in.
      return;
    }
    const importedData = dataImportSlice.data.profile;
    screens?.forEach(async (s) => {
      const importedFacts = s?.content
        .filter(contentConfigIsFactConfig)
        .filter((child) => child.props.path && child.props.importedPath);

      if (importedData) saveImportedFacts(importedData, factGraph, importedFacts, collectionId);
    });
  }, [collectionId, factGraph, dataImportSlice, screens, dataWasSaved]);

  const emptyAlertConfig = { warnings: [], errors: [] };
  const subSubCategoryRefs = useRef(new Map<string, MutableRefObject<HTMLHeadingElement>>());

  return sscs.map((ssc: FlowSubSubcategory) => (
    <SubSubCategory
      ssc={ssc}
      collectionId={collectionId}
      alertConfigs={emptyAlertConfig}
      mefAlertConfigs={emptyAlertConfig}
      refs={subSubCategoryRefs}
      key={`${ssc.fullRoute}-${collectionId}`}
      isAfterNextIncompleteScreen={false}
      nextIncompleteScreen={undefined}
      sectionIsComplete={false}
      includesNextIncompleteScreen={false}
    />
  ));
};

export default DataPreview;
