import { ProcessListHeading, ProcessListItem, ProcessList } from '@trussworks/react-uswds';
import useTranslateWithFacts from '../../../hooks/useTranslateWithFacts.js';
import ChecklistSubcategory from '../ChecklistSubcategory/index.js';
import { useFactGraph } from '../../../factgraph/FactGraphContext.js';
import styles from './ChecklistCategory.module.scss';
import { ChecklistSubcategoryProps } from '../ChecklistSubcategory/ChecklistSubcategory.js';
import { FC, MutableRefObject, RefObject } from 'react';
import { handleRefFromRoute } from '../../SummaryAlert/summaryHelpers.js';

interface ChecklistCategoryProps {
  route: string;
  categoryActive: boolean;
  subcategories: ChecklistSubcategoryProps[];
  categoryRefs: MutableRefObject<Map<string, MutableRefObject<HTMLAnchorElement>>>;
}

const ChecklistCategory: FC<ChecklistCategoryProps> = ({ route, subcategories, categoryActive, categoryRefs }) => {
  const { factGraph } = useFactGraph();
  const { t } = useTranslateWithFacts(factGraph);

  return (
    <>
      <ProcessListItem className={styles.checklistCategory}>
        <ProcessListHeading type='h2'>{t(`checklist.${route}.heading`)}</ProcessListHeading>
        {categoryActive && (
          <ProcessList className={styles.subcategoryList}>
            {subcategories.map((sub) => (
              <ChecklistSubcategory
                ref={handleRefFromRoute(sub.subcategoryRoute, categoryRefs) as RefObject<HTMLAnchorElement>}
                key={sub.subcategoryRoute}
                {...sub}
              />
            ))}
          </ProcessList>
        )}
      </ProcessListItem>
    </>
  );
};
export default ChecklistCategory;
