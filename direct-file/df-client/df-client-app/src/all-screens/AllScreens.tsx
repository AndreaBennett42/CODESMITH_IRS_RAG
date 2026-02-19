import { useState, useMemo } from 'react';
import styles from './AllScreens.module.scss';
import AllScreensHeader from './AllScreensHeader.js';
import AllScreensContent, { AllScreensFilterSet, ShowType } from './AllScreensContent.js';
import AllScreensMenu from './AllScreensMenu.js';
import { AllScreensContext } from './AllScreensContext.js';

function AllScreens() {
  const [showType, setShowType] = useState<ShowType>(`screens`);
  const [showConditionalScreenLogic, setShowConditionalScreenLogic] = useState(true);
  const [showModals, setShowModals] = useState(false);
  const [showDataviews, setShowDataviews] = useState(false);
  const [hideAlerts, setHideAlerts] = useState(false);
  const [filters, setFilters] = useState<AllScreensFilterSet>({
    tax: undefined,
    batch: undefined,
    status: undefined,
    workflow: undefined,
  });

  const handleChangeShowType = (selectedShowType: ShowType) => {
    setShowType(selectedShowType);
  };

  const [expanded, setExpanded] = useState(false);
  const toggleExpanded = (): void => {
    setExpanded((prvExpanded) => !prvExpanded);
  };

  const screenContent = useMemo(() => {
    return (
      <AllScreensContent
        showType={showType}
        showConditionalScreenLogic={showConditionalScreenLogic}
        hideAlerts={hideAlerts}
        showModals={showModals}
        showDataviews={showDataviews}
        filters={filters}
      />
    );
  }, [showType, showConditionalScreenLogic, hideAlerts, showModals, filters, showDataviews]);

  return (
    <AllScreensContext>
      <div className='screens-container'>
        <AllScreensHeader onMenuToggle={toggleExpanded} expanded={expanded} showType={showType} />
        <div className={styles.mainScreen}>
          <div className={expanded ? `` : `${styles.flyoutMenuHidden}`}>
            <AllScreensMenu
              onShowTypeChange={handleChangeShowType}
              showType={showType}
              showConditionalScreenLogic={showConditionalScreenLogic}
              onConditionalScreenLogicToggle={setShowConditionalScreenLogic}
              hideAlerts={hideAlerts}
              setHideAlerts={setHideAlerts}
              filters={filters}
              setFilters={setFilters}
              setShowModals={setShowModals}
              showModals={showModals}
              showDataviews={showDataviews}
              setShowDataviews={setShowDataviews}
            />
          </div>
          {screenContent}
        </div>
      </div>
    </AllScreensContext>
  );
}

export default AllScreens;
