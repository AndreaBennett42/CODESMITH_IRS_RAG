import styles from '../AllScreens.module.scss';
import { Suspense } from 'react';
import { AuthorizeStateScreenContent } from '../../screens/AuthorizeStateScreen/AuthorizeStateScreen.js';
import { useAllScreensStateTaxesSettingsContext } from './AllScreensPostSubmissionSettings.js';
import { DEFAULT_FACT_GRAPH, DEFAULT_TAX_RETURNS_SLICE_STATE_DATA } from './stateTaxesSettingsDefaults.js';
import { FactGraphContext } from '../../factgraph/FactGraphContext.js';
import { SubmissionStatusContext } from '../../context/SubmissionStatusContext/SubmissionStatusContext.js';
import { TaxReturnDetails } from '../../pages/TaxReturnDetails/TaxReturnDetails.js';
import { TaxReturnsContext } from '../../context/TaxReturnsContext.js';

export const AllScreensPostSubmission = () => {
  const {
    currentTaxReturn,
    hasTaxReturnsFetchError,
    isFetchingTaxReturns,
    fetchStateProfileHookResponse,
    submissionStatusContext,
    transferDisabled,
  } = useAllScreensStateTaxesSettingsContext();
  return (
    <FactGraphContext.Provider value={{ factGraph: DEFAULT_FACT_GRAPH }}>
      <SubmissionStatusContext.Provider value={submissionStatusContext}>
        <div>
          <div>
            <h2>Post Submission Screens</h2>
            <div className={styles.subcontentContainer}>
              <div className={styles.screenOuterContainer}>
                <h2>State Taxes</h2>
                <h3>Authorize State Screen</h3>
                <div className={styles.screenHeader}>
                  <span className={styles.screenRoute}>authorize-state</span>
                </div>
                <div className={styles.modalContainer}>
                  <div className={styles.screenContainer}>
                    <Suspense fallback={<div>Loading...</div>}>
                      <AuthorizeStateScreenContent
                        currentTaxReturn={currentTaxReturn}
                        hasTaxReturnsFetchError={hasTaxReturnsFetchError}
                        isFetchingTaxReturns={isFetchingTaxReturns}
                        fetchStateProfileHookResponse={fetchStateProfileHookResponse}
                        transferDisabled={transferDisabled}
                      />
                    </Suspense>
                  </div>
                </div>
              </div>
              <div>
                <div className={styles.subcontentContainer}></div>
              </div>
              <div className={styles.screenOuterContainer}>
                <h2>Tax Return Details</h2>
                <h3>{submissionStatusContext.submissionStatus.status} Return</h3>
                <div className={styles.screenHeader}>
                  <span className={styles.screenRoute}>federal-tax-return</span>
                </div>
                <div className={styles.modalContainer}>
                  <div className={styles.screenContainer}>
                    <Suspense fallback={<div>Loading...</div>}></Suspense>
                    {currentTaxReturn && (
                      <TaxReturnsContext.Provider
                        value={{
                          ...DEFAULT_TAX_RETURNS_SLICE_STATE_DATA,
                          // eslint-disable-next-line @typescript-eslint/no-empty-function
                          fetchTaxReturns: () => {},
                        }}
                      >
                        <TaxReturnDetails />
                      </TaxReturnsContext.Provider>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </SubmissionStatusContext.Provider>
    </FactGraphContext.Provider>
  );
};
