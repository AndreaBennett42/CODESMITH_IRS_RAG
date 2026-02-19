import { useIsReturnEditable } from './useIsReturnEditable.js';
import { useSaveAndPersist } from './useSaveAndPersist.js';

export const useSaveAndPersistIfPossible = () => {
  const { isReturnEditable } = useIsReturnEditable();
  const saveAndPersist = useSaveAndPersist();
  return async () => {
    if (isReturnEditable) {
      return await saveAndPersist();
    }
    return new Promise<{ hasPersistError: boolean }>((resolve) => {
      resolve({
        hasPersistError: false,
      });
    });
  };
};
