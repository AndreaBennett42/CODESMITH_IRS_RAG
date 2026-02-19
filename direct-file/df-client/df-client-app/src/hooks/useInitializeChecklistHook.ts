import { useEffect, useRef } from 'react';

export const useInitializeChecklist = () => {
  const hasDispatched = useRef(false);

  useEffect(() => {
    if (!hasDispatched.current) {
      hasDispatched.current = true;
    }
  });
};
