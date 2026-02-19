import { useEffect, useState } from 'react';
import { isNavigator } from '../misc/misc.js';

export interface NetworkStatus {
  online: boolean | undefined;
  prevOnlineStatus: boolean | undefined;
}

const nav:
  | (Navigator & Partial<Record<'connection' | 'mozConnection' | 'webkitConnection', NetworkStatus>>)
  | undefined = isNavigator ? navigator : undefined;

function getConnectionState(prevStatus: NetworkStatus) {
  const online = nav?.onLine;

  // Only return a new state object if the underlying values would change
  if (prevStatus.online !== online || prevStatus.prevOnlineStatus !== prevStatus.online) {
    return {
      online,
      prevOnlineStatus: prevStatus.online,
    };
  } else {
    return prevStatus;
  }
}

export default function useNetworkConnectionStatus(
  initialNetworkStatus: NetworkStatus,
  intervalTime: number
): NetworkStatus {
  const [networkStatus, setNetworkStatus] = useState<NetworkStatus>(initialNetworkStatus);
  useEffect(() => {
    const interval = setInterval(() => {
      setNetworkStatus((prevNetworkStatus) => getConnectionState(prevNetworkStatus));
    }, intervalTime);
    return () => clearInterval(interval);
  }, [intervalTime]);
  return networkStatus;
}
