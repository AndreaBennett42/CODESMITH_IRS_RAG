import { createContext } from 'react';
import { NetworkStatus } from '../hooks/useNetworkConnectionStatus.js';

export const NetworkConnectionContext = createContext<NetworkStatus>({
  online: undefined,
  prevOnlineStatus: undefined,
});
