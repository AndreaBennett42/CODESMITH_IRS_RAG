import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App.js';
import './styles/App.scss';
import { initPrivacyHandler } from './auth/storage.js';
import reportWebVitals from './reportWebVitals.js';
import { initI18n } from './i18n.js';
import { interceptor } from './hooks/useApiHook.js';
import { CommonOverrideBanner, commonOverrideOverrideDateIfRequired } from '@irs/df-common';

commonOverrideOverrideDateIfRequired();
initPrivacyHandler();
window.fetch = interceptor(fetch);
initI18n().then(() => {
  const root = createRoot(document.getElementById(`root`) as HTMLElement);
  root.render(
    <StrictMode>
      <CommonOverrideBanner />
      <App />
    </StrictMode>
  );

  // If you want to start measuring performance in your app, pass a function
  // to log results (for example: reportWebVitals(logger.info))
  // or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
  reportWebVitals();
});
