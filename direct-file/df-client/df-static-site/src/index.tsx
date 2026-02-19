import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App.js';
import './styles/styles.scss';
//import reportWebVitals from './reportWebVitals.js';
import { commonInitI18n } from 'df-i18n';
import { resources } from './locales/index.js';

commonInitI18n(resources).then(() => {
  const root = createRoot(document.getElementById(`root`) as HTMLElement);
  root.render(
    <StrictMode>
      <App />
    </StrictMode>
  );
});

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(logger.info))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
//reportWebVitals();
