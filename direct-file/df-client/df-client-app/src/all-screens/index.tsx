import { createRoot } from 'react-dom/client';
import { initI18n } from '../i18n.js';
import AllScreens from './AllScreens.js';
import '../styles/App.scss';
import { StrictMode } from 'react';

initI18n();

const root = createRoot(document.getElementById(`root`) as HTMLElement);
root.render(
  <StrictMode>
    <AllScreens />
  </StrictMode>
);
