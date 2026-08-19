import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';

import 'dockview-react/dist/styles/dockview.css';
import '@porthole/framework/theme.css';
import './inspection.css';

import { InspectionApp } from './InspectionApp.js';

const container = document.getElementById('root');
if (!container) throw new Error('#root missing from index.html');

createRoot(container).render(
  <StrictMode>
    <InspectionApp />
  </StrictMode>,
);
