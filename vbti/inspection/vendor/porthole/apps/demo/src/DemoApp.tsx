/**
 * The demo dashboard.
 *
 * This is what an app built on Porthole looks like end to end: wrap in a
 * BusProvider, hand PortholeDashboard a panel list and a layout, done. There is
 * no app-specific transport code, and there is no "demo mode" — the data comes
 * from a Python process on the real socket, exactly as it would from a robot.
 */

import { BusProvider, PortholeDashboard, useBusStatus, useBusHello } from '@porthole/framework';
import type { LayoutSpec } from '@porthole/framework';

import { demoPanelDefinitions } from './panelDefinitions.js';

// Order matters: the last panel added to a group is the active tab, so the 3D
// scene sits in front of the point cloud rather than behind it.
const demoLayout: LayoutSpec = [
  { id: 'cloud', type: 'point-cloud' },
  { id: 'cell', type: 'scene-view', position: { referencePanel: 'cloud', direction: 'within' } },
  { id: 'views', type: 'media-grid', position: { referencePanel: 'cell', direction: 'right' } },
  { id: 'decide', type: 'command', position: { referencePanel: 'views', direction: 'below' } },
  { id: 'events', type: 'event-log', position: { referencePanel: 'cell', direction: 'below' } },
];

function StatusBar() {
  const status = useBusStatus();
  const hello = useBusHello();

  return (
    <div className="porthole-statusbar">
      <span className="porthole-status-dot" data-status={status} />
      <span>{status}</span>
      {hello ? (
        <>
          <span>{hello.app}</span>
          <span>{hello.server}</span>
          <span>{hello.topics.length} topics</span>
        </>
      ) : null}
    </div>
  );
}

export function DemoApp() {
  return (
    <BusProvider>
      <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
        <div style={{ flex: 1, minHeight: 0 }}>
          <PortholeDashboard
            panels={demoPanelDefinitions}
            layout={demoLayout}
            storageKey="demo"
          />
        </div>
        <StatusBar />
      </div>
    </BusProvider>
  );
}
