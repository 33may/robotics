/**
 * Choice + note + submit. The generic shape of "a human decides something".
 *
 * REGISTRY COMPONENT — copy and edit freely; see docs/component-model.md.
 *
 * The backend publishes the available options; this panel renders them and
 * sends back what was picked. It has no idea what the options mean and no
 * ability to act on one. That is protocol §7: a command is a statement of
 * intent by a human, and every check that matters happens in Python.
 */

import { useEffect, useState } from 'react';
import { useSendCommand, useTopicPayload } from '@porthole/framework';
import type { PanelProps } from '@porthole/framework';

export interface CommandPanelConfig extends Record<string, unknown> {
  /** Topic publishing `{ prompt, options: [{id, label, meta?}] }`. */
  requestTopic: string;
  /** Command name sent on submit. */
  command: string;
  /** Show a free-text field alongside the options. */
  allowNote: boolean;
}

export const commandPanelDefaultConfig: CommandPanelConfig = {
  requestTopic: 'decision/request',
  command: 'decision/answer',
  allowNote: true,
};

interface DecisionOption {
  id: string;
  label: string;
  meta?: string;
}

interface DecisionRequest {
  id?: string;
  prompt?: string;
  options?: DecisionOption[];
}

export function CommandPanel({ config }: PanelProps<CommandPanelConfig>) {
  const request = useTopicPayload<DecisionRequest>(config.requestTopic);
  const sendCommand = useSendCommand();

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [note, setNote] = useState('');
  const [failed, setFailed] = useState(false);

  // A new request invalidates the previous answer. Without this, a stale
  // selection can be submitted against a question that is no longer being asked.
  useEffect(() => {
    setSelectedId(null);
    setNote('');
    setFailed(false);
  }, [request?.id]);

  const options = request?.options ?? [];

  function submit() {
    if (!selectedId) return;
    const delivered = sendCommand({
      cmd: config.command,
      ...(request?.id !== undefined ? { id: request.id } : {}),
      choice: selectedId,
      ...(config.allowNote && note ? { note } : {}),
    });
    setFailed(!delivered);
  }

  return (
    <div className="porthole-command-panel">
      {request?.prompt ? <p className="porthole-command-prompt">{request.prompt}</p> : null}

      <ul className="porthole-command-options">
        {options.map((option) => (
          <li key={option.id}>
            <button
              type="button"
              data-selected={option.id === selectedId}
              onClick={() => setSelectedId(option.id)}
            >
              <span>{option.label}</span>
              {option.meta ? <span className="porthole-command-meta">{option.meta}</span> : null}
            </button>
          </li>
        ))}
      </ul>

      {config.allowNote ? (
        <textarea
          className="porthole-command-note"
          value={note}
          rows={3}
          onChange={(event) => setNote(event.target.value)}
        />
      ) : null}

      <button
        type="button"
        className="porthole-command-submit"
        disabled={!selectedId}
        onClick={submit}
      >
        send
      </button>

      {failed ? <p className="porthole-command-failed">not delivered — backend offline</p> : null}
    </div>
  );
}
