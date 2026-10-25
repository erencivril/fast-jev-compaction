import { describe, expect, it } from 'vitest';
import plugin, {
  applyToOpenCode,
  compactOpenCode,
  OPENROUTER_SYSTEM_ONE_URL,
  resolveOpenCodeConfig,
  resultText,
  toTranscript,
  type OpenCodeCompaction,
  type OpenCodeMessage,
} from '../opencode/index.ts';

const fileA = 'export const a = 1;\n'.repeat(50);

function user(text: string, id: string): OpenCodeMessage {
  return { id, role: 'user', content: [{ type: 'text', text }], metadata: {} };
}

function call(id: string, name: string, input: Record<string, unknown>): OpenCodeMessage {
  return {
    id: `msg-${id}`,
    role: 'assistant',
    content: [
      { type: 'reasoning', text: `thinking about ${id}` },
      { type: 'tool-call', id, name, input, providerExecuted: false },
    ],
  };
}

function result(id: string, name: string, value: string, type: 'text' | 'error' = 'text'): OpenCodeMessage {
  return { role: 'tool', content: [{ type: 'tool-result', id, name, result: { type, value } }] };
}

function transcript(): OpenCodeMessage[] {
  return [
    user('Fix the failing test.', 'msg-0'),
    call('call-1', 'read', { path: 'src/a.ts' }),
    result('call-1', 'read', fileA),
    { role: 'system', content: [{ type: 'text', text: 'The Code Mode tool catalog has changed.' }] },
    call('call-2', 'bash', { command: 'npm test' }),
    result('call-2', 'bash', 'FAIL b.test.ts: expected 2 to be 3\n'.repeat(30), 'error'),
    call('call-3', 'glob', { pattern: '**/*.ts' }),
    result('call-3', 'glob', 'src/a.ts\nsrc/b.ts\n'.repeat(40)),
    { id: 'msg-8', role: 'assistant', content: [{ type: 'text', text: 'Fixing now.' }] },
    user('go ahead', 'msg-9'),
  ];
}

function jevFetch(answer: (name: string) => number, bodies: string[] = [], urls: string[] = []) {
  return (async (url: string, init?: { body?: string }) => {
    urls.push(url);
    bodies.push(init?.body ?? '');
    const { questions } = JSON.parse(init?.body ?? '{}') as { questions: Record<string, unknown> };
    const answers = Object.fromEntries(
      Object.keys(questions).map((key) => [key, { type: 'noul', noul: answer(key) }]),
    );
    return new Response(JSON.stringify({ answers }), { status: 200 });
  }) as unknown as typeof fetch;
}

/** t1 (read): drop the call; t2 (bash): keep the call, truncate the result; t3 (glob): keep. */
const answers = (name: string) =>
  ({ call_t1: 0.1, result_t1: 0.1, call_t2: 0.9, result_t2: 0.2, call_t3: 0.9, result_t3: 0.9 })[
    name
  ] ?? 1;

const config = { ...resolveOpenCodeConfig({ preserveRecentMessages: 2 }, {}), apiKey: 'test-key' };

describe('toTranscript', () => {
  it('maps tool calls and results, leaves system messages out', () => {
    const messages = transcript();
    const { transcript: out, sources } = toTranscript(messages);
    expect(out).toHaveLength(9);
    expect(sources).not.toContain(3);
    expect(out[1]).toEqual({
      role: 'assistant',
      text: '',
      toolUses: [{ tool_use_id: 'call-1', tool: 'read', input: { path: 'src/a.ts' } }],
    });
    expect(out[2]?.role).toBe('user');
    expect(out[2]?.toolResults).toEqual([{ tool_use_id: 'call-1', text: fileA, isError: false }]);
    expect(out[4]?.toolResults?.[0]?.isError).toBe(true);
  });

  it('reads json and content results as text', () => {
    expect(resultText({ type: 'json', value: { ok: true } })).toBe('{"ok":true}');
    expect(
      resultText({ type: 'content', value: [{ type: 'text', text: 'hi' }, { type: 'media' }] }),
    ).toBe('hi\n[media]');
  });
});

describe('compactOpenCode', () => {
  it('drops, truncates and keeps parts by Jev decision, at the OpenRouter endpoint', async () => {
    const messages = transcript();
    const urls: string[] = [];
    const bodies: string[] = [];
    const outcome = await compactOpenCode(messages, config, jevFetch(answers, bodies, urls));
    expect(urls).toEqual([OPENROUTER_SYSTEM_ONE_URL]);
    expect(JSON.parse(bodies[0] ?? '{}').model).toBe('jev-latest');
    expect(outcome.replaced).toBe(true);
    if (!outcome.replaced) return;
    const out = outcome.messages;

    // call-1 is gone with its result; the system message stays where it was.
    expect(JSON.stringify(out)).not.toContain('call-1');
    expect(out[0]).toBe(messages[0]);
    expect(out[1]).toBe(messages[3]);

    // call-2 keeps its call (reasoning included) and an error result truncated to a head + note.
    expect(out[2]).toBe(messages[4]);
    const truncated = out[3]?.content[0] as { result: { type: string; value: string } };
    expect(truncated.result.type).toBe('error');
    expect(truncated.result.value).toContain('fast-jev-compaction truncated');
    expect(out[3]).not.toBe(messages[5]);

    // call-3 and the recent messages are the original objects.
    expect(out.slice(4)).toEqual(messages.slice(6));
    expect(out[4]).toBe(messages[6]);
    expect(out).toHaveLength(8);
  });

  it('removes an assistant message that only held reasoning and a dropped call', () => {
    const messages = transcript();
    const { transcript: t } = toTranscript(messages);
    const out = applyToOpenCode(
      messages,
      t,
      {
        messages: t,
        decisions: [{ id: 't1', tool: 'read', action: 'drop_call', reason: 'call_dropped', keepCall: 0, keepResult: 0 }],
        stats: {} as never,
      },
      2,
    );
    expect(out.map((m) => m.id ?? m.role)).not.toContain('msg-call-1');
    expect(out).toHaveLength(messages.length - 2);
  });

  it('does not replace when the reduction is below the minimum', async () => {
    const outcome = await compactOpenCode(transcript(), { ...config, minReductionRatio: 0.99 }, jevFetch(() => 0.9));
    expect(outcome.replaced).toBe(false);
    expect(outcome.note).toContain('fallback to built-in summary');
  });

  it('throws without a key', async () => {
    await expect(compactOpenCode(transcript(), { ...config, apiKey: undefined }, jevFetch(() => 1))).rejects.toThrow(
      'OPENROUTER_API_KEY',
    );
  });
});

describe('resolveOpenCodeConfig', () => {
  it('takes the option key first, then OPENROUTER_API_KEY, then TYPESAFE_API_KEY', () => {
    expect(resolveOpenCodeConfig({ apiKey: 'opt' }, { OPENROUTER_API_KEY: 'or' }).apiKey).toBe('opt');
    expect(resolveOpenCodeConfig({}, { OPENROUTER_API_KEY: 'or', TYPESAFE_API_KEY: 'ts' }).apiKey).toBe('or');
    expect(resolveOpenCodeConfig({}, { TYPESAFE_API_KEY: 'ts' }).apiKey).toBe('ts');
    expect(resolveOpenCodeConfig({}, {}).apiKey).toBeUndefined();
  });

  it('defaults to OpenRouter and accepts the Claude Code plugin options', () => {
    const resolved = resolveOpenCodeConfig({ keepThreshold: 0.7, baseUrl: 'https://example.test/systemone' }, {});
    expect(resolved.keepThreshold).toBe(0.7);
    expect(resolved.baseUrl).toBe('https://example.test/systemone');
    expect(resolveOpenCodeConfig({}, {}).baseUrl).toBe(OPENROUTER_SYSTEM_ONE_URL);
  });
});

describe('plugin', () => {
  it('registers a compaction hook that leaves result unset on failure', async () => {
    let hook: ((event: OpenCodeCompaction) => Promise<void> | void) | undefined;
    const previous = { or: process.env.OPENROUTER_API_KEY, ts: process.env.TYPESAFE_API_KEY };
    delete process.env.OPENROUTER_API_KEY;
    delete process.env.TYPESAFE_API_KEY;
    try {
      await plugin.setup({
        options: {},
        session: {
          hook: async (_name, callback) => {
            hook = callback;
            return { dispose: async () => {} };
          },
        },
      });
    } finally {
      if (previous.or !== undefined) process.env.OPENROUTER_API_KEY = previous.or;
      if (previous.ts !== undefined) process.env.TYPESAFE_API_KEY = previous.ts;
    }
    const event: OpenCodeCompaction = { sessionID: 'ses_1', messages: transcript() };
    await hook?.(event);
    expect(event.result).toBeUndefined();
  });
});
