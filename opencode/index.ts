import { compact, reductionRatio } from '../src/compact.js';
import { JevClient } from '../src/client.js';
import { collectToolCalls, resolveOptions } from '../src/index.js';
import type { CompactResult, Message, ToolResult, ToolUse } from '../src/types.js';
import {
  decisionLogLines,
  resolveHookConfig,
  summarize,
  type HookConfig,
} from '../hooks/fast-jev.js';

/** Jev's System One endpoint as OpenRouter serves it; takes an OpenRouter key. */
export const OPENROUTER_SYSTEM_ONE_URL = 'https://openrouter.ai/api/v1/systemone';

// The subset of `@opencode/ai`'s `Message` the adapter reads. Parts it does not
// know (reasoning, media, effort, ...) are carried through untouched.
export type OpenCodePart = { type: string; [key: string]: unknown };

export type OpenCodeToolCallPart = OpenCodePart & {
  type: 'tool-call';
  id: string;
  name: string;
  input: unknown;
};

export type OpenCodeToolResultPart = OpenCodePart & {
  type: 'tool-result';
  id: string;
  name: string;
  result: { type: 'json' | 'text' | 'error' | 'content'; value: unknown };
};

export type OpenCodeMessage = {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: readonly OpenCodePart[];
  [key: string]: unknown;
};

/** The `compaction` session hook's event (`SessionCompaction` in `@opencode/plugin`). */
export type OpenCodeCompaction = {
  readonly sessionID: string;
  messages: OpenCodeMessage[];
  result?: {
    summary: string;
    native?: Record<string, unknown>;
    metadata?: Record<string, unknown>;
  };
};

export type OpenCodePluginContext = {
  readonly options: Readonly<Record<string, unknown>>;
  readonly session: {
    hook: (
      name: 'compaction',
      callback: (event: OpenCodeCompaction) => Promise<void> | void,
    ) => Promise<{ dispose: () => Promise<void> }>;
  };
};

export type OpenCodeConfig = HookConfig & { baseUrl: string };

function isToolCall(part: OpenCodePart): part is OpenCodeToolCallPart {
  return part.type === 'tool-call' && typeof part.id === 'string';
}

function isToolResult(part: OpenCodePart): part is OpenCodeToolResultPart {
  return part.type === 'tool-result' && typeof part.id === 'string';
}

function stringify(value: unknown): string {
  if (typeof value === 'string') return value;
  try {
    return JSON.stringify(value) ?? '';
  } catch {
    return String(value);
  }
}

/** A tool result's value as the text Jev's state counts and a truncation keeps a head of. */
export function resultText(result: OpenCodeToolResultPart['result']): string {
  if (result.type === 'content' && Array.isArray(result.value)) {
    return result.value
      .map((item: { type?: string; text?: unknown }) =>
        item?.type === 'text' ? stringify(item.text) : `[${item?.type ?? 'part'}]`,
      )
      .join('\n');
  }
  return stringify(result.value);
}

function textOf(message: OpenCodeMessage): string {
  return message.content
    .filter((part) => part.type === 'text' && typeof part.text === 'string')
    .map((part) => part.text as string)
    .join('\n');
}

/**
 * Maps OpenCode messages onto the library's transcript shape. System messages
 * are OpenCode's own (tool catalogs, instructions) and are not shown to Jev;
 * `sources[i]` is the index in `messages` the i-th transcript message came from.
 */
export function toTranscript(messages: readonly OpenCodeMessage[]): {
  transcript: Message[];
  sources: number[];
} {
  const transcript: Message[] = [];
  const sources: number[] = [];
  messages.forEach((message, index) => {
    if (message.role === 'system') return;
    const toolUses: ToolUse[] = message.content.filter(isToolCall).map((part) => ({
      tool_use_id: part.id,
      tool: part.name,
      input:
        part.input && typeof part.input === 'object'
          ? (part.input as Record<string, unknown>)
          : { input: part.input },
    }));
    const toolResults: ToolResult[] = message.content.filter(isToolResult).map((part) => ({
      tool_use_id: part.id,
      text: resultText(part.result),
      isError: part.result.type === 'error',
    }));
    const entry: Message = {
      role: message.role === 'assistant' ? 'assistant' : 'user',
      text: textOf(message),
      toolUses,
    };
    if (toolResults.length > 0) entry.toolResults = toolResults;
    transcript.push(entry);
    sources.push(index);
  });
  return { transcript, sources };
}

/**
 * Applies the library's decisions to the OpenCode messages: a dropped call
 * loses its tool-call and tool-result parts, a dropped result keeps the head
 * and note the library wrote. A message left with no text and no tool parts
 * is removed; untouched messages are returned as the same objects.
 */
export function applyToOpenCode(
  messages: readonly OpenCodeMessage[],
  transcript: readonly Message[],
  result: CompactResult,
  preserveRecentMessages: number,
): OpenCodeMessage[] {
  const calls = collectToolCalls(transcript, preserveRecentMessages);
  const toolUseIds = new Map(calls.map((call) => [call.id, call.tool_use_id]));
  const dropped = new Set<string>();
  const truncated = new Map<string, string>();
  const kept = new Map<string, string>();
  for (const message of result.messages) {
    for (const toolResult of message.toolResults ?? []) {
      kept.set(toolResult.tool_use_id, toolResult.text);
    }
  }
  for (const decision of result.decisions) {
    const id = toolUseIds.get(decision.id);
    if (!id) continue;
    if (decision.action === 'drop_call') dropped.add(id);
    if (decision.action === 'drop_result') {
      const text = kept.get(id);
      if (text !== undefined) truncated.set(id, text);
    }
  }
  const output: OpenCodeMessage[] = [];
  for (const message of messages) {
    const touched = message.content.some(
      (part) =>
        ((isToolCall(part) || isToolResult(part)) && dropped.has(part.id)) ||
        (isToolResult(part) && truncated.has(part.id)),
    );
    if (!touched) {
      output.push(message);
      continue;
    }
    const content = message.content
      .filter((part) => !((isToolCall(part) || isToolResult(part)) && dropped.has(part.id)))
      .map((part) => {
        if (!isToolResult(part)) return part;
        const text = truncated.get(part.id);
        if (text === undefined || text === resultText(part.result)) return part;
        return { ...part, result: { type: part.result.type === 'error' ? 'error' : 'text', value: text } };
      });
    const substantive = content.some(
      (part) =>
        isToolCall(part) ||
        isToolResult(part) ||
        (part.type === 'text' && typeof part.text === 'string' && part.text.trim().length > 0),
    );
    if (!substantive) continue;
    output.push({ ...message, content });
  }
  return output;
}

function optionString(options: Readonly<Record<string, unknown>>, key: string): string | undefined {
  const value = options[key];
  return typeof value === 'string' && value.length > 0 ? value : undefined;
}

/**
 * The plugin's options from `opencode.json` (same keys as the Claude Code
 * plugin's `userConfig`, plus `baseUrl`). The key falls back to
 * `OPENROUTER_API_KEY`, then `TYPESAFE_API_KEY`.
 */
export function resolveOpenCodeConfig(
  options: Readonly<Record<string, unknown>>,
  env: Readonly<Record<string, string | undefined>> = process.env,
): OpenCodeConfig {
  const config: OpenCodeConfig = {
    // resolveHookConfig type-checks every value it reads, so any JSON object will do.
    ...resolveHookConfig(options as Parameters<typeof resolveHookConfig>[0]),
    baseUrl: optionString(options, 'baseUrl') ?? OPENROUTER_SYSTEM_ONE_URL,
  };
  const apiKey = config.apiKey || env['OPENROUTER_API_KEY'] || env['TYPESAFE_API_KEY'];
  if (apiKey) config.apiKey = apiKey;
  else delete config.apiKey;
  return config;
}

export type OpenCodeCompactionOutcome =
  | { replaced: true; messages: OpenCodeMessage[]; result: CompactResult; note: string }
  | { replaced: false; result?: CompactResult; note: string };

/**
 * Runs Jev over an OpenCode compaction's messages. `replaced` is false when
 * the reduction is below `minReductionRatio`; throws when the key is missing
 * or Jev fails.
 */
export async function compactOpenCode(
  messages: readonly OpenCodeMessage[],
  config: OpenCodeConfig,
  fetchFn?: typeof fetch,
): Promise<OpenCodeCompactionOutcome> {
  if (!config.apiKey) throw new Error('OPENROUTER_API_KEY is not configured');
  const { transcript } = toTranscript(messages);
  const asker = new JevClient({
    apiKey: config.apiKey,
    model: config.model,
    baseUrl: config.baseUrl,
    fetch: fetchFn,
  });
  const result = await compact(transcript, asker, config);
  if (reductionRatio(result) < config.minReductionRatio) {
    return {
      replaced: false,
      result,
      note: `fallback to built-in summary (below ${Math.round(config.minReductionRatio * 100)}% minimum: ${summarize(result)})`,
    };
  }
  const { preserveRecentMessages } = resolveOptions(config);
  const compacted = applyToOpenCode(messages, transcript, result, preserveRecentMessages);
  return {
    replaced: true,
    messages: compacted,
    result,
    note: `kept ${compacted.length}/${messages.length} messages, no summary (${summarize(result)})`,
  };
}

/**
 * OpenCode V2 server plugin: answers the session `compaction` hook with the
 * Jev-pruned messages as the replacement window (`native.messages`), so
 * OpenCode replays them instead of writing a summary. On errors or too little
 * reduction it leaves `result` unset and OpenCode summarizes as usual.
 */
export default {
  id: 'fast-jev-compaction',
  async setup(ctx: OpenCodePluginContext) {
    const config = resolveOpenCodeConfig(ctx.options);
    const registration = await ctx.session.hook('compaction', async (event) => {
      try {
        const outcome = await compactOpenCode(event.messages, config);
        if (outcome.result) {
          for (const line of decisionLogLines(outcome.result)) {
            console.log(`fast-jev-compaction ${event.sessionID} ${line}`);
          }
        }
        console.log(`fast-jev-compaction ${event.sessionID}: ${outcome.note}`);
        if (!outcome.replaced) return;
        event.result = {
          summary: `fast-jev-compaction: ${outcome.note}`,
          native: { messages: outcome.messages },
          metadata: { fastJevCompaction: outcome.result.stats },
        };
      } catch (error) {
        console.log(
          `fast-jev-compaction ${event.sessionID}: fallback to built-in summary (${
            error instanceof Error ? error.message : String(error)
          })`,
        );
      }
    });
    return () => registration.dispose();
  },
};
