// Telegram bot <-> OpenRouter (Nemotron). Zero dependencies, Node.js 20.6+.
// Run: node bot.js  (или npm start)

// Загружаем .env сами, из папки рядом с bot.js — работает при любом способе
// запуска и из любой текущей папки. Переменные окружения из системы приоритетнее.
const fs = require('node:fs');
const path = require('node:path');

function loadEnv(file) {
  if (!fs.existsSync(file)) return console.warn(`Файл не найден: ${file}`);
  for (const line of fs.readFileSync(file, 'utf8').replace(/^﻿/, '').split(/\r?\n/)) {
    const m = line.match(/^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$/);
    if (!m || line.trim().startsWith('#')) continue;
    const value = m[2].replace(/^(['"])(.*)\1$/, '$2');
    if (process.env[m[1]] === undefined) process.env[m[1]] = value;
  }
}
loadEnv(path.join(__dirname, '.env'));

const TG_TOKEN = process.env.TELEGRAM_BOT_TOKEN;
const OR_KEY = process.env.OPENROUTER_API_KEY;
const MODELS = (process.env.MODELS ||
  'nvidia/nemotron-3-ultra-550b-a55b:free,nvidia/nemotron-3.5-lightning:free')
  .split(',').map(s => s.trim()).filter(Boolean);
// Модели для фото: нужен vision. nemotron-3-ultra картинки НЕ принимает (404 от OpenRouter).
const VISION_MODELS = (process.env.VISION_MODELS ||
  'nvidia/nemotron-3.5-super-vl:free,google/gemma-4-31b-it:free')
  .split(',').map(s => s.trim()).filter(Boolean);
const TEXT_MAX_TOKENS = 2000;
const VISION_MAX_TOKENS = 4096; // reasoning-модель тратит часть лимита на «мысли»
const DEFAULT_PHOTO_PROMPT = 'Опиши, что на картинке.';
const PHOTO_INSTRUCTION =
  'Отвечай кратко и по делу. Если на картинке текст, просто напиши его. Не пиши свои рассуждения шаг за шагом.';
const FAIL_MESSAGE_PHOTO =
  'Не удалось распознать изображение: ни одна из моделей не смогла дать ответ. Попробуйте отправить фото с более четким запросом или повторите позже.';
const FAIL_MESSAGE_EMPTY_TEXT =
  'Модели задумались слишком сильно и не смогли дать ответ. Попробуйте переформулировать вопрос или задать его короче.';
const FAIL_MESSAGE_GENERIC = 'Не удалось получить ответ, попробуй ещё раз чуть позже.';
const REQUEST_TIMEOUT_MS = 120_000;

// Отказ одной модели. kind: 'empty' — 200 OK, но нет финального текста (content пуст,
// часто весь лимит ушёл в reasoning); 'error' — HTTP-ошибка, таймаут, сеть.
class ModelFailure extends Error {
  constructor(kind, message) { super(message); this.kind = kind; }
}
// Все модели цепочки отказали.
class AllModelsFailedError extends Error {
  constructor(failures) {
    super(`Все модели отказали: ${failures.map(f => `${f.model} [${f.kind}]`).join(', ')}`);
    this.failures = failures;
    this.allEmpty = failures.length > 0 && failures.every(f => f.kind === 'empty');
  }
}
const MAX_PHOTO_BYTES = 5 * 1024 * 1024;
const SYSTEM_PROMPT = process.env.SYSTEM_PROMPT ||
  'Ты дружелюбный помощник в Telegram. Отвечай кратко и по делу, на языке собеседника.';
const HISTORY_LIMIT = 12; // сообщений на чат
const TG_MAX = 4096;

if (!TG_TOKEN || !OR_KEY) {
  console.error('Нужны TELEGRAM_BOT_TOKEN и OPENROUTER_API_KEY в .env');
  process.exit(1);
}

const TG = `https://api.telegram.org/bot${TG_TOKEN}`;
const histories = new Map(); // chatId -> [{role, content}]

async function tg(method, body = {}) {
  const res = await fetch(`${TG}/${method}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!data.ok) throw new Error(`Telegram ${method}: ${data.description}`);
  return data.result;
}

// Скачивает фото из Telegram (getFile -> file/bot<token>/<file_path>) и возвращает data URL.
async function photoToDataUrl(photoSizes) {
  // Telegram присылает несколько размеров одного фото; берём самый большой в пределах лимита.
  const candidates = [...photoSizes].reverse();
  const pick = candidates.find(p => !p.file_size || p.file_size <= MAX_PHOTO_BYTES) ?? candidates[candidates.length - 1];
  const file = await tg('getFile', { file_id: pick.file_id });
  const res = await fetch(`https://api.telegram.org/file/bot${TG_TOKEN}/${file.file_path}`);
  if (!res.ok) throw new Error(`Не удалось скачать фото: HTTP ${res.status}`);
  const buf = Buffer.from(await res.arrayBuffer());
  console.log(`[Telegram] Фото скачано: ${pick.width}x${pick.height}, ${(buf.length / 1024).toFixed(0)} КБ`);
  return `data:image/jpeg;base64,${buf.toString('base64')}`; // Telegram отдаёт фото как JPEG
}

// Запрос к ОДНОЙ модели. Бросает ModelFailure при ошибке или пустом ответе.
async function askOneModel(messages, model, maxTokens) {
  let res;
  try {
    res = await fetch('https://openrouter.ai/api/v1/chat/completions', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${OR_KEY}`,
        'Content-Type': 'application/json',
        'X-Title': 'Telegram Nemotron Bot',
      },
      // Перебор моделей делаем сами (см. askModel): серверный фолбэк `models`
      // срабатывает только на ошибках API, но не на 200 OK с пустым content.
      body: JSON.stringify({
        model,
        messages: [{ role: 'system', content: SYSTEM_PROMPT }, ...messages],
        max_tokens: maxTokens,
      }),
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    });
  } catch (err) {
    throw new ModelFailure('error', `сеть/таймаут: ${err.name}: ${err.message}`);
  }
  const raw = await res.text(); // читаем как текст, чтобы не потерять тело ошибки не-JSON
  let data;
  try { data = JSON.parse(raw); } catch { data = null; }

  if (!res.ok) {
    console.error(`[OpenRouter] ${model}: HTTP ${res.status} ${res.statusText}`);
    console.error('[OpenRouter] Тело ответа целиком:', raw);
    throw new ModelFailure('error', data?.error?.message || `HTTP ${res.status}`);
  }
  const choice = data?.choices?.[0];
  console.log(`[OpenRouter] ${model}: 200 OK, ответила: ${data?.model}, finish_reason: ${choice?.finish_reason}`);
  const text = choice?.message?.content?.trim();
  if (!text) {
    const reasoning = choice?.message?.reasoning;
    console.error(`[OpenRouter] ${model}: пустой content` +
      (reasoning ? ` (но есть reasoning, ${String(reasoning).length} симв.)` : '') +
      `, finish_reason: ${choice?.finish_reason}, usage: ${JSON.stringify(data?.usage)}`);
    if (!reasoning) console.error('[OpenRouter] Ответ целиком:', raw);
    throw new ModelFailure('empty', 'пустой content');
  }
  return { text, model: data?.model ?? model };
}

// Перебирает модели по порядку: следующая пробуется и при ошибке API, и при пустом ответе.
async function askModel(messages, models = MODELS, maxTokens = TEXT_MAX_TOKENS) {
  const failures = [];
  for (const [i, model] of models.entries()) {
    try {
      const { text, model: usedModel } = await askOneModel(messages, model, maxTokens);
      console.log(`[OpenRouter] ✔ Финальный ответ дала модель: ${usedModel} (попытка ${i + 1} из ${models.length})`);
      return text;
    } catch (err) {
      if (!(err instanceof ModelFailure)) throw err;
      failures.push({ model, kind: err.kind, reason: err.message });
      const next = models[i + 1];
      console.warn(`[OpenRouter] ✘ ${model}: ${err.kind} — ${err.message}.` +
        (next ? ` Пробую следующую: ${next}` : ' Моделей больше нет.'));
    }
  }
  throw new AllModelsFailedError(failures);
}

function splitMessage(text) {
  const parts = [];
  while (text.length > TG_MAX) {
    let cut = text.lastIndexOf('\n', TG_MAX);
    if (cut < TG_MAX / 2) cut = TG_MAX;
    parts.push(text.slice(0, cut));
    text = text.slice(cut).trimStart();
  }
  if (text) parts.push(text);
  return parts;
}

async function handleMessage(msg) {
  const chatId = msg.chat.id;
  const isPhoto = Array.isArray(msg.photo) && msg.photo.length > 0;
  const text = (isPhoto ? msg.caption : msg.text)?.trim() ?? '';
  if (!text && !isPhoto) {
    const kinds = Object.keys(msg).filter(k => ['document', 'sticker', 'voice', 'video', 'audio'].includes(k));
    console.warn(`[Telegram] Сообщение без текста проигнорировано (chat ${chatId}, тип: ${kinds.join(',') || 'неизвестно'})`);
    return;
  }

  if (text === '/start') {
    histories.delete(chatId);
    return tg('sendMessage', {
      chat_id: chatId,
      text: 'Привет! Я бот на Nemotron. Просто напиши мне что-нибудь. /reset — сбросить диалог.',
    });
  }
  if (text === '/reset') {
    histories.delete(chatId);
    return tg('sendMessage', { chat_id: chatId, text: 'Контекст очищен.' });
  }

  const history = histories.get(chatId) ?? [];
  // В историю кладём только текст: base64 картинки раздул бы каждый следующий запрос.
  const prompt = text || DEFAULT_PHOTO_PROMPT;
  history.push({ role: 'user', content: isPhoto ? `[Фото] ${prompt}` : text });

  const typing = setInterval(() => tg('sendChatAction', { chat_id: chatId, action: 'typing' }).catch(() => {}), 4000);
  tg('sendChatAction', { chat_id: chatId, action: 'typing' }).catch(() => {});
  try {
    let answer;
    if (isPhoto) {
      const imageUrl = await photoToDataUrl(msg.photo);
      const messages = [
        ...history.slice(0, -1),
        { role: 'user', content: [
          { type: 'text', text: `${prompt}\n\n${PHOTO_INSTRUCTION}` },
          { type: 'image_url', image_url: { url: imageUrl } },
        ] },
      ];
      answer = await askModel(messages, VISION_MODELS, VISION_MAX_TOKENS);
    } else {
      answer = await askModel(history); // текст -> старая текстовая модель
    }
    history.push({ role: 'assistant', content: answer });
    histories.set(chatId, history.slice(-HISTORY_LIMIT));
    for (const part of splitMessage(answer)) {
      await tg('sendMessage', { chat_id: chatId, text: part });
    }
  } catch (err) {
    history.pop(); // не сохраняем неудачный запрос
    console.error('Ошибка:', err.message);
    let reply = FAIL_MESSAGE_GENERIC;
    if (isPhoto && err instanceof AllModelsFailedError) reply = FAIL_MESSAGE_PHOTO;
    else if (err instanceof AllModelsFailedError && err.allEmpty) reply = FAIL_MESSAGE_EMPTY_TEXT;
    await tg('sendMessage', { chat_id: chatId, text: reply }).catch(() => {});
  } finally {
    clearInterval(typing);
  }
}

async function main() {
  const me = await tg('getMe');
  console.log(`Бот @${me.username} запущен. Текст: ${MODELS.join(' -> ')}; фото: ${VISION_MODELS.join(' -> ')}`);
  let offset = 0;
  for (;;) {
    try {
      const updates = await tg('getUpdates', { offset, timeout: 30, allowed_updates: ['message'] });
      for (const u of updates) {
        offset = u.update_id + 1;
        if (u.message) handleMessage(u.message).catch(e => console.error(e.message));
      }
    } catch (err) {
      console.error('Polling:', err.message);
      await new Promise(r => setTimeout(r, 5000));
    }
  }
}

main().catch(e => { console.error(e.message); process.exit(1); });
