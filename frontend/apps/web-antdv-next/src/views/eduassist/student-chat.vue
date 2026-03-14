<script lang="ts" setup>
import { computed, onMounted, ref, watch } from 'vue';
import { useRoute } from 'vue-router';

import { message } from 'antdv-next';

import { askQuestion, listCourses, type Course } from '#/api/eduassist';

import MoocShell from './components/mooc-shell.vue';
import { useEduAssistContext } from './use-eduassist-context';

const route = useRoute();
const {
  appendChatMessage,
  clearChatMessages,
  getChatMessages,
  setStudentCourse,
  state,
} = useEduAssistContext();

const loading = ref(false);
const sending = ref(false);
const courses = ref<Course[]>([]);
const selectedCourseId = ref<null | number>(null);
const question = ref('');
const SHARED_CHAT_KEY = 0;
const QA_UI_TIMEOUT_MS = 130_000;

const activeCourse = computed(() =>
  courses.value.find((item) => item.id === selectedCourseId.value) || null,
);

const messages = computed(() => getChatMessages(SHARED_CHAT_KEY));

const historyPayload = computed(() =>
  messages.value.slice(-8).map((item) => ({
    content: item.content,
    role: item.role,
  })),
);

function escapeHtml(raw: string) {
  return raw
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function formatInline(raw: string) {
  const escaped = escapeHtml(raw);
  return escaped.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
}

function normalizeStructuredText(content: string) {
  let text = (content || '').replaceAll('\r\n', '\n').trim();
  if (!text) {
    return '';
  }

  // If model returns one long line, insert line breaks by common structure markers.
  if (!text.includes('\n')) {
    text = text
      .replace(/(主要内容包括[:：]?)/g, '\n$1\n')
      .replace(/(^|[。；;：:])\s*(\d+\.\s*)/g, '$1\n$2')
      .replace(/([。；;])\s*([-*•]\s+)/g, '$1\n$2')
      .replace(/\s+([-*•]\s+\*\*[^*]+\*\*[:：]?)/g, '\n$1');
  }

  return text.replace(/\n{3,}/g, '\n\n').trim();
}

function normalizeQuestionForApi(questionText: string) {
  return questionText.replace(/[\s'"`‘’“”＇＂，,。！？?；;：]+$/g, '').trim();
}

function renderAssistantHtml(content: string) {
  const normalized = normalizeStructuredText(content);
  const lines = normalized.split('\n');
  const html: string[] = [];
  let inList = false;

  const closeList = () => {
    if (inList) {
      html.push('</ul>');
      inList = false;
    }
  };

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) {
      closeList();
      html.push('<div class="qa-gap"></div>');
      continue;
    }

    if (/^[-*•]\s+/.test(line)) {
      if (!inList) {
        html.push('<ul class="qa-list">');
        inList = true;
      }
      const itemText = line.replace(/^[-*•]\s+/, '');
      html.push(`<li>${formatInline(itemText)}</li>`);
      continue;
    }

    closeList();
    if (/^\d+\.\s+/.test(line)) {
      html.push(`<p class="qa-number">${formatInline(line)}</p>`);
      continue;
    }
    if (/^主要内容包括[:：]?$/.test(line)) {
      html.push(`<p class="qa-heading">${formatInline(line)}</p>`);
      continue;
    }
    html.push(`<p class="qa-paragraph">${formatInline(line)}</p>`);
  }

  closeList();
  const joined = html.join('').replace(/(<div class="qa-gap"><\/div>){2,}/g, '<div class="qa-gap"></div>');
  return joined || `<p class="qa-paragraph">${formatInline(normalized || content || '')}</p>`;
}

async function fetchCourses() {
  loading.value = true;
  try {
    const allCourses = await listCourses();
    courses.value = allCourses.filter((item) => item.joined);
  } catch (error: any) {
    message.error(error?.detail || error?.message || '加载课程失败');
  } finally {
    loading.value = false;
  }
}

function syncSelectedCourse() {
  const queryCourse = Number(route.query.courseId || 0);
  const contextCourse = state.studentCourseId || null;
  const picked = queryCourse > 0 ? queryCourse : contextCourse;
  selectedCourseId.value = picked;
  setStudentCourse(picked);
}

function handleCourseChange(value: string) {
  if (value === 'all') {
    selectedCourseId.value = null;
    return;
  }
  const courseId = Number(value);
  selectedCourseId.value = Number.isFinite(courseId) && courseId > 0 ? courseId : null;
}

function handleCourseSelectChange(event: Event) {
  const target = event.target as HTMLSelectElement | null;
  if (!target) {
    return;
  }
  handleCourseChange(target.value);
}

async function withTimeout<T>(promise: Promise<T>, timeoutMs: number, messageText: string): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | null = null;
  try {
    const timeoutPromise = new Promise<never>((_resolve, reject) => {
      timer = setTimeout(() => reject(new Error(messageText)), timeoutMs);
    });
    return await Promise.race([promise, timeoutPromise]);
  } finally {
    if (timer) {
      clearTimeout(timer);
    }
  }
}

async function sendQuestion() {
  if (!question.value.trim()) {
    message.warning('请输入问题');
    return;
  }

  const content = question.value.trim();
  const apiQuestion = normalizeQuestionForApi(content) || content;
  question.value = '';
  appendChatMessage(SHARED_CHAT_KEY, { content, role: 'user' });
  sending.value = true;

  try {
    const resp = await withTimeout(
      askQuestion({
        course_id: selectedCourseId.value ?? undefined,
        history: historyPayload.value,
        question: apiQuestion,
      }),
      QA_UI_TIMEOUT_MS,
      '问答请求超时，请重试',
    );
    appendChatMessage(SHARED_CHAT_KEY, {
      content: resp.answer,
      role: 'assistant',
      sourceType: resp.source_type,
      references: resp.references,
    });
  } catch (error: any) {
    message.error(error?.detail || error?.message || '提问失败');
    appendChatMessage(SHARED_CHAT_KEY, {
      content: '抱歉，当前回答失败，请稍后重试。',
      role: 'assistant',
      sourceType: 1,
    });
  } finally {
    sending.value = false;
  }
}

function clearCurrentChat() {
  clearChatMessages(SHARED_CHAT_KEY);
}

watch(selectedCourseId, (value) => {
  setStudentCourse(value ?? null);
});

onMounted(async () => {
  await fetchCourses();
  syncSelectedCourse();
});
</script>

<template>
  <div class="p-5">
    <MoocShell
      title="智能问答"
      :course-name="activeCourse ? `当前课程：${activeCourse.name}` : '当前范围：全部课程（未指定）'"
    >
      <template #sidebar>
        <ul class="side-list">
          <li>支持多轮问答与上下文记忆</li>
          <li>默认携带最近 8 轮对话历史</li>
          <li>回答会展示资料引用来源</li>
        </ul>
      </template>

      <template #toolbar>
        <button class="plain-btn" type="button" @click="clearCurrentChat">清空当前对话</button>
      </template>

      <div class="toolbar-row">
        <select
          class="course-select"
          :value="selectedCourseId === null ? 'all' : String(selectedCourseId)"
          @change="handleCourseSelectChange"
        >
          <option value="all">全部课程（不指定，综合检索）</option>
          <option v-for="course in courses" :key="course.id" :value="String(course.id)">
            {{ course.id }} - {{ course.name }}
          </option>
        </select>
      </div>

      <div class="chat-board">
        <div v-if="messages.length === 0" class="placeholder">请输入你想问的问题，点击发送开始对话</div>
        <div v-for="(item, idx) in messages" :key="idx" :class="['chat-row', item.role === 'user' ? 'chat-row-user' : 'chat-row-assistant']">
          <div :class="[
            'chat-bubble',
            item.role === 'user'
              ? 'chat-bubble-user'
              : item.sourceType === 1
                ? 'chat-bubble-extension'
                : 'chat-bubble-rag',
          ]">
            <div
              v-if="item.role === 'assistant'"
              class="assistant-rich"
              v-html="renderAssistantHtml(item.content)"
            ></div>
            <div v-else>{{ item.content }}</div>
            <div v-if="item.references?.length" class="references">引用：{{ item.references.join(' / ') }}</div>
          </div>
        </div>
      </div>

      <div class="send-row">
        <input
          v-model="question"
          class="question-input"
          :disabled="sending"
          placeholder="请输入问题"
          @keyup.enter="sendQuestion"
        />
        <button class="primary-btn" type="button" :disabled="sending" @click="sendQuestion">
          {{ sending ? '发送中...' : '发送' }}
        </button>
      </div>
    </MoocShell>
  </div>
</template>

<style scoped>
.side-list {
  color: #33485d;
  display: flex;
  flex-direction: column;
  font-size: 13px;
  gap: 8px;
}

.toolbar-row {
  margin-bottom: 10px;
}

.course-select {
  border: 1px solid #d6e0e8;
  border-radius: 10px;
  font-size: 14px;
  min-width: 320px;
  padding: 8px 10px;
}

.chat-board {
  background: linear-gradient(180deg, #f8fbfc 0%, #f2f7f6 100%);
  border: 1px solid #e6eeee;
  border-radius: 12px;
  margin-bottom: 10px;
  max-height: 500px;
  min-height: 320px;
  overflow-y: auto;
  padding: 14px;
}

.placeholder {
  color: #6f7c8a;
  font-size: 14px;
  padding: 8px 0;
}

.chat-row {
  display: flex;
  margin-bottom: 10px;
}

.chat-row-user {
  justify-content: flex-end;
}

.chat-row-assistant {
  justify-content: flex-start;
}

.chat-bubble {
  border-radius: 12px;
  font-size: 14px;
  line-height: 1.5;
  max-width: 80%;
  padding: 10px 12px;
  white-space: pre-wrap;
}

.chat-bubble-user {
  background: linear-gradient(135deg, #18c39b 0%, #12ad88 100%);
  color: #fff;
}

.chat-bubble-rag {
  background: #fff;
  border: 1px solid #deebea;
  color: #23354a;
}

.chat-bubble-extension {
  background: #fff7e4;
  border: 1px solid #f3dfab;
  color: #855f18;
}

.assistant-rich {
  color: inherit;
  line-height: 1.7;
  white-space: normal;
}

.assistant-rich :deep(.qa-paragraph) {
  margin: 0 0 6px;
}

.assistant-rich :deep(.qa-heading) {
  font-size: 15px;
  font-weight: 700;
  margin: 8px 0 6px;
}

.assistant-rich :deep(.qa-number) {
  font-size: 15px;
  font-weight: 700;
  margin: 10px 0 4px;
}

.assistant-rich :deep(.qa-list) {
  list-style: disc;
  margin: 0 0 8px 22px;
  padding: 0;
}

.assistant-rich :deep(.qa-list li) {
  margin-bottom: 4px;
}

.assistant-rich :deep(strong) {
  color: #0f3350;
  font-weight: 700;
}

.assistant-rich :deep(.qa-gap) {
  height: 6px;
}

.references {
  color: #6b7f92;
  font-size: 12px;
  margin-top: 6px;
}

.send-row {
  display: grid;
  gap: 10px;
  grid-template-columns: 1fr auto;
}

.question-input {
  border: 1px solid #d6e0e8;
  border-radius: 10px;
  font-size: 14px;
  padding: 10px 12px;
}

.question-input:focus {
  border-color: #12b886;
  outline: none;
}

.primary-btn,
.plain-btn {
  border-radius: 10px;
  cursor: pointer;
  font-size: 13px;
  padding: 9px 12px;
}

.primary-btn {
  background: linear-gradient(135deg, #14bf8c, #0ea47a);
  border: 0;
  color: #fff;
  font-weight: 600;
}

.plain-btn {
  background: #fff;
  border: 1px solid #d8e4e9;
  color: #27506a;
}

.primary-btn:disabled,
.plain-btn:disabled {
  cursor: not-allowed;
  opacity: 0.6;
}

@media (max-width: 980px) {
  .course-select {
    min-width: 100%;
  }

  .send-row {
    grid-template-columns: 1fr;
  }
}
</style>
