<script lang="ts" setup>
import { computed, onMounted, ref, watch } from 'vue';
import { useRoute, useRouter } from 'vue-router';

import { message } from 'antdv-next';

import {
  getCourseDetail,
  listAssignments,
  listCourses,
  listMySubmissions,
  submitAssignment,
  type AssignmentItem,
  type Course,
  type MySubmissionItem,
} from '#/api/eduassist';

import MoocShell from './components/mooc-shell.vue';
import { useEduAssistContext } from './use-eduassist-context';

type AssignmentDraft = {
  code: string;
  content: string;
};

const route = useRoute();
const router = useRouter();
const { setStudentCourse } = useEduAssistContext();

const loading = ref(false);
const submitting = ref(false);
const currentCourse = ref<Course | null>(null);
const joinedCourses = ref<Course[]>([]);
const assignments = ref<AssignmentItem[]>([]);
const mySubmissions = ref<MySubmissionItem[]>([]);

const selectedAssignmentId = ref<null | number>(null);
const selectedCourseId = ref(0);

const submitContent = ref('');
const submitCode = ref('');
const draftMap = ref<Record<string, AssignmentDraft>>({});

const historyPreview = ref<MySubmissionItem | null>(null);

const routeCourseId = computed(() => Number(route.params.courseId || 0));
const queryCourseId = computed(() => Number(route.query.courseId || 0));
const canPickCourse = computed(() => !routeCourseId.value);
const noJoinedCourses = computed(() => joinedCourses.value.length === 0);

const courseId = computed(() => {
  if (routeCourseId.value > 0) return routeCourseId.value;
  if (selectedCourseId.value > 0) return selectedCourseId.value;
  if (queryCourseId.value > 0) return queryCourseId.value;
  return 0;
});

const courseName = computed(() =>
  currentCourse.value ? `当前课程：${currentCourse.value.name}` : '请先选择课程',
);

const activeAssignment = computed(() =>
  assignments.value.find((item) => item.id === selectedAssignmentId.value) || null,
);

const selectedHistory = computed(() => {
  if (!selectedAssignmentId.value) return [];
  return mySubmissions.value.filter((item) => item.assignment_id === selectedAssignmentId.value);
});

function assignmentStats(assignmentId: number) {
  const rows = mySubmissions.value.filter((item) => item.assignment_id === assignmentId);
  const latest = rows.find((item) => item.latest) || null;
  return {
    attempts: rows.length,
    latestScore: latest?.final_score,
    latestState: latest ? '已提交' : '未提交',
  };
}

function draftKey(cid: number, aid: number) {
  return `${cid}:${aid}`;
}

function saveCurrentDraft() {
  if (!courseId.value || !selectedAssignmentId.value) return;
  draftMap.value[draftKey(courseId.value, selectedAssignmentId.value)] = {
    code: submitCode.value,
    content: submitContent.value,
  };
}

function loadDraftFor(assignmentId: null | number) {
  if (!assignmentId || !courseId.value) {
    submitContent.value = '';
    submitCode.value = '';
    return;
  }
  const draft = draftMap.value[draftKey(courseId.value, assignmentId)];
  submitContent.value = draft?.content || '';
  submitCode.value = draft?.code || '';
}

function setActiveAssignment(assignmentId: null | number) {
  if (selectedAssignmentId.value === assignmentId) return;
  saveCurrentDraft();
  selectedAssignmentId.value = assignmentId;
  loadDraftFor(assignmentId);
}

function useAssignment(assignment: AssignmentItem) {
  setActiveAssignment(assignment.id);
}

function syncCourseSelectionFromRoute(forceFromRoute = false) {
  if (routeCourseId.value > 0) {
    selectedCourseId.value = routeCourseId.value;
    return;
  }
  if ((forceFromRoute || !selectedCourseId.value) && queryCourseId.value > 0) {
    selectedCourseId.value = queryCourseId.value;
    return;
  }
  if (!selectedCourseId.value && joinedCourses.value.length > 0) {
    selectedCourseId.value = joinedCourses.value[0]!.id;
  }
}

async function loadJoinedCourses() {
  const all = await listCourses();
  joinedCourses.value = all.filter((item) => !!item.joined);
}

async function syncMenuRouteQuery() {
  if (!canPickCourse.value || !selectedCourseId.value) return false;
  if (queryCourseId.value === selectedCourseId.value) return false;
  await router.replace({
    path: '/eduassist/student/assignments',
    query: { courseId: String(selectedCourseId.value) },
  });
  return true;
}

async function fetchData() {
  loading.value = true;
  try {
    await loadJoinedCourses();
    syncCourseSelectionFromRoute();

    const queryUpdated = await syncMenuRouteQuery();
    if (queryUpdated) return;

    if (!courseId.value) {
      currentCourse.value = null;
      assignments.value = [];
      mySubmissions.value = [];
      setActiveAssignment(null);
      return;
    }

    const [course, list, mine] = await Promise.all([
      getCourseDetail(courseId.value),
      listAssignments(courseId.value),
      listMySubmissions({ courseId: courseId.value }),
    ]);

    currentCourse.value = course;
    assignments.value = list;
    mySubmissions.value = mine;
    setStudentCourse(courseId.value);

    const hasCurrent = list.some((item) => item.id === selectedAssignmentId.value);
    if (!hasCurrent) {
      setActiveAssignment(list[0]?.id || null);
    } else {
      loadDraftFor(selectedAssignmentId.value);
    }
  } catch (error: any) {
    message.error(error?.detail || error?.message || '加载作业信息失败');
  } finally {
    loading.value = false;
  }
}

async function submitCurrentAssignment() {
  if (!activeAssignment.value) {
    message.warning('请先选择作业');
    return;
  }

  const content = submitContent.value.trim();
  const code = submitCode.value.trim();

  if (activeAssignment.value.type === 'code') {
    if (!code) {
      message.warning('代码作业请填写代码内容');
      return;
    }
  } else if (!content) {
    message.warning('文本作业请填写提交内容');
    return;
  }

  submitting.value = true;
  try {
    const resp = await submitAssignment(activeAssignment.value.id, {
      code: code || undefined,
      content: content || undefined,
    });
    message.success(resp?.feedback ? '提交成功，已生成AI评语' : '提交成功');

    saveCurrentDraft();
    await fetchData();
  } catch (error: any) {
    message.error(error?.detail || error?.message || '提交作业失败');
  } finally {
    submitting.value = false;
  }
}

async function handleCourseChange() {
  saveCurrentDraft();
  setActiveAssignment(null);
  const queryUpdated = await syncMenuRouteQuery();
  if (!queryUpdated) {
    await fetchData();
  }
}

async function backToCourse() {
  if (!courseId.value) {
    await router.push('/eduassist/student/courses');
    return;
  }
  await router.push(`/eduassist/student/course/${courseId.value}`);
}

function openHistoryPreview(row: MySubmissionItem) {
  historyPreview.value = row;
}

function closeHistoryPreview() {
  historyPreview.value = null;
}

watch(
  () => [route.params.courseId, route.query.courseId],
  async () => {
    saveCurrentDraft();
    syncCourseSelectionFromRoute(true);
    await fetchData();
  },
);

onMounted(async () => {
  syncCourseSelectionFromRoute(true);
  await fetchData();
});
</script>

<template>
  <div class="p-5">
    <MoocShell
      title="作业中心"
      :course-name="courseName"
    >
      <template #sidebar>
        <ul class="side-list">
          <li>先在左侧选择作业</li>
          <li>支持在截止前多次提交</li>
          <li>下方可查看历史提交与反馈</li>
        </ul>
      </template>

      <template #toolbar>
        <div v-if="canPickCourse" class="toolbar-left">
          <span>课程：</span>
          <select v-model.number="selectedCourseId" class="course-select" @change="handleCourseChange">
            <option :value="0" disabled>请选择课程</option>
            <option v-for="item in joinedCourses" :key="item.id" :value="item.id">
              {{ item.id }} - {{ item.name }}
            </option>
          </select>
        </div>
        <button class="plain-btn" type="button" @click="backToCourse">返回课程详情</button>
        <span class="pill">作业 {{ assignments.length }}</span>
      </template>

      <section v-if="loading" class="state-card">正在加载作业数据...</section>
      <section v-else-if="noJoinedCourses" class="state-card">
        你还没有加入任何课程，请先到“我的课程”加入课程。
      </section>
      <section v-else-if="!courseId" class="state-card">请先选择课程</section>
      <section v-else-if="assignments.length === 0" class="state-card">当前课程暂无作业</section>
      <section v-else class="layout">
        <aside class="assignment-list">
          <article
            v-for="item in assignments"
            :key="item.id"
            :class="['assignment-item', selectedAssignmentId === item.id && 'assignment-item-active']"
            @click="useAssignment(item)"
          >
            <h4>{{ item.title }}</h4>
            <p>{{ item.description || '暂无作业描述' }}</p>
            <div class="meta-row">
              <span>类型：{{ item.type }}</span>
              <span>提交：{{ assignmentStats(item.id).attempts }} 次</span>
              <span>最终分数：{{ assignmentStats(item.id).latestScore ?? '--' }}</span>
            </div>
            <div class="state-tag">{{ assignmentStats(item.id).latestState }}</div>
          </article>
        </aside>

        <div v-if="activeAssignment" class="detail-panel">
          <section class="submit-panel">
            <h3>提交作业：{{ activeAssignment.title }}</h3>
            <p class="panel-tip">{{ activeAssignment.description || '请按课程要求提交作业内容。' }}</p>

            <textarea
              v-if="activeAssignment.type !== 'code'"
              v-model="submitContent"
              class="editor"
              placeholder="请输入文本作业内容..."
            />
            <textarea
              v-else
              v-model="submitCode"
              class="editor"
              placeholder="请输入代码内容..."
            />

            <div class="action-row">
              <button class="submit-btn" type="button" :disabled="submitting" @click="submitCurrentAssignment">
                {{ submitting ? '提交中...' : '提交作业' }}
              </button>
            </div>
          </section>

          <section class="history-panel">
            <h3>我的提交记录</h3>
            <table class="history-table">
              <thead>
                <tr>
                  <th>提交序号</th>
                  <th>提交时间</th>
                  <th>最终分数</th>
                  <th>反馈摘要</th>
                  <th>状态</th>
                  <th>操作</th>
                </tr>
              </thead>
              <tbody>
                <tr v-if="selectedHistory.length === 0">
                  <td colspan="6" class="empty-row">当前作业暂无提交记录</td>
                </tr>
                <tr v-for="row in selectedHistory" :key="row.id">
                  <td>第 {{ row.attempt_no }} 次</td>
                  <td>{{ row.created_at }}</td>
                  <td>{{ row.final_score ?? '--' }}</td>
                  <td>{{ (row.final_feedback || '暂无反馈').slice(0, 30) }}</td>
                  <td>{{ row.latest ? '最新' : '-' }}</td>
                  <td>
                    <button class="link-btn" type="button" @click="openHistoryPreview(row)">查看内容</button>
                  </td>
                </tr>
              </tbody>
            </table>
          </section>
        </div>
      </section>

      <div v-if="historyPreview" class="modal-mask" @click.self="closeHistoryPreview">
        <div class="modal-card">
          <h4>提交详情</h4>
          <p><strong>提交序号：</strong>第 {{ historyPreview.attempt_no }} 次</p>
          <p><strong>提交时间：</strong>{{ historyPreview.created_at }}</p>
          <p><strong>最终分数：</strong>{{ historyPreview.final_score ?? '--' }}</p>
          <p><strong>状态：</strong>{{ historyPreview.latest ? '最新' : '历史记录' }}</p>

          <p><strong>提交内容：</strong></p>
          <pre>{{ historyPreview.content || historyPreview.code || '暂无提交内容' }}</pre>

          <p><strong>AI 评语：</strong></p>
          <pre>{{ historyPreview.ai_feedback || '暂无 AI 评语' }}</pre>

          <p><strong>教师反馈：</strong></p>
          <pre>{{ historyPreview.teacher_feedback || '暂无教师反馈' }}</pre>

          <p><strong>最终反馈：</strong></p>
          <pre>{{ historyPreview.final_feedback || '暂无最终反馈' }}</pre>

          <div class="modal-actions">
            <button class="plain-btn" type="button" @click="closeHistoryPreview">关闭</button>
          </div>
        </div>
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

.toolbar-left {
  align-items: center;
  display: flex;
  font-size: 13px;
  gap: 8px;
}

.course-select {
  background: #fff;
  border: 1px solid #d8e4e9;
  border-radius: 10px;
  color: #27506a;
  cursor: pointer;
  font-size: 13px;
  min-width: 220px;
  padding: 6px 10px;
}

.pill {
  background: rgba(18, 184, 134, 0.15);
  border-radius: 999px;
  color: #107156;
  font-size: 12px;
  padding: 4px 10px;
}

.plain-btn {
  background: #fff;
  border: 1px solid #d8e4e9;
  border-radius: 10px;
  color: #27506a;
  cursor: pointer;
  font-size: 13px;
  padding: 7px 10px;
}

.state-card {
  border: 1px solid #e5eaf0;
  border-radius: 12px;
  color: #64748b;
  font-size: 14px;
  padding: 24px;
  text-align: center;
}

.layout {
  display: grid;
  gap: 12px;
  grid-template-columns: 340px 1fr;
}

.assignment-list {
  border: 1px solid #e5eaf0;
  border-radius: 12px;
  display: flex;
  flex-direction: column;
  gap: 10px;
  max-height: 760px;
  overflow: auto;
  padding: 10px;
}

.assignment-item {
  border: 1px solid #e7edf4;
  border-radius: 10px;
  cursor: pointer;
  padding: 10px;
}

.assignment-item-active {
  border-color: #6ea4fb;
  box-shadow: 0 0 0 2px rgba(32, 116, 255, 0.1);
}

.assignment-item h4 {
  color: #1f2b3d;
  font-size: 16px;
  font-weight: 700;
}

.assignment-item p {
  color: #66778c;
  font-size: 13px;
  margin: 6px 0;
}

.meta-row {
  color: #4b6078;
  display: flex;
  flex-direction: column;
  font-size: 12px;
  gap: 2px;
}

.state-tag {
  background: #edf5ff;
  border-radius: 999px;
  color: #285cae;
  display: inline-block;
  font-size: 12px;
  margin-top: 8px;
  padding: 2px 8px;
}

.detail-panel {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.submit-panel,
.history-panel {
  border: 1px solid #e5eaf0;
  border-radius: 12px;
  padding: 12px;
}

.submit-panel h3,
.history-panel h3 {
  color: #1b2b40;
  font-size: 18px;
  font-weight: 700;
}

.panel-tip {
  color: #617286;
  font-size: 13px;
  margin: 6px 0 10px;
}

.editor {
  border: 1px solid #d8e1ec;
  border-radius: 10px;
  font-size: 14px;
  min-height: 180px;
  padding: 10px;
  resize: vertical;
  width: 100%;
}

.editor:focus {
  border-color: #6ea4fb;
  outline: none;
}

.action-row {
  margin-top: 10px;
}

.submit-btn {
  background: linear-gradient(135deg, #14bf8c, #0ea47a);
  border: 0;
  border-radius: 10px;
  color: #fff;
  cursor: pointer;
  font-size: 13px;
  font-weight: 600;
  padding: 8px 14px;
}

.submit-btn:disabled {
  cursor: not-allowed;
  opacity: 0.6;
}

.history-table {
  border-collapse: collapse;
  margin-top: 8px;
  width: 100%;
}

.history-table th,
.history-table td {
  border-bottom: 1px solid #e7edf4;
  font-size: 13px;
  padding: 8px;
  text-align: left;
}

.history-table th {
  background: #f8fbff;
  color: #425872;
  font-weight: 700;
}

.empty-row {
  color: #66788d;
  text-align: center;
}

.link-btn {
  background: #edf5ff;
  border: 1px solid #d1e6ff;
  border-radius: 8px;
  color: #245fb0;
  cursor: pointer;
  font-size: 12px;
  padding: 3px 8px;
}

.modal-mask {
  align-items: center;
  background: rgba(15, 23, 42, 0.35);
  display: flex;
  inset: 0;
  justify-content: center;
  position: fixed;
  z-index: 2000;
}

.modal-card {
  background: #fff;
  border-radius: 12px;
  max-height: 82vh;
  overflow: auto;
  padding: 16px;
  width: min(760px, 92vw);
}

.modal-card h4 {
  color: #1b2b40;
  font-size: 20px;
  font-weight: 700;
  margin-bottom: 10px;
}

.modal-card p {
  color: #30465e;
  font-size: 14px;
  margin: 6px 0;
}

.modal-card pre {
  background: #f8fbff;
  border: 1px solid #e3ebf5;
  border-radius: 8px;
  color: #27384c;
  font-size: 13px;
  line-height: 1.6;
  margin: 6px 0 10px;
  padding: 10px;
  white-space: pre-wrap;
  word-break: break-word;
}

.modal-actions {
  display: flex;
  justify-content: flex-end;
  margin-top: 10px;
}

@media (max-width: 1100px) {
  .layout {
    grid-template-columns: 1fr;
  }
}
</style>
