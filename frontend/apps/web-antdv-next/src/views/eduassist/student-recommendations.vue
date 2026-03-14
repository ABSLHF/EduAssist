<script lang="ts" setup>
import { computed, onMounted, ref } from 'vue';
import { useRoute, useRouter } from 'vue-router';

import { message } from 'antdv-next';

import {
  getRecommendationReport,
  getRecommendations,
  listCourses,
  type Course,
  type LearningReport,
  type RecommendationItem,
} from '#/api/eduassist';

import MoocShell from './components/mooc-shell.vue';
import { useEduAssistContext } from './use-eduassist-context';

const route = useRoute();
const router = useRouter();
const { setStudentCourse, state } = useEduAssistContext();

const loading = ref(false);
const courses = ref<Course[]>([]);
const selectedCourseId = ref<null | number>(null);
const report = ref<LearningReport | null>(null);
const reportError = ref('');
const legacyFallbackUsed = ref(false);
const fallbackItems = ref<RecommendationItem[]>([]);

const activeCourse = computed(() =>
  courses.value.find((item) => item.id === selectedCourseId.value) || null,
);

function priorityLabel(priority: string) {
  if (priority === 'high') return '高优先级';
  if (priority === 'medium') return '中优先级';
  return '建议关注';
}

function sourceLabel(source: string) {
  switch (source) {
    case 'assignment': {
      return '作业';
    }
    case 'qa': {
      return '问答';
    }
    case 'kg': {
      return '图谱连接';
    }
    case 'material': {
      return '资料';
    }
    case 'event': {
      return '行为';
    }
    case 'fallback': {
      return '兜底';
    }
    default: {
      return '画像';
    }
  }
}

function assignmentScoreText(value?: null | number) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '--';
  return Number(value).toFixed(1);
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

function pickCourse() {
  const queryCourse = Number(route.query.courseId || 0);
  const contextCourse = state.studentCourseId || 0;
  const fallbackCourse = courses.value[0]?.id || null;
  const picked = queryCourse || contextCourse || fallbackCourse;
  selectedCourseId.value = picked;
  if (picked) {
    setStudentCourse(picked);
  }
}

async function loadRecommendations() {
  if (!selectedCourseId.value) return;

  loading.value = true;
  report.value = null;
  fallbackItems.value = [];
  legacyFallbackUsed.value = false;
  reportError.value = '';

  try {
    report.value = await getRecommendationReport(selectedCourseId.value);
  } catch (error: any) {
    reportError.value = error?.detail || error?.message || '学习报告加载失败';
    try {
      const fallback = await getRecommendations(selectedCourseId.value);
      fallbackItems.value = fallback.items || [];
      legacyFallbackUsed.value = true;
    } catch (fallbackError: any) {
      message.error(fallbackError?.detail || fallbackError?.message || '加载推荐失败');
    }
  } finally {
    loading.value = false;
  }
}

async function onCourseChange() {
  if (selectedCourseId.value) {
    setStudentCourse(selectedCourseId.value);
  }
  await loadRecommendations();
}

function openCourseMaterials() {
  if (!selectedCourseId.value) return;
  router.push(`/eduassist/student/course/${selectedCourseId.value}`);
}

function reasonTag(reason: string) {
  if (reason.includes('问答')) return '问答依据';
  if (reason.includes('作业')) return '作业依据';
  if (reason.includes('图谱连接') || reason.includes('图谱')) return '图谱依据';
  if (reason.includes('兜底') || reason.includes('通用')) return '兜底推荐';
  return '综合依据';
}

onMounted(async () => {
  await fetchCourses();
  pickCourse();
  await loadRecommendations();
});
</script>

<template>
  <div class="p-5">
    <MoocShell
      title="学习内容推荐"
      :course-hint="'基于学习行为推荐知识点'"
      :course-name="activeCourse ? `当前课程：${activeCourse.name}` : ''"
    >
      <template #sidebar>
        <ul class="side-list">
          <li>先补弱项，再做巩固与拓展</li>
          <li>每条建议都附带证据与资料入口</li>
          <li>可随课程切换实时刷新报告</li>
        </ul>
      </template>

      <template #toolbar>
        <span v-if="legacyFallbackUsed" class="pill pill-fallback">兼容模式</span>
      </template>

      <div class="toolbar-row">
        <select v-model.number="selectedCourseId" class="course-select" @change="onCourseChange">
          <option v-for="course in courses" :key="course.id" :value="course.id">
            {{ course.id }} - {{ course.name }}
          </option>
        </select>
        <button class="plain-btn" type="button" :disabled="loading" @click="loadRecommendations">
          {{ loading ? '刷新中...' : '刷新报告' }}
        </button>
      </div>

      <div v-if="report" class="report-wrap">
        <div class="bucket-grid">
          <section class="bucket-col">
            <h4>必复习</h4>
            <p class="bucket-hint">优先修补薄弱知识点</p>
            <div v-if="report.must_review.length === 0" class="bucket-empty">当前暂无该类建议</div>
            <article v-for="item in report.must_review" :key="`mr-${item.knowledge_point}`" class="bucket-item">
              <header>
                <strong>{{ item.knowledge_point }}</strong>
                <span class="priority priority-high">{{ priorityLabel(item.priority) }}</span>
              </header>
              <p>{{ item.reason }}</p>
              <ul class="evidence-list">
                <li
                  v-for="(e, idx) in item.evidence.filter((entry) => entry.source !== 'kg')"
                  :key="`mre-${idx}`"
                >
                  <span class="e-source">{{ sourceLabel(e.source) }}</span>
                  <span class="e-value">{{ e.value }}</span>
                </li>
              </ul>
              <div class="material-row">
                <span class="material-title">关联资料：</span>
                <span v-if="item.recommended_materials.length === 0">暂无</span>
                <span v-else class="material-items">
                  {{ item.recommended_materials.map((m) => m.title).join('、') }}
                </span>
              </div>
              <div v-if="item.weakness_basis?.length" class="detail-row">
                <span class="detail-title">推荐依据：</span>
                <ul class="detail-list">
                  <li v-for="(basis, idx) in item.weakness_basis" :key="`basis-${idx}`">{{ basis }}</li>
                </ul>
              </div>
              <div v-if="item.related_assignments?.length" class="detail-row">
                <span class="detail-title">相关作业：</span>
                <ul class="detail-list">
                  <li v-for="(ref, idx) in item.related_assignments" :key="`ra-${idx}`">
                    {{ ref.title }}（低分 {{ ref.low_score_count }} 次，最近分数 {{ assignmentScoreText(ref.last_score) }}）
                  </li>
                </ul>
              </div>
              <div v-if="item.recommend_reason" class="detail-row">
                <span class="detail-title">推荐原因：</span>
                <span class="detail-text">{{ item.recommend_reason }}</span>
              </div>
            </article>
          </section>

          <section class="bucket-col">
            <h4>应巩固</h4>
            <p class="bucket-hint">保持稳定并强化核心能力</p>
            <div v-if="report.need_consolidate.length === 0" class="bucket-empty">当前暂无该类建议</div>
            <article
              v-for="item in report.need_consolidate"
              :key="`nc-${item.knowledge_point}`"
              class="bucket-item"
            >
              <header>
                <strong>{{ item.knowledge_point }}</strong>
                <span class="priority">{{ priorityLabel(item.priority) }}</span>
              </header>
              <p>{{ item.reason }}</p>
              <ul class="evidence-list">
                <li
                  v-for="(e, idx) in item.evidence.filter((entry) => entry.source !== 'kg')"
                  :key="`nce-${idx}`"
                >
                  <span class="e-source">{{ sourceLabel(e.source) }}</span>
                  <span class="e-value">{{ e.value }}</span>
                </li>
              </ul>
              <div class="material-row">
                <span class="material-title">关联资料：</span>
                <span v-if="item.recommended_materials.length === 0">暂无</span>
                <span v-else class="material-items">
                  {{ item.recommended_materials.map((m) => m.title).join('、') }}
                </span>
              </div>
              <div v-if="item.weakness_basis?.length" class="detail-row">
                <span class="detail-title">推荐依据：</span>
                <ul class="detail-list">
                  <li v-for="(basis, idx) in item.weakness_basis" :key="`basis-${idx}`">{{ basis }}</li>
                </ul>
              </div>
              <div v-if="item.related_assignments?.length" class="detail-row">
                <span class="detail-title">相关作业：</span>
                <ul class="detail-list">
                  <li v-for="(ref, idx) in item.related_assignments" :key="`ra-${idx}`">
                    {{ ref.title }}（低分 {{ ref.low_score_count }} 次，最近分数 {{ assignmentScoreText(ref.last_score) }}）
                  </li>
                </ul>
              </div>
              <div v-if="item.recommend_reason" class="detail-row">
                <span class="detail-title">推荐原因：</span>
                <span class="detail-text">{{ item.recommend_reason }}</span>
              </div>
            </article>
          </section>

          <section class="bucket-col">
            <h4>可拓展</h4>
            <p class="bucket-hint">在已掌握基础上扩展能力</p>
            <div v-if="report.need_explore.length === 0" class="bucket-empty">当前暂无该类建议</div>
            <article v-for="item in report.need_explore" :key="`ne-${item.knowledge_point}`" class="bucket-item">
              <header>
                <strong>{{ item.knowledge_point }}</strong>
                <span class="priority">{{ priorityLabel(item.priority) }}</span>
              </header>
              <p>{{ item.reason }}</p>
              <ul class="evidence-list">
                <li
                  v-for="(e, idx) in item.evidence.filter((entry) => entry.source !== 'kg')"
                  :key="`nee-${idx}`"
                >
                  <span class="e-source">{{ sourceLabel(e.source) }}</span>
                  <span class="e-value">{{ e.value }}</span>
                </li>
              </ul>
              <div class="material-row">
                <span class="material-title">关联资料：</span>
                <span v-if="item.recommended_materials.length === 0">暂无</span>
                <span v-else class="material-items">
                  {{ item.recommended_materials.map((m) => m.title).join('、') }}
                </span>
              </div>
              <div v-if="item.weakness_basis?.length" class="detail-row">
                <span class="detail-title">推荐依据：</span>
                <ul class="detail-list">
                  <li v-for="(basis, idx) in item.weakness_basis" :key="`basis-${idx}`">{{ basis }}</li>
                </ul>
              </div>
              <div v-if="item.related_assignments?.length" class="detail-row">
                <span class="detail-title">相关作业：</span>
                <ul class="detail-list">
                  <li v-for="(ref, idx) in item.related_assignments" :key="`ra-${idx}`">
                    {{ ref.title }}（低分 {{ ref.low_score_count }} 次，最近分数 {{ assignmentScoreText(ref.last_score) }}）
                  </li>
                </ul>
              </div>
              <div v-if="item.recommend_reason" class="detail-row">
                <span class="detail-title">推荐原因：</span>
                <span class="detail-text">{{ item.recommend_reason }}</span>
              </div>
            </article>
          </section>
        </div>

        <div class="go-materials">
          <button class="plain-btn" type="button" @click="openCourseMaterials">前往课程资料</button>
        </div>
      </div>

      <div v-else-if="legacyFallbackUsed" class="legacy-wrap">
        <p class="legacy-tip">
          学习报告暂不可用，已自动切换到兼容推荐列表。
          <span v-if="reportError">原因：{{ reportError }}</span>
        </p>
        <div v-if="fallbackItems.length === 0" class="empty-block">当前暂无推荐结果</div>
        <div v-else class="recommend-grid">
          <article
            v-for="(item, index) in fallbackItems"
            :key="`${item.knowledge_point}-${index}`"
            class="recommend-card"
          >
            <h4>{{ item.knowledge_point }}</h4>
            <div class="reason-tag">{{ reasonTag(item.reason) }}</div>
            <p>{{ item.reason }}</p>
          </article>
        </div>
      </div>

      <div v-else class="empty-block">当前暂无可展示的学习推荐</div>
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

.pill {
  background: rgba(18, 184, 134, 0.15);
  border-radius: 999px;
  color: #107156;
  font-size: 12px;
  padding: 4px 10px;
}

.pill-fallback {
  background: #fff1f0;
  color: #b4472b;
}

.toolbar-row {
  align-items: center;
  display: flex;
  gap: 10px;
  margin-bottom: 12px;
}

.course-select {
  border: 1px solid #d6e0e8;
  border-radius: 10px;
  font-size: 14px;
  min-width: 320px;
  padding: 8px 10px;
}

.plain-btn {
  background: #fff;
  border: 1px solid #d8e4e9;
  border-radius: 10px;
  color: #27506a;
  cursor: pointer;
  font-size: 13px;
  padding: 9px 12px;
}

.plain-btn:disabled {
  cursor: not-allowed;
  opacity: 0.6;
}

.report-wrap {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.summary-card {
  background: linear-gradient(180deg, #ffffff 0%, #f8fcfb 100%);
  border: 1px solid #deebea;
  border-radius: 12px;
  padding: 14px;
}

.summary-grid {
  display: grid;
  gap: 12px;
  grid-template-columns: repeat(4, minmax(0, 1fr));
}

.summary-label {
  color: #688198;
  font-size: 12px;
  margin-bottom: 6px;
}

.summary-value {
  color: #1f2a37;
  font-size: 15px;
  font-weight: 700;
}

.summary-note {
  background: #fff7e4;
  border-radius: 8px;
  color: #8a5a12;
  font-size: 12px;
  margin-top: 12px;
  padding: 8px 10px;
}

.bucket-grid {
  display: grid;
  gap: 12px;
  grid-template-columns: repeat(3, minmax(0, 1fr));
}

.bucket-col {
  background: #fff;
  border: 1px solid #deebea;
  border-radius: 12px;
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 12px;
}

.bucket-col h4 {
  color: #1f2a37;
  font-size: 16px;
  font-weight: 700;
  margin: 0;
}

.bucket-hint {
  color: #688198;
  font-size: 12px;
  margin: 0;
}

.bucket-empty {
  color: #64748b;
  font-size: 13px;
  padding: 8px 0;
}

.bucket-item {
  background: #f8fcfb;
  border: 1px solid #e4eeec;
  border-radius: 10px;
  padding: 10px;
}

.bucket-item header {
  align-items: center;
  display: flex;
  justify-content: space-between;
}

.bucket-item strong {
  color: #1f2a37;
  font-size: 14px;
}

.priority {
  background: #eef4ff;
  border-radius: 999px;
  color: #2758ad;
  font-size: 11px;
  padding: 2px 8px;
}

.priority-high {
  background: #fff1f0;
  color: #c23b22;
}

.bucket-item p {
  color: #4f6479;
  font-size: 13px;
  line-height: 1.5;
  margin: 8px 0;
}

.evidence-list {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin: 10px 0 0;
  padding: 0;
}

.evidence-list li {
  align-items: flex-start;
  display: flex;
  gap: 6px;
  list-style: none;
}

.e-source {
  background: #e8f4ff;
  border-radius: 999px;
  color: #255cae;
  flex-shrink: 0;
  font-size: 11px;
  padding: 1px 8px;
}

.e-value {
  color: #4d6176;
  font-size: 12px;
}

.material-row {
  border-top: 1px dashed #d8e8e5;
  color: #4d6176;
  font-size: 12px;
  margin-top: 8px;
  padding-top: 8px;
}

.material-title {
  color: #2d4e6a;
  font-weight: 600;
}

.material-items {
  color: #506a83;
}

.detail-row {
  border-top: 1px dashed #d8e8e5;
  color: #4d6176;
  font-size: 12px;
  margin-top: 8px;
  padding-top: 8px;
}

.detail-title {
  color: #2d4e6a;
  display: inline-block;
  font-weight: 600;
  margin-bottom: 4px;
}

.detail-list {
  margin: 4px 0 0;
  padding-left: 18px;
}

.detail-list li {
  line-height: 1.5;
  margin: 2px 0;
}

.detail-text {
  color: #4d6176;
  display: block;
  line-height: 1.5;
}

.go-materials {
  display: flex;
  justify-content: flex-end;
}

.legacy-tip {
  background: #fff7e4;
  border-radius: 10px;
  color: #8a5a12;
  font-size: 13px;
  margin-bottom: 12px;
  padding: 10px;
}

.empty-block {
  color: #64748b;
  font-size: 14px;
  padding: 18px 0;
  text-align: center;
}

.recommend-grid {
  display: grid;
  gap: 12px;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
}

.recommend-card {
  background: linear-gradient(180deg, #ffffff 0%, #f8fcfb 100%);
  border: 1px solid #deebea;
  border-radius: 12px;
  min-height: 126px;
  padding: 12px;
}

.recommend-card h4 {
  color: #1f2a37;
  font-size: 16px;
  font-weight: 700;
}

.recommend-card p {
  color: #607185;
  font-size: 13px;
  line-height: 1.5;
  margin: 8px 0 10px;
}

.reason-tag {
  background: #eef4ff;
  border-radius: 999px;
  color: #2758ad;
  display: inline-block;
  font-size: 12px;
  margin-top: 8px;
  padding: 2px 8px;
}

@media (max-width: 1300px) {
  .summary-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .bucket-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 980px) {
  .toolbar-row {
    flex-direction: column;
  }

  .course-select {
    min-width: 100%;
    width: 100%;
  }

  .summary-grid,
  .bucket-grid {
    grid-template-columns: 1fr;
  }
}
</style>


