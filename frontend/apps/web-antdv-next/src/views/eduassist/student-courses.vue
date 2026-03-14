<script lang="ts" setup>
import { computed, onMounted, ref } from 'vue';
import { useRouter } from 'vue-router';

import { message } from 'antdv-next';

import { listCourses, type Course } from '#/api/eduassist';

import MoocShell from './components/mooc-shell.vue';
import { useEduAssistContext } from './use-eduassist-context';

const router = useRouter();
const { setStudentCourse } = useEduAssistContext();

const loading = ref(false);
const courses = ref<Course[]>([]);
const joinedExpanded = ref(true);
const unjoinedExpanded = ref(true);

const joinedCourses = computed(() => courses.value.filter((item) => !!item.joined));
const unjoinedCourses = computed(() => courses.value.filter((item) => !item.joined));

async function fetchCourses() {
  loading.value = true;
  try {
    courses.value = await listCourses();
  } catch (error: any) {
    message.error(error?.detail || error?.message || '加载课程失败');
  } finally {
    loading.value = false;
  }
}

async function openCourseDetail(course: Course) {
  setStudentCourse(course.id);
  await router.push(`/eduassist/student/course/${course.id}`);
}

function toggleJoined() {
  joinedExpanded.value = !joinedExpanded.value;
}

function toggleUnjoined() {
  unjoinedExpanded.value = !unjoinedExpanded.value;
}

onMounted(() => {
  void fetchCourses();
});
</script>

<template>
  <div class="p-5">
    <MoocShell title="课程">
      <template #toolbar>
        <span class="pill">全部课程数 {{ courses.length }}</span>
      </template>

      <div v-if="loading" class="loading-text">正在加载课程...</div>
      <div v-else-if="courses.length === 0" class="empty-block">暂无可用课程</div>
      <div v-else class="group-list">
        <section class="group-card">
          <button class="group-head" type="button" @click="toggleJoined">
            <span class="group-title">已加入课程（{{ joinedCourses.length }}）</span>
            <span class="group-arrow">{{ joinedExpanded ? '收起' : '展开' }}</span>
          </button>
          <div v-if="joinedExpanded" class="group-body">
            <div v-if="joinedCourses.length === 0" class="group-empty">暂无已加入课程</div>
            <div v-else class="course-grid">
              <article v-for="course in joinedCourses" :key="course.id" class="course-card">
                <div class="card-head">
                  <h4>{{ course.name }}</h4>
                  <span class="status-tag status-tag-joined">已加入</span>
                </div>
                <p class="desc">{{ course.description || '暂无课程简介' }}</p>
                <div class="meta-row">
                  <span>学生 {{ course.student_count ?? 0 }}</span>
                  <span>资料 {{ course.material_count ?? 0 }}</span>
                </div>
                <button class="primary-btn" type="button" @click="openCourseDetail(course)">
                  查看课程详情
                </button>
              </article>
            </div>
          </div>
        </section>

        <section class="group-card">
          <button class="group-head" type="button" @click="toggleUnjoined">
            <span class="group-title">未加入课程（{{ unjoinedCourses.length }}）</span>
            <span class="group-arrow">{{ unjoinedExpanded ? '收起' : '展开' }}</span>
          </button>
          <div v-if="unjoinedExpanded" class="group-body">
            <div v-if="unjoinedCourses.length === 0" class="group-empty">暂无未加入课程</div>
            <div v-else class="course-grid">
              <article v-for="course in unjoinedCourses" :key="course.id" class="course-card">
                <div class="card-head">
                  <h4>{{ course.name }}</h4>
                  <span class="status-tag status-tag-unjoined">未加入</span>
                </div>
                <p class="desc">{{ course.description || '暂无课程简介' }}</p>
                <div class="meta-row">
                  <span>学生 {{ course.student_count ?? 0 }}</span>
                  <span>资料 {{ course.material_count ?? 0 }}</span>
                </div>
                <button class="primary-btn" type="button" @click="openCourseDetail(course)">
                  查看课程详情
                </button>
              </article>
            </div>
          </div>
        </section>
      </div>
    </MoocShell>
  </div>
</template>

<style scoped>
.pill {
  background: rgba(18, 184, 134, 0.15);
  border-radius: 999px;
  color: #107156;
  font-size: 12px;
  padding: 4px 10px;
}

.loading-text,
.empty-block {
  color: #64748b;
  font-size: 14px;
  padding: 14px 0;
  text-align: center;
}

.group-list {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.group-card {
  border: 1px solid #e5eaf0;
  border-radius: 12px;
  overflow: hidden;
}

.group-head {
  align-items: center;
  background: #f8fbff;
  border: 0;
  color: #213549;
  cursor: pointer;
  display: flex;
  justify-content: space-between;
  padding: 12px 14px;
  width: 100%;
}

.group-title {
  font-size: 16px;
  font-weight: 700;
}

.group-arrow {
  color: #4c6b8b;
  font-size: 13px;
}

.group-body {
  padding: 14px;
}

.group-empty {
  color: #6b7d8f;
  font-size: 13px;
  padding: 8px 0;
}

.course-grid {
  display: grid;
  gap: 14px;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
}

.course-card {
  border: 1px solid #e5eaf0;
  border-radius: 12px;
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 14px;
}

.card-head {
  align-items: center;
  display: flex;
  justify-content: space-between;
}

.card-head h4 {
  color: #1d2b3a;
  font-size: 17px;
  font-weight: 700;
}

.status-tag {
  border-radius: 999px;
  font-size: 12px;
  padding: 3px 8px;
}

.status-tag-joined {
  background: rgba(18, 184, 134, 0.15);
  color: #0d7858;
}

.status-tag-unjoined {
  background: rgba(100, 116, 139, 0.16);
  color: #45586d;
}

.desc {
  color: #5b6c7d;
  font-size: 13px;
  line-height: 1.5;
  min-height: 40px;
}

.meta-row {
  color: #2f4a60;
  display: flex;
  font-size: 13px;
  gap: 14px;
}

.primary-btn {
  align-self: flex-start;
  background: linear-gradient(135deg, #14bf8c, #0ea47a);
  border: 0;
  border-radius: 10px;
  color: #fff;
  cursor: pointer;
  font-size: 13px;
  font-weight: 600;
  padding: 9px 12px;
}
</style>
