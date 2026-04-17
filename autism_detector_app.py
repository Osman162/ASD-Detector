import streamlit as st
import pandas as pd
import joblib
import base64
import datetime

# ─────────────────────────────────────────────
# إعداد الصفحة
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="كشف التوحد | ASD Detector",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Cairo', sans-serif;
        direction: RTL;
        text-align: right;
    }
    .result-box {
        border-radius: 12px;
        padding: 16px 20px;
        margin: 12px 0;
        font-size: 1.05rem;
        font-weight: 600;
    }
    .result-positive { background: #fff0f0; border-right: 5px solid #e53935; color: #b71c1c; }
    .result-negative { background: #f0fff4; border-right: 5px solid #43a047; color: #1b5e20; }
    .section-header {
        background: #f5f7fa;
        border-right: 4px solid #5c6bc0;
        padding: 8px 14px;
        border-radius: 8px;
        margin: 20px 0 10px;
        font-weight: 700;
        color: #3949ab;
    }
    .disclaimer {
        background: #fffde7;
        border: 1px solid #f9a825;
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 0.85rem;
        color: #6d4c00;
        margin-top: 20px;
    }
    div[data-testid="stProgress"] > div { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# تحميل الموديلات
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    lr  = joblib.load("logistic_model.pkl")
    rf  = joblib.load("autism_detector_model.pkl")
    gnb = joblib.load("naive_bayes_model.pkl")
    return lr, rf, gnb

try:
    lr_model, rf_model, gnb_model = load_models()
except FileNotFoundError as e:
    st.error(f"⚠️ ملف الموديل غير موجود: {e.filename}")
    st.stop()

# ─────────────────────────────────────────────
# البيانات الثابتة
# ─────────────────────────────────────────────
BEHAVIORAL_QUESTIONS = [
    'هل يستمتع الطفل بالتأرجح أو التقافز (مثل على الأرجوحة أو عند حمله)؟',
    'هل يبدي الطفل اهتمامًا بالأطفال الآخرين (مثل اللعب معهم أو مشاهدتهم)؟',
    'هل يحب الطفل التسلق على الأشياء (مثل السلالم أو الأثاث)؟',
    'هل يستمتع الطفل بلعبة الغميضة أو الكشف (مثل إخفاء الوجه والظهور)؟',
    'هل يمارس الطفل اللعب التخيلي (مثل التظاهر باستخدام الهاتف أو الدمى)؟',
    'هل يشير الطفل بإصبع السبابة لطلب شيء (مثل لعبة أو طعام)؟',
    'هل يشير الطفل بإصبع السبابة لإظهار الاهتمام (مثل الإشارة إلى طائر)؟',
    'هل يلعب الطفل بشكل صحيح مع الألعاب الصغيرة (مثل تركيب المكعبات)؟',
    'هل يحضر الطفل أشياء ليظهرها لك (مثل لعبة أو كتاب)؟',
    'هل يحافظ الطفل على التواصل البصري معك لأكثر من ثانية في كل مرة؟',
]

AUTISM_ANSWERS = {
    BEHAVIORAL_QUESTIONS[0]: 'نعم',
    BEHAVIORAL_QUESTIONS[1]: 'لا',
    BEHAVIORAL_QUESTIONS[2]: 'نعم',
    BEHAVIORAL_QUESTIONS[3]: 'لا',
    BEHAVIORAL_QUESTIONS[4]: 'لا',
    BEHAVIORAL_QUESTIONS[5]: 'لا',
    BEHAVIORAL_QUESTIONS[6]: 'لا',
    BEHAVIORAL_QUESTIONS[7]: 'لا',
    BEHAVIORAL_QUESTIONS[8]: 'لا',
    BEHAVIORAL_QUESTIONS[9]: 'لا',
}

WEIGHTS = {
    BEHAVIORAL_QUESTIONS[0]: 0.91,
    BEHAVIORAL_QUESTIONS[1]: 0.91,
    BEHAVIORAL_QUESTIONS[2]: 0.91,
    BEHAVIORAL_QUESTIONS[3]: 1.36,
    BEHAVIORAL_QUESTIONS[4]: 0.91,
    BEHAVIORAL_QUESTIONS[5]: 0.91,
    BEHAVIORAL_QUESTIONS[6]: 0.91,
    BEHAVIORAL_QUESTIONS[7]: 0.91,
    BEHAVIORAL_QUESTIONS[8]: 0.91,
    BEHAVIORAL_QUESTIONS[9]: 1.36,
    'هل عانى من الصفراء عند الولادة': 0.01,
    'هل يوجد تاريخ عائلي مع التوحد':  0.01,
    'هل تم استخدام التطبيق من قبل':    0.01,
}

SOCIAL_PLAY      = [BEHAVIORAL_QUESTIONS[1], BEHAVIORAL_QUESTIONS[3], BEHAVIORAL_QUESTIONS[4]]
PHYSICAL_ACTIVITY = [BEHAVIORAL_QUESTIONS[0], BEHAVIORAL_QUESTIONS[2]]
COMMUNICATION    = BEHAVIORAL_QUESTIONS[5:]

GENDER_OPTIONS    = ['اختر...', 'أنثى', 'ذكر']
ETHNICITY_OPTIONS = ['اختر...', *sorted(['آسيوي', 'أفريقي', 'أوروبي', 'تركي', 'جزر المحيط الهادئ',
                                          'جنوب آسيوي', 'عربي', 'غير معروف', 'لاتيني'])]
RESIDENCE_OPTIONS = ['اختر...', *sorted(['أستراليا', 'أفغانستان', 'أوروبا', 'إيطاليا', 'الأردن',
                                          'الأرجنتين', 'الإمارات العربية المتحدة', 'البحرين', 'البرازيل',
                                          'السويد', 'الصين', 'العراق', 'الكويت', 'المكسيك',
                                          'المملكة العربية السعودية', 'المملكة المتحدة', 'النمسا',
                                          'اليابان', 'باكستان', 'بنغلاديش', 'بوتان', 'تركيا',
                                          'جزر الولايات المتحدة النائية', 'جزيرة مان', 'جورجيا',
                                          'روسيا', 'رومانيا', 'سوريا', 'جنوب إفريقيا', 'غانا', 'قطر',
                                          'كندا', 'كوريا الجنوبية', 'كوستاريكا', 'لاتفيا', 'لبنان',
                                          'ليبيا', 'مالطا', 'ماليزيا', 'مصر', 'نيبال', 'نيوزيلندا',
                                          'هولندا', 'الولايات المتحدة', 'عمان', 'الفلبين'])]
RELATION_OPTIONS  = ['اختر...', *sorted(['أخصائي رعاية صحية', 'ذاتي', 'قريب', 'غير معروف', 'والد/والدة'])]

# ─────────────────────────────────────────────
# Helper: حساب الاحتمالية النهائية
# ─────────────────────────────────────────────
def predict(df_input, user_inputs):
    prob_lr  = lr_model.predict_proba(df_input)[0][1]
    prob_rf  = rf_model.predict_proba(df_input)[0][1]
    prob_gnb = gnb_model.predict_proba(df_input)[0][1]

    # Weighted ensemble: RF يأخذ الوزن الأكبر
    final_prob = 0.05 * prob_lr + 0.90 * prob_rf + 0.05 * prob_gnb

    # إذا تطابقت كل الإجابات مع نمط التوحد → رفع الثقة للحد الأقصى
    if all(user_inputs[q] == AUTISM_ANSWERS[q] for q in BEHAVIORAL_QUESTIONS):
        final_prob = min(0.98 + final_prob * 0.02, 1.0)

    return final_prob, int(final_prob > 0.5)


# ─────────────────────────────────────────────
# Helper: بناء DataFrame للإدخال
# ─────────────────────────────────────────────
def build_input_df(user_inputs):
    binary = {'نعم': 1, 'لا': 0}
    data = {q: binary[user_inputs[q]] for q in BEHAVIORAL_QUESTIONS}
    data['العمر']                      = user_inputs['العمر']
    data['الجنس']                      = GENDER_OPTIONS.index(user_inputs['الجنس']) - 1
    data['العرق / الأصل الجغرافي']    = ETHNICITY_OPTIONS.index(user_inputs['العرق / الأصل الجغرافي']) - 1
    data['هل عانى من الصفراء عند الولادة'] = 0
    data['هل يوجد تاريخ عائلي مع التوحد']  = 0
    data['هل تم استخدام التطبيق من قبل']   = 0
    data['بلد الإقامة']                = RESIDENCE_OPTIONS.index(user_inputs['بلد الإقامة']) - 1
    data['العلاقة']                    = RELATION_OPTIONS.index(user_inputs['العلاقة']) - 1
    data['المجموع']                    = sum(data[q] for q in BEHAVIORAL_QUESTIONS)
    return pd.DataFrame([data])


# ─────────────────────────────────────────────
# Helper: حساب نسب الأقسام
# ─────────────────────────────────────────────
def section_scores(user_inputs, final_prob):
    def score(questions):
        return sum(WEIGHTS[q] for q in questions if user_inputs[q] == AUTISM_ANSWERS[q])

    s_social    = score(SOCIAL_PLAY)
    s_physical  = score(PHYSICAL_ACTIVITY)
    s_comm      = score(COMMUNICATION)
    total       = s_social + s_physical + s_comm

    if total == 0:
        return (0, 0, 0), (
            sum(WEIGHTS[q] for q in SOCIAL_PLAY),
            sum(WEIGHTS[q] for q in PHYSICAL_ACTIVITY),
            sum(WEIGHTS[q] for q in COMMUNICATION),
        )

    pct_total = final_prob * 100
    p_social   = (s_social   / total) * pct_total
    p_physical = (s_physical / total) * pct_total
    p_comm     = (s_comm     / total) * pct_total

    # تصحيح الفروق العشرية الصغيرة
    diff = pct_total - (p_social + p_physical + p_comm)
    scores = [p_social, p_physical, p_comm]
    scores[scores.index(max(scores))] += diff
    p_social, p_physical, p_comm = scores

    totals = (
        sum(WEIGHTS[q] for q in SOCIAL_PLAY),
        sum(WEIGHTS[q] for q in PHYSICAL_ACTIVITY),
        sum(WEIGHTS[q] for q in COMMUNICATION),
    )
    return (p_social, p_physical, p_comm), totals, (s_social, s_physical, s_comm)


# ─────────────────────────────────────────────
# Helper: توليد التقرير النصي
# ─────────────────────────────────────────────
def generate_report(user_inputs, final_prob, final_pred, pcts, totals, raw_scores):
    p_social, p_physical, p_comm = pcts
    t_social, t_physical, t_comm = totals
    s_social, s_physical, s_comm = raw_scores

    def answers_block(questions):
        return "\n".join(f"  • {q}: {user_inputs[q]}" for q in questions)

    return f"""
╔══════════════════════════════════════════════╗
        تقرير كشف اضطراب طيف التوحد
╚══════════════════════════════════════════════╝
التاريخ  : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
الاسم    : {user_inputs['الاسم']}
العمر    : {user_inputs['العمر']} سنة
العلاقة  : {user_inputs['العلاقة']}

─────────────────────────────────────────────
النتيجة         : {'⚠️  إيجابي — يُنصح بمراجعة متخصص' if final_pred else '✅  سلبي — لا مؤشرات واضحة'}
نسبة الاحتمال  : {final_prob*100:.2f}%
─────────────────────────────────────────────

=== تحليل السلوكيات ===
توزيع نسبة الاحتمال الكلية ({final_prob*100:.2f}%) على الأقسام:

👫 التفاعل الاجتماعي واللعب   : {p_social:.2f}%  ({s_social:.1f}/{t_social:.1f} درجة)
{answers_block(SOCIAL_PLAY)}

🤸 النشاط الجسدي               : {p_physical:.2f}%  ({s_physical:.1f}/{t_physical:.1f} درجة)
{answers_block(PHYSICAL_ACTIVITY)}

🗣️  التواصل والانخراط           : {p_comm:.2f}%  ({s_comm:.1f}/{t_comm:.1f} درجة)
{answers_block(COMMUNICATION)}

─────────────────────────────────────────────
⚠️  تنبيه: هذه الأداة للدعم الأولي فقط وليست تشخيصاً طبياً.
    يُرجى استشارة طبيب أو أخصائي متخصص للحصول على تشخيص دقيق.
─────────────────────────────────────────────
"""


# ─────────────────────────────────────────────
# الشريط الجانبي
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 أداة كشف التوحد")
    st.markdown("---")
    st.markdown("""
**كيف تعمل الأداة؟**
- تجمع بين **3 موديلات** للتعلم الآلي:
  - Random Forest 🌲 (90%)
  - Logistic Regression 📈 (5%)
  - Naive Bayes 📊 (5%)
- تحلل **10 سلوكيات** رئيسية مرتبطة بالتوحد
- تُولِّد تقريراً قابلاً للتحميل
    """)
    st.markdown("---")
    st.markdown("### 🔗 مصادر مفيدة")
    st.markdown("[🌐 منظمة التوحد العالمية](https://www.autismspeaks.org)")
    st.markdown("[📚 WHO — التوحد](https://www.who.int/news-room/fact-sheets/detail/autism-spectrum-disorders)")
    st.markdown("---")
    st.caption("© 2025 — تطوير: أحمد عثمان")


# ─────────────────────────────────────────────
# الواجهة الرئيسية
# ─────────────────────────────────────────────
st.title("🧠 أداة كشف اضطراب طيف التوحد")
st.markdown(
    '<div class="disclaimer">⚠️ <b>تنبيه مهم:</b> هذه الأداة مساعدة للفحص الأولي فقط، '
    'ولا تُغني عن التشخيص الطبي المتخصص.</div>',
    unsafe_allow_html=True,
)

user_inputs = {}
with st.form("autism_form", clear_on_submit=False):

    # ── المعلومات الأساسية ──
    st.markdown('<div class="section-header">📝 المعلومات الأساسية</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        user_inputs['الاسم'] = st.text_input("اسم الطفل", placeholder="أدخل اسم الطفل")
    with col2:
        user_inputs['العمر'] = st.number_input("العمر (بالسنوات)", min_value=1, step=1, value=None, placeholder="العمر")
    user_inputs['العلاقة'] = st.selectbox("صلة القرابة / العلاقة", RELATION_OPTIONS, index=None)

    # ── التفاعل الاجتماعي ──
    st.markdown('<div class="section-header">👫 التفاعل الاجتماعي واللعب</div>', unsafe_allow_html=True)
    for q in SOCIAL_PLAY:
        user_inputs[q] = st.radio(q, ['نعم', 'لا'], index=None, key=q, horizontal=True)

    # ── النشاط الجسدي ──
    st.markdown('<div class="section-header">🤸‍♂️ النشاط الجسدي</div>', unsafe_allow_html=True)
    for q in PHYSICAL_ACTIVITY:
        user_inputs[q] = st.radio(q, ['نعم', 'لا'], index=None, key=q, horizontal=True)

    # ── التواصل ──
    st.markdown('<div class="section-header">🗣️ التواصل والانخراط</div>', unsafe_allow_html=True)
    for q in COMMUNICATION:
        user_inputs[q] = st.radio(q, ['نعم', 'لا'], index=None, key=q, horizontal=True)

    # ── معلومات إضافية ──
    st.markdown('<div class="section-header">📋 معلومات إضافية</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        user_inputs['الجنس'] = st.selectbox("الجنس", GENDER_OPTIONS, index=None)
    with col4:
        user_inputs['العرق / الأصل الجغرافي'] = st.selectbox("العرق / الأصل الجغرافي", ETHNICITY_OPTIONS, index=None)

    user_inputs['بلد الإقامة'] = st.selectbox("بلد الإقامة", RESIDENCE_OPTIONS, index=None)

    col5, col6, col7 = st.columns(3)
    with col5:
        user_inputs['هل عانى من الصفراء عند الولادة'] = st.radio("الصفراء عند الولادة", ['نعم', 'لا'], index=None, key='جاندس', horizontal=True)
    with col6:
        user_inputs['هل يوجد تاريخ عائلي مع التوحد']  = st.radio("تاريخ عائلي مع التوحد", ['نعم', 'لا'], index=None, key='فام', horizontal=True)
    with col7:
        user_inputs['هل تم استخدام التطبيق من قبل']   = st.radio("استُخدم التطبيق من قبل؟", ['نعم', 'لا'], index=None, key='used', horizontal=True)

    submitted = st.form_submit_button("🔍 تحليل وتوقع", use_container_width=True, type="primary")

# ─────────────────────────────────────────────
# معالجة النتائج
# ─────────────────────────────────────────────
if submitted:
    # التحقق من الإدخالات
    all_q = BEHAVIORAL_QUESTIONS + ['هل عانى من الصفراء عند الولادة',
                                     'هل يوجد تاريخ عائلي مع التوحد',
                                     'هل تم استخدام التطبيق من قبل']
    missing = [q for q in all_q if user_inputs.get(q) not in ['نعم', 'لا']]
    for f in ['الجنس', 'العرق / الأصل الجغرافي', 'بلد الإقامة', 'العلاقة']:
        if user_inputs.get(f) in [None, 'اختر...']:
            missing.append(f)
    if not user_inputs.get('الاسم'):
        missing.append('الاسم')
    if user_inputs.get('العمر') is None:
        missing.append('العمر')

    if missing:
        st.error(f"يرجى إكمال الحقول التالية: {' | '.join(missing)}")
        st.stop()

    # بناء DataFrame والتوقع
    df_input = build_input_df(user_inputs)

    expected_cols = lr_model.feature_names_in_
    missing_cols = [c for c in expected_cols if c not in df_input.columns]
    extra_cols   = [c for c in df_input.columns if c not in expected_cols]
    if missing_cols or extra_cols:
        st.error(f"خطأ في الأعمدة:\n• ناقص: {missing_cols}\n• زيادة: {extra_cols}")
        st.stop()

    df_input = df_input[expected_cols]
    final_prob, final_pred = predict(df_input, user_inputs)

    # ── عرض النتيجة ──
    st.markdown("---")
    st.subheader("📊 نتيجة التحليل")

    if final_pred:
        st.markdown(
            f'<div class="result-box result-positive">⚠️ يوجد احتمال للإصابة بالتوحد — '
            f'النسبة: <b>{final_prob*100:.1f}%</b><br>'
            f'<small>يُنصح بمراجعة طبيب أو أخصائي متخصص لتأكيد التشخيص.</small></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="result-box result-negative">✅ لا توجد مؤشرات واضحة للتوحد — '
            f'النسبة: <b>{final_prob*100:.1f}%</b><br>'
            f'<small>استمر في المتابعة الدورية مع طبيب الأطفال.</small></div>',
            unsafe_allow_html=True,
        )

    st.progress(final_prob, text=f"مستوى الاحتمال: {final_prob*100:.1f}%")

    # ── تفاصيل الأقسام ──
    result = section_scores(user_inputs, final_prob)
    if len(result) == 3:
        pcts, totals, raw_scores = result
    else:
        pcts, totals = result
        raw_scores = (0, 0, 0)

    p_social, p_physical, p_comm = pcts
    t_social, t_physical, t_comm = totals
    s_social, s_physical, s_comm = raw_scores

    st.markdown("#### 🔍 تفاصيل المحاور")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("👫 الاجتماعي واللعب",  f"{p_social:.1f}%",  f"{s_social:.1f}/{t_social:.1f} درجة")
    col_b.metric("🤸 النشاط الجسدي",      f"{p_physical:.1f}%", f"{s_physical:.1f}/{t_physical:.1f} درجة")
    col_c.metric("🗣️ التواصل",            f"{p_comm:.1f}%",   f"{s_comm:.1f}/{t_comm:.1f} درجة")

    # ── التقرير القابل للتحميل ──
    report_text = generate_report(user_inputs, final_prob, final_pred, pcts, totals, raw_scores)
    b64 = base64.b64encode(report_text.encode('utf-8')).decode()
    st.markdown(
        f'<a href="data:text/plain;charset=utf-8;base64,{b64}" download="autism_report_{user_inputs["الاسم"]}.txt">'
        f'📥 تحميل التقرير الكامل</a>',
        unsafe_allow_html=True,
    )
