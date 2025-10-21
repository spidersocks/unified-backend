import os
import json
import time
from typing import Optional, Dict, List

# Optional: pandas is available in this repo (used by dialogflow_app/data.py)
# We use it to read "Publish to web" CSVs from Google Sheets when configured.
try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # Fallback if pandas is not available

# Optional course display-name map (for nicer localized names)
try:
    # DISPLAY_NAMES = { canonical: {"en": "...","zh-hk":"...","zh-cn":"..."} }
    from dialogflow_app.data import DISPLAY_NAMES  # type: ignore
except Exception:  # pragma: no cover
    DISPLAY_NAMES = {}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INFO_JSON_PATH = os.path.join(BASE_DIR, "data", "little_scholars_info.json")


def _norm_lang(lang_code: str) -> str:
    if not lang_code:
        return "en"
    lc = lang_code.lower()
    if lc.startswith("zh-hk"):
        return "zh-hk"
    if lc.startswith("zh-cn") or lc == "zh":
        return "zh-cn"
    return "en"


def _join_lines(lines: List[str]) -> str:
    return "\n".join([line for line in lines if line and str(line).strip()])


def _format_course_display(canonical: str, lang_code: str) -> str:
    """
    Localized course name with a cross-language alias in parentheses to avoid confusion.
      - For zh requests: zh name (EN name)
      - For en requests: EN name (HK name if available, else CN)
    Falls back to the canonical string if no map is found.
    """
    en = DISPLAY_NAMES.get(canonical, {}).get("en", "") if DISPLAY_NAMES else ""
    zh_hk = DISPLAY_NAMES.get(canonical, {}).get("zh-hk", "") if DISPLAY_NAMES else ""
    zh_cn = DISPLAY_NAMES.get(canonical, {}).get("zh-cn", "") if DISPLAY_NAMES else ""

    lc = _norm_lang(lang_code)
    if lc == "en":
        base = en or canonical
        alt = zh_hk or zh_cn
        if alt and alt != base:
            return f"{base} ({alt})"
        return base
    else:
        primary = zh_hk if lc == "zh-hk" else (zh_cn or zh_hk)
        primary = primary or en or canonical
        alt = en if en and en != primary else ""
        return f"{primary} ({alt})" if alt else primary


# ===========================
# Content store (Google Sheets)
# ===========================

class _Csv:
    def __init__(self, url: str):
        self.url = url
        self.df = pd.DataFrame() if pd is not None else None
        self.loaded = False

    def load(self):
        if pd is None or not self.url:
            return
        try:
            self.df = pd.read_csv(self.url)
            self.loaded = True
        except Exception as e:
            print(f"[WARN] Failed to read CSV: {self.url} error={e}", flush=True)
            self.df = pd.DataFrame()
            self.loaded = False


class ContentStore:
    """
    Loads multi-language business data from a Google Sheet "catalog" CSV, where each row contains:
      section,url
    Each URL is the "Publish to web" CSV URL of a sheet tab with a known schema.

    Set INFO_SHEET_CATALOG_URL env var to enable. If not set or load fails, this store remains inactive.
    """

    def __init__(self, catalog_env_var: str = "INFO_SHEET_CATALOG_URL"):
        self.catalog_env_var = catalog_env_var
        self.catalog_url = os.environ.get(self.catalog_env_var, "")
        self.urls: Dict[str, str] = {}
        self.loaded_at = 0.0

        # Tables
        self.display_names = _Csv("")              # canonical,en,zh-HK,zh-CN
        self.institution = _Csv("")                # key,en,zh-HK,zh-CN
        self.contact = _Csv("")                    # key,en,zh-HK,zh-CN
        self.ages = _Csv("")                       # course_key,en,zh-HK,zh-CN
        self.schedules = _Csv("")                  # course_key,en,zh-HK,zh-CN
        self.class_size_cats = _Csv("")            # category_key,en_label,zh-HK_label,zh-CN_label,en_value,zh-HK_value,zh-CN_value
        self.course_class_size_map = _Csv("")      # course_key,category_key
        self.tuition_groups = _Csv("")             # group_key,en_label,zh-HK_label,zh-CN_label,en_value,zh-HK_value,zh-CN_value
        self.course_tuition_map = _Csv("")         # course_key,group_key
        self.enrollment_steps = _Csv("")           # step_number,en,zh-HK,zh-CN
        self.policy_absence = _Csv("")             # key,en,zh-HK,zh-CN
        self.policy_refund = _Csv("")              # key,en,zh-HK,zh-CN
        self.common_objections = _Csv("")          # key,en,zh-HK,zh-CN
        self.promotions = _Csv("")                 # key,en,zh-HK,zh-CN
        self.success_stories = _Csv("")            # key,en,zh-HK,zh-CN

        self.active = False
        self.reload()

    def _read_catalog(self) -> Optional["pd.DataFrame"]:
        if pd is None or not self.catalog_url:
            return None
        try:
            return pd.read_csv(self.catalog_url)
        except Exception as e:
            print(f"[WARN] Catalog load failed: {e}", flush=True)
            return None

    def reload(self):
        if not self.catalog_url or pd is None:
            print(f"[INFO] {self.catalog_env_var} not set or pandas not available; ContentStore inactive.", flush=True)
            self.active = False
            return
        cat = self._read_catalog()
        if cat is None or "section" not in cat.columns or "url" not in cat.columns:
            print("[WARN] Catalog CSV invalid; ContentStore inactive.", flush=True)
            self.active = False
            return
        self.urls = dict(zip(cat["section"], cat["url"]))

        def bind(csv_obj: _Csv, name: str):
            csv_obj.url = self.urls.get(name, "")
            csv_obj.load()

        bind(self.display_names, "display_names")
        bind(self.institution, "institution")
        bind(self.contact, "contact")
        bind(self.ages, "ages")
        bind(self.schedules, "schedules")
        bind(self.class_size_cats, "class_size_categories")
        bind(self.course_class_size_map, "course_class_size_map")
        bind(self.tuition_groups, "tuition_groups")
        bind(self.course_tuition_map, "course_tuition_map")
        bind(self.enrollment_steps, "enrollment_steps")
        bind(self.policy_absence, "policy_absence")
        bind(self.policy_refund, "policy_refund")
        bind(self.common_objections, "common_objections")
        bind(self.promotions, "promotions")
        bind(self.success_stories, "success_stories")

        self.loaded_at = time.time()
        self.active = True
        print("[INFO] ContentStore loaded from catalog.", flush=True)

    # ----- helpers -----

    @staticmethod
    def _lang_col(lang: str) -> str:
        lc = _norm_lang(lang)
        return "zh-HK" if lc == "zh-hk" else ("zh-CN" if lc == "zh-cn" else "en")

    def _get_display_name(self, canonical: str, lang: str) -> str:
        # Prefer in-sheet display names if present
        if self.display_names.loaded and not self.display_names.df.empty and "canonical" in self.display_names.df.columns:
            df = self.display_names.df
            row = df[df["canonical"] == canonical]
            if not row.empty:
                col = self._lang_col(lang)
                if col in row.columns:
                    val = str(row.iloc[0][col]).strip()
                    if val:
                        return val
        # Fallback to static map
        if DISPLAY_NAMES:
            return DISPLAY_NAMES.get(canonical, {}).get(_norm_lang(lang), canonical)
        return canonical

    def format_course_name(self, canonical: str, lang: str) -> str:
        zh = self._get_display_name(canonical, "zh-HK") or self._get_display_name(canonical, "zh-CN")
        en = self._get_display_name(canonical, "en")
        req = _norm_lang(lang)
        if req == "en":
            alt = zh if zh and zh != en else ""
            return f"{en} ({alt})" if alt else en
        else:
            primary = self._get_display_name(canonical, "zh-HK") if req == "zh-hk" else self._get_display_name(canonical, "zh-CN")
            primary = primary or zh or en or canonical
            alt = en if en and en != primary else ""
            return f"{primary} ({alt})" if alt else primary

    # ----- accessors (single-language outputs) -----

    def institution_intro(self, lang: str) -> Optional[str]:
        if not self.institution.loaded or self.institution.df.empty:
            return None
        df = self.institution.df
        col = self._lang_col(lang)

        def v(key: str) -> str:
            rows = df[df["key"] == key]
            return str(rows.iloc[0][col]).strip() if (not rows.empty and col in rows.columns) else ""

        full = v("FullInstitutionName")
        phil = v("FoundingPhilosophy")
        edu = v("EducationalPhilosophy")

        if _norm_lang(lang) == "zh-hk":
            parts = [
                f"機構名稱：{full}" if full else "",
                "我們的理念：" if phil else "",
                phil,
                "教育理念：" if edu else "",
                edu,
            ]
        elif _norm_lang(lang) == "zh-cn":
            parts = [
                f"机构名称：{full}" if full else "",
                "我们的理念：" if phil else "",
                phil,
                "教育理念：" if edu else "",
                edu,
            ]
        else:
            parts = [
                f"Institution: {full}" if full else "",
                "Founding philosophy:" if phil else "",
                phil,
                "Educational philosophy:" if edu else "",
                edu,
            ]
        return _join_lines(parts) or None

    def contact_info(self, lang: str) -> Optional[str]:
        if not self.contact.loaded or self.contact.df.empty:
            return None
        df = self.contact.df
        col = self._lang_col(lang)

        def rows_for(key: str) -> List[str]:
            rows = df[df["key"] == key]
            out: List[str] = []
            if rows.empty or col not in rows.columns:
                return out
            for _, r in rows.iterrows():
                val = str(r.get(col, "")).strip()
                if val:
                    out.append(val)
            return out

        phones = rows_for("Phone")
        email = rows_for("Email")
        addr = rows_for("Address")
        fb = rows_for("Facebook")
        ig = rows_for("Instagram")
        wa = rows_for("WhatsApp")
        web = rows_for("OfficialWebsite")
        maplnk = rows_for("MapLink")

        if _norm_lang(lang) == "zh-hk":
            parts = [
                "聯絡資料：",
                f"電話：{'; '.join(phones)}" if phones else "",
                f"電郵：{email[0]}" if email else "",
                f"地址：{addr[0]}" if addr else "",
                "社交平台：",
                f"- Facebook：{fb[0]}" if fb else "",
                f"- Instagram：{ig[0]}" if ig else "",
                f"- WhatsApp：{wa[0]}" if wa else "",
                f"- 官方網站：{web[0]}" if web else "",
                f"- 地圖：{maplnk[0]}" if maplnk else "",
            ]
        elif _norm_lang(lang) == "zh-cn":
            parts = [
                "联系资料：",
                f"电话：{'; '.join(phones)}" if phones else "",
                f"电邮：{email[0]}" if email else "",
                f"地址：{addr[0]}" if addr else "",
                "社交平台：",
                f"- Facebook：{fb[0]}" if fb else "",
                f"- Instagram：{ig[0]}" if ig else "",
                f"- WhatsApp：{wa[0]}" if wa else "",
                f"- 官网：{web[0]}" if web else "",
                f"- 地图：{maplnk[0]}" if maplnk else "",
            ]
        else:
            parts = [
                "Contact information:",
                f"Phone: {', '.join(phones)}" if phones else "",
                f"Email: {email[0]}" if email else "",
                f"Address: {addr[0]}" if addr else "",
                "Social:",
                f"- Facebook: {fb[0]}" if fb else "",
                f"- Instagram: {ig[0]}" if ig else "",
                f"- WhatsApp: {wa[0]}" if wa else "",
                f"- Website: {web[0]}" if web else "",
                f"- Map: {maplnk[0]}" if maplnk else "",
            ]
        return _join_lines(parts) or None

    def age_for_course(self, canonical: Optional[str], lang: str) -> Optional[str]:
        if not self.ages.loaded or self.ages.df.empty:
            return None
        df = self.ages.df
        col = self._lang_col(lang)
        if not canonical:
            lines = []
            for _, r in df.iterrows():
                key = str(r.get("course_key", "")).strip()
                val = str(r.get(col, "")).strip()
                if key and val:
                    lines.append(f"- {_format_course_display(key, lang)}: {val}")
            if not lines:
                return None
            title = "各課程年齡：" if _norm_lang(lang) == "zh-hk" else ("各课程年龄：" if _norm_lang(lang) == "zh-cn" else "Target ages:")
            return f"{title}\n" + "\n".join(lines)
        row = df[df["course_key"] == canonical]
        if row.empty or col not in row.columns:
            return None
        val = str(row.iloc[0][col]).strip()
        return f"{_format_course_display(canonical, lang)}: {val}" if val else None

    def schedule_for_course(self, canonical: Optional[str], lang: str) -> Optional[str]:
        if not self.schedules.loaded or self.schedules.df.empty:
            return None
        df = self.schedules.df
        col = self._lang_col(lang)
        if not canonical:
            lines = []
            for _, r in df.iterrows():
                key = str(r.get("course_key", "")).strip()
                val = str(r.get(col, "")).strip()
                if key and val:
                    lines.append(f"- {_format_course_display(key, lang)}: {val}")
            if not lines:
                return None
            title = "上課時間：" if _norm_lang(lang) == "zh-hk" else ("上课时间：" if _norm_lang(lang) == "zh-cn" else "Class schedules:")
            return f"{title}\n" + "\n".join(lines)
        row = df[df["course_key"] == canonical]
        if row.empty or col not in row.columns:
            return None
        val = str(row.iloc[0][col]).strip()
        return f"{_format_course_display(canonical, lang)}: {val}" if val else None

    def class_size_for_course(self, canonical: Optional[str], lang: str) -> Optional[str]:
        if not self.class_size_cats.loaded or self.class_size_cats.df.empty:
            return None
        df_cat = self.class_size_cats.df
        df_map = self.course_class_size_map.df if self.course_class_size_map.loaded else pd.DataFrame()
        req = _norm_lang(lang)

        val_col = "en_value" if req == "en" else ("zh-HK_value" if req == "zh-hk" else "zh-CN_value")
        label_col = "en_label" if req == "en" else ("zh-HK_label" if req == "zh-hk" else "zh-CN_label")

        if canonical and not df_map.empty:
            mrow = df_map[df_map["course_key"] == canonical]
            if not mrow.empty:
                cat_key = str(mrow.iloc[0]["category_key"])
                crow = df_cat[df_cat["category_key"] == cat_key]
                if not crow.empty and val_col in crow.columns:
                    val = str(crow.iloc[0][val_col]).strip()
                    lab = str(crow.iloc[0].get(label_col, "")).strip()
                    label = lab or _format_course_display(canonical, lang)
                    return f"{label}: {val}" if val else None

        # Otherwise list all categories
        lines = []
        for _, r in df_cat.iterrows():
            lab = str(r.get(label_col, "")).strip()
            val = str(r.get(val_col, "")).strip()
            if lab and val:
                lines.append(f"- {lab}: {val}")
        title = "班級人數：" if req == "zh-hk" else ("班级人数：" if req == "zh-cn" else "Class size:")
        return f"{title}\n" + "\n".join(lines) if lines else None

    def _tuition_group_value(self, gkey: str, lang: str) -> Optional[str]:
        if not self.tuition_groups.loaded or self.tuition_groups.df.empty:
            return None
        df = self.tuition_groups.df
        row = df[df["group_key"] == gkey]
        if row.empty:
            return None
        req = _norm_lang(lang)
        val_col = "en_value" if req == "en" else ("zh-HK_value" if req == "zh-hk" else "zh-CN_value")
        lab_col = "en_label" if req == "en" else ("zh-HK_label" if req == "zh-hk" else "zh-CN_label")
        val = str(row.iloc[0].get(val_col, "")).strip()
        lab = str(row.iloc[0].get(lab_col, "")).strip()
        if not val:
            return None
        return f"{lab}: {val}" if lab else val

    def tuition_for_course(self, canonical: Optional[str], lang: str) -> Optional[str]:
        if not self.tuition_groups.loaded or self.tuition_groups.df.empty:
            return None

        # Course → group?
        if canonical and self.course_tuition_map.loaded and not self.course_tuition_map.df.empty:
            m = self.course_tuition_map.df
            row = m[m["course_key"] == canonical]
            if not row.empty:
                gkey = str(row.iloc[0]["group_key"])
                gv = self._tuition_group_value(gkey, lang)
                if gv:
                    return gv

        # Otherwise list all groups
        lines = []
        for _, r in self.tuition_groups.df.iterrows():
            gkey = str(r.get("group_key", "")).strip()
            if not gkey:
                continue
            gv = self._tuition_group_value(gkey, lang)
            if gv:
                lines.append(f"- {gv}")
        if not lines:
            return None

        req = _norm_lang(lang)
        title = "課程收費：" if req == "zh-hk" else ("课程收费：" if req == "zh-cn" else "Tuition:")
        return f"{title}\n" + "\n".join(lines)

    def enrollment_process(self, lang: str) -> Optional[str]:
        if not self.enrollment_steps.loaded or self.enrollment_steps.df.empty:
            return None
        df = self.enrollment_steps.df
        col = self._lang_col(lang)
        if "step_number" in df.columns:
            df_sorted = df.sort_values(by="step_number")
        else:
            df_sorted = df
        steps = []
        for _, r in df_sorted.iterrows():
            txt = str(r.get(col, "")).strip()
            if txt:
                steps.append(f"- {txt}")
        if not steps:
            return None
        req = _norm_lang(lang)
        header = "報名流程：" if req == "zh-hk" else ("报名流程：" if req == "zh-cn" else "Enrollment process:")
        return f"{header}\n" + "\n".join(steps)

    def policy_absence_makeup(self, lang: str) -> Optional[str]:
        if not self.policy_absence.loaded or self.policy_absence.df.empty:
            return None
        df = self.policy_absence.df
        col = self._lang_col(lang)
        order = [
            "NotificationRequirement", "MakeupArrangement", "WrittenNotificationRequirement",
            "MedicalCertificateRequirement", "NoNoticeMissedClasses", "MakeupDeadline",
            "InstructorTimeGuarantee", "FreeMakeupQuota", "ExceedingFreeMakeup",
            "ShortTermAbsence", "LongTermAbsence", "CourseValidity"
        ]
        lines = []
        for k in order:
            rows = df[df["key"] == k]
            if rows.empty or col not in rows.columns:
                continue
            txt = str(rows.iloc[0][col]).strip()
            if txt:
                lines.append(f"- {txt}")
        if not lines:
            return None
        req = _norm_lang(lang)
        title = "補課與請假政策：" if req == "zh-hk" else ("补课与请假政策：" if req == "zh-cn" else "Absence and Make-up policy:")
        return f"{title}\n" + "\n".join(lines)

    def policy_refund(self, lang: str) -> Optional[str]:
        if not self.policy_refund.loaded or self.policy_refund.df.empty:
            return None
        df = self.policy_refund.df
        col = self._lang_col(lang)
        order = ["RefundPolicy", "ContinuationAssumption", "WithdrawalNotice", "ClassTransferInquiry"]
        lines = []
        for k in order:
            rows = df[df["key"] == k]
            if rows.empty or col not in rows.columns:
                continue
            txt = str(rows.iloc[0][col]).strip()
            if txt:
                lines.append(f"- {txt}")
        if not lines:
            return None
        req = _norm_lang(lang)
        title = "退款與轉班政策：" if req == "zh-hk" else ("退款与转班政策：" if req == "zh-cn" else "Refund and transfer policy:")
        return f"{title}\n" + "\n".join(lines)

    def common_obj(self, key: str, lang: str, zh_hk_title: str, zh_cn_title: str, en_title: str) -> Optional[str]:
        if not self.common_objections.loaded or self.common_objections.df.empty:
            return None
        df = self.common_objections.df
        col = self._lang_col(lang)
        rows = df[df["key"] == key]
        if rows.empty or col not in rows.columns:
            return None
        val = str(rows.iloc[0][col]).strip()
        if not val:
            return None
        title = zh_hk_title if _norm_lang(lang) == "zh-hk" else (zh_cn_title if _norm_lang(lang) == "zh-cn" else en_title)
        return f"{title}\n- {val}"

    def promotions(self, lang: str) -> Optional[str]:
        if not self.promotions.loaded or self.promotions.df.empty:
            return None
        col = self._lang_col(lang)
        vals = [str(r.get(col, "")).strip() for _, r in self.promotions.df.iterrows() if str(r.get(col, "")).strip()]
        return "\n".join(vals) if vals else None

    def success_stories(self, lang: str) -> Optional[str]:
        if not self.success_stories.loaded or self.success_stories.df.empty:
            return None
        col = self._lang_col(lang)
        lines = []
        for _, r in self.success_stories.df.iterrows():
            key = str(r.get("key", "")).strip()
            val = str(r.get(col, "")).strip()
            if key and val:
                lines.append(f"- {key}: {val}")
        if not lines:
            return None
        req = _norm_lang(lang)
        title = "成功個案：" if req == "zh-hk" else ("成功个案：" if req == "zh-cn" else "Success stories:")
        return f"{title}\n" + "\n".join(lines)


# Instantiate the content store once (no-op if not configured)
try:
    CONTENT_STORE = ContentStore()
except Exception as e:  # pragma: no cover
    print(f"[WARN] ContentStore init failed: {e}", flush=True)
    CONTENT_STORE = None


# ===========================
# Fallback JSON (existing file)
# ===========================

INFO_DATA: Dict = {}
try:
    with open(INFO_JSON_PATH, "r", encoding="utf-8") as f:
        INFO_DATA = json.load(f)
    print("[INFO] Loaded little_scholars_info.json", flush=True)
except Exception as e:  # pragma: no cover
    print(f"[CRITICAL ERROR] Failed to load little_scholars_info.json: {e}", flush=True)
    INFO_DATA = {}


def get_admin_redirect(lang_code: str) -> str:
    """
    Generic handoff message to staff with contact details.
    Uses ContentStore.contact_info when available to ensure single-language output,
    otherwise falls back to JSON.
    """
    # Prefer single-language contact info if available
    if CONTENT_STORE and CONTENT_STORE.active:
        v = CONTENT_STORE.contact_info(lang_code)
        if v:
            return v

    lang = _norm_lang(lang_code)
    contact = INFO_DATA.get("InstitutionAndBrandCoreInfo", {}).get("ContactInformation", {})
    phones = contact.get("Phone", [])
    email = contact.get("Email", "")
    address = contact.get("Address", "")
    social = contact.get("SocialMedia", {})
    whatsapp = social.get("WhatsApp", "")
    website = social.get("OfficialWebsite", "")

    if lang == "zh-hk":
        msg = [
            "此問題超出機械人可解答範圍，請聯絡我們的職員：",
            f"電話：{'; '.join(phones) if phones else ''}".strip(),
            f"WhatsApp：{whatsapp}".strip() if whatsapp else "",
            f"電郵：{email}".strip() if email else "",
            f"地址：{address}".strip() if address else "",
            f"網站：{website}".strip() if website else "",
        ]
    elif lang == "zh-cn":
        msg = [
            "此问题超出机器人可解答范围，请联系我们的职员：",
            f"电话：{'; '.join(phones) if phones else ''}".strip(),
            f"WhatsApp：{whatsapp}".strip() if whatsapp else "",
            f"电邮：{email}".strip() if email else "",
            f"地址：{address}".strip() if address else "",
            f"网站：{website}".strip() if website else "",
        ]
    else:
        msg = [
            "This question is beyond the bot’s current scope. Please contact our staff:",
            f"Phone: {', '.join(phones) if phones else ''}".strip(),
            f"WhatsApp: {whatsapp}".strip() if whatsapp else "",
            f"Email: {email}".strip() if email else "",
            f"Address: {address}".strip() if address else "",
            f"Website: {website}".strip() if website else "",
        ]
    return _join_lines(msg)


# ===========================
# Mapping for JSON fallback
# ===========================

# Age/Schedule sections use specific course labels such as "Phonics" and "Language Arts".
AGE_SCHEDULE_KEY_MAP = {
    "Playgroups": "Playgroups",
    "Phonics": "Phonics",
    "LanguageArts": "Language Arts",
    "Clevercal": "Clevercal (Math)",
    "Alludio": "Alludio (Games)",
    "ToddlerCharRecognition": "寶寶愛認字(普通話/ 廣東話)",
    "MandarinPinyin": "魔法拼音班",
    "ChineseLanguageArts": "中文語文課(普通話/ 廣東話)",
    "PrivateClass": "Private class 私人課",
}

# Tuition section groups English courses under "English" and Chinese under "中文".
TUITION_KEY_MAP = {
    "Playgroups": "Playgroups",
    "Phonics": "English",
    "LanguageArts": "English",
    "Clevercal": "Clevercal (Math)",
    "Alludio": "Alludio (Games)",
    "ToddlerCharRecognition": "中文",
    "MandarinPinyin": "中文",
    "ChineseLanguageArts": "中文",
    "PrivateClass": "Private class 私人課",
}

# Course → class size category in JSON fallback
CLASS_SIZE_COURSE_TO_KEY = {
    "LanguageArts": "English Language Arts",
    "Phonics": "中英數課程",
    "Clevercal": "中英數課程",
    "ChineseLanguageArts": "中英數課程",
    "MandarinPinyin": "中英數課程",
    "ToddlerCharRecognition": "中英數課程",
    # Playgroups / Alludio / PrivateClass -> fall back to listing
}


# ===========================
# Single entry-point helpers
# ===========================

def _institution_intro(lang: str) -> Optional[str]:
    # Prefer ContentStore (single-language)
    if CONTENT_STORE and CONTENT_STORE.active:
        v = CONTENT_STORE.institution_intro(lang)
        if v:
            return v

    # JSON fallback (may include mixed-language text)
    intro = INFO_DATA.get("InstitutionAndBrandCoreInfo", {}).get("InstitutionIntroduction", {})
    full = intro.get("FullInstitutionName", "")
    phil = intro.get("FoundingPhilosophy", "")
    edu = intro.get("EducationalPhilosophy", "")
    if _norm_lang(lang) == "zh-hk":
        return _join_lines([
            f"機構名稱：{full}" if full else "",
            "我們的理念：" if phil else "",
            phil,
            "教育理念：" if edu else "",
            edu,
        ])
    elif _norm_lang(lang) == "zh-cn":
        return _join_lines([
            f"机构名称：{full}" if full else "",
            "我们的理念：" if phil else "",
            phil,
            "教育理念：" if edu else "",
            edu,
        ])
    else:
        return _join_lines([
            f"Institution: {full}" if full else "",
            "Founding philosophy:" if phil else "",
            phil,
            "Educational philosophy:" if edu else "",
            edu,
        ])


def _contact_info(lang: str) -> Optional[str]:
    if CONTENT_STORE and CONTENT_STORE.active:
        v = CONTENT_STORE.contact_info(lang)
        if v:
            return v

    contact = INFO_DATA.get("InstitutionAndBrandCoreInfo", {}).get("ContactInformation", {})
    phones = contact.get("Phone", [])
    email = contact.get("Email", "")
    address = contact.get("Address", "")
    social = contact.get("SocialMedia", {})

    if _norm_lang(lang) == "zh-hk":
        return _join_lines([
            "聯絡資料：",
            f"電話：{'; '.join(phones)}" if phones else "",
            f"電郵：{email}" if email else "",
            f"地址：{address}" if address else "",
            "社交平台：",
            f"- Facebook：{social.get('Facebook','')}" if social.get("Facebook") else "",
            f"- Instagram：{social.get('Instagram','')}" if social.get("Instagram") else "",
            f"- WhatsApp：{social.get('WhatsApp','')}" if social.get("WhatsApp") else "",
            f"- 官方網站：{social.get('OfficialWebsite','')}" if social.get("OfficialWebsite") else "",
            f"- 地圖：{social.get('MapLink','')}" if social.get("MapLink") else "",
        ])
    elif _norm_lang(lang) == "zh-cn":
        return _join_lines([
            "联系资料：",
            f"电话：{'; '.join(phones)}" if phones else "",
            f"电邮：{email}" if email else "",
            f"地址：{address}" if address else "",
            "社交平台：",
            f"- Facebook：{social.get('Facebook','')}" if social.get("Facebook") else "",
            f"- Instagram：{social.get('Instagram','')}" if social.get("Instagram") else "",
            f"- WhatsApp：{social.get('WhatsApp','')}" if social.get("WhatsApp") else "",
            f"- 官网：{social.get('OfficialWebsite','')}" if social.get("OfficialWebsite") else "",
            f"- 地图：{social.get('MapLink','')}" if social.get("MapLink") else "",
        ])
    else:
        return _join_lines([
            "Contact information:",
            f"Phone: {', '.join(phones)}" if phones else "",
            f"Email: {email}" if email else "",
            f"Address: {address}" if address else "",
            "Social:",
            f"- Facebook: {social.get('Facebook','')}" if social.get("Facebook") else "",
            f"- Instagram: {social.get('Instagram','')}" if social.get("Instagram") else "",
            f"- WhatsApp: {social.get('WhatsApp','')}" if social.get("WhatsApp") else "",
            f"- Website: {social.get('OfficialWebsite','')}" if social.get("OfficialWebsite") else "",
            f"- Map: {social.get('MapLink','')}" if social.get("MapLink") else "",
        ])


def _enrollment_process(lang: str) -> Optional[str]:
    if CONTENT_STORE and CONTENT_STORE.active:
        v = CONTENT_STORE.enrollment_process(lang)
        if v:
            return v

    steps = INFO_DATA.get("CourseDetails", {}).get("EnrollmentProcess", {}).get("Steps", [])
    if not steps:
        return None
    header = "報名流程：" if _norm_lang(lang) == "zh-hk" else ("报名流程：" if _norm_lang(lang) == "zh-cn" else "Enrollment process:")
    bullet = "\n".join([f"- {s}" for s in steps])
    return f"{header}\n{bullet}"


def _course_age(lang: str, coursename: Optional[str]) -> Optional[str]:
    if CONTENT_STORE and CONTENT_STORE.active:
        v = CONTENT_STORE.age_for_course(coursename, lang)
        if v:
            return v

    ages = INFO_DATA.get("CourseDetails", {}).get("TargetStudentAge", {})
    if not ages:
        return None
    if not coursename:
        lines = [f"- {k}: {v}" for k, v in ages.items()]
        title = "各課程年齡：" if _norm_lang(lang) == "zh-hk" else ("各课程年龄：" if _norm_lang(lang) == "zh-cn" else "Target ages:")
        return f"{title}\n" + "\n".join(lines)

    key = AGE_SCHEDULE_KEY_MAP.get(coursename, coursename)
    val = ages.get(key)
    if not val:
        print(f"[WARN] _course_age: no age found for coursename={coursename} resolved_key={key}", flush=True)
        return None
    if _norm_lang(lang) == "zh-hk":
        return f"{key} 適合年齡：{val}"
    elif _norm_lang(lang) == "zh-cn":
        return f"{key} 适合年龄：{val}"
    else:
        return f"{key} target age: {val}"


def _course_schedule(lang: str, coursename: Optional[str]) -> Optional[str]:
    if CONTENT_STORE and CONTENT_STORE.active:
        v = CONTENT_STORE.schedule_for_course(coursename, lang)
        if v:
            return v

    sched = INFO_DATA.get("CourseDetails", {}).get("ClassSchedule", {})
    if not sched:
        return None
    if not coursename:
        lines = [f"- {k}: {v}" for k, v in sched.items()]
        title = "上課時間：" if _norm_lang(lang) == "zh-hk" else ("上课时间：" if _norm_lang(lang) == "zh-cn" else "Class schedules:")
        return f"{title}\n" + "\n".join(lines)

    key = AGE_SCHEDULE_KEY_MAP.get(coursename, coursename)
    val = sched.get(key)
    if not val:
        print(f"[WARN] _course_schedule: no schedule found for coursename={coursename} resolved_key={key}", flush=True)
        return None
    if _norm_lang(lang) == "zh-hk":
        return f"{key} 上課安排：{val}"
    elif _norm_lang(lang) == "zh-cn":
        return f"{key} 上课安排：{val}"
    else:
        return f"{key} schedule: {val}"


def _class_size(lang: str, coursename: Optional[str] = None) -> Optional[str]:
    if CONTENT_STORE and CONTENT_STORE.active:
        v = CONTENT_STORE.class_size_for_course(coursename, lang)
        if v:
            return v

    cs = INFO_DATA.get("CourseDetails", {}).get("ClassSize", {})
    if not cs:
        return None

    if coursename:
        category = CLASS_SIZE_COURSE_TO_KEY.get(coursename)
        if category and cs.get(category):
            val = cs[category]
            if _norm_lang(lang) == "zh-hk":
                return f"{category} 班級人數：{val}"
            elif _norm_lang(lang) == "zh-cn":
                return f"{category} 班级人数：{val}"
            else:
                return f"{category} class size: {val}"

    lines = [f"- {k}: {v}" for k, v in cs.items()]
    title = "班級人數：" if _norm_lang(lang) == "zh-hk" else ("班级人数：" if _norm_lang(lang) == "zh-cn" else "Class size:")
    return f"{title}\n" + "\n".join(lines)


def _tuition(lang: str, coursename: Optional[str]) -> Optional[str]:
    if CONTENT_STORE and CONTENT_STORE.active:
        v = CONTENT_STORE.tuition_for_course(coursename, lang)
        if v:
            # Append localized disclaimer
            req = _norm_lang(lang)
            disclaimer = {
                "en": "Note: Fees are indicative and may change. Please confirm with our staff.",
                "zh-hk": "備註：以上費用僅供參考，或有調整；以中心最新通知為準。",
                "zh-cn": "备注：以上费用仅供参考，或有调整；以中心最新通知为准。",
            }[req]
            return _join_lines([v, disclaimer])

    fees = INFO_DATA.get("CourseDetails", {}).get("TuitionAndPayment", {}).get("GroupClassFee", {})
    if not fees:
        return None

    disclaimer_en = "Note: Fees are indicative and may change. Please confirm with our staff."
    disclaimer_zh_hk = "備註：以上費用僅供參考，或有調整；以中心最新通知為準。"
    disclaimer_zh_cn = "备注：以上费用仅供参考，或有调整；以中心最新通知为准。"

    if coursename:
        key = TUITION_KEY_MAP.get(coursename, coursename)
        val = fees.get(key)
        if val:
            if _norm_lang(lang) == "zh-hk":
                return _join_lines([f"{key} 收費：{val}", disclaimer_zh_hk])
            elif _norm_lang(lang) == "zh-cn":
                return _join_lines([f"{key} 收费：{val}", disclaimer_zh_cn])
            else:
                return _join_lines([f"{key} fee: {val}", disclaimer_en])
        else:
            print(f"[WARN] _tuition: no tuition found for coursename={coursename} resolved_key={key}", flush=True)

    lines = [f"- {k}: {v}" for k, v in fees.items()]
    title = "課程收費：" if _norm_lang(lang) == "zh-hk" else ("课程收费：" if _norm_lang(lang) == "zh-cn" else "Tuition:")
    body = f"{title}\n" + "\n".join(lines)
    if _norm_lang(lang) == "zh-hk":
        return _join_lines([body, disclaimer_zh_hk])
    elif _norm_lang(lang) == "zh-cn":
        return _join_lines([body, disclaimer_zh_cn])
    else:
        return _join_lines([body, disclaimer_en])


def _trial_class(lang: str) -> Optional[str]:
    if CONTENT_STORE and CONTENT_STORE.active:
        # If you prefer to keep trial policy in the sheets, place it under promotions/common_objections or a dedicated tab.
        pass
    txt = INFO_DATA.get("PoliciesAndFAQs", {}).get("TrialClassPolicy", {}).get("Description", "")
    return txt or None


def _absence_makeup_policy(lang: str) -> Optional[str]:
    if CONTENT_STORE and CONTENT_STORE.active:
        v = CONTENT_STORE.policy_absence_makeup(lang)
        if v:
            return v

    pol = INFO_DATA.get("PoliciesAndFAQs", {}).get("AbsenceAndMakeupPolicy", {})
    if not pol:
        return None
    order = [
        "NotificationRequirement", "MakeupArrangement", "WrittenNotificationRequirement",
        "MedicalCertificateRequirement", "NoNoticeMissedClasses", "MakeupDeadline",
        "InstructorTimeGuarantee", "FreeMakeupQuota", "ExceedingFreeMakeup",
        "ShortTermAbsence", "LongTermAbsence", "CourseValidity"
    ]
    lines = [f"- {pol[k]}" for k in order if pol.get(k)]
    title = "補課與請假政策：" if _norm_lang(lang) == "zh-hk" else ("补课与请假政策：" if _norm_lang(lang) == "zh-cn" else "Absence and Make-up policy:")
    return f"{title}\n" + "\n".join(lines)


def _refund_transfer_policy(lang: str) -> Optional[str]:
    if CONTENT_STORE and CONTENT_STORE.active:
        v = CONTENT_STORE.policy_refund(lang)
        if v:
            return v

    pol = INFO_DATA.get("PoliciesAndFAQs", {}).get("RefundAndClassTransferPolicy", {})
    if not pol:
        return None
    order = ["RefundPolicy", "ContinuationAssumption", "WithdrawalNotice", "ClassTransferInquiry"]
    lines = [f"- {pol[k]}" for k in order if pol.get(k)]
    title = "退款與轉班政策：" if _norm_lang(lang) == "zh-hk" else ("退款与转班政策：" if _norm_lang(lang) == "zh-cn" else "Refund and transfer policy:")
    return f"{title}\n" + "\n".join(lines)


def _common_obj(lang: str, key: str, zh_hk_title: str, zh_cn_title: str, en_title: str) -> Optional[str]:
    if CONTENT_STORE and CONTENT_STORE.active:
        v = CONTENT_STORE.common_obj(key, lang, zh_hk_title, zh_cn_title, en_title)
        if v:
            return v

    box = INFO_DATA.get("PoliciesAndFAQs", {}).get("CommonObjectionHandling", {})
    val = box.get(key)
    if not val:
        return None
    title = zh_hk_title if _norm_lang(lang) == "zh-hk" else (zh_cn_title if _norm_lang(lang) == "zh-cn" else en_title)
    return f"{title}\n- {val}"


def _promotions(lang: str) -> Optional[str]:
    if CONTENT_STORE and CONTENT_STORE.active:
        v = CONTENT_STORE.promotions(lang)
        if v:
            return v

    txt = INFO_DATA.get("PoliciesAndFAQs", {}).get("PromotionsAndEvents", {}).get("Description", "")
    if not txt:
        txt = INFO_DATA.get("MarketingAndPromotionInfo", {}).get("PromotionsAndEvents", "")
    return txt or None


def _success_stories(lang: str) -> Optional[str]:
    if CONTENT_STORE and CONTENT_STORE.active:
        v = CONTENT_STORE.success_stories(lang)
        if v:
            return v

    succ = INFO_DATA.get("PoliciesAndFAQs", {}).get("SuccessStories", {})
    if not succ:
        succ = INFO_DATA.get("MarketingAndPromotionInfo", {}).get("SuccessStories", {})
    if not succ:
        return None
    lines = [f"- {k}: {v}" for k, v in succ.items()]
    title = "成功個案：" if _norm_lang(lang) == "zh-hk" else ("成功个案：" if _norm_lang(lang) == "zh-cn" else "Success stories:")
    return f"{title}\n" + "\n".join(lines)


TOPIC_HANDLERS = {
    # Organization and contact
    "InstitutionIntroduction": lambda lang, _: _institution_intro(lang),
    "ContactInformation":      lambda lang, _: _contact_info(lang),
    "SocialMedia":             lambda lang, _: _promotions(lang) if False else _contact_info(lang),  # If you create a dedicated Social tab, swap here
    # Enrollment and logistics
    "EnrollmentProcess":       lambda lang, _: _enrollment_process(lang),
    "ClassSize":               lambda lang, course: _class_size(lang, course),
    # Course-specific
    "TargetStudentAge":        lambda lang, course: _course_age(lang, course),
    "ClassSchedule":           lambda lang, course: _course_schedule(lang, course),
    "TuitionAndPayment":       lambda lang, course: _tuition(lang, course),
    "Tuition":                 lambda lang, course: _tuition(lang, course),
    # Policies & FAQs
    "TrialClassPolicy":        lambda lang, _: _trial_class(lang),
    "AbsenceAndMakeupPolicy":  lambda lang, _: _absence_makeup_policy(lang),
    "RefundAndClassTransferPolicy": lambda lang, _: _refund_transfer_policy(lang),
    "TeacherNativeSpeaker":    lambda lang, _: _common_obj(lang, "TeacherNativeSpeaker",
                                                           "老師是否母語老師？", "老师是否母语老师？", "Are teachers native speakers?"),
    "PublicHolidaySchedule":   lambda lang, _: _common_obj(lang, "PublicHolidaySchedule",
                                                           "公眾假期安排", "公众假期安排", "Public holiday schedule"),
    "WeatherPolicy":           lambda lang, _: _common_obj(lang, "WeatherPolicy",
                                                           "天氣安排", "天气安排", "Weather policy"),
    "OtherCenters":            lambda lang, _: _common_obj(lang, "OtherCenters",
                                                           "有沒有其他分校？", "有没有其他分校？", "Do you have other centers?"),
    "PromotionsAndEvents":     lambda lang, _: _promotions(lang),
    "SuccessStories":          lambda lang, _: _success_stories(lang),
}


def get_info(topic: Optional[str], lang_code: str, coursename: Optional[str] = None) -> Optional[str]:
    """
    Returns an info text for a given topic and optional course name.
      - topic: canonical value from @InfoTopic or fixed mapping
      - coursename: canonical value from @CourseName (optional)
    Will prefer Google Sheet content (single-language outputs) if INFO_SHEET_CATALOG_URL is configured,
    otherwise falls back to bundled JSON (which may contain mixed-language lines).
    """
    if not topic:
        return None
    lang = _norm_lang(lang_code)
    handler = TOPIC_HANDLERS.get(topic)
    if not handler:
        print(f"[WARN] Unknown topic requested: {topic}", flush=True)
        return None
    try:
        return handler(lang, coursename)
    except Exception as e:
        print(f"[ERROR] get_info failed for topic={topic}, course={coursename}: {e}", flush=True)
        return None