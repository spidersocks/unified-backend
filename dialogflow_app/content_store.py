import os
import time
import pandas as pd
from typing import Dict, Optional, List

LANG_EN = "en"
LANG_ZH_HK = "zh-HK"
LANG_ZH_CN = "zh-CN"

def norm_lang(lang_code: str) -> str:
    if not lang_code:
        return LANG_EN
    lc = lang_code.lower()
    if lc.startswith("zh-hk"):
        return LANG_ZH_HK
    if lc.startswith("zh-cn") or lc == "zh":
        return LANG_ZH_CN
    return LANG_EN

def _read_csv(url: str) -> Optional[pd.DataFrame]:
    if not url:
        return None
    try:
        return pd.read_csv(url)
    except Exception as e:
        print(f"[WARN] Failed to read CSV: {url} error={e}", flush=True)
        return None

class ContentStore:
    """
    Loads multi-language business data from a Google Sheet 'catalog' CSV, where each row contains:
      section,url

    Each 'url' is the published-to-web CSV URL for a sheet tab with a known schema.

    This store returns language-specific strings (no mixed-language blobs).
    """

    def __init__(self, catalog_env_var: str = "INFO_SHEET_CATALOG_URL"):
        self.catalog_env_var = catalog_env_var
        self.catalog_url = os.environ.get(self.catalog_env_var, "")
        self.urls: Dict[str, str] = {}
        self.loaded_at = 0.0

        # Data containers
        self.display_names = pd.DataFrame()           # canonical,en,zh-HK,zh-CN
        self.institution = pd.DataFrame()             # key,en,zh-HK,zh-CN
        self.contact = pd.DataFrame()                 # key,en,zh-HK,zh-CN (Phone may be multiple rows)
        self.ages = pd.DataFrame()                    # course_key,en,zh-HK,zh-CN
        self.schedules = pd.DataFrame()               # course_key,en,zh-HK,zh-CN
        self.class_size_cats = pd.DataFrame()         # category_key,en_label,zh-HK_label,zh-CN_label,en_value,zh-HK_value,zh-CN_value
        self.course_class_size_map = pd.DataFrame()   # course_key,category_key
        self.tuition_groups = pd.DataFrame()          # group_key,en_label,zh-HK_label,zh-CN_label,en_value,zh-HK_value,zh-CN_value
        self.course_tuition_map = pd.DataFrame()      # course_key,group_key
        self.enrollment_steps = pd.DataFrame()        # step_number,en,zh-HK,zh-CN
        self.policy_absence = pd.DataFrame()          # key,en,zh-HK,zh-CN
        self.policy_refund = pd.DataFrame()           # key,en,zh-HK,zh-CN
        self.common_objections = pd.DataFrame()       # key,en,zh-HK,zh-CN
        self.promotions = pd.DataFrame()              # key,en,zh-HK,zh-CN (usually single row)
        self.success_stories = pd.DataFrame()         # key,en,zh-HK,zh-CN (or counts in en,zh-HK,zh-CN)

        self.reload()

    def reload(self):
        if not self.catalog_url:
            print(f"[INFO] {self.catalog_env_var} not set; ContentStore disabled.", flush=True)
            return
        cat = _read_csv(self.catalog_url)
        if cat is None or "section" not in cat.columns or "url" not in cat.columns:
            print("[WARN] Catalog CSV missing or invalid; ContentStore disabled.", flush=True)
            return
        self.urls = dict(zip(cat["section"], cat["url"]))

        def load(section: str) -> pd.DataFrame:
            url = self.urls.get(section, "")
            df = _read_csv(url)
            return df if df is not None else pd.DataFrame()

        self.display_names = load("display_names")
        self.institution = load("institution")
        self.contact = load("contact")
        self.ages = load("ages")
        self.schedules = load("schedules")
        self.class_size_cats = load("class_size_categories")
        self.course_class_size_map = load("course_class_size_map")
        self.tuition_groups = load("tuition_groups")
        self.course_tuition_map = load("course_tuition_map")
        self.enrollment_steps = load("enrollment_steps")
        self.policy_absence = load("policy_absence")
        self.policy_refund = load("policy_refund")
        self.common_objections = load("common_objections")
        self.promotions = load("promotions")
        self.success_stories = load("success_stories")

        self.loaded_at = time.time()
        print("[INFO] ContentStore loaded from catalog.", flush=True)

    # ========== Helpers ==========

    def _lang_col(self, lang: str) -> str:
        L = norm_lang(lang)
        return L

    def _get_display_name(self, canonical: str, lang: str) -> str:
        if self.display_names.empty or "canonical" not in self.display_names.columns:
            return canonical
        row = self.display_names[self.display_names["canonical"] == canonical]
        if row.empty:
            return canonical
        col = self._lang_col(lang)
        if col not in row.columns:
            return canonical
        val = str(row.iloc[0].get(col, "")).strip()
        return val or canonical

    def format_course_name(self, canonical: str, lang: str) -> str:
        """
        Returns localized course name and includes a recognizable alias in parentheses
        for clarity across languages:
          - If lang is zh-* => zh name (EN name)
          - If lang is en   => EN name (HK name if available, else CN)
        """
        zh = self._get_display_name(canonical, LANG_ZH_HK) or self._get_display_name(canonical, LANG_ZH_CN)
        en = self._get_display_name(canonical, LANG_EN)
        req = norm_lang(lang)
        if req == LANG_EN:
            alt = zh if zh and zh != en else ""
            return f"{en} ({alt})" if alt else en
        else:
            alt = en if en and en != zh else ""
            primary = zh if req == LANG_ZH_HK else self._get_display_name(canonical, LANG_ZH_CN)
            primary = primary or zh or en or canonical
            return f"{primary} ({alt})" if alt else primary

    # ========== Accessors returning single-language text ==========

    def institution_intro(self, lang: str) -> Optional[str]:
        if self.institution.empty:
            return None
        col = self._lang_col(lang)
        def v(key: str) -> str:
            rows = self.institution[self.institution["key"] == key]
            if rows.empty or col not in rows.columns:
                return ""
            return str(rows.iloc[0][col]).strip()
        full = v("FullInstitutionName")
        phil = v("FoundingPhilosophy")
        edu  = v("EducationalPhilosophy")
        if lang.startswith("zh-hk"):
            parts = [
                f"機構名稱：{full}" if full else "",
                "我們的理念：" if phil else "",
                phil,
                "教育理念：" if edu else "",
                edu
            ]
        elif lang.startswith("zh-cn") or lang == "zh":
            parts = [
                f"机构名称：{full}" if full else "",
                "我们的理念：" if phil else "",
                phil,
                "教育理念：" if edu else "",
                edu
            ]
        else:
            parts = [
                f"Institution: {full}" if full else "",
                "Founding philosophy:" if phil else "",
                phil,
                "Educational philosophy:" if edu else "",
                edu
            ]
        lines = [p for p in parts if p]
        return "\n".join(lines) if lines else None

    def contact_info(self, lang: str) -> Optional[str]:
        if self.contact.empty:
            return None
        col = self._lang_col(lang)
        # multiple rows allowed for Phone; Social keys too
        def rows_for(key: str) -> List[str]:
            df = self.contact[self.contact["key"] == key]
            if df.empty:
                return []
            vals = []
            for _, r in df.iterrows():
                if col in r and pd.notna(r[col]) and str(r[col]).strip():
                    vals.append(str(r[col]).strip())
            return vals

        phones = rows_for("Phone")
        email  = rows_for("Email")
        addr   = rows_for("Address")
        fb     = rows_for("Facebook")
        ig     = rows_for("Instagram")
        wa     = rows_for("WhatsApp")
        web    = rows_for("OfficialWebsite")
        maplnk = rows_for("MapLink")

        if lang.startswith("zh-hk"):
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
        elif lang.startswith("zh-cn") or lang == "zh":
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
        return "\n".join([p for p in parts if p])

    def age_for_course(self, canonical: Optional[str], lang: str) -> Optional[str]:
        if self.ages.empty:
            return None
        col = self._lang_col(lang)
        if not canonical:
            # List all
            lines = []
            for _, r in self.ages.iterrows():
                ck = str(r["course_key"])
                v = str(r.get(col, "")).strip()
                if v:
                    lines.append(f"- {self.format_course_name(ck, lang)}: {v}")
            title = "各課程年齡：" if lang.startswith("zh-hk") else ("各课程年龄：" if lang.startswith("zh-cn") else "Target ages:")
            return f"{title}\n" + "\n".join(lines)
        df = self.ages[self.ages["course_key"] == canonical]
        if df.empty or col not in df.columns:
            return None
        v = str(df.iloc[0][col]).strip()
        if not v:
            return None
        return f"{self.format_course_name(canonical, lang)}: {v}"

    def schedule_for_course(self, canonical: Optional[str], lang: str) -> Optional[str]:
        if self.schedules.empty:
            return None
        col = self._lang_col(lang)
        if not canonical:
            lines = []
            for _, r in self.schedules.iterrows():
                ck = str(r["course_key"])
                v = str(r.get(col, "")).strip()
                if v:
                    lines.append(f"- {self.format_course_name(ck, lang)}: {v}")
            title = "上課時間：" if lang.startswith("zh-hk") else ("上课时间：" if lang.startswith("zh-cn") else "Class schedules:")
            return f"{title}\n" + "\n".join(lines)
        df = self.schedules[self.schedules["course_key"] == canonical]
        if df.empty or col not in df.columns:
            return None
        v = str(df.iloc[0][col]).strip()
        if not v:
            return None
        return f"{self.format_course_name(canonical, lang)}: {v}"

    def class_size_for_course(self, canonical: Optional[str], lang: str) -> Optional[str]:
        if self.class_size_cats.empty:
            return None
        col_val = self._lang_col(lang).replace("zh-HK", "zh-HK_value").replace("zh-CN", "zh-CN_value")
        if col_val == "en":
            col_val = "en_value"
        # specific course?
        if canonical and not self.course_class_size_map.empty:
            m = self.course_class_size_map[self.course_class_size_map["course_key"] == canonical]
            if not m.empty:
                cat = str(m.iloc[0]["category_key"])
                cdf = self.class_size_cats[self.class_size_cats["category_key"] == cat]
                if not cdf.empty and col_val in cdf.columns:
                    val = str(cdf.iloc[0][col_val]).strip()
                    label_col = self._lang_col(lang) + "_label" if self._lang_col(lang) != "en" else "en_label"
                    label = str(cdf.iloc[0].get(label_col, "")).strip()
                    label = label or self.format_course_name(canonical, lang)
                    return f"{label}: {val}"
        # list all categories
        lines = []
        for _, r in self.class_size_cats.iterrows():
            val = str(r.get(col_val, "")).strip()
            if not val:
                continue
            label_col = self._lang_col(lang) + "_label" if self._lang_col(lang) != "en" else "en_label"
            label = str(r.get(label_col, "")).strip()
            if label and val:
                lines.append(f"- {label}: {val}")
        title = "班級人數：" if lang.startswith("zh-hk") else ("班级人数：" if lang.startswith("zh-cn") else "Class size:")
        return f"{title}\n" + "\n".join(lines) if lines else None

    def tuition_for_course(self, canonical: Optional[str], lang: str) -> Optional[str]:
        if self.tuition_groups.empty:
            return None
        # group mapping
        col_val = self._lang_col(lang).replace("zh-HK", "zh-HK_value").replace("zh-CN", "zh-CN_value")
        if col_val == "en":
            col_val = "en_value"
        label_col = self._lang_col(lang) + "_label" if self._lang_col(lang) != "en" else "en_label"

        def group_value(gkey: str) -> Optional[str]:
            g = self.tuition_groups[self.tuition_groups["group_key"] == gkey]
            if g.empty:
                return None
            val = str(g.iloc[0].get(col_val, "")).strip()
            lab = str(g.iloc[0].get(label_col, "")).strip()
            if not val:
                return None
            if lab:
                return f"{lab}: {val}"
            return val

        if canonical and not self.course_tuition_map.empty:
            m = self.course_tuition_map[self.course_tuition_map["course_key"] == canonical]
            if not m.empty:
                gkey = str(m.iloc[0]["group_key"])
                gv = group_value(gkey)
                if gv:
                    return gv

        # list all
        lines = []
        for _, r in self.tuition_groups.iterrows():
            gkey = str(r["group_key"])
            gv = group_value(gkey)
            if gv:
                lines.append(f"- {gv}")
        title = "課程收費：" if lang.startswith("zh-hk") else ("课程收费：" if lang.startswith("zh-cn") else "Tuition:")
        return f"{title}\n" + "\n".join(lines) if lines else None

    def enrollment_process(self, lang: str) -> Optional[str]:
        if self.enrollment_steps.empty:
            return None
        col = self._lang_col(lang)
        steps = []
        for _, r in self.enrollment_steps.sort_values(by=self.enrollment_steps.columns[0]).iterrows():
            txt = str(r.get(col, "")).strip()
            if txt:
                steps.append(f"- {txt}")
        if not steps:
            return None
        header = "報名流程：" if lang.startswith("zh-hk") else ("报名流程：" if lang.startswith("zh-cn") else "Enrollment process:")
        return f"{header}\n" + "\n".join(steps)

    def policy_absence_makeup(self, lang: str) -> Optional[str]:
        if self.policy_absence.empty:
            return None
        col = self._lang_col(lang)
        order = [
            "NotificationRequirement", "MakeupArrangement", "WrittenNotificationRequirement",
            "MedicalCertificateRequirement", "NoNoticeMissedClasses", "MakeupDeadline",
            "InstructorTimeGuarantee", "FreeMakeupQuota", "ExceedingFreeMakeup",
            "ShortTermAbsence", "LongTermAbsence", "CourseValidity"
        ]
        lines = []
        for k in order:
            rows = self.policy_absence[self.policy_absence["key"] == k]
            if rows.empty or col not in rows.columns:
                continue
            txt = str(rows.iloc[0][col]).strip()
            if txt:
                lines.append(f"- {txt}")
        if not lines:
            return None
        title = "補課與請假政策：" if lang.startswith("zh-hk") else ("补课与请假政策：" if lang.startswith("zh-cn") else "Absence and Make-up policy:")
        return f"{title}\n" + "\n".join(lines)

    def policy_refund(self, lang: str) -> Optional[str]:
        if self.policy_refund.empty:
            return None
        col = self._lang_col(lang)
        order = ["RefundPolicy", "ContinuationAssumption", "WithdrawalNotice", "ClassTransferInquiry"]
        lines = []
        for k in order:
            rows = self.policy_refund[self.policy_refund["key"] == k]
            if rows.empty or col not in rows.columns:
                continue
            txt = str(rows.iloc[0][col]).strip()
            if txt:
                lines.append(f"- {txt}")
        if not lines:
            return None
        title = "退款與轉班政策：" if lang.startswith("zh-hk") else ("退款与转班政策：" if lang.startswith("zh-cn") else "Refund and transfer policy:")
        return f"{title}\n" + "\n".join(lines)

    def common_obj(self, key: str, lang: str, zh_hk_title: str, zh_cn_title: str, en_title: str) -> Optional[str]:
        if self.common_objections.empty:
            return None
        col = self._lang_col(lang)
        rows = self.common_objections[self.common_objections["key"] == key]
        if rows.empty or col not in rows.columns:
            return None
        val = str(rows.iloc[0][col]).strip()
        if not val:
            return None
        title = zh_hk_title if lang.startswith("zh-hk") else (zh_cn_title if lang.startswith("zh-cn") else en_title)
        return f"{title}\n- {val}"

    def promotions(self, lang: str) -> Optional[str]:
        if self.promotions.empty:
            return None
        col = self._lang_col(lang)
        # single row or multiple; concatenate
        vals = [str(r.get(col, "")).strip() for _, r in self.promotions.iterrows() if str(r.get(col, "")).strip()]
        return "\n".join(vals) if vals else None

    def success_stories(self, lang: str) -> Optional[str]:
        if self.success_stories.empty:
            return None
        col = self._lang_col(lang)
        lines = []
        for _, r in self.success_stories.iterrows():
            key = str(r.get("key", "")).strip()
            val = str(r.get(col, "")).strip()
            if key and val:
                lines.append(f"- {key}: {val}")
        if not lines:
            return None
        title = "成功個案：" if lang.startswith("zh-hk") else ("成功个案：" if lang.startswith("zh-cn") else "Success stories:")
        return f"{title}\n" + "\n".join(lines)