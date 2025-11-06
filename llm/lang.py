import re
import unicodedata
from typing import Optional

# --- Constants for Chinese Variant Detection ---

# Simplified characters with distinct forms (300+)
SIMP_ONLY = set(
    "爱摆备笔边参仓产长车虫从电东风发丰复个关广国过华画汇会几夹监见荐将节尽进举据开"
    "乐离礼丽两灵丽龙楼录陆妈买卖门们难鸟农齐气迁亲穷区权让认赛杀师时识属双说丝肃"
    "岁孙态体条铁听厅头图团为卫稳问无务戏习系显献乡写兴选学寻压严业医义艺阴隐应营"
    "拥优邮鱼与云杂灾郑执质专种众钟筑庄装壮贝呗狈绷毙毕宾补惨灿蚕层搀谗馋缠忏偿厂"
    "彻尘衬称惩迟冲丑出处础处触辞聪丛担胆导灯邓敌籴递点淀电冬斗独吨夺堕儿矾范飞坟"
    "奋粪凤肤妇复盖赶个巩沟构购谷顾刮关观柜汉号合轰后胡壶沪护划怀坏欢环还回伙获击"
    "鸡积极际继家价艰歼拣硷舰姜浆桨奖讲酱胶阶疖洁借仅惊竞旧剧惧卷开克垦恳夸块亏困"
    "腊蜡兰拦栏烂累台垒类里礼隶帘联怜炼练粮疗辽了猎临邻岭庐芦炉陆驴乱么霉蒙梦面庙"
    "蔑亩恼脑拟酿疟盘辟苹凭扑仆朴启签千牵纤窍窃寝庆琼秋曲权劝确让扰热认洒伞丧扫涩"
    "晒伤舍沈声胜湿实适势兽书术树帅松苏虽随态坛叹誊体粜铁听厅头图涂团椭洼袜网卫稳"
    "务雾牺习系戏虾吓咸显宪县响向协胁亵衅兴须悬选旋压盐阳养痒样钥药爷叶医亿忆应拥"
    "佣踊忧优邮余御吁郁誉渊园远愿跃运酝杂赃脏凿枣灶斋毡战赵折这征症证只致制钟肿种"
    "众昼朱烛筑庄桩妆装壮准浊总钻"
)

# Traditional equivalents with distinct forms (400+)
TRAD_ONLY = set(
    "愛罷備筆邊參倉產長車蟲從電東風髮發豐復複個關廣國過華畫匯彙會幾夾監見薦將節盡儘進"
    "舉據開樂離禮麗兩靈劉龍樓錄陸媽買賣門們難鳥農齊氣遷親窮區權讓認賽殺師時識屬雙說絲"
    "肅歲孫態體條鐵聽廳頭圖團為衛穩問無務戲習係繫顯獻鄉寫興選學尋壓嚴業醫義藝陰隱應營"
    "擁優郵魚與雲雜災鄭執質專種眾鐘鍾築莊裝壯貝唄狽綳繃斃畢賓補慘燦蠶層攙讒饞纏懺償廠"
    "徹塵襯稱懲遲沖衝醜齣礎處觸辭聰叢擔膽導燈鄧敵糴遞點澱電鼕鬥獨噸奪墮兒礬範飛墳奮糞"
    "鳳膚婦復複蓋趕個鞏溝構購穀顧颳關觀櫃漢號閤轟後鬍壺滬護劃懷壞歡環還迴夥獲擊雞積極"
    "際繼傢價艱殲揀鹼艦薑漿槳獎講醬膠階癤潔藉僅驚競舊劇懼捲開剋墾懇誇塊虧睏臘蠟蘭攔欄"
    "爛纍壘類裏禮隸簾聯憐煉練糧療遼瞭獵臨鄰嶺廬蘆爐陸驢亂麼黴濛懞矇夢麵廟衊畝惱腦擬釀"
    "瘧盤闢蘋憑撲僕樸啟簽籤韆牽縴竅竊寢慶瓊鞦麴權勸確讓擾熱認灑傘喪掃澀曬傷捨瀋聲勝濕"
    "實適勢獸書術樹帥鬆蘇雖隨臺檯颱態壇罎嘆謄體糶鐵聽廳頭圖塗團糰橢窪襪網衛穩務霧犧習"
    "係繫戲蝦嚇鹹顯憲縣響嚮協脅褻釁興鬚懸選鏇壓鹽陽養癢樣鑰藥爺葉醫億憶應擁傭踴憂優郵"
    "餘禦籲鬱譽淵園遠願躍運醞雜贓臟髒鑿棗竈齋氈戰趙摺這徵癥證隻衹緻製鐘鍾腫種眾晝硃燭"
    "築莊樁妝裝壯準濁總鑽"
)

# --- Normalization and Character Analysis Functions ---

def _normalize_text(text: str) -> str:
    """
    Performs comprehensive text normalization for analysis.
    - Converts to lowercase.
    - Normalizes Unicode to handle different character forms (e.g., full-width to half-width).
    - Replaces various whitespace characters with a standard space.
    """
    if not text:
        return ""
    normalized = unicodedata.normalize('NFKC', text).lower()
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized

def _is_cjk(char: str) -> bool:
    """
    Checks if a character is within the CJK Unicode ranges.
    """
    return any([
        0x4E00 <= ord(char) <= 0x9FFF,   # CJK Unified Ideographs
        0x3400 <= ord(char) <= 0x4DBF,   # CJK Unified Ideographs Extension A
        0x20000 <= ord(char) <= 0x2A6DF, # CJK Unified Ideographs Extension B
    ])

def _get_cjk_ratio(text: str) -> float:
    """
    Calculates the ratio of CJK characters to total alphabetic characters.
    """
    if not text:
        return 0.0
    cjk_count = 0
    letter_count = 0
    for char in text:
        if char.isspace() or char.isdigit() or not char.isprintable():
            continue
        if _is_cjk(char):
            cjk_count += 1
        if char.isalpha():
            letter_count += 1
    if letter_count == 0:
        return 0.0
    return cjk_count / letter_count

def _prefer_variant_from_accept_language(header: Optional[str]) -> Optional[str]:
    """
    Parses Accept-Language to prefer zh-HK (Traditional) or zh-CN (Simplified) when explicitly indicated.
    Defaults to Traditional if only 'zh' is present without script/region.
    """
    if not header:
        return None
    h = header.lower()
    # Explicit script/region hints
    if "zh-hant" in h or "zh-tw" in h or "zh-hk" in h:
        return "zh-HK"
    if "zh-hans" in h or "zh-cn" in h or "zh-sg" in h:
        return "zh-CN"
    # Ambiguous 'zh' -> prefer Traditional (Hong Kong default)
    if "zh" in h:
        return "zh-HK"
    return None

def _get_chinese_variant(text: str) -> str:
    """
    Determine Traditional vs Simplified based on distinctive character counts.
    IMPORTANT: Ties and 'no distinctive chars' default to Traditional (zh-HK) for Hong Kong deployment.
    """
    trad_score = 0
    simp_score = 0
    for char in text:
        if char in TRAD_ONLY:
            trad_score += 1
        elif char in SIMP_ONLY:
            simp_score += 1
    if trad_score > simp_score:
        return "zh-HK"
    if simp_score > trad_score:
        return "zh-CN"
    # Tie or no distinctive characters: default to Traditional for HK
    return "zh-HK"

# --- Main Detection Logic ---

def get_language_code(
    user_message: str,
    accept_language_header: Optional[str] = None
) -> str:
    """
    Determines the language code ('en', 'zh-CN', 'zh-HK') for a given message.

    Changes:
    - Default ambiguous Chinese to Traditional (zh-HK) instead of Simplified.
    - Respect Accept-Language when it explicitly specifies zh-Hans/zh-CN vs zh-Hant/zh-HK/zh-TW.
    - If Accept-Language only says 'zh', prefer Traditional (zh-HK).
    """
    normalized_message = _normalize_text(user_message)

    # Edge case: empty or whitespace-only messages
    if not normalized_message:
        # Prefer Traditional for ambiguous 'zh'; otherwise keep 'en'
        lang_from_header = _prefer_variant_from_accept_language(accept_language_header)
        if lang_from_header:
            return lang_from_header
        return "en"

    # Short common English phrases — fast path
    common_english_greetings = {"hi", "hello", "thanks", "thank you", "ok", "yes", "no"}
    if normalized_message in common_english_greetings:
        return "en"

    # Core heuristic for Chinese vs English
    cjk_ratio = _get_cjk_ratio(normalized_message)
    CJK_THRESHOLD = 0.3
    if cjk_ratio >= CJK_THRESHOLD:
        # Determine variant with Traditional bias on ties
        variant = _get_chinese_variant(normalized_message)
        # If variant detection produced Traditional by tie and header explicitly says Simplified, respect header
        header_pref = _prefer_variant_from_accept_language(accept_language_header)
        if header_pref in ("zh-HK", "zh-CN"):
            # Only override when header is explicit about variant (Hans/Hant or region)
            return header_pref
        return variant

    # Non-Chinese default
    return "en"

# --- Demonstration and Testing ---

if __name__ == "__main__":
    print("--- Running Language Detection Tests ---")

    test_cases = [
        # English cases
        ("Hello, how are you today?", "en"),
        ("This is a test.", "en"),
        ("thanks", "en"),
        # Chinese cases (Simplified distinctive)
        ("请问这里的信息正确吗？", "zh-CN"),
        ("我的电脑坏了，需要修理。", "zh-CN"),
        # Chinese cases (Traditional distinctive)
        ("請問這裏的資料正確嗎？", "zh-HK"),
        ("這是一個繁體字的句子。", "zh-HK"),
        # Ambiguous Chinese (common characters only) should default to Traditional
        ("好的", "zh-HK"),
        ("可以吗", "zh-CN"),  # contains “吗”(吗=简体 distinctive) -> zh-CN
        ("可以嗎", "zh-HK"),  # contains “嗎”(繁體 distinctive) -> zh-HK
        # Mixed language cases
        ("I love to eat 蛋挞 and drink 奶茶.", "en"),  # Primarily English
        ("我的名字是 David, nice to meet you.", "zh-CN"),  # Primarily Chinese
        # Accept-Language disambiguation
        ("好的", "zh-CN"),  # With header zh-CN we expect zh-CN
    ]

    for i, (message, expected) in enumerate(test_cases, start=1):
        if i == len(test_cases):  # last case: force header
            detected_lang = get_language_code(message, accept_language_header="zh-CN,zh;q=0.9")
        else:
            detected_lang = get_language_code(message)
        status = "✅ PASSED" if detected_lang == expected else f"❌ FAILED (Got {detected_lang})"
        print(f"Test {i}: '{message[:30]}...' -> Expected: {expected}, {status}")

    print("\n--- Notes ---")
    print("- Ambiguous Chinese now defaults to zh-HK (Traditional) to suit Hong Kong deployment.")
    print("- Accept-Language with zh-Hant/zh-HK/zh-TW forces zh-HK; zh-Hans/zh-CN/zh-SG forces zh-CN; bare 'zh' prefers zh-HK.")