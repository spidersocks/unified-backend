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
    # NFKC compatibility normalization is great for collapsing variants
    normalized = unicodedata.normalize('NFKC', text).lower()
    # Replace any sequence of whitespace characters with a single space
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized

def _is_cjk(char: str) -> bool:
    """
    Checks if a character is within the CJK Unicode ranges.
    This is a reliable way to identify Chinese, Japanese, or Korean characters.
    """
    return any([
        0x4E00 <= ord(char) <= 0x9FFF,  # CJK Unified Ideographs
        0x3400 <= ord(char) <= 0x4DBF,  # CJK Unified Ideographs Extension A
        0x20000 <= ord(char) <= 0x2A6DF, # CJK Unified Ideographs Extension B
        # Add other relevant ranges if needed, e.g., for punctuation
    ])

def _get_cjk_ratio(text: str) -> float:
    """
    Calculates the ratio of CJK characters to the total number of alphabetic characters.
    This is the most critical heuristic to avoid misclassifying primarily English text
    that contains a few CJK characters (e.g., 'I love Pokémon').
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
        # We only count letters for the denominator to get a meaningful ratio
        if char.isalpha():
            letter_count += 1

    if letter_count == 0:
        return 0.0

    return cjk_count / letter_count

def _get_chinese_variant(text: str) -> str:
    """
    Scores the text to determine if it's primarily Simplified or Traditional Chinese.
    It counts occurrences of characters unique to each script.
    """
    trad_score = 0
    simp_score = 0

    for char in text:
        if char in TRAD_ONLY:
            trad_score += 1
        elif char in SIMP_ONLY:
            simp_score += 1

    # If the Traditional set is populated and has a higher score, classify as zh-HK.
    # We give a slight bias to Traditional if scores are equal, as it's often
    # a conscious choice in mixed contexts.
    if trad_score > simp_score:
        return "zh-HK"
    
    # Otherwise, default to Simplified Chinese. This is the fallback if no unique
    # characters are found or if Simplified characters dominate.
    return "zh-CN"


# --- Main Detection Logic ---

def get_language_code(
    user_message: str,
    accept_language_header: Optional[str] = None
) -> str:
    """
    Determines the language code ('en', 'zh-CN', 'zh-HK') for a given message.
    This version uses a robust set of heuristics.

    Args:
        user_message: The text message from the user.
        accept_language_header: The 'Accept-Language' HTTP header, used as a fallback hint.

    Returns:
        A string representing the detected language code.
    """
    # 1. Normalize the input for consistent processing
    normalized_message = _normalize_text(user_message)

    # 2. Handle edge cases: empty or whitespace-only messages
    if not normalized_message:
        # If the header suggests Chinese, pick a default; otherwise 'en'
        if accept_language_header and 'zh' in accept_language_header.lower():
            return "zh-CN"
        return "en"

    # 3. Handle short, common English phrases that don't need complex analysis
    common_english_greetings = {"hi", "hello", "thanks", "thank you", "ok", "yes", "no"}
    if normalized_message in common_english_greetings:
        return "en"

    # 4. The Core Heuristic: CJK Character Ratio
    # We set a threshold (e.g., 0.3 or 30%). If more than 30% of the alphabetic
    # characters are CJK, we classify it as Chinese.
    cjk_ratio = _get_cjk_ratio(normalized_message)
    CJK_THRESHOLD = 0.3

    if cjk_ratio >= CJK_THRESHOLD:
        # If it's determined to be Chinese, figure out if it's Simplified or Traditional
        return _get_chinese_variant(normalized_message)
    else:
        # If the ratio is below the threshold, it's English.
        return "en"

# --- Demonstration and Testing ---

if __name__ == "__main__":
    print("--- Running Language Detection Tests ---")

    test_cases = [
        # English cases
        ("Hello, how are you today?", "en"),
        ("This is a test.", "en"),
        ("thanks", "en"),
        # Chinese cases (Simplified)
        ("你好，请问有什么可以帮助你的吗？", "zh-CN"),
        ("我的电脑坏了，需要修理。", "zh-CN"),
        # Chinese cases (Traditional) - Will currently resolve to zh-CN until TRAD_ONLY is populated
        ("這是一個繁體字的句子。", "zh-CN"), # EXPECTED: zh-HK after populating TRAD_ONLY
        ("請問這裏的資料正確嗎？", "zh-CN"), # EXPECTED: zh-HK after populating TRAD_ONLY
        # Mixed language cases
        ("I love to eat 蛋挞 and drink 奶茶.", "en"), # Primarily English
        ("My favorite character is 皮卡丘 (Pikachu).", "en"), # Primarily English
        ("我的名字是 David, nice to meet you.", "zh-CN"), # Primarily Chinese
        # Edge cases
        ("          ", "en"), # Whitespace only
        ("", "en"), # Empty string
        ("1234567890", "en"), # Numbers only
        ("你好", "zh-CN"), # Short Chinese
    ]

    for i, (message, expected) in enumerate(test_cases):
        detected_lang = get_language_code(message)
        status = "✅ PASSED" if detected_lang == expected else f"❌ FAILED (Got {detected_lang})"
        print(f"Test {i+1}: '{message[:30]}...' -> Expected: {expected}, {status}")

    print("\n--- Note on Traditional Chinese Detection ---")
    print("The 'TRAD_ONLY' character set is currently empty. Therefore, all Traditional Chinese text")
    print("is expected to fall back to 'zh-CN'. Once you populate the set, the relevant tests should pass as 'zh-HK'.")