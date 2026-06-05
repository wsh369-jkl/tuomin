"""Shared entity helpers for local Chinese desensitization recognizers."""

from __future__ import annotations

import re
from typing import Dict, Iterable, Iterator, List, Optional

from app.core.recognizer_base import RecognizerResult


ORG_SUFFIX_TERMS = (
    "股份有限公司",
    "有限责任公司",
    "有限公司",
    "集团有限公司",
    "集团",
    "分公司",
    "子公司",
    "商行",
    "合作社",
    "联合社",
    "联社",
    "工作室",
    "经营部",
    "门市部",
    "营业部",
    "办事处",
    "基金会",
    "联合会",
    "俱乐部",
    "实验室",
    "门诊部",
    "诊所",
    "学校",
    "大学",
    "学院",
    "医院",
    "协会",
    "学会",
    "银行",
    "人民法院",
    "中级人民法院",
    "高级人民法院",
    "仲裁委员会",
    "人民检察院",
    "公安局",
    "派出所",
    "律师事务所",
    "会计师事务所",
    "研究院",
    "研究所",
    "服务中心",
    "技术中心",
    "管理委员会",
    "委员会",
)
ORG_SUFFIX_PATTERN = (
    r"(?:股份有限公司|有限责任公司|有限公司|集团有限公司|集团|分公司|子公司|"
    r"商行|合作社|联合社|联社|工作室|经营部|门市部|营业部|办事处|基金会|联合会|俱乐部|实验室|门诊部|诊所|"
    r"学校|大学|学院|医院|协会|学会|"
    r"银行(?:股份有限公司)?(?:[\u4e00-\u9fa5A-Za-z0-9]{0,12}(?:支行|分行))?|"
    r"人民法院|中级人民法院|高级人民法院|仲裁委员会|人民检察院|公安局|派出所|"
    r"律师事务所|会计师事务所|研究院|研究所|服务中心|技术中心|管理委员会|委员会)"
)
ORG_PATTERN = re.compile(rf"[\u4e00-\u9fa5A-Za-z0-9（）()·\-]{{2,48}}?{ORG_SUFFIX_PATTERN}")
PERSON_PATTERN = re.compile(r"(?<![\u4e00-\u9fa5])[\u4e00-\u9fa5]{2,8}(?:·[\u4e00-\u9fa5]{2,8})?(?![\u4e00-\u9fa5])")
DATE_PATTERN = re.compile(r"\d{4}[\u5e74/-]\d{1,2}[\u6708/-](?:\d{1,2}|xx|XX)[\u65e5\u53f7]?")
INLINE_SPACE_CONTENT_CHAR = re.compile(r"[\u4e00-\u9fa5A-Za-z0-9]")
INLINE_SPACE_CJK_CHAR = re.compile(r"[\u4e00-\u9fa5]")
INLINE_SPACE_DIGIT_CHAR = re.compile(r"\d")
COMMON_SINGLE_SURNAMES = {
    "赵", "钱", "孙", "李", "周", "吴", "郑", "王", "冯", "陈", "褚", "卫",
    "蒋", "沈", "韩", "杨", "朱", "秦", "尤", "许", "何", "吕", "施", "张",
    "孔", "曹", "严", "华", "金", "魏", "陶", "姜", "戚", "谢", "邹", "喻",
    "柏", "水", "窦", "章", "云", "苏", "潘", "葛", "奚", "范", "彭", "郎",
    "鲁", "韦", "昌", "马", "苗", "凤", "花", "方", "俞", "任", "袁", "柳",
    "酆", "鲍", "史", "唐", "费", "廉", "岑", "薛", "雷", "贺", "倪", "汤",
    "滕", "殷", "罗", "毕", "郝", "邬", "安", "常", "乐", "于", "时", "傅",
    "皮", "卞", "齐", "康", "伍", "余", "元", "卜", "顾", "孟", "平", "黄",
    "和", "穆", "萧", "尹", "姚", "邵", "湛", "汪", "祁", "毛", "禹", "狄",
    "米", "贝", "明", "臧", "计", "伏", "成", "戴", "谈", "宋", "茅", "庞",
    "熊", "纪", "舒", "屈", "项", "祝", "董", "梁", "杜", "阮", "蓝", "闵",
    "席", "季", "麻", "强", "贾", "路", "娄", "危", "江", "童", "颜", "郭",
    "梅", "盛", "林", "刁", "钟", "徐", "邱", "骆", "高", "夏", "蔡", "田",
    "樊", "胡", "凌", "霍", "虞", "万", "支", "柯", "昝", "管", "卢", "莫",
    "经", "房", "裘", "缪", "干", "解", "应", "宗", "丁", "宣", "贲", "邓",
    "郁", "单", "杭", "洪", "包", "诸", "左", "石", "崔", "吉", "钮", "龚",
    "程", "嵇", "邢", "滑", "裴", "陆", "荣", "翁", "荀", "羊", "於", "惠",
    "甄", "麴", "家", "封", "芮", "羿", "储", "靳", "汲", "邴", "糜", "松",
    "井", "段", "富", "巫", "乌", "焦", "巴", "弓", "牧", "隗", "山", "谷",
    "车", "侯", "宓", "蓬", "全", "郗", "班", "仰", "秋", "仲", "伊", "宫",
    "宁", "仇", "栾", "暴", "甘", "斜", "厉", "戎", "祖", "武", "符", "刘",
    "景", "詹", "束", "龙", "叶", "幸", "司", "韶", "郜", "黎", "蓟", "薄",
    "印", "宿", "白", "怀", "蒲", "邰", "从", "鄂", "索", "咸", "籍", "赖",
    "卓", "蔺", "屠", "蒙", "池", "乔", "阴", "胥", "能", "苍", "双", "闻",
    "莘", "党", "翟", "谭", "贡", "劳", "逄", "姬", "申", "扶", "堵", "冉",
    "宰", "郦", "雍", "却", "璩", "桑", "桂", "濮", "牛", "寿", "通", "边",
    "扈", "燕", "冀", "郏", "浦", "尚", "农", "温", "别", "庄", "晏", "柴",
    "瞿", "阎", "充", "慕", "连", "茹", "习", "宦", "艾", "鱼", "容", "向",
    "古", "易", "慎", "戈", "廖", "庾", "终", "暨", "居", "衡", "步", "都",
    "耿", "满", "弘", "匡", "国", "文", "寇", "广", "禄", "阙", "东", "欧",
    "殳", "沃", "利", "蔚", "越", "夔", "隆", "师", "巩", "厍", "聂", "晁",
    "勾", "敖", "融", "冷", "訾", "辛", "阚", "那", "简", "饶", "空", "曾",
    "毋", "沙", "乜", "养", "鞠", "须", "丰", "巢", "关", "蒯", "相", "查",
    "后", "荆", "红", "游", "竺", "权", "逯", "盖", "益", "桓", "公", "晋",
    "楚", "阎", "法", "汝", "鄢", "涂", "钦", "岳", "帅", "缑", "亢", "况",
    "郈", "有", "琴", "归", "海", "墨", "哈", "谯", "笪", "年", "爱", "阳",
    "佟",
}
COMMON_COMPOUND_SURNAMES = (
    "欧阳", "太史", "端木", "上官", "司马", "东方", "独孤", "南宫", "万俟", "闻人",
    "夏侯", "诸葛", "尉迟", "公羊", "赫连", "澹台", "皇甫", "宗政", "濮阳", "公冶",
    "太叔", "申屠", "公孙", "慕容", "仲孙", "钟离", "长孙", "宇文", "司徒", "鲜于",
    "司空", "闾丘", "子车", "亓官", "司寇", "巫马", "公西", "颛孙", "壤驷", "公良",
    "漆雕", "乐正", "宰父", "谷梁", "拓跋", "夹谷", "轩辕", "令狐", "段干", "百里",
    "呼延", "东郭", "南门", "羊舌", "微生", "公户", "公玉", "公仪", "梁丘", "公仲",
    "公上", "公门", "公山", "公坚", "左丘", "公伯", "西门", "公祖", "第五", "公乘",
    "贯丘", "公皙", "南荣", "东里", "东宫", "仲长", "子书", "子桑", "即墨", "达奚",
    "褚师",
)

ROLE_LABELS = {
    "甲方",
    "乙方",
    "丙方",
    "申请人",
    "被申请人",
    "原告",
    "被告",
    "第三人",
    "控告人",
    "被控告人",
    "举报人",
    "被举报人",
    "申诉人",
    "被申诉人",
    "起诉人",
    "自诉人",
    "法定代表人",
    "法定代理人",
    "法人代表",
    "负责人",
    "联系人",
    "委托代理人",
    "委托诉讼代理人",
    "诉讼代理人",
    "签署人",
    "收款单位",
    "付款单位",
}

POSITION_LABEL_TERMS = {
    "岗位",
    "职位",
    "职务",
    "岗位职责",
    "工作岗位",
}

IDENTITY_REFERENCE_TERMS = {
    "甲方",
    "乙方",
    "丙方",
    "委托方",
    "受托方",
    "发包人",
    "承包人",
    "采购人",
    "供应商",
    "上诉人",
    "被上诉人",
    "原审原告",
    "原审被告",
    "一审原告",
    "一审被告",
    "申请人",
    "被申请人",
    "原告",
    "被告",
    "第三人",
    "被执行人",
    "申请执行人",
    "案外人",
    "控告人",
    "被控告人",
    "举报人",
    "被举报人",
    "申诉人",
    "被申诉人",
    "起诉人",
    "自诉人",
    "法定代表人",
    "法定代理人",
    "法人代表",
    "负责人",
    "联系人",
    "委托代理人",
    "委托诉讼代理人",
    "诉讼代理人",
    "代理人",
    "签署人",
    "经办人",
    "授权代表",
    "项目负责人",
    "收款单位",
    "付款单位",
}

NON_ENTITY_ROLE_TERMS = {
    "上诉人",
    "被上诉人",
    "原审原告",
    "原审被告",
    "一审原告",
    "一审被告",
    "申请人",
    "被申请人",
    "原告",
    "被告",
    "第三人",
    "控告人",
    "举报人",
    "申诉人",
    "起诉人",
    "自诉人",
    "被控告人",
    "被举报人",
    "被申诉人",
    "法定代表人",
    "法定代理人",
    "法人代表",
    "负责人",
    "联系人",
    "委托代理人",
    "诉讼代理人",
    "委托诉讼代理人",
    "代理人",
    "签署人",
    "经办人",
}
NON_ENTITY_ROLE_TERMS.update(IDENTITY_REFERENCE_TERMS)

POSITION_TITLE_KEYWORDS = (
    "董事长",
    "副董事长",
    "执行董事",
    "董事",
    "监事",
    "总经理",
    "副总经理",
    "经理",
    "总监",
    "副总监",
    "主管",
    "主任",
    "部长",
    "处长",
    "科长",
    "负责人",
    "专员",
    "助理",
    "顾问",
    "工程师",
    "会计",
    "出纳",
    "法务",
    "财务",
    "销售",
    "采购",
    "项目经理",
    "项目总监",
    "项目主管",
)

ORGANIZATION_MARKERS = (
    "股份有限公司",
    "有限责任公司",
    "有限公司",
    "集团有限公司",
    "集团",
    "公司",
    "分公司",
    "子公司",
    "商行",
    "合作社",
    "事务所",
    "工作室",
    "研究院",
    "研究所",
    "服务中心",
    "技术中心",
    "经营部",
    "门市部",
    "营业部",
    "办事处",
    "基金会",
    "联合会",
    "俱乐部",
    "实验室",
    "门诊部",
    "诊所",
    "委员会",
    "管理委员会",
    "仲裁委员会",
    "人民法院",
    "法院",
    "检察院",
    "公安局",
    "派出所",
    "银行",
    "支行",
    "分行",
    "政府",
    "学校",
    "大学",
    "学院",
    "医院",
    "协会",
    "学会",
    "中心",
)

IDENTITY_REFERENCE_PREFIX_TERMS = tuple(
    sorted(
        {
            "上诉人",
            "被上诉人",
            "原审原告",
            "原审被告",
            "一审原告",
            "一审被告",
            "申请人",
            "被申请人",
            "原告",
            "被告",
            "第三人",
            "控告人",
            "被控告人",
            "举报人",
            "被举报人",
            "申诉人",
            "被申诉人",
            "起诉人",
            "自诉人",
            "法定代表人",
            "法定代理人",
            "法人代表",
            "负责人",
            "联系人",
            "委托代理人",
            "委托诉讼代理人",
            "诉讼代理人",
            "代理人",
            "签署人",
            "经办人",
            "授权代表",
            "项目负责人",
        },
        key=len,
        reverse=True,
    )
)

GENERIC_ORGANIZATION_TERMS = {
    "公司",
    "有限公司",
    "有限责任公司",
    "股份有限公司",
    "集团",
    "集团有限公司",
    "分公司",
    "子公司",
    "企业",
    "单位",
    "机构",
    "部门",
    "中心",
    "委员会",
    "管理委员会",
    "事务所",
    "研究院",
    "研究所",
    "法院",
    "人民法院",
    "检察院",
    "公安局",
    "派出所",
    "银行",
    "支行",
    "分行",
}

SENSITIVE_ENTITY_TYPES = {
    "PERSON",
    "PERSON_NAME",
    "ORGANIZATION",
    "COMPANY_NAME",
    "LOCATION",
    "ADDRESS",
    "PROJECT",
    "BANK_NAME",
    "ACCOUNT_NAME",
    "CONTRACT_NO",
    "CASE_NO",
    "COURT",
    "ALIAS",
}

P0_ENTITY_TYPES = {
    "CN_ID_CARD",
    "CN_PHONE",
    "LANDLINE_PHONE",
    "CN_BANK_CARD",
    "CN_CREDIT_CODE",
    "EMAIL_ADDRESS",
    "CONTRACT_NO",
    "CASE_NO",
    "TAX_NO",
    "URL",
}


def clean_candidate_text(value: str, *, max_len: int = 120) -> str:
    text = re.sub(r"\s+", " ", value or "").strip()
    text = text.strip(" \t\r\n:：,，;；。")
    text = re.sub(r"(?:（盖章）|\(盖章\)|盖章|签章|签字)$", "", text).strip()
    if len(text) > max_len:
        text = text[:max_len].strip()
    return text


def normalize_entity_text(value: str) -> str:
    return re.sub(r"[\s:：，,。；;（）()《》【】\"“”'`]", "", value or "")


def sanitize_recognition_text(text: str) -> tuple[str, list[int]]:
    """Remove intrusive inline whitespace without touching structural newlines.

    The legal source files sometimes contain OCR- or copy-induced spaces inside
    ordinary body text, which breaks short-name recognition. We only remove
    horizontal whitespace runs whose left and right neighbors are both content
    characters, then keep an index map so downstream spans can be projected back
    to the original source text.
    """

    if not text:
        return "", []

    sanitized_chars: list[str] = []
    index_map: list[int] = []
    text_length = len(text)
    cursor = 0
    while cursor < text_length:
        char = text[cursor]
        if char in "\r\n":
            sanitized_chars.append(char)
            index_map.append(cursor)
            cursor += 1
            continue
        if char.isspace():
            space_end = cursor + 1
            while space_end < text_length and text[space_end].isspace() and text[space_end] not in "\r\n":
                space_end += 1
            left_char = sanitized_chars[-1] if sanitized_chars else ""
            right_char = text[space_end] if space_end < text_length else ""
            if _is_intrusive_inline_space(left_char, right_char):
                cursor = space_end
                continue
            for original_index in range(cursor, space_end):
                sanitized_chars.append(text[original_index])
                index_map.append(original_index)
            cursor = space_end
            continue
        sanitized_chars.append(char)
        index_map.append(cursor)
        cursor += 1
    return "".join(sanitized_chars), index_map


def _is_intrusive_inline_space(left_char: str, right_char: str) -> bool:
    if not INLINE_SPACE_CONTENT_CHAR.fullmatch(left_char):
        return False
    if not INLINE_SPACE_CONTENT_CHAR.fullmatch(right_char):
        return False
    return bool(
        INLINE_SPACE_CJK_CHAR.fullmatch(left_char)
        or INLINE_SPACE_CJK_CHAR.fullmatch(right_char)
        or INLINE_SPACE_DIGIT_CHAR.fullmatch(left_char)
        or INLINE_SPACE_DIGIT_CHAR.fullmatch(right_char)
    )


def remap_sanitized_span(index_map: list[int], start: int, end: int) -> Optional[tuple[int, int]]:
    if start < 0 or end <= start:
        return None
    if end - 1 >= len(index_map) or start >= len(index_map):
        return None
    return index_map[start], index_map[end - 1] + 1


def remap_results_to_source(
    results: Iterable[RecognizerResult],
    source_text: str,
    index_map: list[int],
) -> List[RecognizerResult]:
    remapped: list[RecognizerResult] = []
    for result in results:
        span = remap_sanitized_span(index_map, result.start, result.end)
        if span is None:
            continue
        start, end = span
        remapped.append(
            RecognizerResult(
                entity_type=result.entity_type,
                start=start,
                end=end,
                score=result.score,
                text=source_text[start:end],
                source=result.source,
                metadata=dict(result.metadata or {}),
            )
        )
    return remapped


def is_identity_reference_term(value: str) -> bool:
    normalized = normalize_entity_text(value)
    return bool(normalized) and normalized in NON_ENTITY_ROLE_TERMS


def strip_identity_reference_prefix(value: str) -> str:
    normalized = normalize_entity_text(value)
    if not normalized:
        return ""
    for prefix in IDENTITY_REFERENCE_PREFIX_TERMS:
        if not normalized.startswith(prefix) or len(normalized) <= len(prefix):
            continue
        remainder = normalized[len(prefix) :].lstrip(":：，,、")
        if remainder:
            return remainder
    return ""


def has_identity_reference_prefix(value: str) -> bool:
    return bool(strip_identity_reference_prefix(value))


def is_generic_organization_term(value: str) -> bool:
    normalized = normalize_entity_text(value)
    return bool(normalized) and normalized in GENERIC_ORGANIZATION_TERMS


def is_org_like_text(value: str) -> bool:
    normalized = normalize_entity_text(value)
    if not normalized or is_identity_reference_term(normalized):
        return False
    if is_generic_organization_term(normalized):
        return False
    prefixed = strip_identity_reference_prefix(normalized)
    if prefixed:
        if len(prefixed) < 2 or is_generic_organization_term(prefixed):
            return False
        normalized = prefixed
    if ORG_PATTERN.search(normalized):
        return True
    return any(marker in normalized for marker in ORGANIZATION_MARKERS)


def looks_like_natural_person_name(value: str) -> bool:
    text = re.sub(r"\s+", "", value or "")
    if not 2 <= len(text) <= 8:
        return False
    if re.fullmatch(r"[\u4e00-\u9fa5]{2,4}", text):
        for surname in COMMON_COMPOUND_SURNAMES:
            if text.startswith(surname) and 1 <= len(text) - len(surname) <= 2:
                return True
        return text[0] in COMMON_SINGLE_SURNAMES
    return bool(re.fullmatch(r"[\u4e00-\u9fa5]{1,2}·[\u4e00-\u9fa5]{2,8}", text))


def looks_like_organization_short_name(value: str) -> bool:
    normalized = normalize_entity_text(value)
    if not 2 <= len(normalized) <= 6:
        return False
    if is_identity_reference_term(normalized) or is_position_title(normalized):
        return False
    if is_generic_organization_term(normalized):
        return False
    if any(token in normalized for token in ("省", "市", "区", "县", "镇", "乡", "村", "路", "街", "号", "室")):
        return False
    return re.fullmatch(r"[\u4e00-\u9fa5A-Za-z0-9]{2,6}", normalized) is not None


def is_position_title(value: str) -> bool:
    normalized = normalize_entity_text(value)
    if not normalized:
        return False
    if normalized in POSITION_LABEL_TERMS:
        return True
    if is_identity_reference_term(normalized):
        return False
    if is_org_like_text(normalized):
        return False
    if any(
        token in normalized
        for token in (
            "省",
            "市",
            "区",
            "县",
            "镇",
            "乡",
            "村",
            "路",
            "街",
            "号",
            "室",
            "法院",
            "银行",
            "项目",
            "地址",
            "账号",
            "电话",
        )
    ):
        return False
    return any(keyword in normalized for keyword in POSITION_TITLE_KEYWORDS)


def is_probable_person(value: str) -> bool:
    text = re.sub(r"\s+", "", value or "")
    if not 2 <= len(text) <= 17:
        return False
    if text in NON_ENTITY_ROLE_TERMS:
        return False
    if any(
        token in text
        for token in (
            "公司",
            "银行",
            "法院",
            "项目",
            "地址",
            "电话",
            "账号",
            "省",
            "市",
            "区",
            "县",
            "镇",
            "乡",
            "村",
            "路",
            "街",
            "号",
            "室",
            "自治区",
            "自治州",
            "自治县",
        )
    ):
        return False
    return looks_like_natural_person_name(text)


def infer_semantic_type(value: str, label: str | None = None) -> str:
    label_text = label or ""
    value_text = value or ""
    normalized_label = normalize_entity_text(label_text)
    normalized_value = normalize_entity_text(value_text)
    if "案号" in label_text:
        return "CASE_NO"
    if "合同" in label_text and "编号" in label_text:
        return "CONTRACT_NO"
    if "开户行" in label_text or "银行" in label_text:
        return "BANK_NAME"
    if "户名" in label_text or "账户名称" in label_text or "账户名" in label_text:
        return "ACCOUNT_NAME"
    if "项目" in label_text or "工程" in label_text:
        return "PROJECT"
    if "地址" in label_text or "住所" in label_text or "送达" in label_text:
        return "ADDRESS"
    if normalized_label in POSITION_LABEL_TERMS or any(token in label_text for token in POSITION_LABEL_TERMS):
        return "POSITION"
    if normalized_label in IDENTITY_REFERENCE_TERMS:
        if "法院" in value_text:
            return "COURT"
        if is_org_like_text(normalized_value):
            return "ORGANIZATION"
        if is_position_title(normalized_value):
            return "POSITION"
        if looks_like_natural_person_name(normalized_value):
            return "PERSON"
    if looks_like_natural_person_name(normalized_value):
        return "PERSON"
    if "法院" in value_text:
        return "COURT"
    if is_org_like_text(value_text):
        return "ORGANIZATION"
    if is_position_title(value_text):
        return "POSITION"
    if is_probable_person(value_text):
        return "PERSON"
    return "ORGANIZATION"


def make_entity(
    *,
    text: str,
    start: int,
    end: int,
    entity_type: str,
    source: str,
    score: float,
    metadata: Optional[Dict] = None,
) -> Optional[RecognizerResult]:
    if start < 0 or end <= start:
        return None
    value = text[start:end]
    if not value:
        return None
    return RecognizerResult(
        entity_type=entity_type,
        start=start,
        end=end,
        score=score,
        text=value,
        source=source,
        metadata=metadata or {},
    )


def iter_exact_matches(text: str, target: str) -> Iterator[tuple[int, int, str]]:
    if not target:
        return
    seen: set[tuple[int, int]] = set()
    start = 0
    while True:
        index = text.find(target, start)
        if index < 0:
            break
        end = index + len(target)
        seen.add((index, end))
        yield index, end, text[index:end]
        start = index + len(target)

    sanitized_text, index_map = sanitize_recognition_text(text)
    sanitized_target, _ = sanitize_recognition_text(target)
    if sanitized_text == text or not sanitized_target:
        return

    search_from = 0
    while True:
        normalized_index = sanitized_text.find(sanitized_target, search_from)
        if normalized_index < 0:
            break
        span = remap_sanitized_span(index_map, normalized_index, normalized_index + len(sanitized_target))
        if span is not None and span not in seen:
            start_index, end_index = span
            seen.add(span)
            yield start_index, end_index, text[start_index:end_index]
        search_from = normalized_index + len(sanitized_target)


def find_value_span(
    source_text: str,
    value: str,
    *,
    search_start: int = 0,
    search_end: Optional[int] = None,
) -> Optional[tuple[int, int]]:
    cleaned = clean_candidate_text(value)
    if not cleaned:
        return None
    bounded = source_text[search_start:search_end]
    local_index = bounded.find(cleaned)
    if local_index >= 0:
        start = search_start + local_index
        return start, start + len(cleaned)

    normalized_value = re.sub(r"\s+", "", cleaned)
    if not normalized_value:
        return None

    normalized_chars: list[str] = []
    index_map: list[int] = []
    for index, char in enumerate(bounded, start=search_start):
        if char.isspace():
            continue
        normalized_chars.append(char)
        index_map.append(index)

    normalized_text = "".join(normalized_chars)
    normalized_index = normalized_text.find(normalized_value)
    if normalized_index < 0:
        return None
    normalized_end = normalized_index + len(normalized_value) - 1
    start = index_map[normalized_index]
    end = index_map[normalized_end] + 1
    return start, end


def deduplicate_results(results: Iterable[RecognizerResult]) -> List[RecognizerResult]:
    seen: set[tuple[str, int, int, str]] = set()
    deduped: list[RecognizerResult] = []
    for result in sorted(results, key=lambda item: (item.start, item.end, item.entity_type, -item.score)):
        key = (result.entity_type, result.start, result.end, result.text)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(result)
    return deduped
