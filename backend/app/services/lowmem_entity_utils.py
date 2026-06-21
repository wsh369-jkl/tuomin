"""Shared entity helpers for local Chinese desensitization recognizers."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, Iterable, Iterator, List, Optional

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
GOVERNMENT_INSTITUTION_SUFFIX_TERMS = (
    "最高人民法院",
    "高级人民法院",
    "中级人民法院",
    "基层人民法院",
    "人民法院",
    "法院",
    "最高人民检察院",
    "人民检察院",
    "检察院",
    "公安局",
    "派出所",
    "仲裁委员会",
    "人民政府",
    "政府",
    "管理委员会",
    "委员会",
    "市场监督管理局",
    "税务局",
    "财政局",
    "司法局",
    "民政局",
    "商务局",
    "教育局",
    "自然资源局",
    "生态环境局",
    "交通运输局",
    "行政审批局",
    "综合行政执法局",
    "住房和城乡建设局",
    "人力资源和社会保障局",
    "卫生健康委员会",
    "发展和改革委员会",
)
FINANCIAL_INSTITUTION_SUFFIX_TERMS = (
    "银行股份有限公司",
    "银行",
    "支行",
    "分行",
    "营业部",
    "分理处",
    "信用社",
    "农村信用合作联社",
    "信用合作联社",
    "联社",
)
ORG_SUFFIX_PATTERN = (
    r"(?:(?:股份有限公司|有限责任公司|集团有限公司|有限公司)"
    r"(?:[\u4e00-\u9fa5A-Za-z0-9（）()·\-]{1,16}分公司)?|"
    r"集团|分公司|子公司|"
    r"商行|合作社|联合社|联社|工作室|经营部|门市部|营业部|办事处|基金会|联合会|俱乐部|实验室|门诊部|诊所|"
    r"学校|大学|学院|医院|协会|学会|"
    r"银行(?:股份有限公司)?(?:[\u4e00-\u9fa5A-Za-z0-9]{0,12}(?:支行|分行))?|"
    r"人民法院|中级人民法院|高级人民法院|仲裁委员会|人民检察院|公安局|派出所|"
    r"律师事务所|会计师事务所|研究院|研究所|服务中心|技术中心|管理委员会|委员会)"
)
ORG_PATTERN = re.compile(rf"[\u4e00-\u9fa5A-Za-z0-9（）()·\-]{{2,48}}?{ORG_SUFFIX_PATTERN}")
_OFFICIAL_REGION_PREFIX = (
    r"(?:(?:中华人民共和国|国家|中国|全国|最高)|"
    r"(?:某某|XX|xx|Xx|xX)|"
    r"(?:[\u4e00-\u9fa5]{2,16}(?:省|自治区|特别行政区|市|地区|自治州|盟|区|县|旗|自治县|自治旗)))"
)
GOVERNMENT_INSTITUTION_SUFFIX_PATTERN = (
    r"(?:" + "|".join(re.escape(item) for item in sorted(GOVERNMENT_INSTITUTION_SUFFIX_TERMS, key=len, reverse=True)) + r")"
)
FINANCIAL_INSTITUTION_SUFFIX_PATTERN = (
    r"(?:" + "|".join(re.escape(item) for item in sorted(FINANCIAL_INSTITUTION_SUFFIX_TERMS, key=len, reverse=True)) + r")"
)
GOVERNMENT_INSTITUTION_PATTERN = re.compile(
    rf"(?:{_OFFICIAL_REGION_PREFIX}[\u4e00-\u9fa5A-Za-z0-9（）()·\-]{{0,24}}?{GOVERNMENT_INSTITUTION_SUFFIX_PATTERN})"
)
FINANCIAL_INSTITUTION_PATTERN = re.compile(
    rf"(?:{_OFFICIAL_REGION_PREFIX})?[\u4e00-\u9fa5A-Za-z0-9（）()·\-]{{2,32}}?"
    rf"(?:银行(?:股份有限公司)?(?:[\u4e00-\u9fa5A-Za-z0-9]{{1,20}}(?:支行|分行|营业部|分理处))?|"
    rf"{FINANCIAL_INSTITUTION_SUFFIX_PATTERN})"
)
OFFICIAL_INSTITUTION_PATTERN = re.compile(
    rf"(?:{GOVERNMENT_INSTITUTION_PATTERN.pattern}|{FINANCIAL_INSTITUTION_PATTERN.pattern})"
)
LEADING_SUBJECT_FUNCTION_WORDS = (
    # Strong multi-character cues that can pollute the left side of a subject.
    # Ambiguous single-character particles such as 由/经/同/和/与/及/向/对
    # are deliberately excluded here: they may be boundary evidence in a
    # sentence, but are not safe global "non-noun" prefixes.
    "是否通过",
    "已经通过",
    "已通过",
    "并通过",
    "通过",
    "经过",
    "经由",
    "根据",
    "依据",
    "按照",
    "要求",
    "通知",
    "说明",
    "确认",
    "请求",
    "判令",
    "并由",
    "后续由",
    "随后由",
    "另由",
    "另行由",
    "同时由",
    "共同由",
    "分别由",
)
NON_SUBJECT_GENERIC_REFERENCE_PREFIXES = (
    *LEADING_SUBJECT_FUNCTION_WORDS,
    "经",
    "由",
    "向",
    "对",
    "与",
    "和",
    "及",
    "以及",
    "同",
)
STRONG_LEADING_SUBJECT_FUNCTION_PREFIXES = tuple(
    dict.fromkeys(
        (
            *LEADING_SUBJECT_FUNCTION_WORDS,
            "将",
            "把",
            "被",
            "为",
        )
    )
)
SUBJECT_SEQUENCE_BOUNDARY_TERMS = (
    "与",
    "和",
    "及",
    "以及",
    "同",
    "通过",
    "经过",
    "经由",
    "根据",
    "依据",
    "按照",
    "要求",
    "通知",
    "说明",
    "确认",
    "请求",
    "判令",
    "由",
    "向",
    "对",
    "将",
    "把",
    "被",
    "为",
)
LEADING_SUBJECT_FUNCTION_PATTERN = re.compile(
    r"^(?:" + "|".join(re.escape(item) for item in sorted(LEADING_SUBJECT_FUNCTION_WORDS, key=len, reverse=True)) + r")+"
)
ADMIN_REGION_PREFIX_PATTERN = re.compile(
    r"(?:"
    r"(?:[\u4e00-\u9fa5]{2,12}(?:省|自治区|特别行政区))?"
    r"(?:[\u4e00-\u9fa5]{2,12}(?:市|地区|自治州|盟))?"
    r"(?:[\u4e00-\u9fa5]{1,12}(?:区|县|旗|市|自治县))?"
    r")$"
)
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
    "委托方",
    "受托方",
    "发包人",
    "承包人",
    "采购人",
    "供应商",
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
            "甲方",
            "乙方",
            "丙方",
            "委托方",
            "受托方",
            "发包人",
            "承包人",
            "采购人",
            "供应商",
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

NON_SUBJECT_EXACT_TERMS = {
    "通过",
    "通过了",
    "经过",
    "经由",
    "根据",
    "依据",
    "按照",
    "通知",
    "说明",
    "提交",
    "确认",
    "要求",
    "请求",
    "判令",
    "承担",
    "负责",
    "继续",
    "履行",
    "履约",
    "结算",
    "缴款",
    "支付",
    "转账",
    "汇款",
    "付款",
    "收款",
    "交付",
    "配合",
    "联系",
    "协商",
    "办理",
    "签订",
    "签署",
    "签约",
    "盖章",
    "落款",
    "对账",
    "施工",
    "供货",
    "我方",
    "对方",
    "双方",
    "各方",
    "一方",
    "另一方",
    "本方",
    "其",
    "该",
    "本",
    "以上",
    "如下",
    "附件",
    "前述",
    "上述",
    "相关",
    "涉案",
    "本案",
    "本合同",
    "本协议",
    "该公司",
    "贵公司",
    "客户公司",
    "公司提交",
    "提交材料",
    "公司认为",
    "公司名称",
}
NON_SUBJECT_ACTION_VERBS = (
    "通过",
    "经过",
    "根据",
    "依据",
    "按照",
    "通知",
    "说明",
    "提交",
    "确认",
    "要求",
    "请求",
    "判令",
    "承担",
    "负责",
    "继续",
    "履行",
    "履约",
    "结算",
    "缴款",
    "支付",
    "转账",
    "汇款",
    "付款",
    "收款",
    "交付",
    "配合",
    "联系",
    "协商",
    "办理",
    "签订",
    "签署",
    "签约",
    "盖章",
    "落款",
    "对账",
    "施工",
    "供货",
)
NON_SUBJECT_OFFICIAL_PREFIX_TOKENS = (
    *NON_SUBJECT_ACTION_VERBS,
    "将",
    "把",
    "至",
    "给",
    "从",
    "下列",
    "以下",
    "如下",
    "上述",
    "前述",
    "相关",
    "指定",
    "个人",
    "费用",
    "款项",
    "货款",
    "服务费",
    "技术服务费",
    "检测费",
)
NON_SUBJECT_PRONOUN_PREFIXES = (
    "该",
    "本",
    "其",
    "前述",
    "上述",
    "相关",
    "涉案",
)
NON_SUBJECT_GENERIC_TAILS = (
    "公司",
    "企业",
    "集团",
    "机构",
    "单位",
    "主体",
    "部门",
    "人员",
    "材料",
    "资料",
    "信息",
    "事项",
    "情况",
)
NON_SUBJECT_GENERIC_ORG_REFERENCE_TAILS = (
    "账户",
    "材料",
    "资料",
    "文件",
    "信息",
    "名称",
    "人员",
    "负责人",
    "联系人",
    "业务",
    "提交",
    "提交材料",
    "提交资料",
    "认为",
    "办理",
    "办理手续",
    "付款",
    "收款",
    "结算",
    "履约",
    "负责",
    "协助",
    "配合",
    "义务",
    "责任",
    "主体",
    "部门",
    "事项",
    "情况",
)
SHORT_ORG_BOUNDARY_ROLE_TAILS = (
    "工作人员",
    "经办人员",
    "项目负责人",
    "授权代表",
    "法定代表人",
    "委托代理人",
    "诉讼代理人",
    "负责人",
    "联系人",
    "经办人",
    "代理人",
    "签署人",
    "代表",
    "员工",
    "人员",
)
SHORT_ORG_BOUNDARY_OBJECT_TAILS = (
    "银行账户",
    "账户",
    "材料",
    "资料",
    "文件",
    "信息",
    "证明",
    "说明",
    "清单",
    "附件",
    "款项",
    "货款",
    "费用",
    "合同",
    "协议",
    "项目",
    "业务",
    "订单",
    "票据",
    "手续",
)
SHORT_ORG_BOUNDARY_ACTION_TERMS = tuple(
    dict.fromkeys(
        (
            *NON_SUBJECT_ACTION_VERBS,
            "审核",
            "补充",
            "维护",
            "管理",
            "提供",
            "发送",
            "接收",
            "出具",
            "移交",
            "处理",
            "完成",
        )
    )
)
_SHORT_ORG_BOUNDARY_ROLE_PATTERN = "|".join(
    re.escape(item) for item in sorted(SHORT_ORG_BOUNDARY_ROLE_TAILS, key=len, reverse=True)
)
_SHORT_ORG_BOUNDARY_OBJECT_PATTERN = "|".join(
    re.escape(item) for item in sorted(SHORT_ORG_BOUNDARY_OBJECT_TAILS, key=len, reverse=True)
)
_SHORT_ORG_BOUNDARY_ACTION_PATTERN = "|".join(
    re.escape(item) for item in sorted(SHORT_ORG_BOUNDARY_ACTION_TERMS, key=len, reverse=True)
)
_SHORT_ORG_BOUNDARY_ADVERB_PATTERN = r"(?:均|皆|共同|分别|各自)?"
SHORT_ORG_NON_SUBJECT_RIGHT_BOUNDARY_PATTERN = re.compile(
    rf"(?:"
    rf"(?P<role_tail>{_SHORT_ORG_BOUNDARY_ROLE_PATTERN})(?:\s*(?P<role_action>{_SHORT_ORG_BOUNDARY_ACTION_PATTERN}))?|"
    rf"(?P<object_tail>{_SHORT_ORG_BOUNDARY_OBJECT_PATTERN})(?:\s*(?P<object_action>{_SHORT_ORG_BOUNDARY_ACTION_PATTERN}))?|"
    rf"{_SHORT_ORG_BOUNDARY_ADVERB_PATTERN}(?P<direct_action>{_SHORT_ORG_BOUNDARY_ACTION_PATTERN})"
    rf")"
)
LOCATION_NOUN_MARKERS = (
    "省",
    "市",
    "区",
    "县",
    "镇",
    "乡",
    "村",
    "路",
    "街",
    "道",
    "号",
    "栋",
    "室",
    "自治区",
    "自治州",
    "自治县",
    "旗",
    "盟",
)
GOVERNMENT_SUBJECT_MARKERS = (
    *GOVERNMENT_INSTITUTION_SUFFIX_TERMS,
    "人民法院",
    "中级人民法院",
    "高级人民法院",
    "法院",
    "人民检察院",
    "检察院",
    "公安局",
    "派出所",
    "仲裁委员会",
    "人民政府",
    "政府",
    "管理委员会",
    "委员会",
)

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


def has_official_region_prefix(value: str) -> bool:
    normalized = normalize_entity_text(value)
    if not normalized:
        return False
    return bool(
        re.match(
            rf"^(?:{_OFFICIAL_REGION_PREFIX}|国务院|中央)",
            normalized,
        )
    )


def is_government_institution_text(value: str) -> bool:
    """Strict shape for courts, governments, arbitration bodies and agencies."""

    normalized = normalize_entity_text(value)
    if not normalized or is_generic_organization_term(normalized):
        return False
    if is_non_subject_generic_org_reference(normalized):
        return False
    if normalized in {"一审法院", "二审法院", "原审法院", "再审法院", "执行法院", "上级法院", "下级法院", "本院", "贵院"}:
        return False
    if normalized in NON_SUBJECT_EXACT_TERMS or normalized in NON_ENTITY_ROLE_TERMS:
        return False
    if not any(marker in normalized for marker in GOVERNMENT_SUBJECT_MARKERS):
        return False
    first_government_marker = min(
        (normalized.find(marker) for marker in GOVERNMENT_SUBJECT_MARKERS if normalized.find(marker) >= 0),
        default=-1,
    )
    if first_government_marker > 0 and any(
        marker in normalized[:first_government_marker]
        for marker in ("有限公司", "有限责任公司", "股份有限公司", "分公司", "子公司", "集团", "公司")
    ):
        return False
    if not GOVERNMENT_INSTITUTION_PATTERN.fullmatch(normalized):
        return False
    return has_official_region_prefix(normalized)


def is_financial_institution_text(value: str) -> bool:
    """Strict shape for financial institutions under the official-institution family."""

    normalized = normalize_entity_text(value)
    if not normalized or is_generic_organization_term(normalized):
        return False
    if is_non_subject_generic_org_reference(normalized):
        return False
    if any(token in normalized for token in ("开户银行", "开户行", "银行账户", "个人银行账户", "账户", "账号", "户名")):
        return False
    if not any(token in normalized for token in ("银行", "支行", "分行", "营业部", "分理处", "信用社", "联社")):
        return False
    bank_index = normalized.find("银行")
    if bank_index > 0:
        prefix = normalized[:bank_index]
        if any(token in prefix for token in NON_SUBJECT_OFFICIAL_PREFIX_TOKENS):
            return False
    if any(
        normalized.startswith(prefix)
        for prefix in (
            "通过",
            "经过",
            "经由",
            "根据",
            "依据",
            "按照",
            "要求",
            "通知",
            "说明",
            "确认",
            "请求",
            "判令",
            "向",
            "对",
            "与",
            "由",
        )
    ):
        return False
    if bank_index > 0 and any(token in normalized[:bank_index] for token in ("有限公司", "有限责任公司", "股份有限公司", "分公司", "子公司", "集团", "公司")):
        return False
    if not FINANCIAL_INSTITUTION_PATTERN.fullmatch(normalized):
        return False
    if has_official_region_prefix(normalized):
        return True
    if re.search(r"^(?:中国|国家|人民|中央|工商|建设|农业|交通|招商|民生|中信|光大|浦发|兴业|华夏|广发|平安|邮储|农商|农村|城市|商业)[\u4e00-\u9fa5A-Za-z0-9]{0,20}银行(?:[\u4e00-\u9fa5A-Za-z0-9]{1,20}(?:支行|分行|营业部|分理处))?$", normalized):
        return True
    if re.fullmatch(r"[\u4e00-\u9fa5A-Za-z0-9]{2,16}(?:农村信用合作联社|信用合作联社|信用社|联社)", normalized):
        return True
    return False


def is_official_institution_text(value: str) -> bool:
    return is_government_institution_text(value) or is_financial_institution_text(value)


def strip_leading_subject_function_words(value: str) -> tuple[str, int]:
    """Remove functional/context words swallowed before a real subject name."""

    text = str(value or "")
    cursor = 0
    while cursor < len(text) and text[cursor] in " \t\r\n:：,，;；。.!！?？、":
        cursor += 1
    stripped = text[cursor:]
    compact = normalize_entity_text(stripped)
    if not compact:
        return text, 0
    match = LEADING_SUBJECT_FUNCTION_PATTERN.match(compact)
    if not match:
        return text, 0
    prefix_len = match.end()
    consumed_non_space = 0
    local_cursor = cursor
    while local_cursor < len(text) and consumed_non_space < prefix_len:
        if text[local_cursor].isspace() or text[local_cursor] in ":：,，;；。.!！?？、":
            local_cursor += 1
            continue
        consumed_non_space += 1
        local_cursor += 1
    while local_cursor < len(text) and text[local_cursor] in " \t\r\n:：,，;；。.!！?？、":
        local_cursor += 1
    cleaned = text[local_cursor:]
    if cleaned and (is_org_like_text_no_function_guard(cleaned) or ORG_PATTERN.search(normalize_entity_text(cleaned))):
        return cleaned, local_cursor
    return text, 0


def is_org_like_text_no_function_guard(value: str) -> bool:
    normalized = normalize_entity_text(value)
    if not normalized or is_identity_reference_term(normalized) or is_generic_organization_term(normalized):
        return False
    if any(token in normalized for token in ("银行", "支行", "分行", "营业部", "分理处", "信用社", "联社")):
        return is_financial_institution_text(normalized)
    if any(marker in normalized for marker in GOVERNMENT_SUBJECT_MARKERS):
        return is_government_institution_text(normalized)
    prefixed = strip_identity_reference_prefix(normalized)
    if prefixed:
        if len(prefixed) < 2 or is_generic_organization_term(prefixed):
            return False
        normalized = prefixed
    if ORG_PATTERN.search(normalized):
        return True
    return any(marker in normalized for marker in ORGANIZATION_MARKERS)


def looks_like_admin_region_prefix(value: str) -> bool:
    normalized = normalize_entity_text(value)
    if not 2 <= len(normalized) <= 18:
        return False
    if is_non_subject_action_or_function_term(normalized):
        return False
    return bool(ADMIN_REGION_PREFIX_PATTERN.fullmatch(normalized)) and any(
        token in normalized for token in ("省", "自治区", "市", "区", "县", "旗", "州", "盟")
    )


def expand_org_span_with_left_region_prefix(
    source_text: str,
    start: int,
    end: int,
) -> Optional[tuple[int, int]]:
    """Include adjacent administrative prefix when it is part of a company name."""

    if not source_text or start < 0 or end <= start or end > len(source_text):
        return None
    value = source_text[start:end]
    if not is_org_like_text(value):
        return None
    left_start = max(0, start - 18)
    left = source_text[left_start:start]
    match = re.search(r"[\u4e00-\u9fa5]{1,18}$", left)
    if not match:
        return None
    prefix = match.group(0)
    if not looks_like_admin_region_prefix(prefix):
        return None
    expanded_start = left_start + match.start()
    boundary_left = source_text[max(0, expanded_start - 1) : expanded_start]
    if boundary_left and re.match(r"[\u4e00-\u9fa5A-Za-z0-9]", boundary_left):
        return None
    expanded_text = source_text[expanded_start:end]
    if not ORG_PATTERN.fullmatch(normalize_entity_text(expanded_text)) and not is_org_like_text_no_function_guard(expanded_text):
        return None
    return expanded_start, end


def is_weak_function_stripped_org(value: str) -> bool:
    normalized = normalize_entity_text(value)
    if not normalized or len(normalized) <= 6:
        return True
    core = re.sub(r"(?:股份有限公司|有限责任公司|集团有限公司|有限公司|集团|公司|分公司)$", "", normalized)
    return len(core) <= 2


@dataclass(frozen=True)
class RecognitionView:
    original_text: str
    sanitized_text: str
    original_to_sanitized: list[int]
    sanitized_to_original: list[int]
    removed_inline_space_count: int
    span_remap_fail_count: int = 0


def build_recognition_view(text: str) -> RecognitionView:
    """Remove intrusive inline whitespace without touching structural newlines.

    The legal source files sometimes contain OCR- or copy-induced spaces inside
    ordinary body text, which breaks short-name recognition. We only remove
    horizontal whitespace runs whose left and right neighbors are both content
    characters, then keep an index map so downstream spans can be projected back
    to the original source text.
    """

    source_text = text or ""
    if not text:
        return RecognitionView(
            original_text=source_text,
            sanitized_text="",
            original_to_sanitized=[],
            sanitized_to_original=[],
            removed_inline_space_count=0,
        )

    sanitized_chars: list[str] = []
    original_to_sanitized: list[int] = [-1] * len(source_text)
    sanitized_to_original: list[int] = []
    removed_inline_space_count = 0
    text_length = len(source_text)
    cursor = 0
    while cursor < text_length:
        char = source_text[cursor]
        if char in "\r\n":
            sanitized_chars.append(char)
            original_to_sanitized[cursor] = len(sanitized_to_original)
            sanitized_to_original.append(cursor)
            cursor += 1
            continue
        if char.isspace():
            space_end = cursor + 1
            while space_end < text_length and source_text[space_end].isspace() and source_text[space_end] not in "\r\n":
                space_end += 1
            left_char = sanitized_chars[-1] if sanitized_chars else ""
            right_char = source_text[space_end] if space_end < text_length else ""
            if _is_intrusive_inline_space(left_char, right_char):
                removed_inline_space_count += space_end - cursor
                cursor = space_end
                continue
            for original_index in range(cursor, space_end):
                sanitized_chars.append(source_text[original_index])
                original_to_sanitized[original_index] = len(sanitized_to_original)
                sanitized_to_original.append(original_index)
            cursor = space_end
            continue
        sanitized_chars.append(char)
        original_to_sanitized[cursor] = len(sanitized_to_original)
        sanitized_to_original.append(cursor)
        cursor += 1
    return RecognitionView(
        original_text=source_text,
        sanitized_text="".join(sanitized_chars),
        original_to_sanitized=original_to_sanitized,
        sanitized_to_original=sanitized_to_original,
        removed_inline_space_count=removed_inline_space_count,
    )


def sanitize_recognition_text(text: str) -> tuple[str, list[int]]:
    view = build_recognition_view(text or "")
    return view.sanitized_text, view.sanitized_to_original


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


def resolve_docx_unit_spans(text: str, units: Iterable[dict]) -> list[dict]:
    """Resolve DOCX text-unit spans with document-order constraints.

    A stale or missing unit span must not fall back to a global first match:
    repeated labels and repeated table values are common in legal documents.
    The fallback therefore searches forward from the previously resolved unit,
    preserving the parser's structural order. Unresolved units are kept with
    diagnostic metadata so they can be counted and reviewed without producing
    writable entities at the wrong location.
    """

    source_text = text or ""
    source_view = build_recognition_view(source_text)
    resolved_units: list[dict] = []
    cursor = 0
    for order_index, raw_unit in enumerate(units or []):
        if not isinstance(raw_unit, dict):
            continue
        unit = dict(raw_unit)
        unit_text = str(unit.get("text") or "")
        start = _coerce_docx_unit_int(unit.get("start"), -1)
        end = _coerce_docx_unit_int(unit.get("end"), -1)
        resolved_start = -1
        resolved_end = -1
        resolution = "unresolved"
        if not unit_text:
            resolution = "missing_text"
        elif (
            start >= 0
            and end > start
            and end <= len(source_text)
            and source_text[start:end] == unit_text
        ):
            resolved_start = start
            resolved_end = end
            resolution = "exact"
            cursor = max(cursor, resolved_end)
        else:
            found = source_text.find(unit_text, max(0, min(cursor, len(source_text))))
            if found >= 0:
                resolved_start = found
                resolved_end = found + len(unit_text)
                resolution = "ordered_forward"
                cursor = max(cursor, resolved_end)
            else:
                sanitized_unit_text, _unit_index_map = sanitize_recognition_text(unit_text)
                if sanitized_unit_text:
                    sanitized_cursor = 0
                    if 0 <= cursor < len(source_view.original_to_sanitized):
                        mapped_cursor = source_view.original_to_sanitized[cursor]
                        sanitized_cursor = max(0, mapped_cursor if mapped_cursor >= 0 else 0)
                    sanitized_found = source_view.sanitized_text.find(sanitized_unit_text, sanitized_cursor)
                    span = remap_sanitized_span(
                        source_view.sanitized_to_original,
                        sanitized_found,
                        sanitized_found + len(sanitized_unit_text),
                    ) if sanitized_found >= 0 else None
                    if span is not None:
                        resolved_start, resolved_end = span
                        resolution = "sanitized_ordered_forward"
                        cursor = max(cursor, resolved_end)
        unit["_resolved_order_index"] = order_index
        unit["_resolved_start"] = resolved_start
        unit["_resolved_end"] = resolved_end
        unit["_span_resolution"] = resolution
        resolved_units.append(unit)
    return resolved_units


def iter_docx_structure_units(source_structure: dict[str, Any] | None) -> Iterator[dict[str, Any]]:
    """Yield the canonical DOCX structure-unit stream used by recognition.

    ``docx_text_units`` is the parser's complete source-map stream. ``pages[*].units``
    is a page-oriented view built from the same units and therefore mostly a
    duplicate view. Feeding both streams directly into ordered span resolution
    pushes the resolver cursor through the document twice, which can turn real
    coverage into misleading duplicate/unresolved diagnostics. Keep raw units as
    the canonical stream and only add page units that are not already present in
    the raw stream.
    """

    if not isinstance(source_structure, dict):
        return

    raw_unit_ids: set[str] = set()
    raw_units = source_structure.get("docx_text_units")
    if isinstance(raw_units, list):
        for unit in raw_units:
            if not isinstance(unit, dict):
                continue
            unit_id = str(unit.get("unit_id") or "").strip()
            if unit_id:
                raw_unit_ids.add(unit_id)
            yield unit

    pages = source_structure.get("pages")
    if not isinstance(pages, list):
        return
    for page in pages:
        if not isinstance(page, dict):
            continue
        page_units = page.get("units")
        if not isinstance(page_units, list):
            continue
        for unit in page_units:
            if not isinstance(unit, dict):
                continue
            unit_id = str(unit.get("unit_id") or "").strip()
            if unit_id and unit_id in raw_unit_ids:
                continue
            yield unit


def docx_structure_unit_inventory(source_structure: dict[str, Any] | None) -> dict[str, int]:
    """Return privacy-safe counters for DOCX structure-unit source views."""

    if not isinstance(source_structure, dict):
        return {
            "raw_docx_text_unit_count": 0,
            "page_docx_unit_count": 0,
            "page_docx_unit_duplicate_raw_id_count": 0,
            "page_docx_unit_unique_extra_count": 0,
            "canonical_docx_structure_unit_count": 0,
            "raw_docx_text_unit_duplicate_id_count": 0,
        }

    raw_unit_ids: set[str] = set()
    duplicate_raw_ids = 0
    raw_count = 0
    raw_units = source_structure.get("docx_text_units")
    if isinstance(raw_units, list):
        for unit in raw_units:
            if not isinstance(unit, dict):
                continue
            raw_count += 1
            unit_id = str(unit.get("unit_id") or "").strip()
            if not unit_id:
                continue
            if unit_id in raw_unit_ids:
                duplicate_raw_ids += 1
            raw_unit_ids.add(unit_id)

    page_count = 0
    duplicate_page_ids = 0
    unique_page_extra = 0
    pages = source_structure.get("pages")
    if isinstance(pages, list):
        for page in pages:
            if not isinstance(page, dict):
                continue
            page_units = page.get("units")
            if not isinstance(page_units, list):
                continue
            for unit in page_units:
                if not isinstance(unit, dict):
                    continue
                page_count += 1
                unit_id = str(unit.get("unit_id") or "").strip()
                if unit_id and unit_id in raw_unit_ids:
                    duplicate_page_ids += 1
                else:
                    unique_page_extra += 1

    return {
        "raw_docx_text_unit_count": raw_count,
        "page_docx_unit_count": page_count,
        "page_docx_unit_duplicate_raw_id_count": duplicate_page_ids,
        "page_docx_unit_unique_extra_count": unique_page_extra,
        "canonical_docx_structure_unit_count": raw_count + unique_page_extra,
        "raw_docx_text_unit_duplicate_id_count": duplicate_raw_ids,
    }


def _coerce_docx_unit_int(value: object, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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


def subject_left_pollution_reason(value: str, entity_type: str = "") -> str:
    """Return why a subject candidate still contains non-subject left context."""

    normalized = normalize_entity_text(value)
    if not normalized:
        return ""
    for prefix in sorted(STRONG_LEADING_SUBJECT_FUNCTION_PREFIXES, key=len, reverse=True):
        if normalized.startswith(prefix) and len(normalized) > len(prefix) + 1:
            remainder = normalized[len(prefix) :]
            if (
                ORG_PATTERN.search(remainder)
                or OFFICIAL_INSTITUTION_PATTERN.search(remainder)
                or looks_like_organization_short_name(remainder)
                or is_org_like_text_no_function_guard(remainder)
            ):
                return "leading_function_prefix"
    previous_subject_boundary = _previous_subject_boundary_index(normalized)
    if previous_subject_boundary >= 0:
        return "previous_subject_prefix"
    if _company_prefix_before_official_marker(normalized):
        return "company_prefix_before_official_institution"
    return ""


def _previous_subject_boundary_index(normalized: str) -> int:
    first_org_end = _first_company_subject_end(normalized)
    if first_org_end < 0 or first_org_end >= len(normalized) - 1:
        return -1
    tail = normalized[first_org_end:]
    for boundary in sorted(SUBJECT_SEQUENCE_BOUNDARY_TERMS, key=len, reverse=True):
        if not tail.startswith(boundary):
            continue
        remainder = tail[len(boundary) :]
        if len(remainder) < 2:
            continue
        if ORG_PATTERN.search(remainder) or OFFICIAL_INSTITUTION_PATTERN.search(remainder):
            return first_org_end
    return -1


def _first_company_subject_end(normalized: str) -> int:
    endings = []
    for suffix in (
        "股份有限公司",
        "有限责任公司",
        "集团有限公司",
        "有限公司",
        "分公司",
        "子公司",
        "集团",
        "公司",
    ):
        index = normalized.find(suffix)
        if index >= 0:
            endings.append(index + len(suffix))
    return min(endings) if endings else -1


def _company_prefix_before_official_marker(normalized: str) -> bool:
    first_company_end = _first_company_subject_end(normalized)
    if first_company_end < 0:
        return False
    official_indices = [
        normalized.find(marker)
        for marker in (*GOVERNMENT_SUBJECT_MARKERS, "银行", "支行", "分行", "营业部", "分理处", "信用社", "联社")
        if normalized.find(marker) >= 0
    ]
    if not official_indices:
        return False
    first_official = min(official_indices)
    return first_company_end <= first_official


def is_non_subject_action_or_function_term(value: str) -> bool:
    """Reject verbs, pronouns and functional words that are not legal subjects.

    This is intentionally conservative. Function/action words are allowed to
    act as boundary evidence elsewhere, but this predicate should only reject
    text that is itself non-subject. A polluted candidate such as
    "通知北京某某有限公司" still contains a real subject and must reach boundary
    repair instead of being dropped here.
    """

    normalized = normalize_entity_text(value)
    if not normalized:
        return False
    if is_non_subject_generic_org_reference(normalized):
        return True
    stripped, consumed = strip_leading_subject_function_words(normalized)
    if consumed > 0:
        stripped_normalized = normalize_entity_text(stripped)
        if stripped_normalized and (
            ORG_PATTERN.search(stripped_normalized)
            or OFFICIAL_INSTITUTION_PATTERN.search(stripped_normalized)
            or is_org_like_text_no_function_guard(stripped_normalized)
        ):
            return False
        return True
    if ORG_PATTERN.search(normalized) and any(
        normalized.startswith(verb) and len(normalized) > len(verb) + 1
        for verb in NON_SUBJECT_ACTION_VERBS
    ):
        return True
    if ORG_PATTERN.search(normalized):
        return False
    if normalized in NON_SUBJECT_EXACT_TERMS:
        return True
    if len(normalized) <= 8:
        if re.fullmatch(
            rf"(?:已|已经|均|共同|另行|继续|仍|应|需|由)?(?:{'|'.join(map(re.escape, NON_SUBJECT_ACTION_VERBS))})(?:了|后|前|中|时|的|义务|责任|材料|资料|事项)?",
            normalized,
        ):
            return True
        if any(normalized.startswith(prefix) for prefix in NON_SUBJECT_PRONOUN_PREFIXES) and any(
            normalized.endswith(tail) for tail in NON_SUBJECT_GENERIC_TAILS
        ):
            return True
    return False


def is_non_subject_generic_org_reference(value: str) -> bool:
    """Detect function/action/pronoun references to generic organization nouns."""

    normalized = normalize_entity_text(value)
    if not normalized:
        return False
    function_prefix = "|".join(
        re.escape(item) for item in sorted(NON_SUBJECT_GENERIC_REFERENCE_PREFIXES, key=len, reverse=True)
    )
    generic_org = "|".join(re.escape(item) for item in sorted(NON_SUBJECT_GENERIC_TAILS, key=len, reverse=True))
    generic_tail = "|".join(
        re.escape(item) for item in sorted(NON_SUBJECT_GENERIC_ORG_REFERENCE_TAILS, key=len, reverse=True)
    )
    action_tail = "|".join(re.escape(item) for item in sorted(NON_SUBJECT_ACTION_VERBS, key=len, reverse=True))
    if re.fullmatch(rf"(?:{function_prefix})+(?:该|本|其|前述|上述|相关|涉案)?(?:{generic_org})(?:{generic_tail})?", normalized):
        return True
    if re.fullmatch(rf"(?:该|本|其|前述|上述|相关|涉案)(?:{generic_org})(?:{generic_tail})?", normalized):
        return True
    if re.fullmatch(rf"(?:{generic_org})(?:{generic_tail})", normalized):
        return True
    if re.fullmatch(rf"(?:{generic_org})(?:{action_tail})(?:{generic_tail})?", normalized):
        return True
    return False


def is_org_like_text(value: str) -> bool:
    normalized = normalize_entity_text(value)
    if not normalized or is_identity_reference_term(normalized):
        return False
    if is_non_subject_action_or_function_term(normalized):
        return False
    if is_generic_organization_term(normalized):
        return False
    if any(token in normalized for token in ("银行", "支行", "分行", "营业部", "分理处", "信用社", "联社")):
        return is_financial_institution_text(normalized)
    if any(marker in normalized for marker in GOVERNMENT_SUBJECT_MARKERS):
        return is_government_institution_text(normalized)
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
    if "·" in text:
        return bool(re.fullmatch(r"[\u4e00-\u9fa5]{1,8}·[\u4e00-\u9fa5]{2,8}", text))
    if not 2 <= len(text) <= 8:
        return False
    if re.fullmatch(r"[\u4e00-\u9fa5]{2,4}", text):
        for surname in COMMON_COMPOUND_SURNAMES:
            if text.startswith(surname) and 1 <= len(text) - len(surname) <= 2:
                return True
        return text[0] in COMMON_SINGLE_SURNAMES
    return False


def looks_like_organization_short_name(value: str) -> bool:
    normalized = normalize_entity_text(value)
    if not 2 <= len(normalized) <= 6:
        return False
    if is_non_subject_action_or_function_term(normalized):
        return False
    if is_identity_reference_term(normalized) or is_position_title(normalized):
        return False
    if is_generic_organization_term(normalized) or normalized in NON_SUBJECT_GENERIC_TAILS:
        return False
    if any(token in normalized for token in ("省", "市", "区", "县", "镇", "乡", "村", "路", "街", "号", "室")):
        return False
    return re.fullmatch(r"[\u4e00-\u9fa5A-Za-z0-9]{2,6}", normalized) is not None


def short_org_boundary_has_strong_surface(value: str) -> bool:
    normalized = normalize_entity_text(value)
    if not normalized:
        return False
    return bool(re.search(r"[A-Za-z0-9]", normalized)) or len(normalized) >= 4


SHORT_ORG_TRAILING_BOUNDARY_PARTICLES = frozenset("均皆各")
SHORT_ORG_LEADING_FUNCTION_PREFIXES = tuple(
    dict.fromkeys(
        (
            *LEADING_SUBJECT_FUNCTION_WORDS,
            "将",
            "把",
            "被",
            "为",
        )
    )
)


def _strip_short_org_leading_function_prefix_span(value: str) -> Optional[tuple[int, int, str]]:
    text = str(value or "")
    if not text:
        return None
    ordered_prefixes = sorted(SHORT_ORG_LEADING_FUNCTION_PREFIXES, key=len, reverse=True)
    compact = normalize_entity_text(text)
    for prefix in ordered_prefixes:
        if not compact.startswith(prefix):
            continue
        consumed_non_space = 0
        local_start = 0
        while local_start < len(text) and consumed_non_space < len(prefix):
            if text[local_start].isspace() or text[local_start] in ":：,，;；。.!！?？、":
                local_start += 1
                continue
            consumed_non_space += 1
            local_start += 1
        while local_start < len(text) and text[local_start] in " \t\r\n:：,，;；。.!！?？、":
            local_start += 1
        stripped = text[local_start:]
        normalized = normalize_entity_text(stripped)
        if _is_valid_short_org_boundary_subject(normalized):
            return local_start, len(text), normalized
    return None


def _is_valid_short_org_boundary_subject(value: str) -> bool:
    normalized = normalize_entity_text(value)
    if not looks_like_organization_short_name(normalized) and not looks_like_short_org_with_company_suffix(normalized):
        return False
    if any(
        normalized.startswith(prefix) and len(normalized) > len(prefix) + 1
        for prefix in SHORT_ORG_LEADING_FUNCTION_PREFIXES
    ):
        return False
    if normalized[-1:] in SHORT_ORG_TRAILING_BOUNDARY_PARTICLES:
        return False
    if any(token in normalized for token in ("均", "皆", "共同", "分别", "各自")):
        return False
    if any(token in normalized for token in NON_SUBJECT_ACTION_VERBS):
        return False
    if any(
        token in normalized
        for token in ORGANIZATION_MARKERS
        if token not in {"公司", "分公司"}
    ):
        return False
    return True


def looks_like_short_org_with_company_suffix(value: str) -> bool:
    normalized = normalize_entity_text(value)
    if not 4 <= len(normalized) <= 9:
        return False
    suffix = ""
    for item in ("分公司", "公司"):
        if normalized.endswith(item):
            suffix = item
            break
    if not suffix:
        return False
    stem = normalized[: -len(suffix)]
    return looks_like_organization_short_name(stem)


def find_short_org_prefix_before_non_subject_boundary(value: str) -> Optional[tuple[int, int, str]]:
    """Find a short organization surface immediately before non-subject context.

    This is the single short-name right-boundary splitter used by deterministic
    recall and boundary repair. It deliberately scores role/object tails above
    direct action words so a phrase like "星河材料提交" cuts at "星河", not
    "星河材料".
    """

    text = str(value or "")
    if not text:
        return None
    local_start = 0
    while local_start < len(text) and text[local_start] in " \t\r\n:：,，;；。.!！?？、":
        local_start += 1
    candidates: list[tuple[tuple[int, int, int], tuple[int, int, str]]] = []
    max_end = min(len(text), local_start + 9)
    for local_end in range(local_start + 2, max_end + 1):
        subject = text[local_start:local_end]
        normalized_subject = normalize_entity_text(subject)
        subject_start = local_start
        if not _is_valid_short_org_boundary_subject(normalized_subject):
            stripped_subject = _strip_short_org_leading_function_prefix_span(subject)
            if not stripped_subject:
                continue
            stripped_start, stripped_end, normalized_subject = stripped_subject
            subject_start = local_start + stripped_start
            if local_start + stripped_end != local_end:
                continue
        if not _is_valid_short_org_boundary_subject(normalized_subject):
            continue
        boundary = SHORT_ORG_NON_SUBJECT_RIGHT_BOUNDARY_PATTERN.match(text[local_end:])
        if not boundary:
            continue
        if boundary.group("role_tail"):
            boundary_kind = "role_tail"
            kind_score = 4
        elif boundary.group("object_tail"):
            boundary_kind = "object_tail"
            kind_score = 4
        else:
            boundary_kind = "action_tail"
            kind_score = 2
        surface_score = 1 if short_org_boundary_has_strong_surface(normalized_subject) else 0
        candidates.append(((kind_score, surface_score, local_end - subject_start), (subject_start, local_end, boundary_kind)))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def is_location_like_text(value: str) -> bool:
    normalized = normalize_entity_text(value)
    if not normalized or len(normalized) < 2:
        return False
    if is_non_subject_action_or_function_term(normalized):
        return False
    if is_identity_reference_term(normalized) or is_position_title(normalized):
        return False
    if is_org_like_text(normalized) or is_generic_organization_term(normalized):
        return False
    if re.search(r"\d{4}年|\d{1,2}月|\d{1,2}日", normalized):
        return False
    if ADMIN_REGION_PREFIX_PATTERN.fullmatch(normalized) and any(token in normalized for token in LOCATION_NOUN_MARKERS):
        return True
    return len(normalized) >= 6 and any(token in normalized for token in LOCATION_NOUN_MARKERS)


def subject_noun_gate(
    entity_type: str,
    value: str,
    *,
    allow_short_org: bool = False,
) -> tuple[bool, str]:
    """Single entrance check for identity-bearing default subject candidates."""

    normalized_type = str(entity_type or "").strip().upper()
    normalized = normalize_entity_text(value)
    if not normalized:
        return False, "empty_text"
    if normalized_type in {"PERSON_NAME", "LEGAL_REPRESENTATIVE", "CONTACT_PERSON", "SIGNATORY"}:
        normalized_type = "PERSON"
    elif normalized_type in {"COMPANY_NAME", "ACCOUNT_NAME"}:
        normalized_type = "ORGANIZATION"
    elif normalized_type == "BANK_NAME":
        if not is_financial_institution_text(normalized):
            return False, "weak_financial_institution_shape"
        normalized_type = "GOVERNMENT"
    elif normalized_type in {"COURT", "GOVERNMENT_AGENCY"}:
        normalized_type = "GOVERNMENT"
    elif normalized_type == "ADDRESS":
        normalized_type = "LOCATION"
    if normalized_type not in {"PERSON", "ORGANIZATION", "LOCATION", "GOVERNMENT"}:
        return False, "not_default_subject_type"
    if normalized_type in {"ORGANIZATION", "GOVERNMENT"}:
        left_pollution = subject_left_pollution_reason(normalized, normalized_type)
        if left_pollution:
            return False, left_pollution
    if is_non_subject_generic_org_reference(normalized):
        return False, "generic_org_reference"
    if is_non_subject_action_or_function_term(normalized):
        return False, "non_subject_action_or_function_term"
    if is_identity_reference_term(normalized):
        return False, "identity_reference_term"
    if normalized_type == "PERSON":
        if is_position_title(normalized):
            return False, "position_title"
        if is_org_like_text(normalized) or is_location_like_text(normalized):
            return False, "not_person_shape"
        return (True, "person_noun_shape") if is_probable_person(normalized) else (False, "weak_person_shape")
    if normalized_type in {"ORGANIZATION", "GOVERNMENT"}:
        if is_position_title(normalized):
            return False, "position_title"
        if is_generic_organization_term(normalized):
            return False, "generic_organization_term"
        if normalized_type == "GOVERNMENT":
            return (
                (True, "government_noun_shape")
                if is_government_institution_text(normalized) or is_financial_institution_text(normalized)
                else (False, "weak_government_shape")
            )
        if is_official_institution_text(normalized):
            return False, "official_institution_requires_government_family"
        if is_org_like_text(normalized):
            return True, "organization_noun_shape"
        if allow_short_org and looks_like_organization_short_name(normalized):
            return True, "short_organization_noun_shape"
        if is_probable_person(normalized) and not is_org_like_text(normalized):
            return False, "person_shape"
        return False, "weak_organization_shape"
    if normalized_type == "LOCATION":
        return (True, "location_noun_shape") if is_location_like_text(normalized) else (False, "weak_location_shape")
    return False, "unsupported_subject_type"


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
    if "·" in text and looks_like_natural_person_name(text):
        return True
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
        return "ORGANIZATION" if is_org_like_text(value_text) else ""
    if "户名" in label_text or "账户名称" in label_text or "账户名" in label_text:
        return "ORGANIZATION" if is_org_like_text(value_text) else ""
    if "项目" in label_text or "工程" in label_text:
        return "PROJECT"
    if "地址" in label_text or "住所" in label_text or "送达" in label_text:
        return "ADDRESS"
    if normalized_label in POSITION_LABEL_TERMS or any(token in label_text for token in POSITION_LABEL_TERMS):
        return "POSITION"
    if normalized_label in IDENTITY_REFERENCE_TERMS:
        if any(token in value_text for token in ("法院", "检察院", "公安局", "派出所", "仲裁委员会", "政府", "委员会")):
            return "GOVERNMENT"
        if is_org_like_text(normalized_value):
            return "ORGANIZATION"
        if is_position_title(normalized_value):
            return "POSITION"
        if looks_like_natural_person_name(normalized_value):
            return "PERSON"
    if looks_like_natural_person_name(normalized_value):
        return "PERSON"
    if any(token in value_text for token in ("法院", "检察院", "公安局", "派出所", "仲裁委员会", "政府", "委员会")):
        return "GOVERNMENT"
    if is_org_like_text(value_text):
        return "ORGANIZATION"
    if is_position_title(value_text):
        return "POSITION"
    if is_probable_person(value_text):
        return "PERSON"
    return ""


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


_SUBJECT_CONTEXT_PREFIX_NOISE = re.compile(
    r"^(?:"
    r"(?:甲方|乙方|丙方|委托方|受托方|发包人|承包人|采购人|供应商|上诉人|被上诉人|"
    r"原审原告|原审被告|一审原告|一审被告|申请人|被申请人|原告|被告|第三人)"
    r"\s*[:：]?\s*"
    r")?"
    r"(?:(?:通过|经过|根据|依据|按照|经由|经|由|并由|后续由|随后由|另由|另行由|"
    r"同时由|共同由|分别由|向|对|与|和|及|以及|同)\s*)+"
)


def expand_subject_span_to_containing_shape(
    source_text: str,
    start: int,
    end: int,
    entity_type: str,
) -> Optional[tuple[int, int]]:
    """Expand a partial subject span to the containing full org/government name."""

    normalized_type = str(entity_type or "").upper()
    if normalized_type not in {"ORGANIZATION", "COMPANY_NAME", "GOVERNMENT", "COURT", "ACCOUNT_NAME", "ALIAS"}:
        return None
    if not source_text or start < 0 or end <= start or end > len(source_text):
        return None

    search_start = max(0, start - 80)
    search_end = min(len(source_text), end + 80)
    window = source_text[search_start:search_end]
    spans: list[tuple[int, int]] = []
    patterns = (
        (OFFICIAL_INSTITUTION_PATTERN,)
        if normalized_type in {"GOVERNMENT", "COURT"}
        else (ORG_PATTERN,)
    )
    for pattern in patterns:
        for match in pattern.finditer(window):
            candidate_start = search_start + match.start()
            candidate_end = search_start + match.end()
            if candidate_start <= start and candidate_end >= end:
                cleaned = _clean_containing_subject_span(
                    source_text,
                    candidate_start,
                    candidate_end,
                    start,
                    end,
                )
                if cleaned is not None:
                    spans.append(cleaned)
    if normalized_type in {"GOVERNMENT", "COURT"}:
        spans = [
            span
            for span in spans
            if is_official_institution_text(source_text[span[0] : span[1]])
        ]
    if not spans:
        return None

    original_len = end - start
    valid = [
        span
        for span in spans
        if span[0] <= start
        and span[1] >= end
        and span[1] - span[0] > original_len
        and normalize_entity_text(source_text[span[0] : span[1]])
    ]
    if not valid:
        return None
    return max(valid, key=lambda span: (span[1] - span[0], -span[0]))


def _clean_containing_subject_span(
    source_text: str,
    candidate_start: int,
    candidate_end: int,
    anchor_start: int,
    anchor_end: int,
) -> Optional[tuple[int, int]]:
    value = source_text[candidate_start:candidate_end]
    if not value:
        return None
    local_start = 0
    local_end = len(value)

    prefix = _SUBJECT_CONTEXT_PREFIX_NOISE.match(value)
    if prefix and candidate_start + prefix.end() <= anchor_start:
        local_start = prefix.end()

    while local_start < local_end and value[local_start] in " \t\r\n:：,，;；。.!！?？、":
        local_start += 1
    while local_end > local_start and value[local_end - 1] in " \t\r\n:：,，;；。.!！?？、":
        local_end -= 1

    cleaned_start = candidate_start + local_start
    cleaned_end = candidate_start + local_end
    if cleaned_start > anchor_start or cleaned_end < anchor_end or cleaned_end <= cleaned_start:
        return None
    normalized = normalize_entity_text(source_text[cleaned_start:cleaned_end])
    if not normalized or is_generic_organization_term(normalized):
        return None
    return cleaned_start, cleaned_end


def deduplicate_results(results: Iterable[RecognizerResult]) -> List[RecognizerResult]:
    seen: set[tuple[str, int, int, str]] = set()
    deduped: list[RecognizerResult] = []
    for result in sorted(results, key=lambda item: (item.start, item.end, item.entity_type, -item.score)):
        key = (result.entity_type, result.start, result.end, result.text)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(result)
    return _drop_locations_embedded_in_complete_subjects(deduped)


def _drop_locations_embedded_in_complete_subjects(results: list[RecognizerResult]) -> List[RecognizerResult]:
    complete_subjects = [
        item
        for item in results
        if item.entity_type in {"ORGANIZATION", "GOVERNMENT"}
        and is_org_like_text_no_function_guard(item.text)
    ]
    if not complete_subjects:
        return results
    kept: list[RecognizerResult] = []
    for item in results:
        if item.entity_type == "LOCATION" and any(
            subject.start <= item.start
            and subject.end >= item.end
            and normalize_entity_text(item.text) in normalize_entity_text(subject.text)
            for subject in complete_subjects
        ):
            continue
        kept.append(item)
    return kept
