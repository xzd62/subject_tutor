"""生成模拟教材到 data/raw，用于端到端验证转换/切块/入库流程。

覆盖的典型结构：
- 多级标题（H1/H2/H3）、超长小节（触发降级切分）、过小小节（触发合并）；
- markdown 表格、围栏内联公式、跨空行的 $$ 公式块；
- 中英混排与英文长段落；
- GBK 编码的 txt（验证编码回退，而非 MarkItDown）。
"""

from __future__ import annotations

from pathlib import Path

from sample_corpus import EXTRA_DOCS, render_doc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

MATH_MD = """# 第一章 有理数

本章学习有理数的概念、数轴、相反数、绝对值以及有理数的加减乘除运算，是初中数学代数部分的起点。

## 1.1 正数和负数

在天气预报中，零上 5 ℃ 记作 $+5$ ℃，零下 5 ℃ 记作 $-5$ ℃；在记账时，收入 200 元记作 $+200$ 元，支出 150 元记作 $-150$ 元。像 $+5$、$+200$ 这样大于 0 的数叫做正数，像 $-5$、$-150$ 这样在正数前面加上符号“-”的数叫做负数。0 既不是正数也不是负数，它是正数和负数的分界。

### 1.1.1 用正负数表示相反意义的量

在同一问题中，分别用正数和负数表示的量具有相反的意义。例如水位上升 3 m 记作 $+3$ m，那么水位下降 2 m 就记作 $-2$ m；如果向东走 4 m 记作 $+4$ m，那么向西走 4 m 就记作 $-4$ m。判断一个量用正数还是负数表示，关键要看清题目规定的“基准”和“正方向”。

### 1.1.2 数轴

规定了原点、正方向和单位长度的直线叫做数轴。任何一个有理数都可以用数轴上的一个点来表示。数轴上右边的点表示的数总比左边的点表示的数大。例如 $-3 < -1 < 0 < 2$。

> 易错提醒：数轴的三要素——原点、正方向、单位长度——缺一不可。

## 1.2 有理数的运算

有理数的加法法则：同号两数相加，取相同的符号，并把绝对值相加；异号两数相加，取绝对值较大的加数的符号，并用较大的绝对值减去较小的绝对值；互为相反数的两个数相加得 0；一个数同 0 相加，仍得这个数。有理数的减法法则：减去一个数，等于加上这个数的相反数，即 $a-b=a+(-b)$。有理数的乘法法则：两数相乘，同号得正，异号得负，并把绝对值相乘；任何数同 0 相乘都得 0。有理数的除法法则：除以一个不等于 0 的数，等于乘这个数的倒数；两数相除，同号得正，异号得负，并把绝对值相除。有理数的混合运算顺序：先算乘方，再算乘除，最后算加减；同级运算从左到右进行；如果有括号，先做括号内的运算，按小括号、中括号、大括号依次进行。在运算中要特别注意符号问题，例如 $-2^2=-4$，而 $(-2)^2=4$，两者意义完全不同。负数的奇次幂是负数，负数的偶次幂是正数；任何不等于 0 的数的 0 次幂都等于 1；0 的任何正整数次幂都是 0。每天坚持 10 分钟口算训练，可以显著提升运算的准确率和速度，减少考试中的无谓失分。

## 1.3 科学记数法与近似数

把一个大于 10 的数表示成 $a\\times10^n$ 的形式（其中 $1\\le a<10$，$n$ 为正整数），这种记数法叫做科学记数法。例如 $6020000=6.02\\times10^6$。近似数与有效数字：从一个数的左边第一个非 0 数字起，到末位数字止，所有的数字都是这个数的有效数字。

## 1.4 本章小结

本章重点是理解正负数的意义，掌握有理数的四则运算。

## 1.5 拓展阅读

| 历史人物 | 贡献 | 年代 |
| --- | --- | --- |
| 刘徽 | 注解《九章算术》，提出正负数概念 | 三国时期 |
| 秦九韶 | 提出正负开方术 | 南宋 |
| 笛卡尔 | 引入坐标系 | 17 世纪 |

## 1.6 数学活动

$$
\\begin{aligned}
(-3)+(-5) &= -(3+5) = -8 \\\\
(-3)\\times(-5) &= 15 \\\\
(-2)^3 &= -8
\\end{aligned}
$$

# 第二章 整式的加减

## 2.1 用字母表示数

用字母表示数可以简明地表达数量关系与运算规律，例如加法交换律可以写成 $a+b=b+a$，乘法分配律可以写成 $a(b+c)=ab+ac$。书写含有字母的式子时，要注意：数与字母相乘时乘号通常省略不写，数字写在字母前面；带分数与字母相乘时要化成假分数；除法运算一般写成分数的形式。
"""

ENGLISH_MD = """# Unit 1 My name's Gina

## Section A 1a-2d

Good morning! My name is Gina. What's your name? Nice to meet you! 早上好！我叫吉娜。你叫什么名字？很高兴认识你！

— What's his name? 他叫什么名字？
— His name is Eric. 他叫埃里克。

重点句型：
- What's your name? My name is ...
- Nice to meet you. Nice to meet you, too.

## Section A Grammar Focus

| 人称代词 | 形容词性物主代词 | 例句 |
| --- | --- | --- |
| I | my | My name is Gina. |
| you | your | What's your name? |
| he | his | His name is Eric. |
| she | her | Her name is Mary. |

## Section B Reading

Hello, everyone. My name is Jenny. I am twelve years old. I am a student in No. 1 Middle School. This is my friend, Tony. He is thirteen. His favorite subject is science. He thinks science is interesting and useful. We are in the same class, and we often help each other with our homework after school. On weekends, we play basketball together. Tony plays it very well. I like reading books in the library. There are many books about history and geography. I want to be a teacher when I grow up, because I like children and I enjoy sharing what I know. 大家好，我叫珍妮，今年十二岁，是第一中学的学生。这是我的朋友托尼，他十三岁，最喜欢的科目是科学，他觉得科学既有趣又有用。我们在同一个班，放学后经常互相帮助完成作业。周末我们一起打篮球，托尼打得很好。我喜欢在图书馆读书，那里有很多历史和地理方面的书。我长大后想当一名老师。

## Section B Vocabulary

| 英文 | 音标 | 中文 |
| --- | --- | --- |
| name | /neɪm/ | 名字 |
| meet | /miːt/ | 遇见 |
| friend | /frend/ | 朋友 |
| subject | /ˈsʌbdʒɪkt/ | 科目 |

## Grammar Tip

形容词性物主代词后面必须接名词。
"""

MATH_TXT = """# 第五章 相交线与平行线

## 5.1 相交线

两条直线相交所成的四个角中，有公共顶点且两边互为反向延长线的两个角叫做对顶角。对顶角相等。

## 5.2 平行线及其判定

在同一平面内，不相交的两条直线叫做平行线。同位角相等，两直线平行；内错角相等，两直线平行；同旁内角互补，两直线平行。
"""


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    files = {
        RAW_DIR / "数学-人教版-七上.md": MATH_MD,
        RAW_DIR / "英语-人教版-七上.md": ENGLISH_MD,
    }
    for path, content in files.items():
        path.write_text(content, encoding="utf-8")
        print(f"已生成 {path} ({len(content)} 字符)")
    txt_path = RAW_DIR / "数学-人教版-七下.txt"
    txt_path.write_bytes(MATH_TXT.encode("gbk"))
    print(f"已生成 {txt_path} ({len(MATH_TXT)} 字符, GBK)")

    for stem, doc in EXTRA_DOCS.items():
        path = RAW_DIR / f"{stem}.md"
        content = render_doc(stem, doc)
        path.write_text(content, encoding="utf-8")
        print(f"已生成 {path} ({len(content)} 字符)")


if __name__ == "__main__":
    main()
