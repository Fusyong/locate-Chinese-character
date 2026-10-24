# hanloc

定位一页图像上的汉字：给出四点外框、阅读顺序和拉正后的字图。识别文字与置信度一并保存，供重建语流使用。识别对错不作为结果是否可用的标准。

第一期版式：

| 版式 | 页面 | 做法 |
| --- | --- | --- |
| `book` | 现代本，横排 | 检出文本行，再按行内墨迹切单字。用识别出的字、汉字和标点的不同宽度对齐；块数对不上时标出可能出错的位置。从左到右、从上到下 |
| `woodblock` | 刻本，竖排 | 整页顺时针转 90° 后检测，长边收到 2000，再映回原图。列从右到左。单字按列内墨迹切开。明显偏小的字收成夹注，先右行后左行 |
| `manuscript` | 写本，竖排 | 与刻本相同的列序、缩放和切字方式。过宽、过窄或对不上墨迹的字标为不确定 |
| `stele` | 碑帖 | 浅字拓片先反相，再检出文本行，长边收到 2000。列从右到左。整页检不出时改试邻近尺寸，仍不行再按墨迹分列 |
| `modelbook` | 墨迹 | 先去掉朱印，其余与碑帖相同。连成一块、分不开的字标为不确定 |

设计说明见 [docs/调研报告.md](docs/调研报告.md)、[docs/碑帖调研.md](docs/碑帖调研.md) 和 [docs/开发计划.md](docs/开发计划.md)。

## 安装

需要 Python 3.11 或 3.12。在项目根目录：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[ocr,view]"
```

`ocr` 安装 RapidOCR 与 ONNX Runtime，定位时会用到。`view` 安装本机核对页，并用 pypdfium2 渲染 PDF。只做几何和裁切、不跑检测时，`pip install -e .` 即可。

## 命令行

```powershell
python -m hanloc locate samples/1.jpg --profile book --out out
python -m hanloc locate samples/2.jpg --profile book --out out
python -m hanloc locate samples/3.jpg --profile book --out out
python -m hanloc locate samples/4.jpg --profile book --out out
python -m hanloc locate samples/5.jpg --profile book --out out
python -m hanloc locate samples/7.jpg --profile book --out out
python -m hanloc locate samples/60.jpg --profile book --out out
python -m hanloc locate samples/61.jpg --profile book --out out
```

`--profile` 可选 `book`（现代本）、`woodblock`（刻本）、`manuscript`（写本）、`stele`（碑帖）、`modelbook`（墨迹），默认 `book`。碑和墓志用 `stele`，信札和手卷用 `modelbook`。各版式都先把长边收到 2000 再检测。若用标尺量过行或列较窄的一边，可加 `--char-px`，检测会按这个尺寸来缩放，长边限制在 960 到 2400 之间。

刻本若要标出版心，加上 `--centerfold`：

- `none`：不标
- `gutter`：筒子页中缝
- `left` / `right`：半页的左缘或右缘

默认不标版心。版心列单独成一条语流，不写入正文。

`out` 目录里有：

- `page.json`：行、单字和语流
- `overlay.png`：行框、单字框和语流序号
- `crops/`：每个字拉正后的图，文件名即字的编号

`page.json` 中，行有 `text` 和 `score`，单字有 `char`、`score` 和 `uncertain`。`quad` 是墨迹单字框，`cell` 是识别回推的字身格；没有字身格时不写 `cell`。语流的 `text` 由其中的字按顺序拼成。`role` 为 `body`（正文）、`commentary`（夹注）、`title`（标题）、`marginalia`（旁注）或 `centerfold`（版心）。四点框的顺序是左上、右上、右下、左下，坐标是原图像素，y 向下。

核对时若改了字或行，改订写在 `correction`。`char`、`text` 和语流的 `text` 仍是识别原文。`correction` 为空字符串表示改为空；没有这个字段表示尚未改订。

## Python

```python
from hanloc import locate, crop_glyph

page = locate("页面.jpg", profile="woodblock")
print(page.flows[0].text)
patch = crop_glyph("页面.jpg", page.glyphs[0], margin=0.15)
```

`margin` 是四周留白相对字框尺寸的比例。`profile` 也可以传入 `hanloc.profiles` 里的 `BOOK`、`WOODBLOCK`、`MANUSCRIPT`、`STELE`、`MODELBOOK`，并用 `dataclasses.replace` 改 `centerfold`。

## 核对页

```powershell
python -m hanloc.view
```

浏览器打开 http://127.0.0.1:8765 ，只监听本机。选择一张或多张图片，或一份 PDF，设好 PDF 分辨率后点「打开」。多张图片按文件名排序。PDF 按页拆开，默认 300 dpi，可在 72 到 600 之间调整。

点「执行 OCR」只识别当前页。图像上方和左侧有像素标尺，指针所在的位置会标在标尺上，并显示坐标。要填的是行或列较窄的一边，界面上写作「字高（行列窄边）」，不是整行或整列的长度。量出这个宽度，填进去再执行；留空则按长边 2000。识别完成后，旁边会给出这次的预期字高，以及去掉白边后、剔除过大过小者、取中段均匀字算出的平均字宽。用上一页、下一页逐页核对。行框、单字框、语流序号、原文、改订、置信度和语流列表可以分别显示或隐藏，正文、夹注、标题、旁注、版心也可以分开看。

置信度低于设定值的单字或行，可以标出、只显示或隐藏。没有置信度的项一并算入。点「处理」后，侧栏「待核」列出这些项。

点击一个字可改订。保存写到 `correction`，不覆盖原来的 `char` 或行 `text`。输入留空再保存，表示改为空；点「清除」则恢复为未改。语流里的原文不改，有改订时另列一行。下载的 JSON 同时保留原文和改订。重新识别当前页会换掉该页结果，已写的改订也会清掉。

## 测试

```powershell
pip install -e ".[dev]"
pytest
```

默认测试使用合成框，不读取 `samples/`，也不加载识别模型。
