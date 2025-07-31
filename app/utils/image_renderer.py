import io
import base64
from PIL import Image, ImageDraw, ImageFont
import os
from flask import current_app

def group_faces(faces):
    """根据作品或'未知角色'对识别结果进行分组"""
    groups = {}
    # 先按作品分组
    for f in faces:
        key = (
            f["identity"] if f["identity"] == "未知角色" else f.get("anime", "未知作品")
        )
        if key not in groups:
            groups[key] = []
        groups[key].append(f)

    # 对每个分组内的角色按分数排序
    for key in groups:
        groups[key].sort(key=lambda x: x["score"], reverse=True)

    return groups


def draw_rounded_rectangle(draw, xy, corner_radius, fill=None, outline=None, width=1):
    """(Helper) Draws a rounded rectangle with separate logic for fill and outline."""
    x1, y1, x2, y2 = xy
    r = corner_radius

    if fill:
        # Draws the flat parts
        draw.rectangle([(x1 + r, y1), (x2 - r, y2)], fill=fill)
        draw.rectangle([(x1, y1 + r), (x2, y2 - r)], fill=fill)
        # Draws the corners
        draw.pieslice([(x1, y1), (x1 + r * 2, y1 + r * 2)], 180, 270, fill=fill)
        draw.pieslice([(x2 - r * 2, y1), (x2, y1 + r * 2)], 270, 360, fill=fill)
        draw.pieslice([(x1, y2 - r * 2), (x1 + r * 2, y2)], 90, 180, fill=fill)
        draw.pieslice([(x2 - r * 2, y2 - r * 2), (x2, y2)], 0, 90, fill=fill)

    if outline:
        # Draws the straight lines
        draw.line([(x1 + r, y1), (x2 - r, y1)], fill=outline, width=width)
        draw.line([(x1 + r, y2), (x2 - r, y2)], fill=outline, width=width)
        draw.line([(x1, y1 + r), (x1, y2 - r)], fill=outline, width=width)
        draw.line([(x2, y1 + r), (x2, y2 - r)], fill=outline, width=width)
        # Draws the corner arcs
        draw.arc([x1, y1, x1 + r * 2, y1 + r * 2], 180, 270, fill=outline, width=width)
        draw.arc([x2 - r * 2, y1, x2, y1 + r * 2], 270, 360, fill=outline, width=width)
        draw.arc([x1, y2 - r * 2, x1 + r * 2, y2], 90, 180, fill=outline, width=width)
        draw.arc([x2 - r * 2, y2 - r * 2, x2, y2], 0, 90, fill=outline, width=width)


def truncate_text(draw, text, font, max_width):
    """(Helper) Truncates text with '...' if it exceeds the maximum width."""
    if draw.textbbox((0, 0), text, font=font)[2] <= max_width:
        return text

    ellipsis = "..."
    ellipsis_width = draw.textbbox((0, 0), ellipsis, font=font)[2]

    for i in range(len(text) - 1, 0, -1):
        truncated_text = text[:i]
        if (
            draw.textbbox((0, 0), truncated_text, font=font)[2] + ellipsis_width
            <= max_width
        ):
            return truncated_text + ellipsis

    # Fallback for very narrow space
    return text[:1] + ellipsis


def render_cards_to_img_bytes(faces):
    """
    使用 Pillow 将识别结果直接绘制成一张图片，返回图片的二进制数据。
    """
    if not faces:
        return None

    # --- 1. 准备阶段: 字体、颜色、尺寸 ---
    try:
        font_path = "C:/Windows/Fonts/msyh.ttc"  # 微软雅黑
        if not os.path.exists(font_path):
            current_app.logger.warning(
                f"字体 '{font_path}' 不存在，将使用默认字体(可能不支持中文)。"
            )
            font_path = None  # Fallback to default
        font_name = ImageFont.truetype(font_path, 20)
        font_meta = ImageFont.truetype(font_path, 15)  # 中号字体
    except IOError:
        current_app.logger.warning("无法加载字体，将使用默认字体(可能不支持中文)。")
        font_name = ImageFont.load_default()
        font_meta = ImageFont.load_default()

    # 样式定义 (模仿CSS)
    BG_COLOR = "#ffffff"
    GROUP_BG_COLOR = "#ffffff"
    CARD_BG_COLOR = "#ffffff"
    TEXT_COLOR = "#212529"
    MUTED_COLOR = "#6c757d"
    BORDER_COLOR = "#e8eaf0"

    CARD_WIDTH = 390
    CARD_HEIGHT = 135
    GROUP_PADDING = 15
    GROUP_MARGIN_BOTTOM = 15  # 减小组间距
    COLUMN_GAP = 20
    ROW_GAP = 20

    IMG_WIDTH = GROUP_PADDING * 2 + CARD_WIDTH * 2 + COLUMN_GAP

    # --- 2. 分组并计算布局 ---
    grouped_faces = group_faces(faces)

    # 按角色数量对分组进行排序，"未知角色" 组排在最后
    unknown_group = grouped_faces.pop("未知角色", None)
    sorted_groups = dict(
        sorted(grouped_faces.items(), key=lambda item: len(item[1]), reverse=True)
    )
    if unknown_group:
        sorted_groups["未知角色"] = unknown_group  # 将未知角色组追加到末尾

    total_height = 0
    group_layouts = []

    for group_name, faces_in_group in sorted_groups.items():
        group_header_height = 55  # 标题区高度
        num_rows = (len(faces_in_group) + 1) // 2
        group_content_height = num_rows * CARD_HEIGHT + (num_rows - 1) * ROW_GAP
        group_height = (
            group_header_height + group_content_height + GROUP_PADDING * 2
        )  # 上下padding

        group_layouts.append(
            {
                "name": group_name,
                "y_start": total_height,
                "height": group_height,
                "faces": faces_in_group,
            }
        )
        total_height += group_height + GROUP_MARGIN_BOTTOM

    total_height += GROUP_PADDING * 2 - GROUP_MARGIN_BOTTOM  # 上下总边距

    # --- 3. 绘图 ---
    image = Image.new("RGB", (IMG_WIDTH, total_height), BG_COLOR)
    draw = ImageDraw.Draw(image)

    current_y = GROUP_PADDING  # Start with top padding
    for layout in group_layouts:
        # 绘制分组背景
        draw_rounded_rectangle(
            draw,
            (
                GROUP_PADDING,
                current_y,
                IMG_WIDTH - GROUP_PADDING,
                current_y + layout["height"],
            ),
            corner_radius=12,
            fill=GROUP_BG_COLOR,
        )

        # 绘制分组标题
        draw.text(
            (GROUP_PADDING * 2, current_y + GROUP_PADDING),
            layout["name"],
            font=font_name,
            fill=TEXT_COLOR,
        )

        # 绘制分割线
        line_y = current_y + 50
        draw.line(
            [(GROUP_PADDING * 2, line_y), (IMG_WIDTH - GROUP_PADDING * 2, line_y)],
            fill=BORDER_COLOR,
            width=1,
        )

        # 绘制卡片
        card_start_y = line_y + 20

        # 居中卡片
        group_width = IMG_WIDTH - GROUP_PADDING * 2
        cards_total_width = CARD_WIDTH * 2 + COLUMN_GAP
        cards_left_margin = (group_width - cards_total_width) // 2
        card_start_x = GROUP_PADDING + cards_left_margin

        col = 0
        row = 0
        for i, face_data in enumerate(layout["faces"]):
            col = i % 2
            row = i // 2

            card_x = card_start_x + col * (CARD_WIDTH + COLUMN_GAP)
            card_y = card_start_y + row * (CARD_HEIGHT + ROW_GAP)

            # --- 绘制单个卡片 ---
            draw_rounded_rectangle(
                draw,
                (card_x, card_y, card_x + CARD_WIDTH, card_y + CARD_HEIGHT),
                corner_radius=12,
                fill=CARD_BG_COLOR,
                outline=BORDER_COLOR,
            )

            # 头像 (带圆角处理)
            try:
                face_img_b64 = face_data["image"]
                face_img_data = base64.b64decode(face_img_b64)
                face_img = (
                    Image.open(io.BytesIO(face_img_data))
                    .resize((64, 64), Image.Resampling.LANCZOS)
                    .convert("RGB")
                )

                # 创建圆角遮罩
                mask = Image.new("L", face_img.size, 0)
                mask_draw = ImageDraw.Draw(mask)
                draw_rounded_rectangle(
                    mask_draw,
                    (0, 0, face_img.size[0], face_img.size[1]),
                    corner_radius=8,
                    fill=255,
                )

                # 使用遮罩粘贴
                image.paste(face_img, (card_x + 20, card_y + 20), mask)

            except Exception as e:
                current_app.logger.error(f"Error processing face image for card: {e}")
                # Draw a placeholder if image fails
                draw.rectangle(
                    (card_x + 20, card_y + 20, card_x + 20 + 64, card_y + 20 + 64),
                    fill="#cccccc",
                )

            # 文本信息
            text_x = card_x + 100

            # 裁切过长的角色名
            name_max_width = CARD_WIDTH - (text_x - card_x) - 15  # 右侧留15px边距
            original_name = face_data.get("name", "未知角色")
            char_name = truncate_text(draw, original_name, font_name, name_max_width)

            draw.text((text_x, card_y + 22), char_name, font=font_name, fill=TEXT_COLOR)

            # 获取名字文本的宽度，以确定后续标签的起始位置
            name_bbox = draw.textbbox((0, 0), char_name, font=font_name)
            name_width = name_bbox[2] - name_bbox[0]
            label_x = text_x + name_width + 8  # 名字后方8像素间距

            # --- 新增: 绘制 "已修正" 状态 ---
            if face_data.get("is_corrected"):
                correction_label_text = "(已修正)"
                correction_label_color = "#dc3545"  # Bootstrap 'danger' red
                draw.text(
                    (label_x, card_y + 24),
                    correction_label_text,
                    font=font_meta,
                    fill=correction_label_color,
                )
                # 为下一个标签更新x坐标
                correction_bbox = draw.textbbox(
                    (0, 0), correction_label_text, font=font_meta
                )
                label_x += correction_bbox[2] - correction_bbox[0] + 5

            # --- 绘制 正面/背面 状态 ---
            try:
                first_class = face_data.get("first_class")
                if first_class == 0:
                    label_text = "(正面)"
                    label_color = "#198754"  # Bootstrap 'success' green
                    draw.text(
                        (label_x, card_y + 24),
                        label_text,
                        font=font_meta,
                        fill=label_color,
                    )
                elif first_class == 1:
                    label_text = "(背面)"
                    label_color = "#0dcaf0"  # Bootstrap 'info' cyan
                    draw.text(
                        (label_x, card_y + 24),
                        label_text,
                        font=font_meta,
                        fill=label_color,
                    )
            except Exception as e:
                current_app.logger.error(f"绘制'正面/背面'状态时出错: {e}")

            draw.text(
                (text_x, card_y + 50),
                face_data["anime"],
                font=font_meta,
                fill=MUTED_COLOR,
            )

            # 进度条
            progress_y = card_y + 115  # 增加与头像的距离
            percent = int(round(face_data.get("score", 0) * 100))

            draw.text(
                (card_x + 20, progress_y - 22),
                "置信度",
                font=font_meta,
                fill=MUTED_COLOR,
            )
            draw.text(
                (card_x + CARD_WIDTH - 60, progress_y - 22),
                f"{percent}%",
                font=font_meta,
                fill=MUTED_COLOR,
            )

            progress_bar_width = CARD_WIDTH - 40
            # 背景条
            draw_rounded_rectangle(
                draw,
                (
                    card_x + 20,
                    progress_y,
                    card_x + 20 + progress_bar_width,
                    progress_y + 6,
                ),
                corner_radius=3,
                fill="#e9ecef",
            )

            # 前景条
            if percent > 0:
                color = (
                    "#198754"
                    if percent >= 80
                    else ("#0dcaf0" if percent >= 60 else "#ffc107")
                )
                fill_width = int(progress_bar_width * (percent / 100.0))
                draw_rounded_rectangle(
                    draw,
                    (card_x + 20, progress_y, card_x + 20 + fill_width, progress_y + 6),
                    corner_radius=3,
                    fill=color,
                )

        current_y += layout["height"] + GROUP_MARGIN_BOTTOM

    # --- 4. 返回结果 ---
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue() 