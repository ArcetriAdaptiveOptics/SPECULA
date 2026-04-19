
class SimulDiagram:
    def __init__(self,
                 param_file: str,
                 title: str,
                 filename: str,
                 colors_on: bool,
                 ):
        self.param_file = param_file
        self.title = title
        self.filename = filename
        self.colors_on = colors_on

        self.connections = []
        self.references = []

    def add_reference(self, start, end):
        self.references.append({'start':start, 'end': end})
    
    def add_connection(self, start, end, start_label, end_label):
        a_connection = {}
        a_connection['start'] = start
        a_connection['end'] = end
        a_connection['start_label'] = start_label
#       a_connection['middle_label'] = self.objs[dest_object].inputs[use_input_name]
        a_connection['end_label'] = end_label
        self.connections.append(a_connection)

    def int_to_rgb(self, val: int, maxval: int, mplcolors):
        val += 1
        if val >= 0 and val < len(mplcolors):
            return mplcolors[val]
        scale = 255 / maxval
        r = int((val * scale * 611) % 256)
        g = int((val * scale * 551) % 256)
        b = int((val * scale * 501) % 256)
        return (1.0 - r / 255.0, 1.0 - g / 255.0, 1.0 - b / 255.0)

    def arrange_in_grid(self, trigger_order, trigger_order_idx):
        from collections import Counter

        rows = []
        center = False
        n_cols = max(trigger_order_idx) + 1
        n_rows = max(dict(Counter(trigger_order_idx)).values())

        orders_to_names = {i: [] for i in range(n_cols)}
        for name, order in zip(trigger_order, trigger_order_idx):
            orders_to_names[order].append(name)

        for ri in range(n_rows):
            r = []
            for ci in range(n_cols):
                col = orders_to_names[ci]
                col_offset = int((n_rows - len(col)) / 2)

                if center:
                    idx = ri - col_offset
                    r.append(col[idx] if 0 <= idx < len(col) else "")
                else:
                    r.append(col[ri] if ri < len(col) else "")
            rows.append(r)

        return rows

    def build(self,
              trigger_order: list,
              trigger_order_idx: list,
              max_rank: int,
              max_target_device_idx: int,
              all_objs_ranks: dict,
              is_dataobj: dict,
            ):
        from orthogram import Color, DiagramDef, write_png, Side, FontWeight, FontStyle, TextOrientation
        import matplotlib.pyplot as plt

        mplcolors = plt.get_cmap("tab10").colors

        title_fontsize = 96
        block_fontsize = 84
        arrow_fontsize = 48
        arrow_base_value = 12.0

        d = DiagramDef(
            label=self.title,
            text_fill=Color(0, 0, 0),
            scale=1.0,
            collapse_connections=False,
            font_size=title_fontsize,
            connection_distance=28
        )

        rows = self.arrange_in_grid(trigger_order, trigger_order_idx)
        row_len = len(rows[0])

        # ---- blocks ----
        for r in rows:
            d.add_row(r)

            for b in r:
                if not b:
                    continue

                target_device_idx = all_objs_ranks.get(b, 0)

                dataobj = is_dataobj.get(b, True)
                fs = FontStyle.ITALIC if not dataobj else FontStyle.NORMAL
                fw = FontWeight.BOLD if not dataobj else FontWeight.NORMAL

                if self.colors_on:
                    stroke = Color(*self.int_to_rgb(target_device_idx, max_rank + 1, mplcolors))
                    fill = Color(*self.int_to_rgb(max_target_device_idx, max_target_device_idx + 1, mplcolors))
                    sw = 12
                else:
                    stroke = Color(0, 0, 0)
                    fill = Color(1, 1, 1)
                    sw = 2

                d.add_block(
                    b,
                    scale=1,
                    label_distance=40,
                    stroke=stroke,
                    fill=fill,
                    stroke_width=sw,
                    min_height=block_fontsize * 3,
                    min_width=450,
                    margin_top=16,
                    margin_bottom=16,
                    margin_left=16,
                    margin_right=16,
                    font_size=block_fontsize,
                    font_weight=fw,
                    font_style=fs,
                )

        # ---- connections ----
        for c in self.connections:
            label = (c["start_label"] or "") + "→" + str(c["end_label"])

            d.add_connection(
                c["start"],
                c["end"],
                buffer_fill=Color(1, 1, 1),
                buffer_width=2,
                stroke_width=2.0,
                stroke=Color(0, 0, 0),
                arrow_base=arrow_base_value,
                exits=[Side.RIGHT, Side.BOTTOM],
                entrances=[Side.LEFT, Side.TOP],
                font_size=arrow_fontsize,
                text_orientation=TextOrientation.HORIZONTAL,
                label=label,
            )

        # ---- references ----
        for c in self.references:
            if c["end"] == "main":
                continue

            d.add_connection(
                c["start"],
                c["end"],
                buffer_fill=Color(1, 1, 1),
                buffer_width=2,
                stroke_width=2.0,
                stroke=Color(0, 0.5, 0),
                arrow_base=arrow_base_value,
                exits=[Side.LEFT],
                entrances=[Side.RIGHT, Side.BOTTOM, Side.TOP],
                stroke_dasharray=[6, 6],
            )

        write_png(d, self.filename)