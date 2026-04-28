
from collections import Counter
from specula.log import get_specula_logger

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
        self.logger = get_specula_logger(__name__)

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

    def int_to_rgb(self, val: int, maxval: int):
        import matplotlib.pyplot as plt
        mplcolors = plt.get_cmap("tab10").colors

        val += 1
        if val >= 0 and val < len(mplcolors):
            return mplcolors[val]
        scale = 255 / maxval
        r = int((val * scale * 611) % 256)
        g = int((val * scale * 551) % 256)
        b = int((val * scale * 501) % 256)
        return (1.0 - r / 255.0, 1.0 - g / 255.0, 1.0 - b / 255.0)

    def arrange_in_grid(self, trigger_order, trigger_order_idx):

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
              all_target_device_idxs: dict,
              all_objs_ranks: dict,
              is_dataobj: dict,
            ):
        # Imports are inside this method, so that they are not executed
        # unless the diagram is actually built.
        from orthogram import Color, DiagramDef, write_png, Side, FontWeight, FontStyle, TextOrientation

        self.logger.info('Building diagram...')        
        title_fontsize = 48*2
        block_fontsize = 42*2
        arrow_fontsize = 24*2
        arrow_base_value = 12.0
        
        d = DiagramDef(label=self.title, text_fill=Color(0, 0, 0), scale=1.0, collapse_connections=False, font_size=title_fontsize, connection_distance=28)
        rows = self.arrange_in_grid(trigger_order, trigger_order_idx)
        row_len = len(rows[0])        

        max_rank = max(x if x is not None else 0 for x in all_objs_ranks.values())
        max_target_device_idx = max(x if x is not None else 0 for x in all_target_device_idxs.values())
     
        # a row is a list of strings, which are labels for the cells        
        for r in rows:
            d.add_row(r)
            for b in r:
                target_device_idx = 0
                target_rank = 0

                target_device_idx = all_target_device_idxs.get(b, 0) or 0
                target_rank = all_objs_ranks.get(b, 0) or 0
                
                if b in is_dataobj and not is_dataobj[b]:
                    fs = FontStyle.ITALIC
                    fb = FontWeight.BOLD
                else:
                    fs = FontStyle.NORMAL
                    fb = FontWeight.NORMAL

                if self.colors_on:
                    cstroke = Color(*self.int_to_rgb(target_rank - 1, max_rank + 1))
                    refcstroke = Color(0,0.5,0)
                    cfill = Color(*self.int_to_rgb(target_device_idx, max_target_device_idx + 1))
                    swidth = 12
                else:
                    cstroke = Color(0,0,0)
                    refcstroke = Color(0,0,0)
                    cfill = Color(1,1,1)
                    swidth = 2

                d.add_block(b,
                            scale=1,
                            label_distance=40,
                            stroke=cstroke,
                            fill=cfill,
                            stroke_width=swidth,
                            min_height=block_fontsize*3,
                            min_width=450,
                            margin_top=16,
                            margin_bottom=16,
                            margin_right=16,
                            margin_left=16,
                            font_size=block_fontsize,
                            font_weight=fb, 
                            font_style=fs)
        
        if self.colors_on:
            legend_row1 = []
            for td in range(max_target_device_idx + 1):
                legend_row1.append("GPU Id=" + str(td))
            d.add_row(legend_row1)
            for td in range(max_target_device_idx + 1):
                d.add_block("GPU Id=" + str(td),
                            label_distance=40,
                            fill=Color(*self.int_to_rgb(td, max_target_device_idx + 1)),
                            stroke=Color(1.0,1.0,1.0),
                            stroke_width=12,
                            min_height=block_fontsize*3,
                            min_width=450,
                            margin_top=16,
                            margin_bottom=16,
                            margin_right=16,
                            margin_left=16,
                            font_size=block_fontsize)

            legend_row2 = []
            ri=0
            base_rank=0
            for rank in range(max_rank + 1):
                legend_row2.append("Rank=" + str(rank)) 
                if int(rank + 1) % row_len == 0 or rank == max_rank:
                    d.add_row(legend_row2)
                    for ii in range(len(legend_row2)):
                        d.add_block("Rank=" + str(ii+base_rank),
                                    label_distance=40,
                                    stroke=Color(*self.int_to_rgb(ii + base_rank - 1, max_rank + 1)), 
                                    stroke_width=12,
                                    min_height=block_fontsize*3,
                                    min_width=450,
                                    margin_top=16,
                                    margin_bottom=16,
                                    margin_right=16,
                                    margin_left=16,
                                    font_size=block_fontsize)
                    legend_row2 = []
                    ri += 1
                    base_rank += row_len            

        for c in self.connections:
            if c['start_label'] is None:
                ostring = ""
            else:
                ostring = str(c['start_label'])
            aconn = d.add_connection( c['start'],
                                      c['end'],
                                      buffer_fill=Color(1.0,1.0,1.0),
                                      buffer_width=2,
                                      stroke_width=2.0,
                                      stroke=Color(0.0,0.0,0.0), 
                                      arrow_base=arrow_base_value,
                                      exits=[Side.RIGHT, Side.BOTTOM],
                                      entrances=[Side.LEFT, Side.TOP],
                                      font_size=arrow_fontsize,
                                      text_orientation=TextOrientation.HORIZONTAL,
                                      label = ostring + "→" + str(c['end_label']) )

        for c in self.references:
            if c['end'] != 'main':
                aconn = d.add_connection( c['start'],
                                          c['end'],
                                          buffer_fill=Color(1.0,1.0,1.0),
                                          buffer_width=2,
                                          stroke_width=2.0,
                                          stroke=refcstroke,
                                          arrow_base=arrow_base_value,
                                          exits=[Side.LEFT],
                                          entrances=[Side.RIGHT, Side.BOTTOM, Side.TOP], 
                                          stroke_dasharray=[6,6] )


        write_png(d, self.filename)
        self.logger.info(f'Diagram saved in {self.filename}')
