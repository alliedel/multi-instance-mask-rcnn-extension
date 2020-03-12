import vis_utils

exporter = vis_utils.FigExporter()

vis_utils.collate_figures(['/home/adelgior/workspace/images/000050_486536_posttrain_standard_inst0.png',
                           '/home/adelgior/workspace/images/000050_486536_posttrain_standard_inst1.png'],
                          'out.png', exporter, font_color=(0, 0, 0))
