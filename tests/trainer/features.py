"""Full list of all feature names used by the RFExtractionAI, and of all feature columns excluded for inference."""

FULL_FEATURE_LIST = [
    'feat_vowel_len',
    'feat_special_len',
    'feat_space_len',
    'feat_digit_len',
    'feat_len',
    'feat_upper_len',
    'feat_date_count',
    'feat_num_count',
    'feat_as_float',
    'feat_unique_char_count',
    'feat_duplicate_count',
    'accented_char_count',
    'feat_year_count',
    'feat_month_count',
    'feat_day_count',
    'feat_substring_count_slash',
    'feat_substring_count_percent',
    'feat_substring_count_e',
    'feat_substring_count_g',
    'feat_substring_count_a',
    'feat_substring_count_u',
    'feat_substring_count_i',
    'feat_substring_count_f',
    'feat_substring_count_s',
    'feat_substring_count_oe',
    'feat_substring_count_ae',
    'feat_substring_count_ue',
    'feat_substring_count_er',
    'feat_substring_count_str',
    'feat_substring_count_k',
    'feat_substring_count_r',
    'feat_substring_count_y',
    'feat_substring_count_en',
    'feat_substring_count_ch',
    'feat_substring_count_sch',
    'feat_substring_count_c',
    'feat_substring_count_ei',
    'feat_substring_count_on',
    'feat_substring_count_ohn',
    'feat_substring_count_n',
    'feat_substring_count_m',
    'feat_substring_count_j',
    'feat_substring_count_h',
    'feat_substring_count_plus',
    'feat_substring_count_minus',
    'feat_substring_count_period',
    'feat_substring_count_comma',
    'feat_starts_with_plus',
    'feat_starts_with_minus',
    'feat_ends_with_plus',
    'feat_ends_with_minus',
    'x0',
    'y0',
    'x1',
    'y1',
    'page_index',
    'area_quadrant_two',
    'area',
    'l_dist0',
    'l_dist1',
    'r_dist0',
    'r_dist1',
    'l0_feat_vowel_len',
    'l0_feat_special_len',
    'l0_feat_space_len',
    'l0_feat_digit_len',
    'l0_feat_len',
    'l0_feat_upper_len',
    'l0_feat_date_count',
    'l0_feat_num_count',
    'l0_feat_as_float',
    'l0_feat_unique_char_count',
    'l0_feat_duplicate_count',
    'l0_accented_char_count',
    'l0_feat_year_count',
    'l0_feat_month_count',
    'l0_feat_day_count',
    'l0_feat_substring_count_slash',
    'l0_feat_substring_count_percent',
    'l0_feat_substring_count_e',
    'l0_feat_substring_count_g',
    'l0_feat_substring_count_a',
    'l0_feat_substring_count_u',
    'l0_feat_substring_count_i',
    'l0_feat_substring_count_f',
    'l0_feat_substring_count_s',
    'l0_feat_substring_count_oe',
    'l0_feat_substring_count_ae',
    'l0_feat_substring_count_ue',
    'l0_feat_substring_count_er',
    'l0_feat_substring_count_str',
    'l0_feat_substring_count_k',
    'l0_feat_substring_count_r',
    'l0_feat_substring_count_y',
    'l0_feat_substring_count_en',
    'l0_feat_substring_count_ch',
    'l0_feat_substring_count_sch',
    'l0_feat_substring_count_c',
    'l0_feat_substring_count_ei',
    'l0_feat_substring_count_on',
    'l0_feat_substring_count_ohn',
    'l0_feat_substring_count_n',
    'l0_feat_substring_count_m',
    'l0_feat_substring_count_j',
    'l0_feat_substring_count_h',
    'l0_feat_substring_count_plus',
    'l0_feat_substring_count_minus',
    'l0_feat_substring_count_period',
    'l0_feat_substring_count_comma',
    'l0_feat_starts_with_plus',
    'l0_feat_starts_with_minus',
    'l0_feat_ends_with_plus',
    'l0_feat_ends_with_minus',
    'l1_feat_vowel_len',
    'l1_feat_special_len',
    'l1_feat_space_len',
    'l1_feat_digit_len',
    'l1_feat_len',
    'l1_feat_upper_len',
    'l1_feat_date_count',
    'l1_feat_num_count',
    'l1_feat_as_float',
    'l1_feat_unique_char_count',
    'l1_feat_duplicate_count',
    'l1_accented_char_count',
    'l1_feat_year_count',
    'l1_feat_month_count',
    'l1_feat_day_count',
    'l1_feat_substring_count_slash',
    'l1_feat_substring_count_percent',
    'l1_feat_substring_count_e',
    'l1_feat_substring_count_g',
    'l1_feat_substring_count_a',
    'l1_feat_substring_count_u',
    'l1_feat_substring_count_i',
    'l1_feat_substring_count_f',
    'l1_feat_substring_count_s',
    'l1_feat_substring_count_oe',
    'l1_feat_substring_count_ae',
    'l1_feat_substring_count_ue',
    'l1_feat_substring_count_er',
    'l1_feat_substring_count_str',
    'l1_feat_substring_count_k',
    'l1_feat_substring_count_r',
    'l1_feat_substring_count_y',
    'l1_feat_substring_count_en',
    'l1_feat_substring_count_ch',
    'l1_feat_substring_count_sch',
    'l1_feat_substring_count_c',
    'l1_feat_substring_count_ei',
    'l1_feat_substring_count_on',
    'l1_feat_substring_count_ohn',
    'l1_feat_substring_count_n',
    'l1_feat_substring_count_m',
    'l1_feat_substring_count_j',
    'l1_feat_substring_count_h',
    'l1_feat_substring_count_plus',
    'l1_feat_substring_count_minus',
    'l1_feat_substring_count_period',
    'l1_feat_substring_count_comma',
    'l1_feat_starts_with_plus',
    'l1_feat_starts_with_minus',
    'l1_feat_ends_with_plus',
    'l1_feat_ends_with_minus',
    'r0_feat_vowel_len',
    'r0_feat_special_len',
    'r0_feat_space_len',
    'r0_feat_digit_len',
    'r0_feat_len',
    'r0_feat_upper_len',
    'r0_feat_date_count',
    'r0_feat_num_count',
    'r0_feat_as_float',
    'r0_feat_unique_char_count',
    'r0_feat_duplicate_count',
    'r0_accented_char_count',
    'r0_feat_year_count',
    'r0_feat_month_count',
    'r0_feat_day_count',
    'r0_feat_substring_count_slash',
    'r0_feat_substring_count_percent',
    'r0_feat_substring_count_e',
    'r0_feat_substring_count_g',
    'r0_feat_substring_count_a',
    'r0_feat_substring_count_u',
    'r0_feat_substring_count_i',
    'r0_feat_substring_count_f',
    'r0_feat_substring_count_s',
    'r0_feat_substring_count_oe',
    'r0_feat_substring_count_ae',
    'r0_feat_substring_count_ue',
    'r0_feat_substring_count_er',
    'r0_feat_substring_count_str',
    'r0_feat_substring_count_k',
    'r0_feat_substring_count_r',
    'r0_feat_substring_count_y',
    'r0_feat_substring_count_en',
    'r0_feat_substring_count_ch',
    'r0_feat_substring_count_sch',
    'r0_feat_substring_count_c',
    'r0_feat_substring_count_ei',
    'r0_feat_substring_count_on',
    'r0_feat_substring_count_ohn',
    'r0_feat_substring_count_n',
    'r0_feat_substring_count_m',
    'r0_feat_substring_count_j',
    'r0_feat_substring_count_h',
    'r0_feat_substring_count_plus',
    'r0_feat_substring_count_minus',
    'r0_feat_substring_count_period',
    'r0_feat_substring_count_comma',
    'r0_feat_starts_with_plus',
    'r0_feat_starts_with_minus',
    'r0_feat_ends_with_plus',
    'r0_feat_ends_with_minus',
    'r1_feat_vowel_len',
    'r1_feat_special_len',
    'r1_feat_space_len',
    'r1_feat_digit_len',
    'r1_feat_len',
    'r1_feat_upper_len',
    'r1_feat_date_count',
    'r1_feat_num_count',
    'r1_feat_as_float',
    'r1_feat_unique_char_count',
    'r1_feat_duplicate_count',
    'r1_accented_char_count',
    'r1_feat_year_count',
    'r1_feat_month_count',
    'r1_feat_day_count',
    'r1_feat_substring_count_slash',
    'r1_feat_substring_count_percent',
    'r1_feat_substring_count_e',
    'r1_feat_substring_count_g',
    'r1_feat_substring_count_a',
    'r1_feat_substring_count_u',
    'r1_feat_substring_count_i',
    'r1_feat_substring_count_f',
    'r1_feat_substring_count_s',
    'r1_feat_substring_count_oe',
    'r1_feat_substring_count_ae',
    'r1_feat_substring_count_ue',
    'r1_feat_substring_count_er',
    'r1_feat_substring_count_str',
    'r1_feat_substring_count_k',
    'r1_feat_substring_count_r',
    'r1_feat_substring_count_y',
    'r1_feat_substring_count_en',
    'r1_feat_substring_count_ch',
    'r1_feat_substring_count_sch',
    'r1_feat_substring_count_c',
    'r1_feat_substring_count_ei',
    'r1_feat_substring_count_on',
    'r1_feat_substring_count_ohn',
    'r1_feat_substring_count_n',
    'r1_feat_substring_count_m',
    'r1_feat_substring_count_j',
    'r1_feat_substring_count_h',
    'r1_feat_substring_count_plus',
    'r1_feat_substring_count_minus',
    'r1_feat_substring_count_period',
    'r1_feat_substring_count_comma',
    'r1_feat_starts_with_plus',
    'r1_feat_starts_with_minus',
    'r1_feat_ends_with_plus',
    'r1_feat_ends_with_minus',
    'x0_relative',
    'x1_relative',
    'y0_relative',
    'y1_relative',
    'relative_position_in_page',
    'first_word_x0',
    'first_word_y0',
    'first_word_x1',
    'first_word_y1',
    'first_word_feat_vowel_len',
    'first_word_feat_special_len',
    'first_word_feat_space_len',
    'first_word_feat_digit_len',
    'first_word_feat_len',
    'first_word_feat_upper_len',
    'first_word_feat_date_count',
    'first_word_feat_num_count',
    'first_word_feat_as_float',
    'first_word_feat_unique_char_count',
    'first_word_feat_duplicate_count',
    'first_word_accented_char_count',
    'first_word_feat_year_count',
    'first_word_feat_month_count',
    'first_word_feat_day_count',
    'first_word_feat_substring_count_slash',
    'first_word_feat_substring_count_percent',
    'first_word_feat_substring_count_e',
    'first_word_feat_substring_count_g',
    'first_word_feat_substring_count_a',
    'first_word_feat_substring_count_u',
    'first_word_feat_substring_count_i',
    'first_word_feat_substring_count_f',
    'first_word_feat_substring_count_s',
    'first_word_feat_substring_count_oe',
    'first_word_feat_substring_count_ae',
    'first_word_feat_substring_count_ue',
    'first_word_feat_substring_count_er',
    'first_word_feat_substring_count_str',
    'first_word_feat_substring_count_k',
    'first_word_feat_substring_count_r',
    'first_word_feat_substring_count_y',
    'first_word_feat_substring_count_en',
    'first_word_feat_substring_count_ch',
    'first_word_feat_substring_count_sch',
    'first_word_feat_substring_count_c',
    'first_word_feat_substring_count_ei',
    'first_word_feat_substring_count_on',
    'first_word_feat_substring_count_ohn',
    'first_word_feat_substring_count_n',
    'first_word_feat_substring_count_m',
    'first_word_feat_substring_count_j',
    'first_word_feat_substring_count_h',
    'first_word_feat_substring_count_plus',
    'first_word_feat_substring_count_minus',
    'first_word_feat_substring_count_period',
    'first_word_feat_substring_count_comma',
    'first_word_feat_starts_with_plus',
    'first_word_feat_starts_with_minus',
    'first_word_feat_ends_with_plus',
    'first_word_feat_ends_with_minus',
]

EXCLUDED_COLUMNS_LIST = [
    'annotation_id',
    'annotation_set_id',
    'category_id',
    'confidence',
    'created_by',
    'custom_offset_string',
    'data_type',
    'document_id',
    'document_id_local',
    'end_offset',
    'first_word_string',
    'id_',
    'id_local',
    'is_correct',
    'l_offset_string0',
    'l_offset_string1',
    'label_has_multiple_top_candidates',
    'label_id',
    'label_name',
    'label_set_id',
    'label_set_name',
    'label_threshold',
    'line_index',
    'normalized',
    'offset_string',
    'page_height',
    'page_index_relative',
    'page_width',
    'r_offset_string0',
    'r_offset_string1',
    'revised',
    'revised_by',
    'start_offset',
    'target',
]