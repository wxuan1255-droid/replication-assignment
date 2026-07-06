"""
published_numbers.py -- reference data module (not a standalone script).

Coefficients, standard errors, and sample sizes hand-transcribed from the
published tables (Tables 1, 3, 4, 5, 6) of:

  Anderson, Michael L. (2017). "The Benefits of College Athletic Success: An
  Application of the Propensity Score Design." The Review of Economics and
  Statistics, 99(1), 119-134.

Used ONLY by 03_tables_figures.py to build the original-vs-replication
comparison tables and the replication-accuracy statistics reported in the
report. These numbers are never used as inputs to estimation -- they exist
solely for comparison after 02_analysis.py has produced this project's own
independent estimates.
"""

# Table 3: All FBS
TABLE3_ORIG = {
    "alumni_ops_athletics":      dict(iv=136.4, iv_se=41.1,  iv_n=637,  ste=191.2, ste_se=65.0,  ste_n=616),
    "alum_non_athl_ops":         dict(iv=227.9, iv_se=171.4, iv_n=637,  ste=-137.4, ste_se=96.1, ste_n=616),
    "alumni_total_giving":       dict(iv=311.9, iv_se=295.7, iv_n=1264, ste=267.4, ste_se=266.9, ste_n=1258),
    "vse_alum_giving_rate":      dict(iv=-0.0001, iv_se=0.0005, iv_n=1293, ste=0.0002, ste_se=0.0007, ste_n=1287),
    "usnews_academic_rep_new":   dict(iv=0.004, iv_se=0.001, iv_n=669, ste=0.003, ste_se=0.002, ste_n=650),
    "applicants":                dict(iv=135.3, iv_se=49.9, iv_n=533, ste=81.1, ste_se=60.4, ste_n=528),
    "acceptance_rate":           dict(iv=-0.003, iv_se=0.001, iv_n=967, ste=-0.003, ste_se=0.002, ste_n=979),
    "firsttime_outofstate":      dict(iv=-0.4, iv_se=3.4, iv_n=1038, ste=1.6, ste_se=5.0, ste_n=962),
    "first_time_instate":        dict(iv=15.2, iv_se=5.8, iv_n=1038, ste=12.6, ste_se=6.4, ste_n=962),
    "sat_25":                    dict(iv=1.8, iv_se=0.6, iv_n=431, ste=0.8, ste_se=0.7, ste_n=426),
}

# Table 4: BCS
TABLE4_ORIG = {
    "alumni_ops_athletics":      dict(iv=190.7, iv_se=68.5,  iv_n=353, ste=245.8, ste_se=100.0, ste_n=341),
    "alum_non_athl_ops":         dict(iv=537.1, iv_se=332.0, iv_n=353, ste=-183.9, ste_se=151.6, ste_n=341),
    "alumni_total_giving":       dict(iv=670.9, iv_se=461.8, iv_n=796, ste=398.9, ste_se=404.1, ste_n=795),
    "vse_alum_giving_rate":      dict(iv=0.0000, iv_se=0.0007, iv_n=805, ste=0.0007, ste_se=0.0009, ste_n=807),
    "usnews_academic_rep_new":   dict(iv=0.004, iv_se=0.002, iv_n=457, ste=0.003, ste_se=0.002, ste_n=453),
    "applicants":                dict(iv=79.4, iv_se=58.9, iv_n=312, ste=56.9, ste_se=73.7, ste_n=312),
    "acceptance_rate":           dict(iv=-0.003, iv_se=0.002, iv_n=642, ste=-0.005, ste_se=0.002, ste_n=658),
    "firsttime_outofstate":      dict(iv=2.1, iv_se=5.0, iv_n=623, ste=10.8, ste_se=5.2, ste_n=571),
    "first_time_instate":        dict(iv=13.8, iv_se=8.0, iv_n=623, ste=14.5, ste_se=6.4, ste_n=571),
    "sat_25":                    dict(iv=0.7, iv_se=0.6, iv_n=268, ste=0.5, ste_se=0.6, ste_n=270),
}

# Table 5: Non-BCS
TABLE5_ORIG = {
    "alumni_ops_athletics":      dict(iv=8.6, iv_se=15.5, iv_n=288, ste=40.1, ste_se=17.1, ste_n=275),
    "alum_non_athl_ops":         dict(iv=49.4, iv_se=41.3, iv_n=288, ste=-24.1, ste_se=49.7, ste_n=275),
    "alumni_total_giving":       dict(iv=-260.1, iv_se=137.8, iv_n=478, ste=-26.2, ste_se=100.2, ste_n=463),
    "vse_alum_giving_rate":      dict(iv=-0.0003, iv_se=0.0006, iv_n=495, ste=-0.0006, ste_se=0.0010, ste_n=480),
    "usnews_academic_rep_new":   dict(iv=0.004, iv_se=0.003, iv_n=214, ste=0.002, ste_se=0.003, ste_n=197),
    "applicants":                dict(iv=96.3, iv_se=74.4, iv_n=226, ste=48.2, ste_se=113.2, ste_n=216),
    "acceptance_rate":           dict(iv=-0.001, iv_se=0.002, iv_n=335, ste=-0.001, ste_se=0.002, ste_n=321),
    "firsttime_outofstate":      dict(iv=-3.2, iv_se=4.7, iv_n=428, ste=-14.1, ste_se=7.9, ste_n=391),
    "first_time_instate":        dict(iv=17.8, iv_se=9.0, iv_n=428, ste=-2.4, ste_se=11.7, ste_n=391),
    "sat_25":                    dict(iv=1.9, iv_se=1.4, iv_n=166, ste=1.3, ste_se=1.4, ste_n=156),
}

# Table 6: weeks 5+
TABLE6_ORIG = {
    "alumni_ops_athletics":      dict(iv=141.6, iv_se=53.8, iv_n=637, ste=110.4, ste_se=67.6, ste_n=374),
    "alum_non_athl_ops":         dict(iv=194.9, iv_se=187.3, iv_n=637, ste=216.6, ste_se=104.8, ste_n=374),
    "alumni_total_giving":       dict(iv=579.8, iv_se=376.6, iv_n=1264, ste=266.7, ste_se=331.7, ste_n=787),
    "vse_alum_giving_rate":      dict(iv=0.0000, iv_se=0.0006, iv_n=1293, ste=0.0016, ste_se=0.0007, ste_n=810),
    "usnews_academic_rep_new":   dict(iv=0.004, iv_se=0.002, iv_n=669, ste=0.003, ste_se=0.003, ste_n=390),
    "applicants":                dict(iv=119.9, iv_se=62.1, iv_n=533, ste=40.0, ste_se=93.3, ste_n=320),
    "acceptance_rate":           dict(iv=-0.002, iv_se=0.002, iv_n=967, ste=-0.005, ste_se=0.002, ste_n=630),
    "firsttime_outofstate":      dict(iv=-2.8, iv_se=4.5, iv_n=1038, ste=7.4, ste_se=5.0, ste_n=604),
    "first_time_instate":        dict(iv=15.4, iv_se=7.5, iv_n=1038, ste=28.3, ste_se=8.7, ste_n=604),
    "sat_25":                    dict(iv=1.9, iv_se=0.8, iv_n=431, ste=1.7, ste_se=1.2, ste_n=252),
}

# Table 1 summary stats: (mean, sd, N) for lag_seasonwins/games/expwins and outcomes, BCS / Non-BCS
TABLE1_ORIG = {
    "BCS": {
        "lag_seasonwins": (5.9, 2.6, 1437), "lag_seasongames": (10.8, 0.8, 1437), "lag_exp_wins": (5.8, 1.9, 1437),
        "alumni_ops_athletics": (3953, 3805, 495), "alum_non_athl_ops": (11600, 14800, 495),
        "alumni_total_giving": (27600, 30900, 1084), "vse_alum_giving_rate": (0.167, 0.077, 1104),
        "usnews_academic_rep_new": (3.499, 0.54, 679), "applicants": (16815, 8043, 480),
        "acceptance_rate": (0.667, 0.185, 1036), "firsttime_outofstate": (1038, 591, 886),
        "first_time_instate": (2811, 1537, 886), "sat_25": (1101, 103, 431),
    },
    "Non-BCS": {
        "lag_seasonwins": (4.6, 2.5, 923), "lag_seasongames": (10.5, 1.0, 923), "lag_exp_wins": (4.7, 1.7, 923),
        "alumni_ops_athletics": (695, 836, 430), "alum_non_athl_ops": (1992, 3569, 430),
        "alumni_total_giving": (5359, 6751, 709), "vse_alum_giving_rate": (0.103, 0.061, 733),
        "usnews_academic_rep_new": (2.712, 0.43, 365), "applicants": (9660, 6403, 360),
        "acceptance_rate": (0.755, 0.158, 555), "firsttime_outofstate": (461, 525, 612),
        "first_time_instate": (2150, 1073, 612), "sat_25": (984, 106, 287),
    }
}
