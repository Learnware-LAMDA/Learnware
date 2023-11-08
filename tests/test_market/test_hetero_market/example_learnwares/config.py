input_shape_list=[20, 30]    # 20-input shape of example learnware 0, 30-input shape of example learnware 1

input_description_list=[
    {
        "Dimension": 20,
        "Description": {     # medical description
            "0": "baseline value: Baseline Fetal Heart Rate (FHR)",
            "1": "accelerations: Number of accelerations per second",
            "2": "fetal_movement: Number of fetal movements per second",
            "3": "uterine_contractions: Number of uterine contractions per second",
            "4": "light_decelerations: Number of LDs per second",
            "5": "severe_decelerations: Number of SDs per second",
            "6": "prolongued_decelerations: Number of PDs per second",
            "7": "abnormal_short_term_variability: Percentage of time with abnormal short term variability",
            "8": "mean_value_of_short_term_variability: Mean value of short term variability",
            "9": "percentage_of_time_with_abnormal_long_term_variability: Percentage of time with abnormal long term variability",
            "10": "mean_value_of_long_term_variability: Mean value of long term variability",
            "11": "histogram_width: Width of the histogram made using all values from a record",
            "12": "histogram_min: Histogram minimum value",
            "13": "histogram_max: Histogram maximum value",
            "14": "histogram_number_of_peaks: Number of peaks in the exam histogram",
            "15": "histogram_number_of_zeroes: Number of zeroes in the exam histogram",
            "16": "histogram_mode: Hist mode",
            "17": "histogram_mean: Hist mean",
            "18": "histogram_median: Hist Median",
            "19": "histogram_variance: Hist variance"
        },
    },
    {
        "Dimension": 30,
        "Description": {     # business description
            "0": "This is a consecutive month number, used for convenience. For example, January 2013 is 0, February 2013 is 1,..., October 2015 is 33.",
            "1": "This is the unique identifier for each shop.",
            "2": "This is the unique identifier for each item.",
            "3": "This is the code representing the city where the shop is located.",
            "4": "This is the unique identifier for the category of the item.",
            "5": "This is the code representing the type of the item.",
            "6": "This is the code representing the subtype of the item.",
            "7": "This is the number of this type of item sold in the shop one month ago.",
            "8": "This is the number of this type of item sold in the shop two months ago.",
            "9": "This is the number of this type of item sold in the shop three months ago.",
            "10": "This is the number of this type of item sold in the shop six months ago.",
            "11": "This is the number of this type of item sold in the shop twelve months ago.",
            "12": "This is the average count of items sold one month ago.",
            "13": "This is the average count of this type of item sold one month ago.",
            "14": "This is the average count of this type of item sold two months ago.",
            "15": "This is the average count of this type of item sold three months ago.",
            "16": "This is the average count of this type of item sold six months ago.",
            "17": "This is the average count of this type of item sold twelve months ago.",
            "18": "This is the average count of items sold in the shop one month ago.",
            "19": "This is the average count of items sold in the shop two months ago.",
            "20": "This is the average count of items sold in the shop three months ago.",
            "21": "This is the average count of items sold in the shop six months ago.",
            "22": "This is the average count of items sold in the shop twelve months ago.",
            "23": "This is the average count of items in the same category sold one month ago.",
            "24": "This is the average count of items in the same category sold in the shop one month ago.",
            "25": "This is the average count of items of the same type sold in the shop one month ago.",
            "26": "This is the average count of items of the same subtype sold in the shop one month ago.",
            "27": "This is the average count of items sold in the same city one month ago.",
            "28": "This is the average count of this type of item sold in the same city one month ago.",
            "29": "This is the average count of items of the same type sold one month ago."
        },
    },
    
]

output_description_list=[
    {
        "Dimension": 1,
        "Description": {     # medical description
            "0": "length of stay: Length of hospital stay (days)"
        },
    },
    {
        "Dimension": 1,
        "Description": {     # business description
            "0": "sales of the item in the next day: Number of items sold in the next day"
        },
    },
    
]

user_description_list=[
    {
        "Dimension": 15,
        "Description": {     # medical description
            "0": "Whether the patient is on thyroxine medication (0: No, 1: Yes)",
            "1": "Whether the patient has been queried about thyroxine medication (0: No, 1: Yes)",
            "2": "Whether the patient is on antithyroid medication (0: No, 1: Yes)",
            "3": "Whether the patient has undergone thyroid surgery (0: No, 1: Yes)",
            "4": "Whether the patient has been queried about hypothyroidism (0: No, 1: Yes)",
            "5": "Whether the patient has been queried about hyperthyroidism (0: No, 1: Yes)",
            "6": "Whether the patient is pregnant (0: No, 1: Yes)",
            "7": "Whether the patient is sick (0: No, 1: Yes)",
            "8": "Whether the patient has a tumor (0: No, 1: Yes)",
            "9": "Whether the patient is taking lithium (0: No, 1: Yes)",
            "10": "Whether the patient has a goitre (enlarged thyroid gland) (0: No, 1: Yes)",
            "11": "Whether TSH (Thyroid Stimulating Hormone) level has been measured (0: No, 1: Yes)",
            "12": "Whether T3 (Triiodothyronine) level has been measured (0: No, 1: Yes)",
            "13": "Whether TT4 (Total Thyroxine) level has been measured (0: No, 1: Yes)",
            "14": "Whether T4U (Thyroxine Utilization) level has been measured (0: No, 1: Yes)"
        },
    }
]