"""
Params library
Define custom parameters for main app
"""

import pandas as pd
import pickle


########### Application log level ##########
LOG_LEVEL = 'DEBUG'
# LOG_LEVEL = 'INFO'

########## App configuration ##########
# APP_TITLE = 'Mortgage Loans'
# MAIN_HEADER = 'Predicting Mortgage Loan Approval'
# # MAIN_IMAGE = 'ames_welcome.jpeg'
# SOURCE_URL = 'https://www.kaggle.com/burak3ergun/loan-data-set'
# GITHUB_LINK = 'https://github.com/Dangee/log-reg-loans-simple'


# color palette
Viridis=[
"#440154", "#440558", "#450a5c", "#450e60", "#451465", "#461969",
"#461d6d", "#462372", "#472775", "#472c7a", "#46307c", "#45337d",
"#433880", "#423c81", "#404184", "#3f4686", "#3d4a88", "#3c4f8a",
"#3b518b", "#39558b", "#37598c", "#365c8c", "#34608c", "#33638d",
"#31678d", "#2f6b8d", "#2d6e8e", "#2c718e", "#2b748e", "#29788e",
"#287c8e", "#277f8e", "#25848d", "#24878d", "#238b8d", "#218f8d",
"#21918d", "#22958b", "#23988a", "#239b89", "#249f87", "#25a186",
"#25a584", "#26a883", "#27ab82", "#29ae80", "#2eb17d", "#35b479",
"#3cb875", "#42bb72", "#49be6e", "#4ec16b", "#55c467", "#5cc863",
"#61c960", "#6bcc5a", "#72ce55", "#7cd04f", "#85d349", "#8dd544",
"#97d73e", "#9ed93a", "#a8db34", "#b0dd31", "#b8de30", "#c3df2e",
"#cbe02d", "#d6e22b", "#e1e329", "#eae428", "#f5e626", "#fde725"]