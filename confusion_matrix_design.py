import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set(font_scale=1.5)
'''
array = [[15,	0,	15,	0,	0,	0,	3],  # when input was A, prediction was A for 9 times, B for 1 time
         [2	,33,	11,	1,	3,	0,	1], # when input was B, prediction was A for 1 time, B for 15 times, C for 3 times
         [1,	1,	95,	0,	6,	1,	6], # when input was C, prediction was A for 5 times, C for 24 times, D for 1 time
         [1	,0	,1,	9,	1,	0,	0],
         [0,	0,	16,	0,	633,	0,	22],
         [0,	1,	1,	0,	1,	11,	0],
         [0	,0,	13,	0	,10,	0,	88]] # when input was D, prediction was B for 4 times, C for 1 time, D for 15 timess
         '''
array = [[92,4,4],  # when input was A, prediction was A for 9 times, B for 1 time
         [0,99,	1], # when input was B, prediction was A for 1 time, B for 15 times, C for 3 times
         [1,	4,95], # when input was C, prediction was A for 5 times, C for 24 times, D for 1 time
         ] # when input was D, prediction was B for 4 times, C for 1 time, D for 15 timess

#df_cm = pd.DataFrame(array, index = [i for i in ["AK",'BCC','BK','Der','MN','VL','Mel']],
 #                 columns = [i for i in ["AK",'BCC','BK','Der','MN','VL','Mel']])
df_cm = pd.DataFrame(array, index = [i for i in ["COVID-19",'Normal','Pneumonia']],
                  columns = [i for i in ["COVID-19",'Normal','Pneumonia']])
plt.figure(figsize = (10,10))
plt.rcParams.update({'font.size': 24})
plt.title('Residual Attention MobileNet')
sns.heatmap(df_cm, annot=True,fmt="d")
plt.show()
'''

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
data = {'y_Predicted': [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0],
        'y_Actual':    [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0]
        }

df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'], margins = True)
plt.figure(figsize = (10,10))
plt.title('Atrous InceptionV3')

sn.heatmap(confusion_matrix, annot=True)

plt.show()
'''