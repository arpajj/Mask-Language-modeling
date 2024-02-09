all_data = {"entailment": [[84,63,80,59,68,27,67,23,24,29],[87,38,90,49,63,21,65,29,33,38], [93,42,43,55,43,65,43,34,38,29,34]],
            "neutral": [[0,0,0,0,0,0,0,0,0,0],[0,0.3,0.3,0.6,0,0,0,0,0,0,0,0.6,0,0,0,0],[0.3,0.3,0,0.3,0,0,0,0,0,0.3,0,0.3,0,0,0,0],],
            "contradiction": [[52,31,53,35,57,70,63,76,99,99],[46,60,29,48,70,66,64,63,95,92],[22,61,55,60,59,46,58,66,57,96,95],],
            }

# all_data = {"entailment": [[84,63,80,59,68],[87,38,90,49,63], [93,42,43,55,43,]],
#             "neutral": [[0,0,0,0,0],[0,0.3,0.3,0.6,0,],[0.3,0.3,0,0.3,0,],],
#             "contradiction": [[52,31,53,35,57],[46,60,29,48,70],[22,61,55,60,59],],
#             }

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.DataFrame(all_data)
# 将数据按照时间分为上半月和下半月
df1 = df["entailment"]
df2 = df["neutral"]
df3 = df["contradiction"]

y1 = df1.values
y2 = df2.values
y3 = df3.values

x1 = np.arange(1,15,5)
print(x1)
x2 = x1+1
x3 = x1+2

box1 = plt.boxplot(y1,positions=x1,patch_artist=True,showmeans=True,
            boxprops={"facecolor": "C0",
                      "edgecolor": "grey",
                      "linewidth": 0.5},
            medianprops={"color": "k", "linewidth": 0.5},
            meanprops={'marker':'+',
                       'markerfacecolor':'k',
                       'markeredgecolor':'k',
                       'markersize':5})
box2 = plt.boxplot(y2,positions=x2,patch_artist=True,showmeans=True,
            boxprops={"facecolor": "C1",
                      "edgecolor": "grey",
                      "linewidth": 0.5},
            medianprops={"color": "k", "linewidth": 0.5},
            meanprops={'marker':'+',
                       'markerfacecolor':'k',
                       'markeredgecolor':'k',
                       'markersize':5})

box3 = plt.boxplot(y3,positions=x3,patch_artist=True,showmeans=True,
            boxprops={"facecolor": "C2",
                      "edgecolor": "grey",
                      "linewidth": 0.5},
            medianprops={"color": "k", "linewidth": 0.5},
            meanprops={'marker':'+',
                       'markerfacecolor':'k',
                       'markeredgecolor':'k',
                       'markersize':5})

city = df.columns
plt.xticks([2,7,12], ["SICK","SNLI","MNLI"], fontsize=11)
# plt.ylim(10,45)
plt.ylabel('Accuracy %',fontsize=11)
plt.grid(axis='y',ls='--',alpha=0.8)

# 给箱体添加图例，每类箱线图中取第一个颜色块用于代表图例
plt.legend(handles=[box1['boxes'][0],box2['boxes'][0],box3['boxes'][0]],labels=['entailment','neutral','contradiction'])

# plt.tight_layout()
plt.savefig('boxplot.png',dpi=600)
plt.show()