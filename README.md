# 경기도 무료급식 및 도시락 배달소 최적입지 선정

## 주제 선정 이유
- 대한민국 심각한 노인 빈곤율(OECD 국가중 43.4%로 가장 높다)
- 경기도 도시지역과 농촌지역 혼재, 복지 불균형 심각

- 무료급식 및 도시락 배달소의 최적 입지를 선정 함으로써 `노인 복지 증진` 및 지역 `복지 불균형`을 해결하고자 함

## 프로젝트 방법
### **1. Clustering**
- 다양한 노인 Data를 기준으로 Clustering을 진행하여 타겟 클러스터를 선정하였음
- `K-means Clustering`을 사용하여 분석 진행
- SSE(Sum of squared errors), Calinski Harabasz score, Silhouette score 를 보고 적절한 k 결정 

``` python
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score

import seaborn as sns
import matplotlib.pyplot as plt

search_range = range(1, 21)
report = {}
for k in search_range:
    temp_dict = {}
    kmeans = KMeans(init='k-means++',
                    algorithm='auto',
                    n_clusters=k,
                    max_iter=1000,
                    random_state=1,
                    verbose=0).fit(data_prime)
    inertia = kmeans.inertia_
    temp_dict['Sum of squared error'] = inertia
    try:
        cluster = kmeans.predict(data_prime)
        chs = calinski_harabasz_score(data_prime, cluster)
        ss = silhouette_score(data_prime, cluster)
        temp_dict['Calinski Harabasz Score'] = chs
        temp_dict['Silhouette Score'] = ss
        report[k] = temp_dict
    except:
        report[k] = temp_dict

report_df = pd.DataFrame(report).T
report_df.plot(figsize=(15, 10),
               xticks=search_range,
               grid=True,
               title=f'Selecting optimal "K"',
               subplots=True,
               marker='o',
               sharex=True)
plt.tight_layout()
```
<img width="400" alt="select_k" src="https://user-images.githubusercontent.com/118239192/210171651-86dd47cc-f711-4f0e-b47e-0d3c6660da97.png">

지표들을 확인 후 본 프로젝트에서는 **k=5**로 결정

- 변수들을 변경해가며 K-means Clustering을 실시하고, Cluter별 특성을 분석하여
  타겟 지역을 선정 하였다

```python
import seaborn as ans
import matplotlib.pyplot as plt
x = data_prime['Variable1']
y = data_prime['Variable2']
n = a

fig, ax = plt.subplots()
ax.scatter(x, y, c=data_prime['k_means_cluster'])
ax.scatter(x=centroids.iloc[:,0], y=centroids.iloc[:,1], marker='D', c='r') 
plt.xlabel('Variable1')
plt.ylabel('Variable2')

for i, txt in enumerate(n):
    ax.annotate(txt, (x[i], y[i]))
```
<img width="300" alt="clustering1" src="https://user-images.githubusercontent.com/118239192/210171560-9439687f-dff6-40d8-ae0b-7432749e31ca.png"><img width="300" alt="clustering2" src="https://user-images.githubusercontent.com/118239192/210171561-6648a9d5-adc6-44fb-9221-193ae8b1b4cc.png">

- 여러 Clustering을 해본 결과 타겟 구역을 `용인시`로 결정하였음 &nbsp;

 (급식소대비 노인수가 많았고 인구수가 많은 시중 가장 노인빈곤율이 높았다)

- 용인시의 경우 수지구, 기흥구, 처인구로 이루어져 있으며 수지구, 기흥구의 경우 인구 밀도가 높고 복지가 잘 되어있는 도심 지역이기 때문에 본 프로젝트에서의 최적입지는 처인구에서만 분석하기로 결정하였음.
- 

### MSD & MCLP 알고리즘을 사용한 최적입지 선정
 - MSD 알고리즘(Minmizing Sum Of Distances)
    - 가중치를 고려한 거리의 합을 최소로 하는 최적입지 선정 알고리즘
    - `거리 기반 알고리즘`
    - 시설의 개수가 1개로 고정 (본 프로젝트에서는 처인구를 인구밀도와 인구수가 비슷하게 세 지역으로 구분 후 사용)
&nbsp;

 - MCLP 알고리즘(Maximal Covering Location Problem)
    - 시설의 개수가 결정되어 있을 때 수요를 최대한 해결할 수 있는 최적입지 선정 알고리즘
    - `수요 기반 알고리즘`

    MSD 알고리즘과 MCLP 알고리즘의 최적입지가 가장 비슷한 지역이 `거리와 수요를 모두 고려`한 
    최종 입지로 판단하고, 최우선순위 입지로 선정하였다.

#### **MSD 알고리즘을 사용한 최적입지**
```python

xlist=[]
ylist=[]
# n은 지역 나누는 간격 수 
for i in range(n):
  xlist.append(Minimum latitude+ i*(min-max of latitude/n))

for i in range(n):
  ylist.append(Minimum longitude+ i*(min-max of longtitude/n))

xylist=[]    
for i in range(n):
  for k in range(n):
    m=xlist[i]
    n=ylist[k]
    l=[m,n]
    xylist.append(l) 

goal= 1000000000000000000000 # Any large Number
answer=0
for i in range(n**2):  
  sum=0
  
  for k in range(m): # m = 타겟 구역의 행정구분 지역 수     
    num=(((xylist[i][0]-x[k])*(xylist[i][0]-x[k])+(xylist[i][1]-y[k])*(xylist[i][1]-y[k]))**(1/2))*z[k]
    sum=sum+num
  if sum<goal:
    goal=sum
    answer=i  
print(goal)
print(answer)
```
- n의 값에 따라 최적입지가 바뀌는 휴리스틱 알고리즘 이기 때문에 n의 값을 늘려가며 오차를 줄인다.
- MSD알고리즘의 최적 입지는
    - 경기도 용인시 처인구 이동면 덕성리 726-5 [37.172372761974,127.205807892381] (black)
    - 경기도용인시 처인구 포곡읍 전대리 192-38 [37.285433578742,127.219621332] (red)
    - 경기도 용인시 처인구 원삼면 사암리 726-5가 [37.197471978341,127.28606468906999] (yellow)
<img width="292" alt="MSD" src="https://user-images.githubusercontent.com/118239192/210173835-fb406092-319b-4604-aac1-ada69a85bcf0.png">

#### **MCLP알고리즘을 사용한 최적입지**
```python
import numpy as np
from scipy.spatial import distance_matrix
from gurobipy import *
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, Point
from numpy import random

def generate_candidate_sites(points,M=100):
    '''
    Generate M candidate sites with the convex hull of a point set
    Input:
        points: a Numpy array with shape of (N,2)
        M: the number of candidate sites to generate
    Return:
        sites: a Numpy array with shape of (M,2)
    '''
    hull = ConvexHull(points)
    polygon_points = points[hull.vertices]
    poly = Polygon(polygon_points)
    min_x, min_y, max_x, max_y = poly.bounds
    sites = []
    while len(sites) < M:
        random_point = Point([random.uniform(min_x, max_x),
                             random.uniform(min_y, max_y)])
        if (random_point.within(poly)):
            sites.append(random_point)
    return np.array([(p.x,p.y) for p in sites])

    """
    Solve maximum covering location problem
    Input:
        points: input points, Numpy array in shape of [N,2]
        K: the number of sites to select
        radius: the radius of circle
        M: the number of candidate sites, which will randomly generated inside
        the ConvexHull wrapped by the polygon
    Return:
        opt_sites: locations K optimal sites, Numpy array in shape of [K,2]
        f: the optimal value of the objective function
    """
def mclp(points,K,radius,M):
    print('----- Configurations -----')
    print('  Number of points %g' % points.shape[0])
    print('  K %g' % K)
    print('  Radius %g' % radius)
    print('  M %g' % M)
    import time
    start = time.time()
    sites = generate_candidate_sites(points,M)
    J = sites.shape[0]
    I = points.shape[0]
    D = distance_matrix(points,sites)
    mask1 = D<=radius
    D[mask1]=1
    D[~mask1]=0
    # Build model
    m = Model()
    # Add variables
    x = {}
    y = {}
    for i in range(I):
      y[i] = m.addVar(vtype=GRB.BINARY, name="y%d" % i)
    for j in range(J):
      x[j] = m.addVar(vtype=GRB.BINARY, name="x%d" % j)

    m.update()
    # Add constraints
    m.addConstr(quicksum(x[j] for j in range(J)) == K)

    for i in range(I):
        m.addConstr(quicksum(x[j] for j in np.where(D[i]==1)[0]) >= y[i])

    m.setObjective(quicksum(y[i]for i in range(I)),GRB.MAXIMIZE)
    m.setParam('OutputFlag', 0)
    m.optimize()
    end = time.time()
    print('----- Output -----')
    print('  Running time : %s seconds' % float(end-start))
    print('  Optimal coverage points: %g' % m.objVal)
    
    solution = []
    if m.status == GRB.Status.OPTIMAL:
        for v in m.getVars():
            # print v.varName,v.x
            if v.x==1 and v.varName[0]=="x":
               solution.append(int(v.varName[1:]))
    opt_sites = sites[solution]
    return opt_sites,m.objVal

def plot_input(points):
    '''
    Plot the result
    Input:
        points: input points, Numpy array in shape of [N,2]
        opt_sites: locations K optimal sites, Numpy array in shape of [K,2]
        radius: the radius of circle
    '''
    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=(8,8))
    plt.scatter(points[:,0],points[:,1],c='C0')
    ax = plt.gca()
    ax.axis('equal')
    ax.tick_params(axis='both',left=False, top=False, right=False,
                       bottom=False, labelleft=False, labeltop=False,
                       labelright=False, labelbottom=False)

def plot_result(points,opt_sites,radius):
    '''
    Plot the result
    Input:
        points: input points, Numpy array in shape of [N,2]
        opt_sites: locations K optimal sites, Numpy array in shape of [K,2]
        radius: the radius of circle
    '''
    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=(8,8))
    plt.scatter(points[:,0],points[:,1],c='C0')
    ax = plt.gca()
    plt.scatter(opt_sites[:,0],opt_sites[:,1],c='C1',marker='+')
    for site in opt_sites:
        circle = plt.Circle(site, radius, color='C1',fill=False,lw=2)
        ax.add_artist(circle)
    ax.axis('equal')
    ax.tick_params(axis='both',left=False, top=False, right=False,
                       bottom=False, labelleft=False, labeltop=False,
                       labelright=False, labelbottom=False)
    return opt_sites[:,0],opt_sites[:,1]
```
- 시설 개수 k는 우수 무료급식 지역 성남시 급식 커버율을 토대로 용인시에서 계산, k=3으로 결정
- 도시락 배달 범위 Radius는 실제 경기도 마북동 무료 도시락 배달을 토대로 사용, R= 0.05
- MCLP 알고리즘 최적 입지는
    - 경기도 용인시 처인구 이동면 시미리 [37.15200452,127.19955221]
    - 경기도 용인시 처인구 양지면 주북리 364-27 [37.24880806,127.2524786]
    - 경기도 용인시 처인구 백암면 근창리 407[37.15747446,127.3566083]
<img width="300" alt="MCLP" src="https://user-images.githubusercontent.com/118239192/210173881-c3ace181-1f5a-46ba-b457-b0a06724f8fc.png">


#### **최우선 최종입지 선정**
- MSD의 <용인시 처인구 이동면 덕성리 726-5>와 MCLP의경기도 용인시 처인구 이동면 시미리가 가장 유사
- 따라서 두 지역 사이에서 도시락 배달/무료급식이 적절히 이루어지기 위해선 `교통이 편리`해야함
- 두 지역 사이에서 교통이 가장 편리한 `용인 공용 버스터미널` 근방을 최우선 최종입지로 선정함

<img width="400" alt="Final" src="https://user-images.githubusercontent.com/118239192/210174910-e71cb5f4-ed48-4a8c-870e-a74437f151b3.png">
