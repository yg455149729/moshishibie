import pandas as pd


def getResource(csvpath):
    frame = pd.read_csv(csvpath)
    return frame


def getUserGraph(frame, userID=1):
    itemList = list(set(frame[frame['userId']==userID]['movieId']))
    graphDict = {'i'+str(item): 1 for item in itemList}
    return graphDict


def getItemGraph(frame, itemID=1):
    userList = list(set(frame[frame['movieId']==itemID]['userId']))
    graphDict = {'u'+str(user): 1 for user in userList}
    return graphDict


def initGraph(frame):
    userList = list(set(frame['userId']))
    itemList = list(set(frame['movieId']))
    G = {'u'+str(user): getUserGraph(frame, user) for user in userList}
    for item in itemList: G['i'+str(item)] = getItemGraph(frame, item)
    return G


def personalRank(G, alpha, userID, iterCount=20):
    rank = {g: 0 for g in G.keys()}
    rank['u'+str(userID)] = 1                                       #根节点为起点选择概率为1,其他顶点为0
    for k in range(iterCount):
        tmp = {g: 0 for g in G.keys()}

        for i, ri in G.items():                                     #遍历每一个顶点
            for j, wij in ri.items():                               #遍历每个顶点连接的顶点
                tmp[j] += alpha * rank[i] / len(ri)
        tmp['u' + str(userID)] += 1 - alpha                    #根顶点r=1，加上1-alpha
        rank = tmp
    series = pd.Series(list(rank.values()), index=list(rank.keys()))
    series = series.sort_values(ascending=False)
    return series                                                   #返回排序后的series


def recommend(frame, series, userID, TopN=10):
    '''
    推荐TopN个用户没有评分的物品
    :param frame: ratings数据
    :param series: series
    :param userID: 目标用户
    :param TopN: TopN
    :return: 推荐物品
    '''
    itemList = ['i'+str(i) for i in list(set(frame[frame['userId']==userID]['movieId']))]
    recommendList = [{u: series[u]} for u in list(series.index) if u not in itemList and 'u' not in u]
    return recommendList[:TopN]


if __name__ == '__main__':
    frame = getResource('ml-latest-small/ratings.csv')
    G = initGraph(frame)
    for i in range(1,20):
        s = personalRank(G, 0.6, 1, iterCount=i )
        r = recommend(frame, s, 1, TopN=10)
        print(r)