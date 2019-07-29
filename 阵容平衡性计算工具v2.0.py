#coding:utf-8
import matplotlib.pyplot as plt
import matplotlib

import configparser
import pandas
import seaborn as sns
sns.set_context("talk")
import numpy as np
import pulp
import copy
from pulp import *
import os

global doDebug
doDebug = False

#初始化线性规划条件
def setupBasicProblem(matrix,HeroList):
    prob = LpProblem("rock_paper_scissors", pulp.LpMaximize)
    #pulp.LpMaximize是目标函数求最大值还是最小值，这里是最大值，也就是求“石头剪刀布问题的最优解”

    the_vars = np.append(HeroList, (["w"]))
    #构建了一个表格索引字典lp_vars（线性规划求解中使用的矩阵索引和从文件里读到的矩阵索引不一样）
    lp_vars = LpVariable.dicts("vrs", the_vars)
    #设置线性规划初始条件（没看懂）
    prob += lpSum([lp_vars['w']]) 

    #设置每个阵容的先验使用占比范围
    #for row_strat in HeroList:
    #    prob += lpSum([1.0 * lp_vars[row_strat]]) >= matrix.loc[row_strat,'使用下限']
    #    prob += lpSum([1.0 * lp_vars[row_strat]]) <= matrix.loc[row_strat,'使用上限']

    for row_strat in HeroList:
        prob += lpSum([1.0 * lp_vars[row_strat]]) >= 0
        prob += lpSum([1.0 * lp_vars[row_strat]]) <= 1

    #设置所有阵容使用占比之和等于1
    prob += lpSum([1.0 * lp_vars[x] for x in HeroList]) == 1    
    #根据胜率矩阵设置线性规划约束条件（没看懂）
    for col_strat in HeroList:
        stratTerms = []
        for row_strat in HeroList:
            if matrix.loc[row_strat, col_strat]>=0:
                stratTerms = stratTerms + [matrix.loc[row_strat, col_strat] * lp_vars[row_strat]]
        allTerms = stratTerms + [-1 * lp_vars['w']]
        prob += lpSum(allTerms) >= 0

    return prob, lp_vars

def getWinRateNow(matrix,myArray,enemyArray):
    winRate = 0.0
    for heroMe in myArray.index.values:
        noRate = 0
        for heroEnemy in enemyArray.index.values:
            if matrix.loc[heroMe,heroEnemy]<0:
                noRate+=enemyArray.loc[heroEnemy]
        for heroEnemy in enemyArray.index.values:
            if matrix.loc[heroMe,heroEnemy]>0:
                if myArray[heroMe]>0 and enemyArray.loc[heroEnemy]>0:
                    winRate += myArray[heroMe]*enemyArray.loc[heroEnemy]*matrix.loc[heroMe,heroEnemy]/(1-noRate)
    return winRate

#计算某个阵容在特定使用占比下的策略胜率
def solveGameWithRowConstraint(matrix, rowname, constraint,HeroList,bestArray):
    print(rowname,constraint)
    #初始化问题
    #pulp.LpMaximize是目标函数求最大值还是最小值，这里是最大值，也就是求“石头剪刀布问题的最优解”
    prob = LpProblem("rock_paper_scissors", pulp.LpMaximize)
    #构建了一个表格索引字典lp_vars（线性规划求解中使用的矩阵索引和从文件里读到的矩阵索引不一样）
    the_vars = np.append(HeroList, (["w"]))
    lp_vars = LpVariable.dicts("vrs", the_vars)

    #设置线性规划求解目标函数
    winRate = []
    for heroMe in HeroList:
        noRate = 0
        for heroEnemy in bestArray.index.values:
            if matrix.loc[heroMe,heroEnemy]<0:
                noRate+=bestArray.loc[heroEnemy]
        for heroEnemy in bestArray.index.values:
            if matrix.loc[heroMe,heroEnemy]>0:
                if bestArray.loc[heroEnemy]>0:
                    winRate += lp_vars[heroMe]*bestArray.loc[heroEnemy]*matrix.loc[heroMe,heroEnemy]/(1-noRate)
    
    prob += lpSum(winRate) 
    #设置每个阵容的先验使用占比范围
    for row_strat in HeroList:
        if row_strat != rowname:
           prob += lpSum([1.0 * lp_vars[row_strat]]) >= matrix.loc[row_strat,'使用下限']
           prob += lpSum([1.0 * lp_vars[row_strat]]) <= matrix.loc[row_strat,'使用上限']

    #设置某个阵容的固定占比
    prob += lpSum(lp_vars[rowname]) == constraint
    #设置所有阵容使用占比之和等于1
    prob += lpSum([1.0 * lp_vars[x] for x in HeroList]) == 1    
    #求解问题
    prob.writeLP("rockpaperscissors.lp")
    cwd = os.getcwd()
    cwd = cwd+'\资源文件\cbc.exe'
    solver = pulp.COIN_CMD(path=cwd)
    prob.solve(solver)

    newBestArray = pandas.Series([value(lp_vars[heroName]) for heroName in HeroList],index=HeroList)
    bestWinRate = getWinRateNow(matrix,newBestArray,bestArray)

    global doDebug
    if doDebug:
        print(bestWinRate)
        print(newBestArray)
        print('-----------------')
    return prob, bestWinRate, newBestArray

#计算某个阵容的策略胜率
def getWinRates(rowname,matrix,division,HeroList,bestArray):
    #以rowname阵容的使用占比（0-1，切割division份）为循环变量，计算rowname阵容在不同使用占比下的策略得分
    probs = np.linspace(0,1,division+1)
    WinRates = pandas.Series([solveGameWithRowConstraint(matrix, rowname, p,HeroList,bestArray)[1] for p in probs], index=probs, name=rowname)
    #pandas.Series是创建数组的函数
    return WinRates

#求解一次最佳阵容
def getMetaOnce(matrix,HeroList,bestArray):
    #初始化问题
    #pulp.LpMaximize是目标函数求最大值还是最小值，这里是最大值，也就是求“石头剪刀布问题的最优解”
    prob = LpProblem("rock_paper_scissors", pulp.LpMaximize)
    #构建了一个表格索引字典lp_vars（线性规划求解中使用的矩阵索引和从文件里读到的矩阵索引不一样）
    the_vars = np.append(HeroList, (["w"]))
    lp_vars = LpVariable.dicts("vrs", the_vars)

    #设置线性规划求解目标函数
    winRate = []
    for heroMe in HeroList:
        noRate = 0
        for heroEnemy in bestArray.index.values:
            if matrix.loc[heroMe,heroEnemy]<0:
                noRate+=bestArray.loc[heroEnemy]
        tmp = []
        for heroEnemy in bestArray.index.values:
            if matrix.loc[heroMe,heroEnemy]>0:
                if bestArray.loc[heroEnemy]>0:
                    winRate += lp_vars[heroMe]*bestArray.loc[heroEnemy]*matrix.loc[heroMe,heroEnemy]/(1-noRate)
                    tmp+=lp_vars[heroMe]*bestArray.loc[heroEnemy]*matrix.loc[heroMe,heroEnemy]/(1-noRate)

    prob += lpSum(winRate) 
    #设置每个阵容的先验使用占比范围
    for row_strat in HeroList:
       prob += lpSum([1.0 * lp_vars[row_strat]]) >= matrix.loc[row_strat,'使用下限']
       prob += lpSum([1.0 * lp_vars[row_strat]]) <= matrix.loc[row_strat,'使用上限']

    #设置所有阵容使用占比之和等于1
    prob += lpSum([1.0 * lp_vars[x] for x in HeroList]) == 1.0    

    #print(prob)
    #求解问题
    prob.writeLP("rockpaperscissors.lp")
    cwd = os.getcwd()
    cwd = cwd+'\资源文件\cbc.exe'
    solver = pulp.COIN_CMD(path=cwd)
    prob.solve(solver)

    newBestArray = pandas.Series([value(lp_vars[heroName]) for heroName in HeroList],index=HeroList)
    bestWinRate = getWinRateNow(matrix,newBestArray,bestArray)
    
    if doDebug:
        print(prob)
        print(bestWinRate)
        print(newBestArray)
    return newBestArray,bestWinRate

def simulate(matrix,HeroList,bestArray,simulateNum,simulateRange,x,y,fig,axes,title):
    for i in range(1,simulateNum):
        #print("-------------------")
        print(((x*2)+y)*100+i,"-------")
        #print("当前环境：")
        #print(bestArray)
        newBestArray,bestWinRate = getMetaOnce(matrix,HeroList,bestArray)
        bestArray+=simulateRange*(newBestArray - bestArray)
        #bestArray = newBestArray

        for heroMe in newBestArray.index.values:
            if newBestArray[heroMe]>0.5:
                print("NB:",heroMe,bestArray[heroMe])
                print("NB:",bestWinRate)
    fig = bestArray.plot(ax=axes[x,y],kind='barh',stacked=True, xticks = np.linspace(0,1,11), legend=False, title=title, fontsize = 7)
    global doDebug
    if doDebug:
        print(bestArray)

    return bestArray,fig

#计算当前的环境下的最优策略(胜率矩阵，需要计算的英雄列表)
def getMeta(matrix,HeroList,nowArray,simulateNum,simulateRange):
    #初始状态，最佳策略是所有阵容均匀使用
    bestArray = nowArray

    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
    fig,axes = matplotlib.pyplot.subplots(2,2,figsize=(12,9))
    
    bestArray,fig = simulate(matrix,HeroList,bestArray,0,simulateRange,0,0,fig,axes,"初始状态")
    x_str = "迭代" + str(simulateNum) + "次"
    bestArray,fig = simulate(matrix,HeroList,bestArray,simulateNum,simulateRange,0,1,fig,axes,x_str)
    x_str = "迭代" + str(2*simulateNum) + "次"
    bestArray,fig = simulate(matrix,HeroList,bestArray,simulateNum,simulateRange,1,0,fig,axes,x_str)
    x_str = "迭代" + str(3*simulateNum) + "次"
    bestArray,fig = simulate(matrix,HeroList,bestArray,simulateNum,simulateRange,1,1,fig,axes,x_str)

    plt.tight_layout()#自动排版，避免超出显示范围
    fig.get_figure().savefig('预测环境.pdf')

    #bestArray = pandas.Series([1/len(HeroList) for heroName in HeroList],index=HeroList)
    
    return bestArray

##计算所有阵容的策略胜率（胜率矩阵，需要计算的英雄列表，0-1的切割数量）
def getAllWinRates(matrix,HeroList,nowArray,usePrediction,division,simulateNum,simulateRange):
    #计算当前的环境下的最优策略(胜率矩阵，需要计算的英雄列表)
    nowArray_copy = copy.deepcopy(nowArray)

    bestArray = getMeta(matrix,HeroList,nowArray,simulateNum,simulateRange)
    bestArray.to_csv('预测环境.csv',encoding="gbk")

    if usePrediction == False:
        bestArray = copy.deepcopy(nowArray_copy)

    #以阵容为循环变量，计算每个阵容在不同占比下面对最优策略的胜率
    return pandas.concat([getWinRates(row,matrix,division,HeroList,bestArray) for row in HeroList], axis=1)   
    #pandas.concat是合并表格的函数

##画一张图（数据，是否排序，标绿阈值，图句柄，坐标轴句柄，当前图是第几行，当前图是第几列，坐标字号，标题字号）
def printByWinRate(winRates,doSort,threshold,fig,axes,figure_x,figure_y,figureNume,axesFontSize,titleFontSize,showNum):
    intervals = winRates.apply(lambda x: pandas.Series([x[x >= threshold].first_valid_index(), x[x >= threshold].last_valid_index()], index = ['minv','maxv'])).T
    intervals['bar1'] = intervals['minv']
    intervals['bar2'] = intervals['maxv'] - intervals['minv']
    intervals['bar3'] = 1 - (intervals['bar1'] + intervals['bar2'])

    ##如果需要，根据标绿部分的情况进行排序
    #先把NaN都替换为0
    intervals = intervals.fillna(0)
    #以maxv和minv排序
    if doSort:
        intervals = intervals.sort_index(by=['maxv','minv'])
    else: #else reverse, it's weird
        intervals = intervals.reindex(index=intervals.index[::-1])

    x_str = "胜率大于" + str(threshold) + "%所需阵容占比"
    if figureNume > 1:
        fig_tmp = intervals[['bar1','bar2','bar3']].tail(showNum).plot(ax=axes[figure_x,figure_y],kind='barh',stacked=True, color=['w','g','w'], xticks = np.linspace(0,1,11), legend=False, title=x_str, fontsize = axesFontSize)
        fig_tmp.axes.title.set_size(titleFontSize)
        return fig,axes
    else:
        fig_tmp = intervals[['bar1','bar2','bar3']].tail(showNum).plot(kind='barh',stacked=True, color=['w','g','w'], xticks = np.linspace(0,1,11), legend=False, title=x_str, fontsize = axesFontSize)
        plt.title(x_str)
        return fig_tmp.get_figure()

##画图（数据，是否排序，标绿的阈值,画4图，坐标字号，标题字号）
def plotIntervals(winRates,doSort,threshold,print4fig,axesFontSize,titleFontSize,showNum):
    #设置显示中文
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

    #根据阈值threshold、threshold2，阵容策略胜率大于threshold2的部分标红，threshold和threshold2之间的标绿(标2种颜色的功能有问题……撤回了)
    
    ##画图
    if print4fig:
        #初始化一个多图格式
        fig,axes = matplotlib.pyplot.subplots(2,2,figsize=(12,9))
        #画不同胜率要求下的阵容分布（数据，是否排序，标绿阈值，图句柄，坐标轴句柄，当前图是第几行，当前图是第几列，坐标字号，标题字号）
        fig,axes = printByWinRate(winRates,doSort,threshold[0],fig,axes,0,0,4,axesFontSize,titleFontSize,showNum)
        fig,axes = printByWinRate(winRates,doSort,threshold[1],fig,axes,0,1,4,axesFontSize,titleFontSize,showNum)
        fig,axes = printByWinRate(winRates,doSort,threshold[2],fig,axes,1,0,4,axesFontSize,titleFontSize,showNum)
        fig,axes = printByWinRate(winRates,doSort,50,fig,axes,1,1,4,axesFontSize,titleFontSize,showNum)
    else:
        fig = printByWinRate(winRates,doSort,threshold[0],None,None,0,0,1,axesFontSize,titleFontSize,showNum)
    
    
    #设置格式（得先画图再设置，否则不生效）
    plt.tight_layout()#自动排版，避免超出显示范围
    plt.show()
    #plt.figure(figsize=(20,6))#想设置长宽比，但失败了
    return fig

def strToBool(str):
    if str == 'True' or str == 'true':
        return True
    else:
        return False

def main():
    #读取配置文件
    cf = configparser.ConfigParser()
    cf.read("配置参数.ini",encoding='utf-8-sig')
    #####可选参数
    #True,False
    #胜率矩阵以1代表100%，反之以100代表100%
    global doDebug
    doDebug = strToBool(cf.get("调试配置", "调试模式"))

    inputFileName = cf.get("输入输出配置", "胜率矩阵文件")
    inputEncode = cf.get("输入输出配置", "胜率矩阵文件编码格式")
    inputCountFileName = cf.get("输入输出配置", "场次矩阵文件")
    inputCountEncode = cf.get("输入输出配置", "场次矩阵文件编码格式")

    oneWinRate = strToBool(cf.get("输入输出配置", "胜率以1为最大"))
    rowWinRate = strToBool(cf.get("输入输出配置", "横向排布"))
    outputFileName = cf.get("输入输出配置", "输出文件名")

    simulateNum = int(cf.get("计算配置", "环境预测次数"))
    simulateRange = float(cf.get("计算配置", "环境预测步长"))

    stepRange = int(cf.get("计算配置", "计算密度"))

    minGameNum = int(cf.get("计算配置", "有效对局场次阈值"))

    usePrediction = strToBool(cf.get("计算配置", "使用迭代预测的环境"))

    onlyPaint = strToBool(cf.get("画图配置", "只画图"))
    paintFileName = cf.get("画图配置", "画图用数据")
    print4fig = strToBool(cf.get("画图配置", "画4图"))
    greenThreshold1 = float(cf.get("画图配置", "标绿胜率阈值1"))
    greenThreshold2 = float(cf.get("画图配置", "标绿胜率阈值2"))
    greenThreshold3 = float(cf.get("画图配置", "标绿胜率阈值3"))
    greenThreshold4 = float(cf.get("画图配置", "标绿胜率阈值4"))
    greenThreshold = [greenThreshold1,greenThreshold2,greenThreshold3,greenThreshold4]
    doSort = strToBool(cf.get("画图配置", "是否排序"))
    showNum = int(cf.get("画图配置", "显示数量"))
    axesFontSize = int(cf.get("画图配置", "坐标轴字号"))
    titleFontSize = int(cf.get("画图配置", "标题字号"))
    #matplotlib.use('PS')

    if onlyPaint:
        
        allWinRates = pandas.read_csv(paintFileName, index_col = 0,encoding="gbk")
    else:
        #csv第一行是列名
        matchups = pandas.read_csv(inputFileName, index_col = 0,encoding=inputEncode)
        matchups.index.name = "row_char"
        HeroList = matchups.index.values
        #print(matchups)
        
        ##csv第一行不是列名
        #matchups = pandas.read_csv('matchups.csv', header=None, index_col = 0,encoding="gbk")
        #matchups.columns = matchups.index.values
        #matchups.columns.name = "col_char"

        #处理胜率矩阵部分
        print(matchups)
        matchupPayoffs = copy.deepcopy(matchups)
        for row_strat in matchupPayoffs.index.values:
            for low_start in matchupPayoffs.index.values:
                if rowWinRate:
                    if oneWinRate:
                        matchupPayoffs.loc[row_strat,low_start] = matchups.loc[row_strat,low_start]*100
                    else:
                        matchupPayoffs.loc[row_strat,low_start] = matchups.loc[row_strat,low_start]
                else:
                    if oneWinRate:
                        matchupPayoffs.loc[row_strat,low_start] = matchups.loc[low_start,row_strat]*100
                    else:
                        matchupPayoffs.loc[row_strat,low_start] = matchups.loc[low_start,row_strat]

        gameNumTotal = 0
        gameNumHero = {}
        if minGameNum>-1:
            GameNum = pandas.read_csv(inputCountFileName, index_col = 0,encoding=inputCountEncode)
            print(GameNum.index.values)
            GameNum = GameNum.fillna(-1)
            for row_strat in HeroList:
                gameNumHero[row_strat] = 0
                for low_start in HeroList:
                    if GameNum.loc[row_strat,low_start] >-1:
                        gameNumTotal += GameNum.loc[row_strat,low_start]
                        gameNumHero[row_strat] += GameNum.loc[row_strat,low_start]
                    if GameNum.loc[row_strat,low_start]<minGameNum:
                        matchupPayoffs.loc[row_strat,low_start] = -1;
                    if row_strat==low_start:
                        matchupPayoffs.loc[row_strat,low_start] = 50;
        else:
            for row_strat in HeroList:
                gameNumHero[row_strat] = 1
                gameNumTotal += 1

        #将不合规的数值替换为-1
        matchupPayoffs=matchupPayoffs.fillna(-1)

        for heroName in matchupPayoffs.index.values:
            if heroName not in HeroList:
                matchupPayoffs=matchupPayoffs.drop([heroName],axis=0)
                matchupPayoffs=matchupPayoffs.drop([heroName],axis=1)
                print(heroName)
        #输出数据清洗之后的结果
        matchupPayoffs.to_csv('中间数据.csv',encoding="gbk")
        print(matchupPayoffs) 

        nowArray = pandas.Series([gameNumHero[heroName]/gameNumTotal for heroName in HeroList],index=HeroList)
        #计算策略胜率（胜率矩阵，需要计算的英雄列表，0-1的切割数量）
        allWinRates = getAllWinRates(matchupPayoffs,HeroList,nowArray,usePrediction,stepRange,simulateNum,simulateRange)
        
        #移除超出使用约束条件的结果
        minUse = 1/stepRange
        probs = np.linspace(0,1,stepRange+1)
        for row_strat in HeroList:
            for low_start in probs:
                if low_start > minUse and low_start > matchupPayoffs.loc[row_strat,'使用上限']:
                    allWinRates.loc[low_start,row_strat] = 0;
                if low_start > minUse and low_start < matchupPayoffs.loc[row_strat,'使用下限']:
                    allWinRates.loc[low_start,row_strat] = 0;

        #输出到csv表
        allWinRates.to_csv(outputFileName+'-表.csv',encoding="gbk")

    #显示策略胜率
    print(allWinRates)
    #显示最高策略胜率（方便调整画图参数）
    maxWinRates = allWinRates.stack().max()
    minWinRates = allWinRates.stack().min()
    print("最高胜率:",maxWinRates,",最低胜率:",minWinRates)
    greenThreshold = [int(maxWinRates-1),int(maxWinRates-2),int(maxWinRates-3),int(maxWinRates-5)]

    #画图（数据，是否排序，标绿的阈值,画4图，坐标字号，标题字号，显示数量）
    img = plotIntervals(allWinRates,doSort,greenThreshold,print4fig,axesFontSize,titleFontSize,showNum)
    img.savefig(outputFileName+'-图.pdf')


if __name__ == "__main__":
    main()
    #command = input("按回车结束")